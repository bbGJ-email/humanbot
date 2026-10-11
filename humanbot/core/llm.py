"""AI 大脑层：OpenAI 兼容接口 + 多供应商路由 + 成本统计。

策划案 §8 / §15.6：
- 所有业务调用统一走这里，按 task 路由到不同 provider
- reply_generate 用强模型，其余结构化小任务用便宜模型
- 记录 token 与成本，支持日预算熔断
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import requests

from ..settings import (
    ALL_TASKS,
    TASK_ANTIAI,
    TASK_EMBEDDING,
    TASK_EMOTION,
    TASK_INTENT,
    TASK_MEMORY,
    TASK_REPLY,
    ModelConfig,
    ProviderConfig,
    Settings,
)
from .crypto import decrypt
from .logging_setup import BUS, get_logger

log = get_logger("llm")

# 粗略计价（USD / 1K tokens），用于成本估算；未知模型按兜底价
_PRICE = {
    "default": (0.0002, 0.0006),
}


def _estimate_cost(model: str, pt: int, ct: int) -> float:
    pin, pout = _PRICE.get(model, _PRICE["default"])
    return round(pt / 1000 * pin + ct / 1000 * pout, 6)


class LLMError(Exception):
    pass


class BudgetExceeded(LLMError):
    pass


# 换一个供应商就能救回来的错误特征。
# 免费模型站最常见的失败是 429（并发限流），其次是 5xx 网关抖动和超时。
_RETRYABLE_HINTS = (
    "429", "500", "502", "503", "504",
    "rate limit", "too many requests", "overload", "overloaded",
    "timeout", "timed out", "超时", "网络错误",
    "返回空内容",          # 推理型模型把额度烧在 reasoning 上，content 为空
)


def is_retryable(exc: BaseException) -> bool:
    """判断这个错误值不值得换供应商重试。

    只有"不是我们自己的错"才重试：限流 / 网关抖动 / 超时。
    配置错误（key 无效、模型名不存在）重试也是白费，直接抛。
    """
    s = str(exc).lower()
    return any(h in s for h in _RETRYABLE_HINTS)


class Provider:
    """单个 OpenAI 兼容供应商的薄封装。"""

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg

    # -- 配置校验 ---------------------------------------------------------

    def ready(self) -> Tuple[bool, str]:
        """检查配置是否可用。返回 (可用, 原因)。

        这里拦住"看起来配了、其实是空的"的供应商，避免真正发出
        一个必然卡住或必然 401 的请求。
        """
        if not (self.cfg.base_url or "").strip():
            return False, "base_url 为空"
        if not (self.cfg.base_url or "").startswith(("http://", "https://")):
            return False, "base_url 必须以 http:// 或 https:// 开头"
        if not (self.cfg.api_key or "").strip():
            return False, "api_key 为空"
        if not (self.cfg.model or "").strip():
            return False, "model 为空"
        return True, ""

    @property
    def base_url(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, Any]], *, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None, json_mode: bool = False,
             timeout: Optional[int] = None) -> Dict[str, Any]:
        """同步对话。返回 {text, prompt_tokens, completion_tokens, raw}."""
        ok, why = self.ready()
        if not ok:
            raise LLMError(f"供应商 {self.cfg.id} 不可用：{why}")

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        try:
            resp = requests.post(
                url, headers=self._headers(), json=payload,
                timeout=timeout or self.cfg.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMError(f"请求超时（{timeout or self.cfg.timeout}s）") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"网络错误: {exc}") from exc

        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"响应不是合法 JSON: {resp.text[:200]}") from exc
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"响应结构异常: {str(data)[:300]}") from exc
        usage = data.get("usage") or {}
        text = text.strip()
        if not text:
            # 少数推理型模型会把 max_tokens 全烧在 reasoning 上，content 直接为空。
            # 这种情况如果放过，上层就会"已读不回"，所以当成可重试错误抛出，
            # 交给故障转移链换一个供应商。
            finish = ""
            try:
                finish = data["choices"][0].get("finish_reason") or ""
            except Exception:
                pass
            raise LLMError(f"模型返回空内容（finish_reason={finish or '?'}）")
        return {
            "text": text,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "raw": data,
        }

    def embed(self, texts: List[str], model: str = "",
              timeout: Optional[int] = None) -> List[List[float]]:
        """调用 embeddings 接口。"""
        ok, why = self.ready()
        if not ok:
            raise LLMError(f"供应商 {self.cfg.id} 不可用：{why}")

        payload = {"model": model or self.cfg.embedding_model or self.cfg.model,
                   "input": texts}
        url = f"{self.base_url}/embeddings"
        # 嵌入是可选增强，超时给短一点，避免拖住主链路
        to = timeout or min(int(self.cfg.timeout or 60), 20)
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=to)
        except requests.exceptions.Timeout as exc:
            raise LLMError(f"embeddings 请求超时（{to}s）") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"embeddings 网络错误: {exc}") from exc

        if resp.status_code >= 400:
            raise LLMError(f"embeddings HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"embeddings 响应不是合法 JSON: {resp.text[:200]}") from exc
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [it.get("embedding") or [] for it in items]


class Brain:
    """大脑：路由 + 调用 + 统计 + 熔断。"""

    def __init__(self, settings: Settings, db: Any = None) -> None:
        self.settings = settings
        self.db = db
        self._spent_today = 0.0
        self._day = ""
        self._degraded = False

    # -- 配置 -------------------------------------------------------------

    def _model_cfg(self) -> ModelConfig:
        return self.settings.model

    def provider_for(self, task: str) -> Provider:
        cfg = self._model_cfg().provider_for(task)
        if cfg is None:
            raise LLMError("没有可用的 AI 供应商，请在控制台配置")
        return Provider(cfg)

    def candidates_for(self, task: str, *, need_embedding: bool = False) -> List[Provider]:
        """按任务返回候选链：主供应商打头，后面跟故障转移链。

        顺序去重，`need_embedding` 时只保留声明支持嵌入的供应商
        （备用链上的对话模型通常不支持，跳过它们而不是发一个必然失败的请求）。
        """
        mc = self._model_cfg()
        chain: List[ProviderConfig] = []
        primary = mc.provider_for(task)
        if primary:
            chain.append(primary)
        for pid in mc.failover:
            cfg = mc.get_provider(pid)
            if cfg:
                chain.append(cfg)

        out: List[Provider] = []
        seen: set = set()
        for cfg in chain:
            if cfg.id in seen or not cfg.enabled:
                continue
            if need_embedding and not cfg.supports_embedding:
                continue
            seen.add(cfg.id)
            out.append(Provider(cfg))
        return out

    def reload(self, settings: Settings) -> None:
        self.settings = settings

    # -- 预算 -------------------------------------------------------------

    def _check_budget(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._spent_today = 0.0
            self._degraded = False
        budget = self._model_cfg().daily_budget_usd
        if budget <= 0:
            return
        if self._spent_today >= budget:
            action = self._model_cfg().budget_action
            if action == "stop":
                raise BudgetExceeded(
                    f"今日预算已用尽 (${self._spent_today:.4f}/${budget})"
                )
            self._degraded = True

    def cost_today(self) -> float:
        if self.db is None:
            return self._spent_today
        row = self.db.query_one(
            "SELECT COALESCE(SUM(cost_usd),0) AS c FROM cost_log WHERE ts>=?",
            (int(time.time()) - 86400,),
        )
        return float(row["c"]) if row else 0.0

    # -- 调用 -------------------------------------------------------------

    def complete(self, task: str, system: str, user: str, *,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 json_mode: bool = False, history: Optional[List[Dict[str, str]]] = None
                 ) -> str:
        """一次性补全，返回纯文本。

        主供应商报限流 / 网关抖动 / 超时时，自动沿故障转移链换下一个再试，
        避免一次偶发 429 就让这条回复直接失败。
        """
        self._check_budget()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        provs = self.candidates_for(task)
        if not provs:
            raise LLMError("没有可用的 AI 供应商，请在控制台配置")

        last_exc: Optional[BaseException] = None
        for idx, prov in enumerate(provs):
            t0 = time.time()
            try:
                res = prov.chat(messages, temperature=temperature,
                                max_tokens=max_tokens, json_mode=json_mode)
            except Exception as exc:
                self._record(task, prov.cfg, 0, 0, 0.0,
                             int((time.time() - t0) * 1000), False, str(exc)[:200])
                last_exc = exc
                nxt = provs[idx + 1] if idx + 1 < len(provs) else None
                if nxt is not None and is_retryable(exc):
                    log.warning("供应商 %s 不可用（%s），改用备用 %s",
                                prov.cfg.id, str(exc)[:80], nxt.cfg.id)
                    BUS.publish("llm_failover", {
                        "task": task, "from": prov.cfg.id, "to": nxt.cfg.id,
                        "error": str(exc)[:160]})
                    continue
                raise
            cost = _estimate_cost(prov.cfg.model, res["prompt_tokens"],
                                  res["completion_tokens"])
            self._spent_today += cost
            self._record(task, prov.cfg, res["prompt_tokens"], res["completion_tokens"],
                         cost, int((time.time() - t0) * 1000), True)
            BUS.publish("llm_call", {
                "task": task, "provider": prov.cfg.id, "model": prov.cfg.model,
                "cost": cost, "latency_ms": int((time.time() - t0) * 1000),
            })
            return res["text"]

        raise last_exc if last_exc is not None else LLMError("调用失败")

    def complete_json(self, task: str, system: str, user: str, **kw: Any) -> Dict[str, Any]:
        """要求返回 JSON，容错解析。"""
        raw = self.complete(task, system, user, json_mode=True, **kw)
        return parse_json_loose(raw)

    def stream(self, task: str, system: str, user: str, *,
               temperature: Optional[float] = None, max_tokens: Optional[int] = None,
               history: Optional[List[Dict[str, str]]] = None) -> AsyncIterator[str]:
        """流式生成（同步迭代器包一层，供流式分句使用）。

        为了不引入 aiohttp 依赖，这里用 requests 的流式读取。
        """
        self._check_budget()
        prov = self.provider_for(task)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        payload = {
            "model": prov.cfg.model,
            "messages": messages,
            "temperature": prov.cfg.temperature if temperature is None else temperature,
            "max_tokens": prov.cfg.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        url = f"{prov.base_url}/chat/completions"
        t0 = time.time()
        pt = ct = 0
        try:
            with requests.post(url, headers=prov._headers(), json=payload,
                               timeout=prov.cfg.timeout, stream=True) as resp:
                if resp.status_code >= 400:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        pt = int(obj["usage"].get("prompt_tokens") or pt)
                        ct = int(obj["usage"].get("completion_tokens") or ct)
                    choices = obj.get("choices") or []
                    if choices:
                        delta = (choices[0].get("delta") or {}).get("content") or ""
                        if delta:
                            yield delta
        except Exception as exc:
            self._record(task, prov.cfg, pt, ct, 0.0, int((time.time() - t0) * 1000),
                         False, str(exc)[:200])
            raise
        cost = _estimate_cost(prov.cfg.model, pt, ct)
        self._spent_today += cost
        self._record(task, prov.cfg, pt, ct, cost, int((time.time() - t0) * 1000), True)

    # -- 向量嵌入 ---------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """获取向量。失败返回 None（调用方退化为关键词召回）。"""
        if not texts:
            return []
        mc = self.settings.memory
        task = TASK_EMBEDDING
        try:
            if mc.embedding_provider:
                # 明确点名了嵌入供应商，就不做故障转移
                cfg = self._model_cfg().get_provider(mc.embedding_provider)
                provs = [Provider(cfg)] if cfg else []
            else:
                provs = self.candidates_for(task, need_embedding=True)
            if not provs:
                return None

            last_exc: Optional[BaseException] = None
            for idx, prov in enumerate(provs):
                ok, why = prov.ready()
                if not ok:
                    # 配置不完整就不去发请求，直接退回关键词召回
                    log.debug("跳过向量嵌入（%s）：%s", prov.cfg.id, why)
                    continue
                try:
                    embs = prov.embed(texts, model=prov.cfg.embedding_model)
                    self._record(task, prov.cfg, 0, 0, 0.0, 0, True)
                    return embs
                except Exception as exc:
                    last_exc = exc
                    self._record(task, prov.cfg, 0, 0, 0.0, 0, False, str(exc)[:200])
                    if idx + 1 < len(provs) and is_retryable(exc):
                        log.warning("嵌入供应商 %s 不可用（%s），换备用",
                                    prov.cfg.id, str(exc)[:80])
                        continue
                    break
            log.warning("向量嵌入失败，退化为关键词召回: %s", last_exc)
            return None
        except Exception as exc:
            log.warning("向量嵌入失败，退化为关键词召回: %s", exc)
            return None

    # -- 内部 -------------------------------------------------------------

    def _record(self, task: str, cfg: ProviderConfig, pt: int, ct: int, cost: float,
                latency: int, ok: bool, error: str = "") -> None:
        if self.db is None:
            return
        try:
            self.db.log_cost(task, cfg.id, cfg.model, pt, ct, cost, latency, ok, error)
        except Exception:
            pass

    # -- 连通性测试 -------------------------------------------------------

    def test_provider(self, cfg: ProviderConfig) -> Dict[str, Any]:
        prov = Provider(cfg)
        t0 = time.time()
        try:
            res = prov.chat([{"role": "user", "content": "hi"}], max_tokens=8,
                            timeout=min(cfg.timeout, 30))
            return {"ok": True, "latency_ms": int((time.time() - t0) * 1000),
                    "sample": res["text"][:60]}
        except Exception as exc:
            return {"ok": False, "latency_ms": int((time.time() - t0) * 1000),
                    "error": str(exc)[:300]}


def parse_json_loose(text: str) -> Dict[str, Any]:
    """从模型输出里尽力抠出 JSON。"""
    if not text:
        return {}
    t = text.strip()
    # 去掉 ```json ``` 包裹
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {"value": obj}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def resolve_api_key(cfg: ProviderConfig, db: Any = None) -> ProviderConfig:
    """若 api_key 是密文形式则解密。"""
    if cfg.api_key and cfg.api_key.startswith("enc:"):
        cfg.api_key = decrypt(cfg.api_key[4:])
    return cfg
