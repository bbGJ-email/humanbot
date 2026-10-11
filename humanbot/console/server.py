"""控制台后端：FastAPI + WebSocket 事件流（策划案 §15）。

要点：
- 绑定外网（0.0.0.0），但启动脚本会打印访问地址提示
- 端口自动探测空闲
- 首次启动自动生成强密码
- API Key 加密存储
- 控制台只读写 DB 与命令队列，不直接操作 SDK 对象
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.crypto import (decrypt, encrypt, gen_password, gen_token,
                           hash_password, verify_password)
from ..core.llm import Provider, resolve_api_key
from ..core.logging_setup import BUS, get_logger
from ..core.persona import (STAGES, STAGE_LABELS, save_persona, validate_persona,
                            load_persona)
from ..core.storage import Database
from ..settings import (ALL_TASKS, ROOT, TASK_LABELS, ProviderConfig, Settings,
                        mask_api_key)

log = get_logger("console")

STATIC_DIR = Path(__file__).parent / "static"

# 媒体文件访问白名单根目录（防止路径穿越）
ASSETS_SAFE = str((ROOT / "assets").resolve())


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def find_free_port(start: int, end: int, host: str = "0.0.0.0") -> int:
    """在 [start, end] 范围内找一个可用端口。"""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    # 兜底：让系统随机分配
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def local_ip() -> str:
    """获取本机对外 IP（用于打印访问地址）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def public_ip() -> Optional[str]:
    """尝试获取公网 IP。"""
    try:
        import requests
        r = requests.get("https://api.ipify.org", timeout=4)
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class LoginReq(BaseModel):
    username: str
    password: str


class PersonaSaveReq(BaseModel):
    data: Dict[str, Any]


class PersonaRawReq(BaseModel):
    yaml_text: str


class PreviewReq(BaseModel):
    message: str
    apply_emotion: bool = True
    apply_antiai: bool = True


class ProviderReq(BaseModel):
    id: str
    name: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    temperature: float = 0.9
    max_tokens: int = 400
    timeout: int = 60
    supports_embedding: bool = False
    embedding_model: str = ""


class RoutingReq(BaseModel):
    routing: Optional[Dict[str, str]] = None
    failover: Optional[List[str]] = None


class ConfigReq(BaseModel):
    boxim: Optional[Dict[str, Any]] = None
    humanize: Optional[Dict[str, Any]] = None
    sleep: Optional[Dict[str, Any]] = None
    physio: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    media: Optional[Dict[str, Any]] = None
    sticker: Optional[Dict[str, Any]] = None
    group: Optional[Dict[str, Any]] = None
    voice: Optional[Dict[str, Any]] = None
    call: Optional[Dict[str, Any]] = None
    model: Optional[Dict[str, Any]] = None


class InjectEmotionReq(BaseModel):
    peer_id: int
    chat_type: str = "private"
    pleasure: Optional[float] = None
    energy: Optional[float] = None
    intimacy: Optional[float] = None
    stress: Optional[float] = None
    annoyance: Optional[int] = None


class MemoryEditReq(BaseModel):
    text: str


class MemoryAddReq(BaseModel):
    peer_id: int
    chat_type: str = "private"
    kind: str = "episode"
    text: str
    importance: float = 0.7


class SendMsgReq(BaseModel):
    peer_id: int
    chat_type: str = "private"
    text: str


class SendImageReq(BaseModel):
    peer_id: int
    chat_type: str = "private"
    prompt: str


class SandboxReq(BaseModel):
    message: str
    rounds: int = 1


class GenerateReq(BaseModel):
    prompt: str


class ActionReq(BaseModel):
    action: str
    payload: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------

class ConsoleState:
    """控制台共享状态。"""

    def __init__(self) -> None:
        self.settings: Settings = None          # type: ignore
        self.db: Database = None                # type: ignore
        self.manager: Any = None                # BotManager
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.password_hash: str = ""
        self.username: str = "admin"
        self.port: int = 0


STATE = ConsoleState()


def build_app(settings: Settings, db: Database, manager: Any,
              port: int = 0) -> FastAPI:
    STATE.settings = settings
    STATE.db = db
    STATE.manager = manager
    STATE.username = settings.console.username
    if port:
        STATE.port = int(port)

    # 密码：配置里有就用配置的，否则生成强密码并落库
    stored = db.get_setting("console_password_hash", "")
    if stored:
        STATE.password_hash = stored
    else:
        pwd = settings.console.password or gen_password(20)
        STATE.password_hash = hash_password(pwd)
        db.set_setting("console_password_hash", STATE.password_hash)
        db.set_setting("console_password_plain", pwd)   # 启动脚本读取并打印
        settings.console.password = pwd
        settings.save()

    app = FastAPI(title="HumanBot Console", docs_url=None, redoc_url=None)

    # ---------- 认证 ----------

    def current_user(request: Request) -> str:
        tok = request.headers.get("X-Auth-Token") or request.cookies.get("hb_token")
        sess = STATE.sessions.get(tok or "")
        if not sess:
            raise HTTPException(status_code=401, detail="未登录")
        if time.time() > sess["expire"]:
            STATE.sessions.pop(tok, None)
            raise HTTPException(status_code=401, detail="会话已过期")
        return sess["user"]

    @app.post("/api/login")
    def login(req: LoginReq, request: Request) -> Dict[str, Any]:
        if req.username != STATE.username or not verify_password(
                req.password, STATE.password_hash):
            STATE.db.log_audit("login_failed", req.username,
                               ip=request.client.host if request.client else "")
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        tok = gen_token(32)
        hours = STATE.settings.console.session_hours
        STATE.sessions[tok] = {"user": req.username,
                               "expire": time.time() + hours * 3600}
        STATE.db.log_audit("login", req.username,
                           ip=request.client.host if request.client else "")
        log.info("控制台登录成功: %s", req.username)
        return {"token": tok, "expire_hours": hours}

    @app.post("/api/logout")
    def logout(request: Request) -> Dict[str, Any]:
        tok = request.headers.get("X-Auth-Token") or ""
        STATE.sessions.pop(tok, None)
        return {"ok": True}

    @app.get("/api/me")
    def me(user: str = Depends(current_user)) -> Dict[str, Any]:
        return {"user": user, "version": "1.0.0"}

    # ---------- 状态 ----------

    @app.get("/api/status")
    def status(user: str = Depends(current_user)) -> Dict[str, Any]:
        bots = manager.all_status() if manager else []
        return {
            "bots": bots,
            "port": STATE.port,
            "ip": local_ip(),
            "spent_today": sum(b.get("cost_today", 0) for b in bots),
            "budget": STATE.settings.model.daily_budget_usd,
            "providers": len(STATE.settings.model.providers),
        }

    @app.get("/api/stats")
    def stats(range: str = "24h", user: str = Depends(current_user)
              ) -> Dict[str, Any]:
        seconds = {"24h": 86400, "7d": 604800, "30d": 2592000}.get(range, 86400)
        since = int(time.time()) - seconds
        rows = STATE.db.query(
            "SELECT role, COUNT(*) AS c FROM messages WHERE ts>=? GROUP BY role",
            (since,))
        counts = {r["role"]: r["c"] for r in rows}
        convs = STATE.db.query_one(
            "SELECT COUNT(*) AS c FROM conversations")["c"]
        cost = STATE.db.query_one(
            "SELECT COALESCE(SUM(cost_usd),0) AS c, COUNT(*) AS n "
            "FROM cost_log WHERE ts>=?", (since,))
        media = STATE.db.query_one(
            "SELECT COUNT(*) AS c FROM media_cache WHERE ts>=?", (since,))["c"]
        return {
            "messages_in": counts.get("user", 0),
            "messages_out": counts.get("bot", 0),
            "conversations": convs,
            "cost": round(float(cost["c"] or 0), 4),
            "llm_calls": cost["n"],
            "media": media,
        }

    @app.get("/api/emotion/history")
    def emotion_history(peer_id: int = 0, hours: int = 168,
                        user: str = Depends(current_user)) -> Dict[str, Any]:
        since = int(time.time()) - hours * 3600
        sql = ("SELECT ts, pleasure, energy, intimacy, stress, mood, annoyance, phase "
               "FROM emotion_log WHERE ts>=?")
        params: List[Any] = [since]
        if peer_id:
            sql += " AND peer_id=?"
            params.append(peer_id)
        sql += " ORDER BY ts ASC LIMIT 2000"
        rows = [dict(r) for r in STATE.db.query(sql, params)]
        return {"points": rows}

    # ---------- 会话 ----------

    @app.get("/api/conversations")
    def conversations(bot_id: int = 0, user: str = Depends(current_user)
                      ) -> Dict[str, Any]:
        if not bot_id and manager:
            bots = manager.bots()
            bot_id = bots[0].bot_id if bots else 0
        rows = STATE.db.query(
            "SELECT * FROM conversations WHERE bot_id=? ORDER BY last_ts DESC LIMIT 200",
            (bot_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["stage_label"] = STAGE_LABELS.get(r["relation"], r["relation"])
            last = STATE.db.query_one(
                "SELECT content, role, ts FROM messages WHERE conv_id=? "
                "ORDER BY id DESC LIMIT 1", (r["id"],))
            d["last_message"] = last["content"] if last else ""
            d["last_role"] = last["role"] if last else ""
            d["paused"] = bool(
                manager and manager.is_muted(bot_id, int(r["peer_id"]),
                                             r["chat_type"] == "group"))
            out.append(d)
        return {"conversations": out}

    @app.get("/api/conversations/{conv_id}/messages")
    def conv_messages(conv_id: int, limit: int = 200,
                      user: str = Depends(current_user)) -> Dict[str, Any]:
        rows = STATE.db.query(
            "SELECT * FROM messages WHERE conv_id=? ORDER BY id ASC LIMIT ?",
            (conv_id, limit))
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["extra"] = json.loads(r["extra_json"] or "{}")
            except Exception:
                d["extra"] = {}
            out.append(d)
        return {"messages": out}

    @app.post("/api/conversations/send")
    def conv_send(req: SendMsgReq, user: str = Depends(current_user)
                  ) -> Dict[str, Any]:
        if manager is None:
            raise HTTPException(400, "运行时未启动")
        ok = manager.manual_send(req.peer_id, req.text, req.chat_type)
        STATE.db.log_audit("manual_send", f"{req.chat_type}:{req.peer_id}")
        return {"ok": ok}

    @app.post("/api/conversations/send-image")
    def conv_send_image(req: SendImageReq, user: str = Depends(current_user)
                        ) -> Dict[str, Any]:
        if manager is None:
            raise HTTPException(400, "运行时未启动")
        path = manager.manual_send_image(req.peer_id, req.prompt, req.chat_type)
        return {"ok": bool(path), "path": path}

    @app.post("/api/conversations/{conv_id}/pause")
    def conv_pause(conv_id: int, paused: bool = True,
                   user: str = Depends(current_user)) -> Dict[str, Any]:
        row = STATE.db.query_one("SELECT * FROM conversations WHERE id=?", (conv_id,))
        if not row:
            raise HTTPException(404, "会话不存在")
        if manager:
            if paused:
                manager.mute(int(row["bot_id"]), int(row["peer_id"]),
                             row["chat_type"] == "group")
            else:
                manager.unmute(int(row["bot_id"]), int(row["peer_id"]),
                               row["chat_type"] == "group")
        STATE.db.execute("UPDATE conversations SET paused=? WHERE id=?",
                         (int(paused), conv_id))
        return {"ok": True}

    @app.delete("/api/conversations/{conv_id}/memory")
    def conv_clear_memory(conv_id: int, user: str = Depends(current_user)
                          ) -> Dict[str, Any]:
        row = STATE.db.query_one("SELECT * FROM conversations WHERE id=?", (conv_id,))
        if not row:
            raise HTTPException(404, "会话不存在")
        n = manager.memory.forget_by_scope(int(row["bot_id"]), int(row["peer_id"]),
                                           row["chat_type"]) if manager else 0
        STATE.db.log_audit("clear_memory", f"conv={conv_id} n={n}")
        return {"ok": True, "deleted": n}

    # ---------- 人设 ----------

    @app.get("/api/persona")
    def get_persona(user: str = Depends(current_user)) -> Dict[str, Any]:
        p = load_persona(STATE.settings.persona_file)
        return {
            "data": p.data,
            "physio_enabled": p.physio_enabled,
            "gender": p.gender,
            "path": str(p.path),
        }

    @app.get("/api/persona/raw")
    def get_persona_raw(user: str = Depends(current_user)) -> Dict[str, Any]:
        path = Path(STATE.settings.persona_file)
        if not path.exists():
            return {"yaml_text": ""}
        return {"yaml_text": path.read_text(encoding="utf-8")}

    @app.put("/api/persona")
    def put_persona(req: PersonaSaveReq, user: str = Depends(current_user)
                    ) -> Dict[str, Any]:
        errs = validate_persona(req.data)
        if errs:
            raise HTTPException(400, "；".join(errs))
        save_persona(STATE.settings.persona_file, req.data)
        # 热更新
        if manager:
            manager.reload_persona()
        STATE.db.log_audit("persona_update", "structured")
        g = str((req.data.get("identity") or {}).get("gender", "")).lower()
        return {"ok": True, "physio_enabled": g in ("female", "f", "女", "女性")}

    @app.put("/api/persona/raw")
    def put_persona_raw(req: PersonaRawReq, user: str = Depends(current_user)
                        ) -> Dict[str, Any]:
        import yaml
        try:
            data = yaml.safe_load(req.yaml_text) or {}
        except yaml.YAMLError as exc:
            raise HTTPException(400, f"YAML 解析失败: {exc}")
        errs = validate_persona(data)
        if errs:
            raise HTTPException(400, "；".join(errs))
        save_persona(STATE.settings.persona_file, data)
        if manager:
            manager.reload_persona()
        STATE.db.log_audit("persona_update", "raw")
        return {"ok": True}

    @app.post("/api/persona/preview")
    def persona_preview(req: PreviewReq, user: str = Depends(current_user)
                        ) -> Dict[str, Any]:
        if manager is None or not manager.bots():
            raise HTTPException(400, "BOT 未启动，无法预览")
        return manager.preview(req.message, req.apply_emotion, req.apply_antiai)

    @app.get("/api/persona/meta")
    def persona_meta(user: str = Depends(current_user)) -> Dict[str, Any]:
        return {
            "stages": [{"key": s, "label": STAGE_LABELS[s]} for s in STAGES],
            "tasks": [{"key": t, "label": TASK_LABELS.get(t, t)} for t in ALL_TASKS],
        }

    # ---------- 模型配置 ----------

    @app.get("/api/providers")
    def list_providers(user: str = Depends(current_user)) -> Dict[str, Any]:
        out = []
        for p in STATE.settings.model.providers:
            d = p.to_dict(mask_key=True)
            out.append(d)
        return {
            "providers": out,
            "routing": STATE.settings.model.routing,
            "failover": STATE.settings.model.failover,
            "budget": STATE.settings.model.daily_budget_usd,
            "budget_action": STATE.settings.model.budget_action,
            "cost_today": manager.brain.cost_today() if manager else 0,
        }

    @app.post("/api/providers")
    def add_provider(req: ProviderReq, user: str = Depends(current_user)
                     ) -> Dict[str, Any]:
        if any(p.id == req.id for p in STATE.settings.model.providers):
            raise HTTPException(400, "供应商 ID 已存在")
        cfg = ProviderConfig.from_dict(req.model_dump())
        STATE.settings.model.providers.append(cfg)
        _sync_providers_to_db()
        STATE.settings.save()
        if manager:
            manager.reload_settings()
        STATE.db.log_audit("provider_add", req.id)
        return {"ok": True}

    @app.put("/api/providers/{pid}")
    def update_provider(pid: str, req: ProviderReq, user: str = Depends(current_user)
                        ) -> Dict[str, Any]:
        for i, p in enumerate(STATE.settings.model.providers):
            if p.id == pid:
                data = req.model_dump()
                # 空 api_key 表示不修改
                if not data.get("api_key"):
                    data["api_key"] = p.api_key
                STATE.settings.model.providers[i] = ProviderConfig.from_dict(data)
                _sync_providers_to_db()
                STATE.settings.save()
                if manager:
                    manager.reload_settings()
                STATE.db.log_audit("provider_update", pid)
                return {"ok": True}
        raise HTTPException(404, "供应商不存在")

    @app.delete("/api/providers/{pid}")
    def delete_provider(pid: str, user: str = Depends(current_user)
                        ) -> Dict[str, Any]:
        before = len(STATE.settings.model.providers)
        STATE.settings.model.providers = [
            p for p in STATE.settings.model.providers if p.id != pid]
        if len(STATE.settings.model.providers) == before:
            raise HTTPException(404, "供应商不存在")
        STATE.db.execute("DELETE FROM provider_config WHERE id=?", (pid,))
        STATE.settings.save()
        if manager:
            manager.reload_settings()
        STATE.db.log_audit("provider_delete", pid)
        return {"ok": True}

    @app.post("/api/providers/{pid}/test")
    def test_provider(pid: str, user: str = Depends(current_user)
                      ) -> Dict[str, Any]:
        cfg = None
        for p in STATE.settings.model.providers:
            if p.id == pid:
                cfg = p
                break
        if cfg is None:
            raise HTTPException(404, "供应商不存在")
        brain = manager.brain if manager else None
        if brain is None:
            from ..core.llm import Brain
            brain = Brain(STATE.settings, STATE.db)
        res = brain.test_provider(cfg)
        STATE.db.log_audit("provider_test", f"{pid} ok={res.get('ok')}")
        return res

    @app.post("/api/providers/test-all")
    def test_all(user: str = Depends(current_user)) -> Dict[str, Any]:
        from ..core.llm import Brain
        brain = manager.brain if manager else Brain(STATE.settings, STATE.db)
        out = {}
        for p in STATE.settings.model.providers:
            out[p.id] = brain.test_provider(p)
        return {"results": out}

    @app.put("/api/routing")
    def put_routing(req: RoutingReq, user: str = Depends(current_user)
                    ) -> Dict[str, Any]:
        if req.routing:
            STATE.settings.model.routing.update(req.routing)
            for task, pid in req.routing.items():
                STATE.db.execute(
                    "INSERT INTO routing_config(task, provider_id) VALUES(?,?) "
                    "ON CONFLICT(task) DO UPDATE SET provider_id=excluded.provider_id",
                    (task, pid))
        if req.failover is not None:
            # 只保留真实存在的供应商 id，避免写进一个永远解析不到的幽灵
            valid = {p.id for p in STATE.settings.model.providers}
            STATE.settings.model.failover = [
                str(x) for x in req.failover if str(x).strip() in valid]
        STATE.settings.save()
        if manager:
            manager.reload_settings()
        STATE.db.log_audit("routing_update", json.dumps({
            "routing": req.routing or {},
            "failover": STATE.settings.model.failover,
        }, ensure_ascii=False))
        return {"ok": True, "failover": STATE.settings.model.failover}

    @app.get("/api/cost")
    def cost(range: str = "7d", user: str = Depends(current_user)
             ) -> Dict[str, Any]:
        seconds = {"24h": 86400, "7d": 604800, "30d": 2592000}.get(range, 604800)
        since = int(time.time()) - seconds
        by_task = [dict(r) for r in STATE.db.query(
            "SELECT task, COUNT(*) AS calls, COALESCE(SUM(cost_usd),0) AS cost, "
            "AVG(latency_ms) AS latency FROM cost_log WHERE ts>=? "
            "GROUP BY task ORDER BY cost DESC", (since,))]
        by_day = [dict(r) for r in STATE.db.query(
            "SELECT date(ts,'unixepoch','localtime') AS day, "
            "COALESCE(SUM(cost_usd),0) AS cost, COUNT(*) AS calls "
            "FROM cost_log WHERE ts>=? GROUP BY day ORDER BY day", (since,))]
        total = STATE.db.query_one(
            "SELECT COALESCE(SUM(cost_usd),0) AS c FROM cost_log WHERE ts>=?",
            (since,))
        return {"by_task": by_task, "by_day": by_day,
                "total": round(float(total["c"] or 0), 4)}

    # ---------- 通话 ----------

    @app.get("/api/calls")
    def calls(bot_id: int = 0, limit: int = 50,
              user: str = Depends(current_user)) -> Dict[str, Any]:
        if not bot_id and manager:
            bots = manager.bots()
            bot_id = bots[0].bot_id if bots else 0
        rows = STATE.db.recent_calls(bot_id, limit)
        return {"calls": [dict(r) for r in rows]}

    @app.get("/api/calls/{call_id}")
    def call_detail(call_id: int, user: str = Depends(current_user)
                    ) -> Dict[str, Any]:
        row = STATE.db.query_one("SELECT * FROM calls WHERE id=?", (call_id,))
        if not row:
            raise HTTPException(404, "通话不存在")
        transcript = [dict(r) for r in STATE.db.call_transcript(call_id)]
        events = [dict(r) for r in STATE.db.call_events(call_id)]
        return {"call": dict(row), "transcript": transcript, "events": events}

    # ---------- 情绪 ----------

    @app.get("/api/emotion/{peer_id}")
    def get_emotion(peer_id: int, chat_type: str = "private",
                    user: str = Depends(current_user)) -> Dict[str, Any]:
        if manager is None or not manager.bots():
            return {"emotion": {}, "annoyance": 0}
        b = manager.bots()[0]
        st = b.emotion.state(b.bot_id, peer_id, chat_type)
        from ..core.emotion import mood_of
        import datetime as _dt
        return {
            "emotion": st.emotion.to_json(),
            "annoyance": st.annoyance,
            "relation": st.relation,
            "mood": mood_of(st.emotion, _dt.datetime.now().hour),
            "cold": b.emotion.is_cold(st),
            "cold_remaining": b.emotion.cold_remaining(st),
            "times_asked_ai": st.times_asked_ai,
        }

    @app.post("/api/emotion/inject")
    def inject_emotion(req: InjectEmotionReq, user: str = Depends(current_user)
                       ) -> Dict[str, Any]:
        if manager is None or not manager.bots():
            raise HTTPException(400, "BOT 未启动")
        b = manager.bots()[0]
        st = b.emotion.state(b.bot_id, req.peer_id, req.chat_type)
        for field in ("pleasure", "energy", "intimacy", "stress"):
            v = getattr(req, field)
            if v is not None:
                setattr(st.emotion, field, float(v))
        if req.annoyance is not None:
            st.annoyance = int(req.annoyance)
        b.emotion.persist(st)
        STATE.db.log_audit("emotion_inject", f"{req.peer_id}")
        return {"ok": True, "emotion": st.emotion.to_json(),
                "annoyance": st.annoyance}

    # ---------- 记忆 ----------

    @app.get("/api/memories")
    def list_memories(peer_id: int, chat_type: str = "private", kind: str = "",
                      include_deleted: bool = False,
                      user: str = Depends(current_user)) -> Dict[str, Any]:
        if manager is None or not manager.bots():
            return {"memories": []}
        b = manager.bots()[0]
        return {"memories": manager.memory.list_memories(
            b.bot_id, peer_id, chat_type, kind, include_deleted)}

    @app.post("/api/memories")
    def add_memory(req: MemoryAddReq, user: str = Depends(current_user)
                   ) -> Dict[str, Any]:
        if manager is None or not manager.bots():
            raise HTTPException(400, "BOT 未启动")
        b = manager.bots()[0]
        mid = manager.memory.add_manual(b.bot_id, req.peer_id, req.chat_type,
                                        req.kind, req.text, req.importance)
        STATE.db.log_audit("memory_add", f"id={mid}")
        return {"ok": True, "id": mid}

    @app.put("/api/memories/{mem_id}")
    def edit_memory(mem_id: int, req: MemoryEditReq,
                    user: str = Depends(current_user)) -> Dict[str, Any]:
        if manager:
            manager.memory.edit(mem_id, req.text)
        STATE.db.log_audit("memory_edit", f"id={mem_id}")
        return {"ok": True}

    @app.delete("/api/memories/{mem_id}")
    def forget_memory(mem_id: int, user: str = Depends(current_user)
                      ) -> Dict[str, Any]:
        if manager:
            manager.memory.forget(mem_id)
        STATE.db.log_audit("memory_forget", f"id={mem_id}")
        return {"ok": True}

    @app.post("/api/memories/{mem_id}/pin")
    def pin_memory(mem_id: int, pinned: bool = True,
                   user: str = Depends(current_user)) -> Dict[str, Any]:
        if manager:
            manager.memory.toggle_pin(mem_id, pinned)
        return {"ok": True}

    @app.post("/api/memories/search")
    def search_memories(peer_id: int, query: str, chat_type: str = "private",
                        user: str = Depends(current_user)) -> Dict[str, Any]:
        if manager is None or not manager.bots():
            return {"results": []}
        b = manager.bots()[0]
        return {"results": manager.memory.search(b.bot_id, peer_id, chat_type, query)}

    # ---------- 媒体 ----------

    @app.get("/api/media")
    def list_media(limit: int = 100, user: str = Depends(current_user)
                   ) -> Dict[str, Any]:
        if manager is None:
            return {"items": []}
        return {"items": manager.media.list_cache(limit)}

    @app.post("/api/media/generate")
    def gen_media(req: GenerateReq, user: str = Depends(current_user)
                  ) -> Dict[str, Any]:
        if manager is None:
            raise HTTPException(400, "未启动")
        path = manager.media.manual_generate(req.prompt)
        STATE.db.log_audit("media_generate", req.prompt[:60])
        return {"ok": bool(path), "path": path}

    @app.get("/api/media/file")
    def get_media_file(path: str, user: str = Depends(current_user)):
        p = Path(path)
        if not p.exists() or ASSETS_SAFE not in str(p.resolve()):
            raise HTTPException(404, "文件不存在")
        return FileResponse(str(p))

    # ---------- 表情包 ----------

    def _sticker_svc() -> Any:
        """拿表情服务：优先用运行中的 BOT，否则用只读实例。

        这样 `--no-bots` 单跑控制台时，表情库页面照样能看、能预览。
        """
        if manager is not None and getattr(manager, "stickers", None) is not None:
            return manager.stickers
        from ..core.stickers import StickerService
        return StickerService(STATE.db, STATE.settings)

    def _interest_model() -> Any:
        if manager is not None and getattr(manager, "interest", None) is not None:
            return manager.interest
        from ..core.interest import InterestModel
        return InterestModel(STATE.db, load_persona(STATE.settings.persona_file),
                             STATE.settings)

    def _first_bot_id() -> int:
        if manager is not None:
            bots = manager.bots()
            if bots:
                return int(bots[0].bot_id)
        row = STATE.db.query_one("SELECT bot_id FROM bots ORDER BY bot_id LIMIT 1")
        return int(row["bot_id"]) if row else 0

    @app.get("/api/stickers/albums")
    def sticker_albums(user: str = Depends(current_user)) -> Dict[str, Any]:
        svc = _sticker_svc()
        return {"albums": svc.list_albums(), "total": svc.count()}

    @app.get("/api/stickers")
    def sticker_list(album_id: int = 0, label: str = "", q: str = "",
                     limit: int = 200, user: str = Depends(current_user)
                     ) -> Dict[str, Any]:
        svc = _sticker_svc()
        if label:
            return {"items": svc.list_by_label(label, limit)}
        sql = ("SELECT sticker_id,album_id,album_name,name,image_url,"
               "thumb_url,width,height,labels,descriptive FROM sticker_catalog "
               "WHERE 1=1")
        params: List[Any] = []
        if album_id:
            sql += " AND album_id=?"
            params.append(album_id)
        if q:
            sql += " AND name LIKE ?"
            params.append(f"%{q}%")
        sql += " ORDER BY album_id, sticker_id LIMIT ?"
        params.append(limit)
        items = []
        for r in STATE.db.query(sql, params):
            d = dict(r)
            d["id"] = d.pop("sticker_id")
            try:
                d["labels"] = json.loads(d["labels"] or "[]")
            except Exception:
                d["labels"] = []
            items.append(d)
        return {"items": items}

    @app.post("/api/stickers/sync")
    def sticker_sync(user: str = Depends(current_user)) -> Dict[str, Any]:
        svc = _sticker_svc()
        if manager is None or getattr(manager, "stickers", None) is None:
            raise HTTPException(400, "BOT 未启动，无法从平台同步")
        n = svc.sync(force=True)
        STATE.db.log_audit("sticker_sync", str(n))
        return {"ok": True, "count": n, "total": svc.count()}

    @app.get("/api/stickers/usage")
    def sticker_usage(limit: int = 50, user: str = Depends(current_user)
                      ) -> Dict[str, Any]:
        bid = _first_bot_id()
        if not bid:
            return {"items": []}
        rows = STATE.db.query(
            "SELECT u.ts, u.chat_id, u.name, c.album_name, c.image_url "
            "FROM sticker_usage u LEFT JOIN sticker_catalog c "
            "ON c.sticker_id = u.sticker_id "
            "WHERE u.bot_id=? ORDER BY u.id DESC LIMIT ?", (bid, limit))
        return {"items": [dict(r) for r in rows]}

    @app.get("/api/stickers/pick")
    def sticker_pick(mood: str = "", user_text: str = "",
                     annoyance: int = 0, label: str = "", limit: int = 8,
                     user: str = Depends(current_user)) -> Dict[str, Any]:
        """预览选图结果（不发出去）。"""
        svc = _sticker_svc()
        if label:
            return {"items": svc.list_by_label(label, limit)}
        brain = manager.brain if (manager is not None) else None
        got = svc.pick(bot_id=_first_bot_id() or 0, chat_id=0, mood=mood,
                       annoyance=annoyance, user_text=user_text, brain=brain)
        return {"items": [got] if got else [],
                "labels": svc.wanted_labels(mood, annoyance, user_text)}

    # ---------- 群聊参与 ----------

    @app.get("/api/groups")
    def group_list(user: str = Depends(current_user)) -> Dict[str, Any]:
        """列出所有出现过的群，带当前热度与参与状态。"""
        bid = _first_bot_id()
        if not bid:
            return {"groups": []}
        im = _interest_model()
        rows = STATE.db.query(
            "SELECT peer_id, title FROM conversations "
            "WHERE bot_id=? AND chat_type='group' ORDER BY last_ts DESC LIMIT 60",
            (bid,))
        out = []
        for r in rows:
            gid = int(r["peer_id"])
            snap = im.snapshot(bid, gid)
            eng = STATE.db.query_one(
                "SELECT * FROM group_engagement WHERE group_id=?", (gid,))
            out.append({
                "group_id": gid,
                "title": r["title"] or str(gid),
                "hot_topics": snap["hot_topics"][:6],
                "momentum": snap["momentum"],
                "ignores": snap["ignores"],
                "replies": int(eng["replies"] or 0) if eng else 0,
                "last_spoke": int(eng["last_spoke_ts"] or 0) if eng else 0,
            })
        return {"groups": out, "interests": im.interest_topics(bid)}

    @app.get("/api/group/{group_id}")
    def group_detail(group_id: int, user: str = Depends(current_user)
                     ) -> Dict[str, Any]:
        bid = _first_bot_id()
        im = _interest_model()
        snap = im.snapshot(bid, group_id) if bid else {}
        row = STATE.db.query_one(
            "SELECT * FROM group_rules WHERE group_id=?", (group_id,))
        snap["rules"] = dict(row) if row else {}
        snap["muted_until"] = int(row["muted_until"] or 0) if row else 0
        # 话题热度明细（含命中次数）
        snap["topics"] = [dict(x) for x in STATE.db.query(
            "SELECT keyword, heat, hits, last_ts FROM group_topics "
            "WHERE bot_id=? AND group_id=? ORDER BY heat DESC LIMIT 30",
            (bid, group_id))]
        eng = STATE.db.query_one(
            "SELECT * FROM group_engagement WHERE group_id=?", (group_id,))
        snap["engagement"] = dict(eng) if eng else {}
        return {"data": snap}

    @app.post("/api/group/{group_id}/interest")
    def group_set_interest(group_id: int, keywords: List[str],
                           user: str = Depends(current_user)) -> Dict[str, Any]:
        STATE.db.execute(
            "INSERT INTO group_rules(group_id,bot_id,interest_keywords) "
            "VALUES(?,?,?) ON CONFLICT(group_id) DO UPDATE SET "
            "interest_keywords=excluded.interest_keywords",
            (group_id, _first_bot_id(),
             json.dumps(keywords, ensure_ascii=False)))
        STATE.db.log_audit("group_interest", f"g={group_id}")
        return {"ok": True}

    @app.post("/api/group/{group_id}/reset")
    def group_reset(group_id: int, user: str = Depends(current_user)
                    ) -> Dict[str, Any]:
        """清空该群的话题热度与参与状态。"""
        STATE.db.execute("DELETE FROM group_topics WHERE group_id=?", (group_id,))
        STATE.db.execute("DELETE FROM group_engagement WHERE group_id=?", (group_id,))
        STATE.db.log_audit("group_reset", str(group_id))
        return {"ok": True}

    # ---------- 沙盒 ----------

    @app.post("/api/sandbox/chat")
    def sandbox_chat(req: SandboxReq, user: str = Depends(current_user)
                     ) -> Dict[str, Any]:
        if manager is None:
            raise HTTPException(400, "未启动")
        return manager.sandbox(req.message, req.rounds)

    @app.get("/api/sandbox/antiai")
    def sandbox_antiai(limit: int = 200, user: str = Depends(current_user)
                       ) -> Dict[str, Any]:
        rows = [dict(r) for r in STATE.db.query(
            "SELECT ts, score, hits, text, passed FROM antiai_stats "
            "ORDER BY id DESC LIMIT ?", (limit,))]
        if rows:
            avg = sum(r["score"] for r in rows) / len(rows)
            passed = sum(1 for r in rows if r["passed"]) / len(rows)
        else:
            avg, passed = 0, 0
        # 统计最常命中的检测项
        counter: Dict[str, int] = {}
        for r in rows:
            try:
                for h in json.loads(r["hits"]):
                    name = h.get("name") if isinstance(h, dict) else str(h)
                    counter[name] = counter.get(name, 0) + 1
            except Exception:
                pass
        top = sorted(counter.items(), key=lambda x: -x[1])[:12]
        return {"avg_score": round(avg, 1), "pass_rate": round(passed, 3),
                "top_hits": [{"name": k, "count": v} for k, v in top],
                "recent": rows[:40]}

    # ---------- 账号行为 ----------

    @app.get("/api/account/actions")
    def account_actions(limit: int = 50, user: str = Depends(current_user)
                        ) -> Dict[str, Any]:
        if manager is None or not manager.bots():
            return {"actions": []}
        return {"actions": manager.bots()[0].account.recent_actions(limit)}

    @app.get("/api/account/life")
    def account_life(limit: int = 50, user: str = Depends(current_user)
                     ) -> Dict[str, Any]:
        rows = STATE.db.query(
            "SELECT * FROM life_events ORDER BY ts DESC LIMIT ?", (limit,))
        return {"events": [dict(r) for r in rows]}

    @app.get("/api/sleep/log")
    def sleep_log(limit: int = 60, user: str = Depends(current_user)
                  ) -> Dict[str, Any]:
        rows = STATE.db.query(
            "SELECT * FROM sleep_log ORDER BY date DESC LIMIT ?", (limit,))
        return {"logs": [dict(r) for r in rows]}

    # ---------- 系统配置 ----------

    @app.get("/api/config")
    def get_config(user: str = Depends(current_user)) -> Dict[str, Any]:
        s = STATE.settings
        return {
            "boxim": {"username": s.boxim.username,
                      "base_url": s.boxim.base_url,
                      "ws_url": s.boxim.ws_url,
                      "ssl_verify": s.boxim.ssl_verify,
                      "password_set": bool(s.boxim.password),
                      "disconnect_on_sleep": s.boxim.disconnect_on_sleep},
            "humanize": s.humanize.__dict__,
            "sleep": s.sleep.__dict__,
            "physio": s.physio.__dict__,
            "memory": s.memory.__dict__,
            "media": s.media.__dict__,
            "sticker": s.sticker.__dict__,
            "group": s.group.__dict__,
            "voice": s.voice.__dict__,
            "call": s.call.__dict__,
            "console": {"host": s.console.host, "port": s.console.port,
                        "username": s.console.username},
            "log_level": s.log_level,
        }

    @app.put("/api/config")
    def put_config(req: ConfigReq, user: str = Depends(current_user)
                   ) -> Dict[str, Any]:
        s = STATE.settings
        if req.boxim:
            b = req.boxim
            if b.get("username") is not None:
                s.boxim.username = b["username"]
            if b.get("password"):
                s.boxim.password = b["password"]
            if b.get("base_url"):
                s.boxim.base_url = b["base_url"]
            if b.get("ws_url"):
                s.boxim.ws_url = b["ws_url"]
            if "ssl_verify" in b:
                s.boxim.ssl_verify = bool(b["ssl_verify"])
            if "disconnect_on_sleep" in b:
                s.boxim.disconnect_on_sleep = bool(b["disconnect_on_sleep"])
        for name, patch in (("humanize", req.humanize), ("sleep", req.sleep),
                            ("physio", req.physio), ("memory", req.memory),
                            ("media", req.media), ("sticker", req.sticker),
                            ("group", req.group), ("voice", req.voice),
                            ("call", req.call)):
            if patch:
                obj = getattr(s, name)
                for k, v in patch.items():
                    if hasattr(obj, k):
                        try:
                            setattr(obj, k, type(getattr(obj, k))(v))
                        except (TypeError, ValueError):
                            pass
        if req.model:
            if "daily_budget_usd" in req.model:
                s.model.daily_budget_usd = float(req.model["daily_budget_usd"])
            if "budget_action" in req.model:
                s.model.budget_action = str(req.model["budget_action"])
        s.save()
        if manager:
            manager.reload_settings()
        STATE.db.log_audit("config_update", "ok")
        return {"ok": True}

    @app.get("/api/audit")
    def get_audit(limit: int = 100, user: str = Depends(current_user)
                  ) -> Dict[str, Any]:
        rows = STATE.db.query(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return {"logs": [dict(r) for r in rows]}

    @app.get("/api/logs/history")
    def logs_history(limit: int = 200, user: str = Depends(current_user)
                     ) -> Dict[str, Any]:
        return {"events": BUS.history(limit)}

    # ---------- 动作 ----------

    @app.post("/api/action")
    def do_action(req: ActionReq, user: str = Depends(current_user)
                  ) -> Dict[str, Any]:
        if manager is None:
            raise HTTPException(400, "未启动")
        res = manager.action(req.action, req.payload)
        STATE.db.log_audit("action", f"{req.action} {json.dumps(req.payload)[:80]}")
        return res

    @app.post("/api/reload")
    def reload_all(user: str = Depends(current_user)) -> Dict[str, Any]:
        if manager:
            manager.reload_settings()
            manager.reload_persona()
        return {"ok": True}

    # ---------- 静态页 ----------

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        f = STATIC_DIR / "index.html"
        if f.exists():
            return HTMLResponse(f.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>HumanBot Console</h1><p>前端文件缺失</p>")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ---------- WebSocket 事件流 ----------

    @app.websocket("/ws/console")
    async def ws_console(ws: WebSocket) -> None:
        tok = ws.query_params.get("token", "")
        sess = STATE.sessions.get(tok)
        if not sess or time.time() > sess["expire"]:
            await ws.close(code=4401)
            return
        await ws.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        loop = asyncio.get_running_loop()

        def on_event(evt: Dict[str, Any]) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, evt)
            except Exception:
                pass

        BUS.subscribe_all(on_event)
        try:
            for evt in BUS.history(80):
                await ws.send_text(json.dumps(evt, ensure_ascii=False))
            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=25)
                    await ws.send_text(json.dumps(evt, ensure_ascii=False))
                except asyncio.TimeoutError:
                    await ws.send_text(json.dumps({"topic": "ping",
                                                   "ts": int(time.time() * 1000)}))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.debug("WS 控制台异常: %s", exc)
        finally:
            BUS.unsubscribe_all(on_event)

    return app


def _sync_providers_to_db() -> None:
    """把 provider 配置加密后写入 DB（便于审计与前端展示）。"""
    for p in STATE.settings.model.providers:
        STATE.db.execute(
            "INSERT OR REPLACE INTO provider_config(id, name, base_url, "
            "api_key_enc, model, enabled, temperature, max_tokens, timeout, "
            "supports_embedding, embedding_model) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (p.id, p.name, p.base_url, encrypt(p.api_key), p.model,
             int(p.enabled), p.temperature, p.max_tokens, p.timeout,
             int(p.supports_embedding), p.embedding_model),
        )


def load_providers_from_db(db: Database, settings: Settings) -> None:
    """启动时从 DB 回填 provider（api_key 解密）。"""
    rows = db.query("SELECT * FROM provider_config")
    if not rows:
        return
    existing = {p.id for p in settings.model.providers}
    for r in rows:
        if r["id"] in existing:
            # 用 DB 里的密文覆盖配置中的明文（统一来源）
            for p in settings.model.providers:
                if p.id == r["id"] and not p.api_key:
                    p.api_key = decrypt(r["api_key_enc"])
            continue
        settings.model.providers.append(ProviderConfig(
            id=r["id"], name=r["name"], base_url=r["base_url"],
            api_key=decrypt(r["api_key_enc"]), model=r["model"],
            enabled=bool(r["enabled"]), temperature=float(r["temperature"]),
            max_tokens=int(r["max_tokens"]), timeout=int(r["timeout"]),
            supports_embedding=bool(r["supports_embedding"]),
            embedding_model=r["embedding_model"],
        ))
