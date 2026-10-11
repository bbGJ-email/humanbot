"""记忆层：工作记忆 / 情节记忆 / 画像记忆（策划案 §4）。

- 向量嵌入走 API（OpenAI 兼容 /embeddings），失败自动退化为关键词打分
- 严格按 (bot_id, peer_id, chat_type) 隔离，群记忆绝不串进私聊
- 支持遗忘权（软删除 + prompt 屏蔽）
"""
from __future__ import annotations

import array
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ..settings import TASK_MEMORY, Settings
from .llm import Brain
from .logging_setup import get_logger
from .storage import Database
from .utils import now, text_hash

log = get_logger("memory")

# 用于"值得记吗"的判断
_MEM_TRIGGER_HINTS = (
    "我", "我的", "喜欢", "讨厌", "害怕", "家", "爸", "妈", "工作",
    "面试", "考试", "生日", "分手", "分手", "病", "累", "难过",
    "开心", "第一次", "以前", "小时候", "梦想", "秘密",
)


def _pack(vec: List[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: Optional[bytes]) -> List[float]:
    if not blob:
        return []
    a = array.array("f")
    try:
        a.frombytes(blob)
    except Exception:
        return []
    return list(a)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _keyword_score(query: str, text: str) -> float:
    """无向量时的退化召回：字符 2-gram 重合。"""
    q = re.sub(r"\s+", "", query or "")
    t = re.sub(r"\s+", "", text or "")
    if len(q) < 2 or len(t) < 2:
        return 0.0
    qg = {q[i:i + 2] for i in range(len(q) - 1)}
    tg = {t[i:i + 2] for i in range(len(t) - 1)}
    if not qg or not tg:
        return 0.0
    return len(qg & tg) / len(qg)


class MemoryStore:
    """三层记忆管理。"""

    def __init__(self, db: Database, brain: Brain, settings: Settings) -> None:
        self.db = db
        self.brain = brain
        self.settings = settings
        self._working: Dict[str, List[Dict[str, str]]] = {}

    # -- 工作记忆 ---------------------------------------------------------

    def _wkey(self, bot_id: int, peer_id: int, chat_type: str) -> str:
        return f"{bot_id}:{peer_id}:{chat_type}"

    def working(self, bot_id: int, peer_id: int, chat_type: str) -> List[Dict[str, str]]:
        return self._working.get(self._wkey(bot_id, peer_id, chat_type), [])

    def push_working(self, bot_id: int, peer_id: int, chat_type: str,
                     role: str, content: str) -> None:
        key = self._wkey(bot_id, peer_id, chat_type)
        seq = self._working.setdefault(key, [])
        seq.append({"role": "assistant" if role == "bot" else "user",
                    "content": content})
        limit = self.settings.memory.working_turns * 2
        if len(seq) > limit:
            del seq[: len(seq) - limit]
        # 同时持久化到 messages 表由调用方负责

    def clear_working(self, bot_id: int, peer_id: int, chat_type: str) -> None:
        self._working.pop(self._wkey(bot_id, peer_id, chat_type), None)

    def load_working_from_db(self, bot_id: int, peer_id: int, chat_type: str,
                            conv_id: int) -> None:
        """进程重启后从 DB 恢复工作记忆。"""
        rows = self.db.recent_messages(conv_id, self.settings.memory.working_turns * 2)
        key = self._wkey(bot_id, peer_id, chat_type)
        self._working[key] = [
            {"role": "assistant" if r["role"] == "bot" else "user",
             "content": r["content"]}
            for r in rows if r["role"] in ("user", "bot")
        ]

    # -- 写入 -------------------------------------------------------------

    def should_remember(self, text: str) -> bool:
        """先做廉价预筛，避免每条消息都调 LLM。"""
        if not text or len(text) < 4:
            return False
        return any(h in text for h in _MEM_TRIGGER_HINTS)

    def extract_and_store(self, bot_id: int, peer_id: int, chat_type: str,
                          user_text: str, bot_text: str = "") -> None:
        """调 LLM 抽取值得记的内容，写入情节/画像记忆。"""
        if not self.settings.memory.enabled or not self.should_remember(user_text):
            return
        sys_p = (
            "你是一个记忆抽取器。从用户的话里判断是否有需要长期记住的信息。"
            "只输出 JSON，不要解释。\n"
            '格式：{"remember": true/false, "kind": "profile|episode", '
            '"text": "用第三人称陈述，不超过40字", "importance": 0.0~1.0}\n'
            "规则：\n"
            "- profile：稳定事实（职业、家人、喜好、性格、居住地）\n"
            "- episode：具体事件（某天发生的事、情绪事件）\n"
            "- 客套、寒暄、无信息量的内容一律 remember=false\n"
            "- text 必须是自足的一句话，脱离上下文也能看懂"
        )
        try:
            obj = self.brain.complete_json(
                TASK_MEMORY, sys_p, f"用户说：{user_text}",
                temperature=0.2, max_tokens=160,
            )
        except Exception as exc:
            log.warning("记忆抽取失败: %s", exc)
            return
        if not obj.get("remember"):
            return
        text = str(obj.get("text") or "").strip()
        kind = str(obj.get("kind") or "episode").strip()
        if kind not in ("profile", "episode"):
            kind = "episode"
        if not text:
            return
        imp = float(obj.get("importance") or 0.5)
        emb = self._embed_one(text)
        mem_id = self.db.add_memory(bot_id, peer_id, chat_type, kind, text,
                                    _pack(emb) if emb else None, imp)
        log.info("记住 [%s] %s (#%d)", kind, text, mem_id)

    def add_manual(self, bot_id: int, peer_id: int, chat_type: str, kind: str,
                   text: str, importance: float = 0.7) -> int:
        emb = self._embed_one(text)
        return self.db.add_memory(bot_id, peer_id, chat_type, kind, text,
                                  _pack(emb) if emb else None, importance)

    def _embed_one(self, text: str) -> Optional[List[float]]:
        if not self.settings.memory.embedding_enabled:
            return None
        res = self.brain.embed_texts([text])
        if not res or not res[0]:
            return None
        return res[0]

    # -- 召回 -------------------------------------------------------------

    def retrieve(self, bot_id: int, peer_id: int, chat_type: str, query: str,
                 top_k: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """返回 (画像片段, 情节片段)。"""
        if not self.settings.memory.enabled or not query:
            return [], []
        mc = self.settings.memory
        k = top_k or mc.episodic_top_k

        rows = self.db.query(
            "SELECT * FROM memories WHERE bot_id=? AND peer_id=? AND chat_type=? "
            "AND deleted=0 ORDER BY pinned DESC, importance DESC, id DESC LIMIT 400",
            (bot_id, peer_id, chat_type),
        )
        if not rows:
            return [], []

        profile: List[str] = []
        episodic = [r for r in rows if r["kind"] == "episode"]
        for r in rows:
            if r["kind"] == "profile":
                profile.append(r["text"])
        profile = profile[: mc.profile_max_items]

        # 情节按相似度排序
        qvec = self._embed_one(query) if mc.embedding_enabled else None
        scored: List[Tuple[float, Any]] = []
        for r in episodic:
            vec = _unpack(r["embedding"])
            if qvec and vec:
                sim = _cosine(qvec, vec)
            else:
                sim = _keyword_score(query, r["text"])
            # 叠加重要度与时效衰减
            age_days = max(0.0, (now() - (r["created_at"] or now())) / 86400.0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            score = sim * 0.7 + float(r["importance"]) * 0.2 + recency * 0.1
            if r["pinned"]:
                score += 0.3
            scored.append((score, r))

        scored.sort(key=lambda x: -x[0])
        picked = [r for s, r in scored[:k] if s > 0.05]
        episodic_texts = [r["text"] for r in picked]

        # 更新命中统计
        for r in picked:
            try:
                self.db.execute(
                    "UPDATE memories SET hits=hits+1, last_hit=? WHERE id=?",
                    (now(), r["id"]),
                )
            except Exception:
                pass
        return profile, episodic_texts

    # -- 遗忘 -------------------------------------------------------------

    def forget(self, mem_id: int) -> None:
        """软删除：标记 deleted，不再进入 prompt。"""
        self.db.execute("UPDATE memories SET deleted=1 WHERE id=?", (mem_id,))

    def forget_by_scope(self, bot_id: int, peer_id: int, chat_type: str) -> int:
        cur = self.db.execute(
            "UPDATE memories SET deleted=1 WHERE bot_id=? AND peer_id=? AND chat_type=?",
            (bot_id, peer_id, chat_type),
        )
        return cur.rowcount

    def list_memories(self, bot_id: int, peer_id: int, chat_type: str,
                      kind: str = "", include_deleted: bool = False) -> List[Dict[str, Any]]:
        sql = ("SELECT id, kind, text, importance, pinned, deleted, hits, created_at "
               "FROM memories WHERE bot_id=? AND peer_id=? AND chat_type=?")
        params: List[Any] = [bot_id, peer_id, chat_type]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        if not include_deleted:
            sql += " AND deleted=0"
        sql += " ORDER BY pinned DESC, importance DESC, id DESC LIMIT 300"
        return [dict(r) for r in self.db.query(sql, params)]

    def toggle_pin(self, mem_id: int, pinned: bool) -> None:
        self.db.execute("UPDATE memories SET pinned=? WHERE id=?",
                        (1 if pinned else 0, mem_id))

    def edit(self, mem_id: int, text: str) -> None:
        emb = self._embed_one(text)
        self.db.execute("UPDATE memories SET text=?, embedding=? WHERE id=?",
                        (text, _pack(emb) if emb else None, mem_id))

    def search(self, bot_id: int, peer_id: int, chat_type: str, query: str,
               top_k: int = 10) -> List[Dict[str, Any]]:
        """语义检索（控制台用）。档案与情节一起排，返回带 score 的结果。"""
        if not query:
            return []
        mc = self.settings.memory
        rows = self.db.query(
            "SELECT id, kind, text, importance, pinned, embedding, created_at, hits "
            "FROM memories WHERE bot_id=? AND peer_id=? AND chat_type=? "
            "AND deleted=0 LIMIT 500",
            (bot_id, peer_id, chat_type),
        )
        if not rows:
            return []

        qvec = self._embed_one(query) if mc.embedding_enabled else None
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            vec = _unpack(r["embedding"])
            if qvec and vec:
                sim = _cosine(qvec, vec)
            else:
                sim = _keyword_score(query, r["text"])
            age_days = max(0.0, (now() - (r["created_at"] or now())) / 86400.0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            score = sim * 0.7 + float(r["importance"]) * 0.2 + recency * 0.1
            if r["pinned"]:
                score += 0.3
            d = dict(r)
            d.pop("embedding", None)
            d["score"] = round(score, 4)
            d["similarity"] = round(sim, 4)
            scored.append((score, d))

        scored.sort(key=lambda x: -x[0])
        return [d for s, d in scored[:top_k]]
