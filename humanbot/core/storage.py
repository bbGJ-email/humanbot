"""SQLite 数据访问层：建表 + 常用查询。"""
from __future__ import annotations

import os
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 表结构（对齐策划案 §11）
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- BOT 本体：配置与实时状态
CREATE TABLE IF NOT EXISTS bots (
    bot_id          INTEGER PRIMARY KEY,          -- BoxIM 用户 ID
    username        TEXT NOT NULL,
    nickname        TEXT DEFAULT '',
    gender          TEXT DEFAULT 'female',        -- 决定生理周期是否启用
    physio_enabled  INTEGER DEFAULT 0,            -- 初始化时计算一次
    persona_yaml    TEXT DEFAULT '',
    persona_version INTEGER DEFAULT 0,
    relation_stage  TEXT DEFAULT 'stranger',      -- stranger/acquaintance/familiar/close/dependent
    emotion_json    TEXT DEFAULT '{}',            -- 四维情绪快照
    annoyance       INTEGER DEFAULT 0,            -- 不满度 0-100
    state           TEXT DEFAULT 'AWAKE',         -- AWAKE/DROWSY/ASLEEP/WAKING
    sleep_at        INTEGER DEFAULT 0,            -- 本次入睡时间戳
    wake_at         INTEGER DEFAULT 0,
    cycle_start     INTEGER DEFAULT 0,            -- 本次生理周期起始日
    cycle_length    INTEGER DEFAULT 28,
    cycle_light     INTEGER DEFAULT 0,            -- 本月是否"很轻"
    last_active     INTEGER DEFAULT 0,
    created_at      INTEGER DEFAULT 0
);

-- 会话：私聊 / 群聊统一管理，保存断线补偿游标
CREATE TABLE IF NOT EXISTS conversations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id       INTEGER NOT NULL,
    peer_id      INTEGER NOT NULL,        -- 私聊=对方 uid；群聊=group_id
    chat_type    TEXT NOT NULL,           -- private | group
    title        TEXT DEFAULT '',
    is_online    INTEGER DEFAULT 0,
    relation     TEXT DEFAULT 'stranger',
    intimacy     REAL DEFAULT 0.0,
    paused       INTEGER DEFAULT 0,       -- 控制台静音
    last_msg_id  INTEGER DEFAULT 0,       -- 断线补偿游标
    last_seq_no  INTEGER DEFAULT 0,
    last_ts      INTEGER DEFAULT 0,
    unread       INTEGER DEFAULT 0,
    UNIQUE(bot_id, peer_id, chat_type)
);

-- 全量消息
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id     INTEGER NOT NULL,
    conv_id    INTEGER NOT NULL,
    msg_id     TEXT,                      -- 服务端消息 ID（去重）
    local_id   TEXT,
    seq_no     INTEGER DEFAULT 0,
    role       TEXT NOT NULL,             -- user | bot | system
    msg_type   INTEGER DEFAULT 0,
    content    TEXT DEFAULT '',
    at_user_ids TEXT DEFAULT '',          -- JSON 数组
    quote_id   INTEGER DEFAULT 0,
    ts         INTEGER DEFAULT 0,
    extra_json TEXT DEFAULT '{}'          -- 延迟/反AI分数等调试信息
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id, ts);
CREATE UNIQUE INDEX IF NOT EXISTS idx_msg_dedup ON messages(conv_id, msg_id)
    WHERE msg_id IS NOT NULL AND msg_id != '';

-- 三层记忆
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id     INTEGER NOT NULL,
    peer_id    INTEGER NOT NULL,
    chat_type  TEXT DEFAULT 'private',    -- 严格隔离：群记忆不串私聊
    kind       TEXT NOT NULL,             -- profile | episodic
    text       TEXT NOT NULL,
    embedding  BLOB,                      -- float32 数组
    importance REAL DEFAULT 0.5,
    pinned     INTEGER DEFAULT 0,
    deleted    INTEGER DEFAULT 0,         -- 软删除（遗忘权）
    hits       INTEGER DEFAULT 0,
    created_at INTEGER DEFAULT 0,
    last_hit   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(bot_id, peer_id, chat_type, kind, deleted);

-- 情绪轨迹
CREATE TABLE IF NOT EXISTS emotion_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id    INTEGER NOT NULL,
    peer_id   INTEGER DEFAULT 0,
    ts        INTEGER DEFAULT 0,
    pleasure  REAL DEFAULT 0,
    energy    REAL DEFAULT 0.6,
    intimacy  REAL DEFAULT 0,
    stress    REAL DEFAULT 0,
    mood      TEXT DEFAULT '',
    annoyance INTEGER DEFAULT 0,
    phase     TEXT DEFAULT ''             -- 生理周期阶段（female）
);

-- 冲突事件时间线
CREATE TABLE IF NOT EXISTS annoyance_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id      INTEGER NOT NULL,
    peer_id     INTEGER NOT NULL,
    ts          INTEGER DEFAULT 0,
    level       INTEGER DEFAULT 1,
    trigger     TEXT DEFAULT '',
    resolved_at INTEGER DEFAULT 0
);

-- 账号行为历史（个签/头像/社交动作）
CREATE TABLE IF NOT EXISTS account_actions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id  INTEGER NOT NULL,
    action  TEXT NOT NULL,
    payload TEXT DEFAULT '{}',
    ts      INTEGER DEFAULT 0
);

-- 生活时间线事件
CREATE TABLE IF NOT EXISTS life_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id     INTEGER NOT NULL,
    ts         INTEGER DEFAULT 0,
    kind       TEXT DEFAULT '',
    content    TEXT DEFAULT '',
    impact     TEXT DEFAULT '{}',
    share_prob REAL DEFAULT 0.5,
    shared     INTEGER DEFAULT 0,
    expires_at INTEGER DEFAULT 0
);

-- 媒体缓存
CREATE TABLE IF NOT EXISTS media_cache (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    hash    TEXT UNIQUE,
    kind    TEXT DEFAULT 'gen',           -- gen | search
    prompt  TEXT DEFAULT '',
    path    TEXT DEFAULT '',
    ts      INTEGER DEFAULT 0,
    used    INTEGER DEFAULT 0
);

-- 群聊规则
CREATE TABLE IF NOT EXISTS group_rules (
    group_id        INTEGER PRIMARY KEY,
    bot_id          INTEGER NOT NULL,
    response_weight REAL DEFAULT 0.25,
    muted_until     INTEGER DEFAULT 0,
    aliases         TEXT DEFAULT '',      -- JSON 数组；被提到名字时算 mention
    interest_keywords TEXT DEFAULT '[]'
);

-- 每晚作息记录（保证"昨晚几点睡"回答一致）
CREATE TABLE IF NOT EXISTS sleep_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id         INTEGER NOT NULL,
    date           TEXT NOT NULL,
    sleep_at       INTEGER DEFAULT 0,
    wake_at        INTEGER DEFAULT 0,
    duration_min   INTEGER DEFAULT 0,
    quality        TEXT DEFAULT 'normal',
    is_all_nighter INTEGER DEFAULT 0,
    sleep_debt     INTEGER DEFAULT 0,
    UNIQUE(bot_id, date)
);

-- 生理周期记录
CREATE TABLE IF NOT EXISTS cycle_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id     INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    length     INTEGER DEFAULT 28,
    light      INTEGER DEFAULT 0,
    UNIQUE(bot_id, start_date)
);

-- 控制台命令队列（主进程轮询消费）
CREATE TABLE IF NOT EXISTS control_commands (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    cmd        TEXT NOT NULL,
    payload    TEXT DEFAULT '{}',
    status     TEXT DEFAULT 'pending',    -- pending | done | failed
    result     TEXT DEFAULT '',
    created_at INTEGER DEFAULT 0,
    done_at    INTEGER DEFAULT 0
);

-- 供应商配置（api_key 加密存储）
CREATE TABLE IF NOT EXISTS provider_config (
    id                TEXT PRIMARY KEY,
    name              TEXT DEFAULT '',
    base_url          TEXT DEFAULT '',
    api_key_enc       TEXT DEFAULT '',
    model             TEXT DEFAULT '',
    enabled           INTEGER DEFAULT 1,
    temperature       REAL DEFAULT 0.9,
    max_tokens        INTEGER DEFAULT 400,
    timeout           INTEGER DEFAULT 60,
    supports_embedding INTEGER DEFAULT 0,
    embedding_model   TEXT DEFAULT ''
);

-- 任务路由
CREATE TABLE IF NOT EXISTS routing_config (
    task        TEXT PRIMARY KEY,
    provider_id TEXT DEFAULT ''
);

-- 调用成本日志
CREATE TABLE IF NOT EXISTS cost_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            INTEGER DEFAULT 0,
    task          TEXT DEFAULT '',
    provider_id   TEXT DEFAULT '',
    model         TEXT DEFAULT '',
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd      REAL DEFAULT 0,
    latency_ms    INTEGER DEFAULT 0,
    ok            INTEGER DEFAULT 1,
    error         TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_cost_ts ON cost_log(ts);

-- 控制台操作审计
CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      INTEGER DEFAULT 0,
    actor   TEXT DEFAULT 'admin',
    action  TEXT DEFAULT '',
    detail  TEXT DEFAULT '',
    ip      TEXT DEFAULT ''
);

-- 系统设置键值对
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT DEFAULT ''
);

-- 反AI检测统计（沙盒压测用）
CREATE TABLE IF NOT EXISTS antiai_stats (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      INTEGER DEFAULT 0,
    score   INTEGER DEFAULT 0,
    hits    TEXT DEFAULT '[]',
    text    TEXT DEFAULT '',
    passed  INTEGER DEFAULT 1
);

-- 生活时间线里的"剧本事件"进度
CREATE TABLE IF NOT EXISTS script_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id     INTEGER NOT NULL,
    key        TEXT NOT NULL,
    fired_at   INTEGER DEFAULT 0,
    UNIQUE(bot_id, key)
);

-- 表情包目录（从 BoxIM /sticker 接口同步而来）
CREATE TABLE IF NOT EXISTS sticker_catalog (
    sticker_id  INTEGER PRIMARY KEY,
    album_id    INTEGER DEFAULT 0,
    album_name  TEXT DEFAULT '',
    name        TEXT DEFAULT '',
    image_url   TEXT DEFAULT '',
    thumb_url   TEXT DEFAULT '',
    width       INTEGER DEFAULT 0,
    height      INTEGER DEFAULT 0,
    descriptive INTEGER DEFAULT 1,     -- 名字是否有语义（数字命名的不参与匹配）
    labels      TEXT DEFAULT '[]',     -- 自动打的情感标签 JSON 数组
    synced_at   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sticker_album ON sticker_catalog(album_id);
CREATE INDEX IF NOT EXISTS idx_sticker_desc ON sticker_catalog(descriptive);

-- 表情包发送记录（冷却 + 防重复）
CREATE TABLE IF NOT EXISTS sticker_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id     INTEGER NOT NULL,
    chat_id    INTEGER DEFAULT 0,
    sticker_id INTEGER DEFAULT 0,
    name       TEXT DEFAULT '',
    ts         INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sticker_usage ON sticker_usage(bot_id, chat_id, ts);

-- 群话题热度（累积 / 衰减，用于"主动参与感兴趣的话题"）
CREATE TABLE IF NOT EXISTS group_topics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id      INTEGER NOT NULL,
    group_id    INTEGER NOT NULL,
    keyword     TEXT NOT NULL,
    heat        REAL DEFAULT 0.0,
    hits        INTEGER DEFAULT 0,
    last_ts     INTEGER DEFAULT 0,
    UNIQUE(bot_id, group_id, keyword)
);

-- 每群参与状态（惯性 / 插话 / 主动开口额度）
CREATE TABLE IF NOT EXISTS group_engagement (
    group_id          INTEGER PRIMARY KEY,
    bot_id            INTEGER NOT NULL,
    last_spoke_ts     INTEGER DEFAULT 0,
    last_interject_ts INTEGER DEFAULT 0,
    proactive_ts      TEXT DEFAULT '',   -- 主动开口的时间戳 JSON 数组
    ignores           INTEGER DEFAULT 0, -- 连续被无视次数
    replies           INTEGER DEFAULT 0
);

-- ============================ 语音通话 ============================

-- 通话记录（每次来电一条；去电预留字段）
CREATE TABLE IF NOT EXISTS calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id           INTEGER NOT NULL,
    peer_id          INTEGER NOT NULL,
    peer_name        TEXT DEFAULT '',
    direction        TEXT DEFAULT 'incoming',   -- incoming | outgoing
    mode             TEXT DEFAULT 'voice',      -- voice | video
    decision         TEXT DEFAULT '',           -- answered | rejected | ignored | busy
    decision_reason  TEXT DEFAULT '',           -- 为什么接/不接（可读）
    ring_delay_ms    INTEGER DEFAULT 0,         -- 实际振铃了多久
    started_at       INTEGER DEFAULT 0,
    answered_at      INTEGER DEFAULT 0,
    ended_at         INTEGER DEFAULT 0,
    duration_sec     INTEGER DEFAULT 0,         -- 接通后的时长
    hangup_by        TEXT DEFAULT '',           -- bot | peer | timeout
    end_reason       TEXT DEFAULT '',           -- normal | timeout | failed
    turn_count       INTEGER DEFAULT 0,
    avg_latency_ms   INTEGER DEFAULT 0,         -- 平均响应延迟（调优用）
    mood             TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_calls_peer ON calls(bot_id, peer_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_time ON calls(bot_id, started_at DESC);

-- 通话转写（按句存，含各环节耗时，方便定位延迟瓶颈）
CREATE TABLE IF NOT EXISTS call_transcript (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id    INTEGER NOT NULL,
    ts         INTEGER DEFAULT 0,
    speaker    TEXT DEFAULT '',      -- bot | peer
    text       TEXT DEFAULT '',
    latency_ms INTEGER DEFAULT 0,    -- 对方说完 → 我方开口
    stt_ms     INTEGER DEFAULT 0,
    llm_ms     INTEGER DEFAULT 0,
    tts_ms     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_transcript_call ON call_transcript(call_id, id);

-- 通话事件（调试与拟人化分析：振铃/接听/打断/填充词/挂断）
CREATE TABLE IF NOT EXISTS call_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id  INTEGER NOT NULL,
    ts       INTEGER DEFAULT 0,
    event    TEXT DEFAULT '',
    detail   TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_callevent_call ON call_events(call_id, id);
"""


def _now() -> int:
    return int(time.time())


class Database:
    """SQLite 封装。线程安全（每线程独立连接）。"""

    def __init__(self, path: str) -> None:
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        self.init_schema()

    # -- 连接管理 ---------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    def init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self._harden()

    def _harden(self) -> None:
        """把库文件权限收到 600。

        库里存着控制台密码哈希与"供启动脚本打印"的明文密码，
        默认 644 会让同机其他用户读到，所以建库后立刻收紧。
        WAL 模式下数据会先写进 -wal 文件，所以三个文件都要收。
        """
        base = Path(self.path)
        for p in (base, Path(str(base) + "-wal"), Path(str(base) + "-shm")):
            try:
                if p.exists():
                    os.chmod(p, 0o600)
            except Exception:
                pass

    def harden_files(self) -> None:
        """外部（如定时任务）可反复调用，保证新生成的 WAL 也被收紧。"""
        self._harden()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """事务上下文。"""
        with self._lock:
            c = self.conn
            try:
                yield c
                c.commit()
            except Exception:
                c.rollback()
                raise

    # -- 通用 -------------------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self.conn.executemany(sql, list(seq))
            self.conn.commit()

    # -- 便捷方法 ---------------------------------------------------------

    def upsert_bot(self, bot_id: int, username: str, **kw: Any) -> None:
        row = self.query_one("SELECT bot_id FROM bots WHERE bot_id=?", (bot_id,))
        if row is None:
            self.execute(
                "INSERT INTO bots(bot_id, username, created_at, last_active, "
                "emotion_json) VALUES(?,?,?,?,?)",
                (bot_id, username, _now(), _now(),
                 json.dumps({"pleasure": 0.0, "energy": 0.6, "intimacy": 0.0,
                             "stress": 0.0})),
            )
        if kw:
            sets = ", ".join(f"{k}=?" for k in kw)
            self.execute(
                f"UPDATE bots SET {sets} WHERE bot_id=?",
                (*kw.values(), bot_id),
            )

    def get_bot(self, bot_id: int) -> Optional[sqlite3.Row]:
        return self.query_one("SELECT * FROM bots WHERE bot_id=?", (bot_id,))

    def get_conversation(self, bot_id: int, peer_id: int,
                         chat_type: str) -> Optional[sqlite3.Row]:
        return self.query_one(
            "SELECT * FROM conversations WHERE bot_id=? AND peer_id=? AND chat_type=?",
            (bot_id, peer_id, chat_type),
        )

    def ensure_conversation(self, bot_id: int, peer_id: int, chat_type: str,
                            title: str = "") -> int:
        row = self.get_conversation(bot_id, peer_id, chat_type)
        if row:
            return int(row["id"])
        cur = self.execute(
            "INSERT INTO conversations(bot_id, peer_id, chat_type, title, last_ts) "
            "VALUES(?,?,?,?,?)",
            (bot_id, peer_id, chat_type, title, _now()),
        )
        return int(cur.lastrowid)

    def save_cursor(self, conv_id: int, last_msg_id: int, last_seq_no: int) -> None:
        # 游标必须推进到「消息 ID」（服务端 id），否则 loadOfflineMessage(minId)
        # 会永远从 0 拉全部历史。这里做类型容错，调用方传 str 也能落库。
        try:
            last_msg_id = int(last_msg_id or 0)
        except (ValueError, TypeError):
            last_msg_id = 0
        try:
            last_seq_no = int(last_seq_no or 0)
        except (ValueError, TypeError):
            last_seq_no = 0
        self.execute(
            "UPDATE conversations SET last_msg_id=?, last_seq_no=?, last_ts=? WHERE id=?",
            (last_msg_id, last_seq_no, _now(), conv_id),
        )

    def conversations_of(self, bot_id: int) -> List[sqlite3.Row]:
        return self.query(
            "SELECT * FROM conversations WHERE bot_id=? ORDER BY last_ts DESC", (bot_id,)
        )

    def insert_message(self, bot_id: int, conv_id: int, role: str, content: str,
                       msg_id: Optional[str] = None, local_id: Optional[str] = None,
                       seq_no: int = 0, msg_type: int = 0, ts: Optional[int] = None,
                       at_user_ids: Optional[List[int]] = None, quote_id: int = 0,
                       extra: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """插入消息，msg_id 重复则忽略（返回 None）。

        注意：messages 表没有对 msg_id 建唯一约束（历史库可能已含重复数据，
        建索引会直接失败），所以 INSERT OR IGNORE 在这里不会去重——
        必须显式查重，否则重启补拉离线消息时会重复入库、重复回复。
        """
        if msg_id:
            dup = self.query_one(
                "SELECT id FROM messages WHERE bot_id=? AND conv_id=? AND msg_id=?",
                (bot_id, conv_id, msg_id),
            )
            if dup:
                return None
        try:
            cur = self.execute(
                "INSERT INTO messages(bot_id, conv_id, msg_id, local_id, "
                "seq_no, role, msg_type, content, at_user_ids, quote_id, ts, extra_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (bot_id, conv_id, msg_id, local_id, seq_no, role, msg_type, content,
                 json.dumps(at_user_ids or []), quote_id, ts or _now(),
                 json.dumps(extra or {}, ensure_ascii=False)),
            )
            return int(cur.lastrowid) if cur.rowcount else None
        except sqlite3.IntegrityError:
            return None

    def recent_messages(self, conv_id: int, limit: int = 20) -> List[sqlite3.Row]:
        rows = self.query(
            "SELECT * FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT ?",
            (conv_id, limit),
        )
        return list(reversed(rows))

    def add_memory(self, bot_id: int, peer_id: int, chat_type: str, kind: str,
                   text: str, embedding: Optional[bytes] = None,
                   importance: float = 0.5) -> int:
        cur = self.execute(
            "INSERT INTO memories(bot_id, peer_id, chat_type, kind, text, embedding, "
            "importance, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (bot_id, peer_id, chat_type, kind, text, embedding, importance, _now()),
        )
        return int(cur.lastrowid)

    def log_emotion(self, bot_id: int, peer_id: int, p: float, e: float, i: float,
                    s: float, mood: str, annoyance: int, phase: str = "") -> None:
        self.execute(
            "INSERT INTO emotion_log(bot_id, peer_id, ts, pleasure, energy, intimacy, "
            "stress, mood, annoyance, phase) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (bot_id, peer_id, _now(), p, e, i, s, mood, annoyance, phase),
        )

    def log_cost(self, task: str, provider_id: str, model: str, pt: int, ct: int,
                 cost: float, latency_ms: int, ok: bool, error: str = "") -> None:
        self.execute(
            "INSERT INTO cost_log(ts, task, provider_id, model, prompt_tokens, "
            "completion_tokens, cost_usd, latency_ms, ok, error) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (_now(), task, provider_id, model, pt, ct, cost, latency_ms, int(ok), error),
        )

    def log_audit(self, action: str, detail: str = "", ip: str = "",
                  actor: str = "admin") -> None:
        self.execute(
            "INSERT INTO audit_log(ts, actor, action, detail, ip) VALUES(?,?,?,?,?)",
            (_now(), actor, action, detail, ip),
        )

    def push_command(self, cmd: str, payload: Dict[str, Any]) -> int:
        cur = self.execute(
            "INSERT INTO control_commands(cmd, payload, created_at) VALUES(?,?,?)",
            (cmd, json.dumps(payload, ensure_ascii=False), _now()),
        )
        return int(cur.lastrowid)

    def pop_commands(self, limit: int = 20) -> List[sqlite3.Row]:
        with self.tx() as c:
            rows = c.execute(
                "SELECT * FROM control_commands WHERE status='pending' "
                "ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
            for r in rows:
                c.execute("UPDATE control_commands SET status='running' WHERE id=?",
                          (r["id"],))
        return rows

    def finish_command(self, cmd_id: int, ok: bool, result: str = "") -> None:
        self.execute(
            "UPDATE control_commands SET status=?, result=?, done_at=? WHERE id=?",
            ("done" if ok else "failed", result, _now(), cmd_id),
        )

    def get_setting(self, key: str, default: str = "") -> str:
        row = self.query_one("SELECT value FROM settings WHERE key=?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO settings(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # -- 语音通话 ---------------------------------------------------------

    def create_call(self, bot_id: int, peer_id: int, **kw: Any) -> int:
        """新建一条通话记录，返回 call_id。"""
        cols = ["bot_id", "peer_id", "started_at"]
        vals: List[Any] = [bot_id, peer_id, int(kw.pop("started_at", 0) or _now())]
        for k in ("peer_name", "direction", "mode", "decision", "decision_reason"):
            if k in kw:
                cols.append(k)
                vals.append(kw.pop(k))
        ph = ",".join(["?"] * len(cols))
        cur = self.execute(f"INSERT INTO calls({','.join(cols)}) VALUES({ph})", vals)
        return int(cur.lastrowid or 0)

    def update_call(self, call_id: int, **fields: Any) -> None:
        """更新通话记录（字段白名单，避免拼出不存在的列）。"""
        allowed = {"peer_name", "decision", "decision_reason", "ring_delay_ms",
                   "answered_at", "ended_at", "duration_sec", "hangup_by",
                   "end_reason", "turn_count", "avg_latency_ms", "mood"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        if not sets:
            return
        vals.append(call_id)
        self.execute(f"UPDATE calls SET {','.join(sets)} WHERE id=?", vals)

    def add_transcript(self, call_id: int, speaker: str, text: str,
                       latency_ms: int = 0, stt_ms: int = 0,
                       llm_ms: int = 0, tts_ms: int = 0) -> int:
        """记一句通话转写。各环节耗时分开存，方便定位延迟瓶颈。"""
        cur = self.execute(
            "INSERT INTO call_transcript(call_id, ts, speaker, text, latency_ms, "
            "stt_ms, llm_ms, tts_ms) VALUES(?,?,?,?,?,?,?,?)",
            (call_id, _now(), speaker, text, latency_ms, stt_ms, llm_ms, tts_ms),
        )
        return int(cur.lastrowid or 0)

    def add_call_event(self, call_id: int, event: str, detail: str = "") -> None:
        """记一条通话事件（振铃/接听/打断/填充词/挂断）。"""
        self.execute(
            "INSERT INTO call_events(call_id, ts, event, detail) VALUES(?,?,?,?)",
            (call_id, _now(), event, detail),
        )

    def recent_calls(self, bot_id: int, limit: int = 30) -> List[sqlite3.Row]:
        return self.query(
            "SELECT * FROM calls WHERE bot_id=? ORDER BY started_at DESC LIMIT ?",
            (bot_id, limit),
        )

    def calls_today(self, bot_id: int) -> int:
        """今天的通话次数（本地日期）。用于每日限额。"""
        start = int(time.mktime(time.strptime(time.strftime("%Y-%m-%d"),
                                              "%Y-%m-%d")))
        row = self.query_one(
            "SELECT COUNT(*) AS c FROM calls WHERE bot_id=? AND started_at>=?",
            (bot_id, start),
        )
        return int(row["c"]) if row else 0

    def call_transcript(self, call_id: int, limit: int = 200) -> List[sqlite3.Row]:
        return self.query(
            "SELECT * FROM call_transcript WHERE call_id=? ORDER BY id LIMIT ?",
            (call_id, limit),
        )

    def call_events(self, call_id: int) -> List[sqlite3.Row]:
        return self.query(
            "SELECT * FROM call_events WHERE call_id=? ORDER BY id", (call_id,),
        )

    def last_call_with(self, bot_id: int, peer_id: int,
                       exclude_id: int = 0) -> Optional[sqlite3.Row]:
        """与某人最近一次通话。冷却判断和"上次通话"上下文都要用。

        exclude_id：决策时本次来电已经先建了一条 calls 记录，必须把它排除，
        否则"上次通话"会查到刚创建的本次记录，导致"距上次通话 0 秒"永远冷却。
        """
        if exclude_id:
            return self.query_one(
                "SELECT * FROM calls WHERE bot_id=? AND peer_id=? AND id!=? "
                "ORDER BY started_at DESC LIMIT 1",
                (bot_id, peer_id, exclude_id),
            )
        return self.query_one(
            "SELECT * FROM calls WHERE bot_id=? AND peer_id=? "
            "ORDER BY started_at DESC LIMIT 1",
            (bot_id, peer_id),
        )
