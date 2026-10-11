"""核心工具：时间、随机、文本处理。"""
from __future__ import annotations

import json
import hashlib
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 时间
# ---------------------------------------------------------------------------

def now() -> int:
    return int(time.time())


def now_ms() -> int:
    return int(time.time() * 1000)


def today_str(d: Optional[datetime] = None) -> str:
    return (d or datetime.now()).strftime("%Y-%m-%d")


def minute_of_day(ts: Optional[int] = None) -> int:
    """当前时间是一天中的第几分钟（0-1439）。"""
    dt = datetime.fromtimestamp(ts) if ts else datetime.now()
    return dt.hour * 60 + dt.minute


def fmt_clock(minute: int) -> str:
    """把分钟数格式化为 HH:MM（支持跨日 >1440）。"""
    m = minute % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def human_time(ts: Optional[int] = None) -> str:
    dt = datetime.fromtimestamp(ts) if ts else datetime.now()
    h = dt.hour
    if h < 6:
        return "凌晨"
    if h < 9:
        return "早上"
    if h < 12:
        return "上午"
    if h < 14:
        return "中午"
    if h < 18:
        return "下午"
    if h < 23:
        return "晚上"
    return "深夜"


# ---------------------------------------------------------------------------
# 随机
# ---------------------------------------------------------------------------

def dice(p: float) -> bool:
    """以概率 p 返回 True。"""
    return random.random() < p


def jitter(base: float, spread: float) -> float:
    return base + random.uniform(-spread, spread)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def gauss_clamp(mu: float, sigma: float, lo: float, hi: float) -> float:
    return clamp(random.gauss(mu, sigma), lo, hi)


def pick(seq: Sequence[Any], default: Any = None) -> Any:
    if not seq:
        return default
    return random.choice(list(seq))


def pick_weighted(pairs: Sequence[tuple]) -> Any:
    """pairs: [(value, weight), ...]"""
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[0][0]
    r = random.random() * total
    acc = 0.0
    for v, w in pairs:
        acc += w
        if r <= acc:
            return v
    return pairs[-1][0]


# ---------------------------------------------------------------------------
# 文本
# ---------------------------------------------------------------------------

_SENT_END = "。！？!?…~～\n"


def split_sentences(text: str, max_chars: int = 45,
                    max_parts: int = 3) -> List[str]:
    """按语义边界把回复切成多条，模拟真人分条发送。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENT_END and len(buf) >= max_chars * 0.6:
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())

    # 合并过短的、拆开过长的
    merged: List[str] = []
    for p in parts:
        if merged and len(merged[-1]) + len(p) <= max_chars:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)

    # 超长再硬切
    final: List[str] = []
    for p in merged:
        while len(p) > max_chars * 1.8:
            final.append(p[:max_chars])
            p = p[max_chars:]
        if p:
            final.append(p)

    if len(final) > max_parts:
        final = final[:max_parts - 1] + ["".join(final[max_parts - 1:])]
    return [p for p in final if p]


_HOMOPHONE = {
    "的": "地", "地": "的", "在": "再", "再": "在",
    "是": "事", "事": "是", "己": "已", "已": "己",
    "那": "哪", "哪": "那", "候": "後", "后": "候",
}


def inject_typo(text: str, rate: float = 0.05) -> Optional[str]:
    """以一定概率制造一个同音错别字，返回错版文本；未触发返回 None。"""
    if not text or len(text) < 6 or random.random() > rate:
        return None
    candidates = [(i, c) for i, c in enumerate(text) if c in _HOMOPHONE]
    if not candidates:
        return None
    idx, ch = random.choice(candidates)
    return text[:idx] + _HOMOPHONE[ch] + text[idx + 1:]


def strip_markdown(text: str) -> str:
    """去掉 markdown 痕迹。"""
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", t, flags=re.S)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.M)
    t = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", t)
    return t.strip()


def text_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()[:16]


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """两段文本的 n-gram 重合度（0-1）。"""
    a, b = (a or "").strip(), (b or "").strip()
    if len(a) < n or len(b) < n:
        return 0.0
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def truncate(text: str, limit: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= limit else t[:limit] + "…"


# ---------------------------------------------------------------------------
# 消息体解析（BoxIM 回调给的 dict 结构）
# ---------------------------------------------------------------------------

def parse_message(raw: Dict[str, Any]) -> Dict[str, Any]:
    """把 BoxIM 推送的消息体标准化。"""
    mtype = int(raw.get("type") or 0)
    content = raw.get("content") or ""
    return {
        "msg_id": str(raw.get("id") or raw.get("messageId") or ""),
        "local_id": str(raw.get("localId") or raw.get("tmpId") or ""),
        "seq_no": int(raw.get("seqNo") or 0),
        "sender_id": int(raw.get("sendId") or raw.get("senderId") or 0),
        "receiver_id": int(raw.get("recvId") or raw.get("receiverId") or 0),
        "group_id": int(raw.get("groupId") or 0),
        "type": mtype,
        # 非文字消息的 content 是一坨 JSON，直接喂给模型会让她答非所问
        # （出现过她回"什么一串数字"）。这里换成人类可读的占位描述。
        "content": (content if mtype == MSG_TYPE_TEXT
                    else _describe_non_text(mtype, content)),
        "content_raw": content,
        "send_time": int(raw.get("sendTime") or now_ms()),
        "send_nick_name": raw.get("sendNickName") or "",
        "send_head_image": raw.get("sendHeadImage") or "",
        "at_user_ids": raw.get("atUserIds") or [],
        "quote_id": int(raw.get("quoteMessageId") or 0),
        "raw": raw,
    }


# -- 消息类型（对齐 SDK 的 MessageType 枚举）---------------------------------
MSG_TYPE_TEXT = 0
MSG_TYPE_IMAGE = 1
MSG_TYPE_FILE = 2
MSG_TYPE_AUDIO = 3
MSG_TYPE_VIDEO = 4
MSG_TYPE_USER_CARD = 5
MSG_TYPE_GROUP_CARD = 6
MSG_TYPE_STICKER = 7
MSG_TYPE_MERGE_FORWARD = 8
MSG_TYPE_RECALL = 10
MSG_TYPE_READED = 11
MSG_TYPE_RECEIPT = 12
MSG_TYPE_TIP_TIME = 20
MSG_TYPE_TIP_TEXT = 21
MSG_TYPE_LOADING = 30
MSG_TYPE_SYSTEM_MESSAGE = 53
MSG_TYPE_FRIEND_ONLINE = 82
MSG_TYPE_ONLINE_STATUS = 82

# 这些才是"人能说的话"，只有它们需要进对话链路。
# 其余（回执、已读、撤回、系统提示、好友/群事件、RTC 信令）都不该被回复——
# 之前不过滤，导致 BOT 会去回 {"id":19032170} 这种系统回执。
CHAT_MESSAGE_TYPES = frozenset({
    MSG_TYPE_TEXT, MSG_TYPE_IMAGE, MSG_TYPE_FILE, MSG_TYPE_AUDIO,
    MSG_TYPE_VIDEO, MSG_TYPE_USER_CARD, MSG_TYPE_GROUP_CARD,
    MSG_TYPE_STICKER, MSG_TYPE_MERGE_FORWARD,
})

_NON_TEXT_LABEL = {
    MSG_TYPE_IMAGE: "[图片]",
    MSG_TYPE_FILE: "[文件]",
    MSG_TYPE_AUDIO: "[语音]",
    MSG_TYPE_VIDEO: "[视频]",
    MSG_TYPE_USER_CARD: "[个人名片]",
    MSG_TYPE_GROUP_CARD: "[群名片]",
    MSG_TYPE_STICKER: "[表情]",
    MSG_TYPE_MERGE_FORWARD: "[合并转发]",
}


def _describe_non_text(mtype: int, content: Any) -> str:
    """把非文字消息渲染成一句可读描述，附带文件名/尺寸等信息。"""
    base = _NON_TEXT_LABEL.get(mtype, "[消息]")
    if not content:
        return base
    if isinstance(content, str) and content.lstrip().startswith("{"):
        try:
            obj = json.loads(content)
        except (ValueError, TypeError):
            return base
    elif isinstance(content, dict):
        obj = content
    else:
        return base
    if not isinstance(obj, dict):
        return base

    extra = ""
    if mtype == MSG_TYPE_FILE:
        extra = str(obj.get("fileName") or obj.get("name") or "")
    elif mtype == MSG_TYPE_AUDIO:
        sec = obj.get("duration") or obj.get("time")
        extra = f"{sec}秒" if sec else ""
    elif mtype == MSG_TYPE_VIDEO:
        sec = obj.get("duration")
        extra = f"{sec}秒" if sec else ""
    elif mtype == MSG_TYPE_STICKER:
        extra = str(obj.get("name") or "")
    return base + (f"（{extra}）" if extra else "")


def http_ts_to_sec(ms: int) -> int:
    """毫秒时间戳转秒。"""
    if not ms:
        return now()
    return int(ms / 1000) if ms > 10 ** 11 else int(ms)
