"""调度层：消息合并防抖 + 优先级队列 + 群聊响应权重（策划案 §6 / §7.1）。"""
from __future__ import annotations

import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core.logging_setup import BUS, get_logger
from ..core.storage import Database
from ..core.utils import dice, now, pick_weighted
from .connection import InboundMessage

log = get_logger("sched")


@dataclass
class QueuedMessage:
    """队列中的一条待处理消息。"""

    msg: InboundMessage
    priority: int = 0            # 越大越优先
    weight: float = 1.0          # 群聊响应权重
    enqueued_at: int = 0

    def __post_init__(self) -> None:
        if not self.enqueued_at:
            self.enqueued_at = now()


@dataclass
class MergeBucket:
    """同一会话的消息合并桶。"""

    bot_id: int = 0
    chat_id: int = 0
    is_group: bool = False
    texts: List[str] = field(default_factory=list)
    msgs: List[InboundMessage] = field(default_factory=list)
    opened_at: float = 0.0
    timer_handle: Any = None
    # 这批消息是不是"她自己想插嘴"（没人 @ 也没提到她）
    proactive: bool = False
    # 触发时的权重（用于事后判断这次开口值不值）
    weight: float = 1.0

    @property
    def key(self) -> str:
        return f"{self.bot_id}:{self.is_group}:{self.chat_id}"


class Scheduler:
    """消息调度：合并、防抖、抢占、群聊权重。"""

    def __init__(self, db: Database, humanize: Any,
                 handler: Callable[[List[InboundMessage], bool], None],
                 interest: Any = None, group_cfg: Any = None) -> None:
        self.db = db
        self.cfg = humanize
        self.handler = handler            # 收到合并完成的一批消息后调用
        self.interest = interest          # InterestModel（可为 None）
        self.gcfg = group_cfg             # GroupConfig（可为 None）

        self._buckets: Dict[str, MergeBucket] = {}
        self._busy: Set[str] = set()      # 正在生成的会话
        self._queue: "asyncio.Queue[Tuple[str, MergeBucket]]" = asyncio.Queue()
        self._lock = asyncio.Lock() if self._has_loop() else None
        self._dnd: Set[str] = set()       # 被静音的会话
        self._burst: Dict[str, deque] = {}  # 群活跃度统计

    @staticmethod
    def _has_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    # -- 群聊权重 ---------------------------------------------------------

    def group_weight(self, msg: InboundMessage, aliases: List[str],
                     interests: List[str],
                     bot_id: int) -> float:
        """按策划案 §7.1 的决策树算响应权重。

        决策树（取最高优先级命中项作为底座）：
            @她            → group_at_weight        (1.0)
            提到她的名字     → group_mention_weight   (0.8)
            感兴趣的话题     → interest_max_weight × 兴趣得分
            完全无关         → group_other_weight     (0.05)

        然后叠加「像真人」的修正项：
            + 参与惯性      刚在这个群说过话 → 更容易接话
            + 话题连续性    同一话题反复出现 → 加成（有上限）
            × 插话抑制      刚回过又回 → 降权，避免刷屏
            × 刷屏抑制      群消息密度过高 → 降权
        """
        text = msg.text or ""
        key = f"{bot_id}:{msg.chat_id}"

        # BOT 互相忽略
        if msg.sender_id == bot_id:
            return 0.0

        g = self.gcfg

        # ---- 1. 底座权重 ----
        if msg.at_me:
            w = self.cfg.group_at_weight
            basis = "at"
        elif any(a and a in text for a in aliases):
            w = self.cfg.group_mention_weight
            basis = "mention"
        else:
            # 兴趣通道：静态关键词 + 学习到的话题热度
            interest_score = 0.0
            if self.interest is not None:
                try:
                    interest_score = self.interest.score(
                        bot_id, msg.chat_id, text)
                except Exception as exc:
                    log.debug("兴趣打分失败: %s", exc)
            # 配置里手写的关键词命中，直接给一个下限
            kw_hit = any(k and k in text for k in interests)
            if kw_hit:
                interest_score = max(interest_score, g.interest_min_score if g else 0.25)

            if (g is not None and interest_score >= g.interest_min_score):
                w = g.interest_max_weight * interest_score
                basis = "interest %.2f" % interest_score
            elif kw_hit:
                w = self.cfg.group_interest_weight
                basis = "keyword"
            else:
                w = self.cfg.group_other_weight if g is None else g.floor_weight
                basis = "other"

        # ---- 2. 参与惯性 ----
        if g is not None and self.interest is not None and basis != "at":
            try:
                mom = self.interest.momentum(msg.chat_id, bot_id,
                                             g.momentum_window_sec)
                if mom > 0:
                    w += mom * g.momentum_bonus
            except Exception:
                pass

        # ---- 3. 话题连续性加成 ----
        if g is not None and self.interest is not None and basis.startswith("interest"):
            try:
                heat = self.interest.heat_of(bot_id, msg.chat_id,
                                             int(g.topic_window_sec))
                if heat:
                    cur = max((v for k, v in heat.items() if k in text),
                              default=0.0)
                    w += min(g.topic_boost_max, cur * 0.06)
            except Exception:
                pass

        # ---- 4. 插话抑制 ----
        if g is not None and self.interest is not None:
            try:
                if self.interest.just_interjected(msg.chat_id, bot_id,
                                                  g.interrupt_window_sec):
                    w *= g.interrupt_penalty
            except Exception:
                pass

        # ---- 5. 刷屏抑制 ----
        dq = self._burst.setdefault(key, deque(maxlen=20))
        t = time.time()
        dq.append(t)
        recent = sum(1 for x in dq if t - x <= 60)
        if recent > 5:
            w *= self.cfg.group_burst_penalty

        log.debug("群权重 %.3f (%s)", w, basis)
        return w

    def _group_rules(self, group_id: int, bot_id: int) -> Tuple[List[str], List[str]]:
        row = self.db.query_one(
            "SELECT * FROM group_rules WHERE group_id=? AND bot_id=?",
            (group_id, bot_id),
        )
        if not row:
            return [], []
        import json
        try:
            aliases = json.loads(row["aliases"] or "[]")
        except Exception:
            aliases = []
        try:
            interests = json.loads(row["interest_keywords"] or "[]")
        except Exception:
            interests = []
        return aliases, interests

    # -- 入队 -------------------------------------------------------------

    def submit(self, msg: InboundMessage, aliases: Optional[List[str]] = None,
               interest_keywords: Optional[List[str]] = None) -> None:
        """消息入口：判断是否响应，然后进合并桶。

        aliases 是"叫到她"的所有可能名字（IM 昵称 + 人设名 + 用户名）。
        """
        is_group = msg.is_group
        al: List[str] = []
        interests: List[str] = interest_keywords or []

        if is_group:
            db_aliases, db_interests = self._group_rules(msg.chat_id, msg.bot_id)
            al = list(db_aliases) + list(aliases or [])
            interests = list(set(interests) | set(db_interests))
            # 去重保序
            seen = set()
            al = [a for a in al if a and not (a in seen or seen.add(a))]
            # 群免打扰
            row = self.db.query_one(
                "SELECT muted_until FROM group_rules WHERE group_id=? AND bot_id=?",
                (msg.chat_id, msg.bot_id),
            )
            if row and int(row["muted_until"] or 0) > now():
                log.debug("群 %s 免打扰中，忽略", msg.chat_id)
                return
            weight = self.group_weight(msg, al, interests, msg.bot_id)

            # 记话题热度：只要在聊就累积，回复了的话权重更高
            responding = weight >= 0.5 or dice(min(0.95, weight * 1.2))
            if self.interest is not None:
                try:
                    self.interest.note(msg.bot_id, msg.chat_id, msg.text,
                                       engaged=responding)
                    # 顺便清理过期热度（低频操作）
                    if dice(0.02):
                        self.interest.decay(msg.bot_id, msg.chat_id)
                except Exception as exc:
                    log.debug("话题累积失败: %s", exc)

            if not responding:
                log.debug("群消息权重 %.2f，选择不响应", weight)
                self._store_only(msg)
                return
            # 是不是"自己想插嘴"：没 @、没提名字都算主动参与
            proactive = not msg.at_me and not any(
                a and a in (msg.text or "") for a in al)
        else:
            weight = 1.0
            proactive = False
            # 私聊如果被静音（控制台操作）
            if self._conv_key(msg) in self._dnd:
                self._store_only(msg)
                return

        self._enqueue(msg, weight, proactive=proactive)

    def _conv_key(self, msg: InboundMessage) -> str:
        return f"{msg.bot_id}:{msg.is_group}:{msg.chat_id}"

    def _store_only(self, msg: InboundMessage) -> None:
        """只入库，不触发回复。"""
        try:
            conv_id = self.db.ensure_conversation(
                msg.bot_id, msg.chat_id, msg.chat_type)
            self.db.insert_message(
                msg.bot_id, conv_id, "user", msg.text, msg_id=msg.msg_id,
                local_id=msg.local_id, seq_no=msg.seq_no, msg_type=msg.msg_type,
                ts=msg.ts, at_user_ids=msg.at_user_ids, quote_id=msg.quote_id,
            )
        except Exception as exc:
            log.warning("入库失败: %s", exc)

    def _enqueue(self, msg: InboundMessage, weight: float,
                 proactive: bool = False) -> None:
        """进合并桶，启动/重置防抖计时器。"""
        key = self._conv_key(msg)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = MergeBucket(bot_id=msg.bot_id, chat_id=msg.chat_id,
                                 is_group=msg.is_group, opened_at=time.time())
            self._buckets[key] = bucket

        bucket.texts.append(msg.text)
        bucket.msgs.append(msg)
        # 主动参与标记：桶里只要有一条是"没人叫她"，就算主动
        bucket.proactive = bucket.proactive or proactive
        bucket.weight = max(bucket.weight, weight)

        # 抢占：明显是"算了/撤回"类，立即触发
        if re.search(r"(算了|当我没说|没事了|不说了)", msg.text or ""):
            self._flush(key)
            return

        # 重置防抖窗口
        self._restart_timer(key)

    def _restart_timer(self, key: str) -> None:
        bucket = self._buckets.get(key)
        if bucket is None:
            return
        if bucket.timer_handle is not None:
            try:
                bucket.timer_handle.cancel()
            except Exception:
                pass
        window = random.uniform(self.cfg.merge_window_min,
                                self.cfg.merge_window_max)
        try:
            loop = asyncio.get_running_loop()
            bucket.timer_handle = loop.call_later(window, self._flush, key)
        except RuntimeError:
            # 无事件循环时立即处理
            self._flush(key)

    def _flush(self, key: str) -> None:
        """防抖窗口到期，交给处理函数。"""
        bucket = self._buckets.pop(key, None)
        if not bucket or not bucket.msgs:
            return
        if bucket.timer_handle is not None:
            try:
                bucket.timer_handle.cancel()
            except Exception:
                pass
        merged_text = " ".join(t for t in bucket.texts if t).strip()
        log.info("处理合并消息（%d 条）: %s", len(bucket.msgs), merged_text[:60])
        BUS.publish("sched", {"event": "flush", "chat_id": bucket.chat_id,
                              "count": len(bucket.msgs), "text": merged_text[:80]})
        try:
            self.handler(bucket.msgs, merged_text, bucket)
        except TypeError:
            # 兼容只接受两个参数的旧 handler
            self.handler(bucket.msgs, merged_text)
        except Exception as exc:
            log.error("调度处理异常: %s", exc)

    # -- 控制 -------------------------------------------------------------

    def mute(self, bot_id: int, chat_id: int, is_group: bool) -> None:
        self._dnd.add(f"{bot_id}:{is_group}:{chat_id}")

    def unmute(self, bot_id: int, chat_id: int, is_group: bool) -> None:
        self._dnd.discard(f"{bot_id}:{is_group}:{chat_id}")

    def is_muted(self, bot_id: int, chat_id: int, is_group: bool) -> bool:
        return f"{bot_id}:{is_group}:{chat_id}" in self._dnd

    def clear(self) -> None:
        for key in list(self._buckets.keys()):
            self._flush(key)

    # -- 防封：发送节流 ---------------------------------------------------

    def min_interval_ok(self, last_sent: float) -> bool:
        return (time.time() - last_sent) >= self.cfg.min_send_interval

    def daily_limit_ok(self, bot_id: int) -> bool:
        row = self.db.query_one(
            "SELECT COUNT(*) AS c FROM messages WHERE bot_id=? AND role='bot' "
            "AND ts >= ?", (bot_id, now() - 86400),
        )
        cnt = int(row["c"]) if row else 0
        if cnt >= self.cfg.daily_send_limit:
            log.warning("BOT %s 已达日发送上限 %d", bot_id, self.cfg.daily_send_limit)
            return False
        return True
