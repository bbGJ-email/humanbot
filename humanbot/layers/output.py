"""输出层：打字延迟 / 分条 / 错别字 / 撤回（策划案 §7）。"""
from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional

from ..core.logging_setup import BUS, get_logger
from ..core.storage import Database
from ..core.utils import (dice, inject_typo, now, pick, split_sentences,
                          strip_markdown)

log = get_logger("output")


class OutputPlanner:
    """把一段文本规划成"真人那样"的发送序列。"""

    def __init__(self, cfg: Any, db: Database) -> None:
        self.cfg = cfg
        self.db = db

    # -- 延迟计算 ---------------------------------------------------------

    def typing_delay(self, text: str, energy: float = 0.6,
                     drowsiness: float = 0.0) -> float:
        """打字延迟：base + 字数*系数，再按能量/睡意调节。"""
        c = self.cfg
        base = c.typing_base + len(text) * c.typing_per_char
        # 能量低 → 回得慢
        base *= (2.0 - max(0.1, energy))
        # 睡前迷糊 → 更慢
        base *= (1.0 + drowsiness * 2.0)
        return min(base, c.typing_cap * (1.0 + drowsiness))

    def read_delay(self) -> float:
        return random.uniform(self.cfg.read_delay_min, self.cfg.read_delay_max)

    def split_gap(self) -> float:
        return random.uniform(self.cfg.split_gap_min, self.cfg.split_gap_max)

    # -- 计划 -------------------------------------------------------------

    def plan(self, text: str, *, energy: float = 0.6, drowsiness: float = 0.0,
             mood: str = "", max_chars: Optional[int] = None,
             allow_typo: bool = True) -> List[Dict[str, Any]]:
        """生成发送计划：[{text, delay, typo, recall}]"""
        if not text:
            return []

        t = strip_markdown(text).strip()
        if not t:
            return []

        # 能量低 → 字数上限下调
        limit = max_chars or self.cfg.split_max_chars
        limit = int(limit * (1.0 - drowsiness * 0.7))
        limit = max(8, limit)

        parts = split_sentences(t, max_chars=limit,
                                max_parts=self.cfg.split_max_parts)

        plan: List[Dict[str, Any]] = []
        for i, p in enumerate(parts):
            item: Dict[str, Any] = {
                "text": p,
                "delay": self.typing_delay(p, energy, drowsiness),
                "typo": False,
                "recall": False,
            }
            # 错别字（仅长句，且概率随睡意翻倍）
            rate = self.cfg.typo_rate * (1.0 + drowsiness)
            if allow_typo and len(p) >= 8 and dice(rate):
                wrong = inject_typo(p, rate=1.0)
                if wrong and wrong != p:
                    item["typo_text"] = wrong
                    item["typo"] = True
            # 撤回重发（发出后 2 秒内）
            if i == 0 and dice(self.cfg.recall_rate) and len(p) > 10:
                item["recall"] = True
                item["recall_text"] = self._shorten(p)
            plan.append(item)

        # 条间停顿
        for i in range(1, len(plan)):
            plan[i]["delay"] = self.split_gap()

        return plan

    @staticmethod
    def _shorten(text: str) -> str:
        """撤回后重发的更短版本。"""
        t = re.sub(r"[，,].*$", "", text).strip()
        if not t:
            t = text[:max(2, len(text) // 2)]
        return t + ("。" if len(t) > 3 and not t.endswith(("…", "。")) else "")

    # -- 执行 -------------------------------------------------------------

    def execute(self, conn: Any, chat_id: int, plan: List[Dict[str, Any]],
                *, is_group: bool = False, at_users: Optional[List[int]] = None,
                quote_id: int = 0, bot_id: int = 0, db: Optional[Database] = None,
                conv_id: int = 0) -> int:
        """按计划逐条发送（阻塞执行）。返回实际发送条数。

        注意：本方法会 sleep，调用方必须放在线程池里跑，不能阻塞事件循环。
        """
        sent = 0
        for item in plan:
            time.sleep(item["delay"])

            text = item.get("typo_text") if item["typo"] else item["text"]
            # send_text 现在返回真实消息 id（失败为 None），撤回要用它
            msg_id = conn.send_text(chat_id, text, is_group=is_group,
                                    at_users=at_users, quote_id=quote_id)
            if not msg_id:
                continue
            sent += 1
            self._record(db, bot_id, conv_id, text)

            # 错别字：1.5 秒后补一条修正
            if item["typo"]:
                time.sleep(1.5)
                fix = f"{item['text']}（打错了）"
                if conn.send_text(chat_id, fix, is_group=is_group):
                    sent += 1
                    self._record(db, bot_id, conv_id, fix)

            # 撤回重发：一定要用上面那条的真实 id，
            # 之前传的是 0，服务端只会回"消息不存在"
            if item["recall"] and item.get("recall_text"):
                time.sleep(2.0)
                if not conn.recall(msg_id, is_group):
                    # 撤回失败就别再补发短版了，否则会出现两句重复内容
                    log.debug("撤回失败，跳过重发")
                    continue
                short = item["recall_text"]
                if conn.send_text(chat_id, short, is_group=is_group):
                    sent += 1
                    self._record(db, bot_id, conv_id, short)

        if sent:
            BUS.publish("outbound", {
                "bot_id": bot_id, "chat_id": chat_id, "is_group": is_group,
                "count": sent,
            })
        return sent

    @staticmethod
    def _record(db: Optional[Database], bot_id: int, conv_id: int, text: str) -> None:
        if db is None or not bot_id or not conv_id:
            return
        try:
            db.insert_message(bot_id, conv_id, "bot", text, ts=now())
            db.execute(
                "UPDATE conversations SET last_ts=? WHERE id=?", (now(), conv_id))
        except Exception:
            pass

    # -- 打字中状态 -------------------------------------------------------

    def send_loading(self, conn: Any, chat_id: int, is_group: bool = False) -> None:
        """发送 LOADING(30) 状态——长回复时用。"""
        try:
            if is_group:
                conn.im.send_group_raw(chat_id, "", 30)
            else:
                conn.im.send_private_raw(chat_id, "", 30)
        except Exception as exc:
            log.debug("发送 LOADING 状态失败: %s", exc)
