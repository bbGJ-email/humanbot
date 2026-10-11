"""账号行为层：个签 / 头像 / 社交动作 / 生活时间线 / 主动消息（策划案 §8）。

这是"她有自己的生活"的落地层——不只是聊天，还会改状态、发动态、找人。
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..core.logging_setup import BUS, get_logger
from ..core.storage import Database
from ..core.utils import dice, now, pick, today_str

log = get_logger("account")


# 个签库（按心境分类）
SIGNATURE_BANK: Dict[str, List[str]] = {
    "平静": ["无事", "还行", "今天也是普普通通的一天", "混日子"],
    "挺开心的": ["今天天气不错", "壮壮发新叶了！", "买到想要的书了", "心情不错"],
    "有点累": ["累", "想睡觉", "上班如上坟", "今天好长"],
    "烦": ["烦", "别烦我", "。”", "不想说话"],
    "很烦，别惹我": ["。", "…", "呵呵", "勿扰"],
    "有点委屈": ["算了", "没什么", "没事"],
    "睡不着": ["又失眠", "有没有人还醒着", "凌晨三点的城市", "睡不着"],
}

# 生活时间线（BOT 自己的世界）
DEFAULT_LIFE_EVENTS: List[Dict[str, Any]] = [
    {"date": "每周一", "event": "闭店休息", "share_prob": 0.4},
    {"date": "每月1号", "event": "交房租，肉疼", "share_prob": 0.5},
    {"date": "每周1~2次", "event": "书店来了奇怪的客人", "share_prob": 0.6},
    {"date": "每周1~2次", "event": "淘到一本旧书", "share_prob": 0.6},
]

# 剧本事件（一次性）
SCRIPT_EVENTS: List[Dict[str, Any]] = [
    {
        "key": "plant_dies",
        "event": "壮壮（龟背竹）死了",
        "impact": {"pleasure": -0.5, "stress": 0.3},
        "duration_days": 3,
        "share_prob": 1.0,
        "min_age_days": 5,
    },
    {
        "key": "ex_appears",
        "event": "前男友在书店门口出现过",
        "impact": {"pleasure": -0.3, "stress": 0.4},
        "duration_days": 2,
        "share_prob": 0.2,
        "min_age_days": 20,
    },
    {
        "key": "found_letter",
        "event": "在旧书里翻到一封几十年前的信",
        "impact": {"pleasure": 0.3},
        "duration_days": 2,
        "share_prob": 0.8,
        "min_age_days": 12,
    },
]

# 主动消息触发话术
PROACTIVE_LINES = [
    "在干嘛",
    "你今天没找我说话",
    "我刚看到个东西想到你了",
    "你今天怎么样",
    "……",
    "诶",
    "睡了吗",
    "今天好累",
]


class AccountBehavior:
    """账号级行为：个签、头像、生活事件、主动消息。"""

    def __init__(self, db: Database, persona: Any, humanize: Any) -> None:
        self.db = db
        self.persona = persona
        self.cfg = humanize
        self._bot_id = 0
        self._started_at = now()
        self._last_sig_change = 0
        self._last_head_change = 0
        self._last_proactive: Dict[int, int] = {}   # peer_id -> ts
        self._ignored_streak: Dict[int, int] = {}

    def bind(self, bot_id: int) -> None:
        self._bot_id = bot_id
        row = self.db.get_bot(bot_id)
        if row and int(row["created_at"] or 0):
            self._started_at = int(row["created_at"])

    def age_days(self) -> float:
        return (now() - self._started_at) / 86400.0

    # -- 个签 -------------------------------------------------------------

    def maybe_update_signature(self, mood: str, prev_mood: str = "",
                               force: bool = False) -> Optional[str]:
        """情绪跃迁时换个签。每周 1~3 次。"""
        # 频率限制：至少间隔 2 天，除非 force
        if not force and (now() - self._last_sig_change) < 2 * 86400:
            return None
        # 只在心境变化时换
        if not force and mood == prev_mood:
            if not dice(0.15):
                return None

        bank = SIGNATURE_BANK.get(mood)
        if not bank:
            return None

        cur = self.db.get_setting(f"sig:{self._bot_id}", "")
        if cur and cur in bank:
            bank = [x for x in bank if x != cur] or bank
        new_sig = pick(bank)
        if new_sig == cur:
            return None

        self.db.set_setting(f"sig:{self._bot_id}", new_sig)
        self._last_sig_change = now()
        self._log_action("signature", {"text": new_sig, "mood": mood})
        log.info("个签 → %s（心境：%s）", new_sig, mood)
        BUS.publish("account", {"event": "signature", "bot_id": self._bot_id,
                                "text": new_sig, "mood": mood})
        return new_sig

    def current_signature(self) -> str:
        return self.db.get_setting(f"sig:{self._bot_id}", "")

    # -- 头像 -------------------------------------------------------------

    def maybe_update_head(self, mood: str, media: Any) -> Optional[str]:
        """2~4 周换一次头像；情绪极端时提前换。"""
        extreme = mood in ("很烦，别惹我", "有点委屈")
        gap = 7 * 86400 if extreme else random.randint(14, 28) * 86400
        if (now() - self._last_head_change) < gap:
            return None
        if media is None:
            return None

        prompt = self._head_prompt(mood)
        path = media.generate(prompt, seed=random.randint(1, 99999))
        if not path:
            return None
        self._last_head_change = now()
        self._log_action("head_image", {"path": path, "mood": mood})
        log.info("换头像（%s）: %s", mood, path)
        BUS.publish("account", {"event": "head_image", "bot_id": self._bot_id,
                                "path": path})
        return path

    @staticmethod
    def _head_prompt(mood: str) -> str:
        base = {
            "很烦，别惹我": "dark grey abstract minimal, moody, no people",
            "烦": "overcast sky over an empty street, desaturated",
            "有点委屈": "rain on a window, blurry, muted colors",
            "挺开心的": "green plant leaf close up, soft daylight",
            "睡不着": "night window with city lights, dark blue",
        }.get(mood, "old book on a wooden desk, warm indoor light")
        return base

    # -- 生活时间线 -------------------------------------------------------

    def tick_life_events(self) -> Optional[Dict[str, Any]]:
        """随机触发生活事件，写入 life_events。"""
        if not dice(0.06):      # 每次调用约 6% 概率
            return None

        pool = list(self.persona.data.get("bot_life_events")
                    or DEFAULT_LIFE_EVENTS)
        ev = pick(pool)
        if not isinstance(ev, dict):
            return None
        share = float(ev.get("share_prob", 0.5))
        content = str(ev.get("event", ""))
        if not content:
            return None
        self.db.execute(
            "INSERT INTO life_events(bot_id, ts, kind, content, impact, share_prob) "
            "VALUES(?,?,?,?,?,?)",
            (self._bot_id, now(), "routine", content, "{}", share),
        )
        log.info("生活事件：%s", content)
        BUS.publish("account", {"event": "life", "bot_id": self._bot_id,
                                "content": content})
        return {"content": content, "share_prob": share}

    def tick_script_events(self) -> Optional[Dict[str, Any]]:
        """检查一次性剧本事件是否该触发。"""
        age = self.age_days()
        for ev in SCRIPT_EVENTS:
            if age < float(ev.get("min_age_days", 0)):
                continue
            key = ev["key"]
            row = self.db.query_one(
                "SELECT 1 FROM script_events WHERE bot_id=? AND key=?",
                (self._bot_id, key))
            if row:
                continue
            # 达龄后 5% 概率触发
            if not dice(0.05):
                continue
            self.db.execute(
                "INSERT OR REPLACE INTO script_events(bot_id, key, fired_at) "
                "VALUES(?,?,?)", (self._bot_id, key, now()))
            self.db.execute(
                "INSERT INTO life_events(bot_id, ts, kind, content, impact, "
                "share_prob, expires_at) VALUES(?,?,?,?,?,?,?)",
                (self._bot_id, now(), "script", ev["event"],
                 json.dumps(ev.get("impact", {})), ev.get("share_prob", 0.5),
                 now() + int(ev.get("duration_days", 2)) * 86400),
            )
            log.info("剧本事件触发：%s", ev["event"])
            BUS.publish("account", {"event": "script", "bot_id": self._bot_id,
                                    "content": ev["event"]})
            return ev
        return None

    def pending_life_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM life_events WHERE bot_id=? AND shared=0 "
            "ORDER BY ts DESC LIMIT ?", (self._bot_id, limit))
        return [dict(r) for r in rows]

    def mark_shared(self, event_id: int) -> None:
        self.db.execute("UPDATE life_events SET shared=1 WHERE id=?", (event_id,))

    def active_impact(self) -> Dict[str, float]:
        """当前生效的剧本事件影响。"""
        rows = self.db.query(
            "SELECT impact FROM life_events WHERE bot_id=? AND kind='script' "
            "AND expires_at > ?", (self._bot_id, now()))
        total: Dict[str, float] = {}
        for r in rows:
            try:
                imp = json.loads(r["impact"] or "{}")
            except Exception:
                continue
            for k, v in imp.items():
                total[k] = total.get(k, 0.0) + float(v)
        return total

    # -- 主动消息 ---------------------------------------------------------

    def should_proactive(self, peer_id: int, *, is_online: bool,
                         hours_since: float, relation: str,
                         hour: int, ignored: int = 0) -> Tuple[bool, str]:
        """判断是否该主动找人（策划案 §7 主动消息规则）。

        硬前提（缺一不可）：
          1. 关系至少到「熟悉」——真人不会找个刚加的人说"我刚看到个东西想到你了"
          2. 之前确实聊过（hours_since 有效，即存在真实历史）
        否则再高的随机概率也不发。
        """
        # 关系门槛：陌生/认识阶段一律不主动
        if relation not in ("familiar", "close", "dependent"):
            return False, ""
        # 从没聊过的人不主动（避免好友列表里捡一个就发）
        if hours_since is None or hours_since < 0:
            return False, ""

        # 频率硬上限：24 小时最多 1 次
        last = self._last_proactive.get(peer_id, 0)
        if last and (now() - last) < 24 * 3600:
            return False, ""
        # 连续 3 次未回复 → 静默 7 天
        if ignored >= 3:
            if last and (now() - last) < 7 * 86400:
                return False, ""
            self._ignored_streak[peer_id] = 0

        reason = ""
        if is_online and hours_since > 6 and relation in ("familiar", "close",
                                                          "dependent"):
            reason = "online_gap"
        elif (hour >= 23 or hour < 2) and hours_since > 1:
            if relation in ("close", "dependent") and dice(0.25):
                reason = "insomnia"
        elif hours_since > 3 and dice(0.08):
            # 低频随机想起（已经聊过、且隔了几小时）
            reason = "random"

        if reason:
            self._last_proactive[peer_id] = now()
            line = pick(PROACTIVE_LINES)
            self._log_action("proactive", {"peer_id": peer_id, "reason": reason,
                                           "text": line})
            log.info("主动消息 → %s（%s）：%s", peer_id, reason, line)
            BUS.publish("account", {"event": "proactive", "bot_id": self._bot_id,
                                    "peer_id": peer_id, "reason": reason,
                                    "text": line})
            return True, line
        return False, ""

    def note_ignored(self, peer_id: int) -> None:
        self._ignored_streak[peer_id] = self._ignored_streak.get(peer_id, 0) + 1

    def note_replied(self, peer_id: int) -> None:
        self._ignored_streak[peer_id] = 0

    # -- 社交动作 ---------------------------------------------------------

    def set_dnd(self, conn: Any, peer_id: int, enabled: bool,
                is_group: bool = False) -> bool:
        """开/关免打扰——情绪武器。"""
        try:
            if is_group:
                conn.im.set_group_dnd(peer_id, enabled)
            else:
                conn.im.set_friend_dnd(peer_id, enabled)
            self._log_action("set_dnd", {"peer_id": peer_id, "enabled": enabled})
            log.info("免打扰 %s → %s", peer_id, enabled)
            return True
        except Exception as exc:
            log.warning("设置免打扰失败: %s", exc)
            return False

    def set_top(self, conn: Any, peer_id: int, enabled: bool,
                is_group: bool = False) -> bool:
        try:
            if is_group:
                conn.im.set_group_top(peer_id, enabled)
            else:
                conn.im.set_friend_top(peer_id, enabled)
            self._log_action("set_top", {"peer_id": peer_id, "enabled": enabled})
            return True
        except Exception as exc:
            log.warning("设置置顶失败: %s", exc)
            return False

    def update_remark(self, conn: Any, peer_id: int, remark: str) -> bool:
        try:
            conn.im.update_friend_remark(peer_id, remark)
            self._log_action("remark", {"peer_id": peer_id, "remark": remark})
            return True
        except Exception as exc:
            log.warning("改备注失败: %s", exc)
            return False

    def add_friend(self, conn: Any, peer_id: int, remark: str = "") -> bool:
        try:
            conn.im.add_friend(peer_id, remark or None)
            self._log_action("add_friend", {"peer_id": peer_id, "remark": remark})
            return True
        except Exception as exc:
            log.warning("加好友失败: %s", exc)
            return False

    def blacklist(self, conn: Any, peer_id: int, enable: bool = True) -> bool:
        """拉黑 / 解除拉黑（L5 用）。"""
        try:
            if enable:
                conn.im.add_to_blacklist(peer_id)
            else:
                conn.im.remove_from_blacklist(peer_id)
            self._log_action("blacklist", {"peer_id": peer_id, "enable": enable})
            log.info("拉黑 %s → %s", peer_id, enable)
            return True
        except Exception as exc:
            log.warning("拉黑操作失败: %s", exc)
            return False

    def delete_chat(self, conn: Any, chat_id: int, is_group: bool = False) -> bool:
        """删空聊天记录——情绪武器。"""
        try:
            if is_group:
                conn.im.delete_group_chat(chat_id)
            else:
                conn.im.delete_private_chat(chat_id)
            self._log_action("delete_chat", {"chat_id": chat_id})
            log.info("删除会话 %s", chat_id)
            return True
        except Exception as exc:
            log.warning("删除会话失败: %s", exc)
            return False

    def add_favorite(self, conn: Any, content: str, msg_id: int = 0) -> bool:
        try:
            conn.im.add_favorite(content=content, messageId=msg_id)
            self._log_action("favorite", {"content": content[:60]})
            return True
        except Exception as exc:
            log.debug("收藏失败: %s", exc)
            return False

    # -- 内部 -------------------------------------------------------------

    def _log_action(self, action: str, payload: Dict[str, Any]) -> None:
        if not self._bot_id:
            return
        try:
            self.db.execute(
                "INSERT INTO account_actions(bot_id, action, payload, ts) "
                "VALUES(?,?,?,?)",
                (self._bot_id, action,
                 json.dumps(payload, ensure_ascii=False), now()),
            )
        except Exception:
            pass

    def recent_actions(self, limit: int = 30) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM account_actions WHERE bot_id=? ORDER BY ts DESC LIMIT ?",
            (self._bot_id, limit))
        return [dict(r) for r in rows]
