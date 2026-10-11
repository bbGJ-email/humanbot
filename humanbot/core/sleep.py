"""睡眠与作息系统（策划案 §6.5）。

四态状态机：AWAKE / DROWSY / ASLEEP / WAKING
- 入睡时间随机漂移（基准 + 多重扰动），避开整点
- 睡着时真正断开 WebSocket，靠服务端离线消息补偿
- 睡前有渐变（延迟递增、字数递减），醒来有渐醒流程
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ..settings import SleepConfig
from .logging_setup import BUS, get_logger
from .storage import Database
from .utils import (clamp, dice, gauss_clamp, minute_of_day, now, pick,
                    today_str)

log = get_logger("sleep")

# 状态常量
AWAKE = "AWAKE"
DROWSY = "DROWSY"
ASLEEP = "ASLEEP"
WAKING = "WAKING"

# 睡前告别话术（策划案 §6.5.3）
SLEEP_PRETEXT = [
    "我去洗澡了",
    "困了，睡了",
    "我要睡觉了，明天聊",
    "眼睛睁不开了",
    "先不聊了，明天还要上班",
    "……",  # 20% 概率用"直接不回"由 p_no_goodbye 控制
]
GOODBYE_LINE = "嗯 晚安"   # 全天唯一一次用"晚安"

# 醒来渐变
WAKE_SHORT = ["早", "刚醒", "嗯我看到了", "起了", "……早"]

# 紧急关键词（醒来优先处理）
URGENT_KEYWORDS = ("急", "出事了", "救命", "在吗在吗", "怎么办", "崩了",
                   "出事了", "快回", "有件事")

# 紧急回复措辞（按对方发消息的时间分层）
URGENT_REPLY = {
    "early":  ["我睡得早…刚看到，怎么了", "刚醒，看到你消息了，怎么了"],
    "late":   ["我睡着了，你还好吗", "刚看到…出什么事了"],
    "dawn":   ["你三点多还没睡？发生什么了", "这个点还没睡？怎么了"],
}


@dataclass
class SleepPlan:
    """每晚睡眠计划。"""

    sleep_minute: int          # 入睡时刻（0-1439 或跨日 >1440）
    wake_minute: int           # 醒来时刻
    is_all_nighter: bool = False
    no_goodbye: bool = False
    sleep_back: bool = False
    oversleep: bool = False
    revenge: bool = False
    date: str = ""
    debt: int = 0

    @property
    def duration_min(self) -> int:
        return max(0, self.wake_minute - self.sleep_minute)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sleep": f"{self.sleep_minute // 60 % 24:02d}:{self.sleep_minute % 60:02d}",
            "wake": f"{self.wake_minute // 60 % 24:02d}:{self.wake_minute % 60:02d}",
            "all_nighter": self.is_all_nighter,
            "no_goodbye": self.no_goodbye,
        }


class SleepScheduler:
    """睡眠调度：计算入睡/醒来时间，管理状态机。"""

    def __init__(self, db: Database, cfg: SleepConfig) -> None:
        self.db = db
        self.cfg = cfg
        self._plans: Dict[int, SleepPlan] = {}      # bot_id -> plan
        self._states: Dict[int, str] = {}           # bot_id -> state
        self._on_sleep: Dict[int, Callable[[], None]] = {}
        self._on_wake: Dict[int, Callable[[], None]] = {}

    # -- 注册回调 ---------------------------------------------------------

    def bind(self, bot_id: int, on_sleep: Callable[[], None],
             on_wake: Callable[[], None]) -> None:
        self._on_sleep[bot_id] = on_sleep
        self._on_wake[bot_id] = on_wake

    def state_of(self, bot_id: int) -> str:
        return self._states.get(bot_id, AWAKE)

    def set_state(self, bot_id: int, state: str) -> None:
        self._states[bot_id] = state
        self.db.execute("UPDATE bots SET state=? WHERE bot_id=?", (state, bot_id))
        BUS.publish("state", {"bot_id": bot_id, "state": state})

    # -- 计划计算 ---------------------------------------------------------

    def plan_for(self, bot_id: int, energy: float = 0.6,
                 pleasure: float = 0.0, weekday: Optional[int] = None
                 ) -> SleepPlan:
        """按策划案 §6.5.2 的五重扰动生成今晚的睡眠计划。"""
        cfg = self.cfg
        d = weekday if weekday is not None else datetime.now().weekday()
        day_str = today_str()
        # 缓存键带上星期几：显式传入 weekday 时不复用别的星期的计划
        cache_key = f"{bot_id}:{day_str}:{d}"

        # 已有同一晚的计划则复用
        cached = self._plans.get(cache_key)
        if cached and cached.date == day_str:
            return cached

        # 昨晚睡眠债
        debt = self._sleep_debt(bot_id)

        base = cfg.base_sleep_minute
        jitter = gauss_clamp(0, cfg.sleep_jitter_sigma, -180, 300)

        # 星期几效应：周一闭店，晚上精神
        weekday_effect = {0: -30, 1: 5, 2: 10, 3: 0, 4: 15, 5: 20, 6: 5}
        jitter += weekday_effect.get(d, 0)

        # 当天情绪
        if energy > 0.75:
            jitter -= 45
        elif energy < 0.3:
            jitter += 60
        if pleasure < -0.3:
            jitter -= 40

        # 睡眠债（熬过夜今晚更晚）
        if debt > 60:
            jitter += min(90, debt * 0.6)

        # 报复性熬夜
        revenge = pleasure < -0.3
        if revenge:
            jitter -= random.randint(40, 90)

        # 7% 整晚失眠
        if dice(cfg.p_all_nighter):
            plan = SleepPlan(sleep_minute=0, wake_minute=0,
                             is_all_nighter=True, date=day_str, debt=debt)
            self._plans[cache_key] = plan
            return plan

        sleep_min = int(clamp(base + jitter, cfg.sleep_earliest, cfg.sleep_latest))
        # 避开整点 ±5 分钟
        if cfg.never_on_the_hour:
            off = sleep_min % 60
            if off <= 5:
                sleep_min += random.randint(6, 12)
            elif off >= 55:
                sleep_min -= random.randint(6, 12)

        # 睡眠时长
        dur_h = gauss_clamp(cfg.duration_mean, cfg.duration_sigma,
                            cfg.duration_min, cfg.duration_max)
        # 睡太晚就睡得短一点（不能无限赖床）
        if sleep_min > 25 * 60:
            dur_h -= 1.0
        # 周末补觉
        if d in (5, 6):
            dur_h += 0.8

        # 醒来时刻统一用"跨日分钟"表达（>=1440 表示次日），
        # 再夹到 05:30 ~ 12:00 的钟点区间，然后换回跨日刻度。
        wake_min = sleep_min + int(dur_h * 60)
        r = random.random()
        oversleep = wake_early = False
        if r < cfg.p_oversleep:
            wake_min += random.randint(30, 90)
            oversleep = True
        elif r < cfg.p_oversleep + cfg.p_wake_early:
            wake_min -= random.randint(30, 60)
            wake_early = True

        wake_clock = wake_min % 1440                       # 钟点
        wake_clock = int(clamp(wake_clock, 5 * 60 + 30, 12 * 60))
        # 换回跨日刻度：若钟点早于入睡钟点，说明跨到了第二天
        if wake_clock < sleep_min % 1440:
            wake_clock += 1440
        if wake_clock <= sleep_min:
            wake_clock = sleep_min + int(cfg.duration_min * 60)

        plan = SleepPlan(
            sleep_minute=sleep_min,
            wake_minute=int(wake_clock),
            no_goodbye=dice(cfg.p_no_goodbye),
            sleep_back=dice(cfg.p_sleep_back),
            oversleep=oversleep,
            revenge=revenge,
            date=day_str,
            debt=debt,
        )
        self._plans[cache_key] = plan
        log.info("今晚作息计划：睡 %02d:%02d → 醒 %02d:%02d%s",
                 plan.sleep_minute // 60, plan.sleep_minute % 60,
                 plan.wake_minute // 60, plan.wake_minute % 60,
                 "（报复性熬夜）" if revenge else "")
        return plan

    def _sleep_debt(self, bot_id: int) -> int:
        row = self.db.query_one(
            "SELECT sleep_debt FROM sleep_log WHERE bot_id=? "
            "ORDER BY date DESC LIMIT 1", (bot_id,),
        )
        return int(row["sleep_debt"]) if row else 0

    # -- 状态判定 ---------------------------------------------------------

    def current_state(self, bot_id: int, energy: float = 0.6,
                      pleasure: float = 0.0) -> str:
        """根据当前时间与计划推断应处的状态。

        时间刻度说明：sleep_minute / wake_minute 用"跨日分钟"表示，
        例如 1430=23:50、1452=次日 00:12、1782=次日 05:42。
        判断时必须先折算成钟点再比区间，否则"凌晨入睡"的计划
        会让 BOT 一整个上午都以为自己还在睡。
        """
        if not self.cfg.enabled:
            return AWAKE
        plan = self.plan_for(bot_id, energy, pleasure)
        if plan.is_all_nighter:
            return AWAKE

        m = minute_of_day()                       # 当前钟点 0~1439
        sm_c = plan.sleep_minute % 1440           # 入睡钟点
        wm_c = plan.wake_minute % 1440            # 起床钟点

        # -- 睡眠区间（含跨午夜）--
        if sm_c <= wm_c:
            asleep = sm_c <= m < wm_c             # 当天内区间
        else:
            asleep = m >= sm_c or m < wm_c        # 跨午夜：晚段 ∪ 早段

        # -- 入睡前的困倦窗口（同样要跨午夜处理）--
        lead = self.cfg.drowsy_lead_min
        d_from = (sm_c - lead) % 1440
        if d_from <= sm_c:
            drowsy = d_from <= m < sm_c
        else:
            drowsy = m >= d_from or m < sm_c

        if drowsy and not asleep:
            return DROWSY
        if asleep:
            return ASLEEP
        # -- 刚醒的迷糊窗口 --
        if wm_c <= m < wm_c + self.cfg.waking_lead_min:
            return WAKING
        return AWAKE

    def drowsiness(self, bot_id: int) -> float:
        """睡意程度 0~1，用于调节延迟与字数。"""
        if not self.cfg.enabled:
            return 0.0
        plan = self.plan_for(bot_id)
        if plan.is_all_nighter:
            return 0.0
        m = minute_of_day()
        sm = plan.sleep_minute
        lead = self.cfg.drowsy_lead_min
        if sm - lead <= m < sm:
            return clamp((m - (sm - lead)) / max(1, lead), 0.0, 1.0)
        if m >= sm - 20 or m < 60:
            return 0.6
        return 0.0

    # -- 状态迁移 ---------------------------------------------------------

    def check_transition(self, bot_id: int, energy: float = 0.6,
                         pleasure: float = 0.0) -> Optional[str]:
        """检查是否发生状态迁移，触发对应回调。返回新状态（无变化返回 None）。"""
        want = self.current_state(bot_id, energy, pleasure)
        cur = self._states.get(bot_id, AWAKE)
        if want == cur:
            return None

        self.set_state(bot_id, want)
        log.info("状态迁移 %s → %s", cur, want)

        if want == ASLEEP and cur != ASLEEP:
            self._record_sleep(bot_id)
            cb = self._on_sleep.get(bot_id)
            if cb:
                try:
                    cb()
                except Exception as exc:
                    log.error("on_sleep 回调异常: %s", exc)
        elif cur == ASLEEP and want in (WAKING, AWAKE):
            self._record_wake(bot_id)
            cb = self._on_wake.get(bot_id)
            if cb:
                try:
                    cb()
                except Exception as exc:
                    log.error("on_wake 回调异常: %s", exc)
        return want

    def _record_sleep(self, bot_id: int) -> None:
        plan = self.plan_for(bot_id)
        self.db.execute(
            "INSERT OR REPLACE INTO sleep_log(bot_id, date, sleep_at, wake_at, "
            "duration_min, quality, is_all_nighter, sleep_debt) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (bot_id, plan.date, now(), 0, 0, "normal", int(plan.is_all_nighter),
             self._sleep_debt(bot_id)),
        )

    def _record_wake(self, bot_id: int) -> None:
        plan = self.plan_for(bot_id)
        duration = plan.duration_min
        # 睡眠债：少于 7 小时则累积
        debt = max(0, int((7 * 60 - duration) / 60 * 60)) if duration else 0
        quality = "good" if duration >= 7 * 60 else ("poor" if duration < 6 * 60 else "normal")
        self.db.execute(
            "UPDATE sleep_log SET wake_at=?, duration_min=?, quality=?, sleep_debt=? "
            "WHERE bot_id=? AND date=?",
            (now(), duration, quality, debt, bot_id, plan.date),
        )

    # -- 睡前/醒后行为生成 -----------------------------------------------

    def goodbye_line(self, bot_id: int) -> Optional[str]:
        """睡前告别。20% 概率不告而别（返回 None）。"""
        plan = self.plan_for(bot_id)
        if plan.no_goodbye:
            return None
        return pick(SLEEP_PRETEXT[:-1] + [GOODBYE_LINE])

    def wake_short_line(self) -> str:
        return pick(WAKE_SHORT)

    def urgent_reply_for(self, msg_ts: int) -> str:
        """按对方发消息的时间生成紧急回复。"""
        dt = datetime.fromtimestamp(msg_ts) if msg_ts else datetime.now()
        h = dt.hour
        if h >= 3 and h < 6:
            return pick(URGENT_REPLY["dawn"])
        if h >= 0 and h < 3:
            return pick(URGENT_REPLY["late"])
        return pick(URGENT_REPLY["early"])

    @staticmethod
    def is_urgent(text: str) -> bool:
        return any(k in (text or "") for k in URGENT_KEYWORDS)

    # -- 对外描述 ---------------------------------------------------------

    def describe(self, bot_id: int) -> Dict[str, Any]:
        plan = self.plan_for(bot_id)
        return {
            "state": self.state_of(bot_id),
            "plan": plan.to_dict(),
            "drowsiness": round(self.drowsiness(bot_id), 2),
        }

    def last_sleep_text(self, bot_id: int) -> str:
        """用于回答"昨晚几点睡的"——必须与记录一致。"""
        row = self.db.query_one(
            "SELECT * FROM sleep_log WHERE bot_id=? ORDER BY date DESC LIMIT 1",
            (bot_id,),
        )
        if not row or not row["sleep_at"]:
            return ""
        dt = datetime.fromtimestamp(int(row["sleep_at"]))
        return f"{dt.hour:02d}:{dt.minute:02d}"
