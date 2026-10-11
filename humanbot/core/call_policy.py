"""接听决策（CallPolicy）。

策划案 §5.1：**关系 + 情绪 + 时段 综合判断**。

关键不是"能不能接"，而是"像不像人在接"：

- 陌生号不接（真人也不接陌生来电）
- 生气时不接（这本身就是关系的一部分）
- 深夜只接亲密的人
- **永远留 8% 的随机不接** —— 真人总有在忙、没听见、不想接的时候，
  每次都秒接才是最大的破绽

决策结果同时决定"振铃多久"，两者一起构成这通电话的性格。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional

from .logging_setup import BUS, get_logger
from .utils import dice

log = get_logger("call")

# 关系阶段由远到近
STAGE_ORDER = ["stranger", "acquaintance", "familiar", "close", "dependent"]
STAGE_LABELS = {
    "stranger": "陌生", "acquaintance": "认识", "familiar": "熟悉",
    "close": "亲密", "dependent": "依赖",
}

# 决策动作
ANSWER = "answer"     # 接听
REJECT = "reject"     # 明确拒接（对方会看到"已拒绝"）
IGNORE = "ignore"     # 不接也不拒，让铃声自然响完（显得"人不在"）
BUSY = "busy"         # 占线


@dataclass
class CallContext:
    """一次来电的全部判断依据。"""

    bot_id: int
    peer_id: int
    peer_name: str = ""
    stage: str = "stranger"          # 关系阶段
    annoyance: int = 0               # 不满度 0~100
    energy: float = 0.6              # 情绪能量
    sleep_state: str = "AWAKE"       # AWAKE / DROWSY / ASLEEP / WAKING
    minute_of_day: int = 0           # 当前分钟（0~1439）
    has_active_call: bool = False    # 是否正在通话
    calls_today: int = 0
    last_call_ts: int = 0            # 与这个人的上次通话时间
    now_ts: int = 0
    is_blacklisted: bool = False
    mode: str = "voice"              # voice | video
    sleep_just_woke: bool = False    # 刚睡醒（影响接起语与延迟）


@dataclass
class CallDecision:
    """决策结果。"""

    action: str = ANSWER
    reason: str = ""
    ring_delay: float = 3.0          # 振铃延迟（秒）
    state: str = ""                  # 接起时的状态标签：sleepy/busy/annoyed/...

    @property
    def answered(self) -> bool:
        return self.action == ANSWER

    def __repr__(self) -> str:
        return "<CallDecision %s %.1fs %s>" % (self.action, self.ring_delay,
                                               self.reason)


class CallPolicy:
    """接听决策器。"""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    # -- 工具 -------------------------------------------------------------

    @staticmethod
    def stage_rank(stage: str) -> int:
        try:
            return STAGE_ORDER.index(stage)
        except ValueError:
            return 0

    @staticmethod
    def is_night(minute: int, start: int, end: int) -> bool:
        """判断是否处在跨零点的夜间区间。"""
        m, s, e = minute % 1440, start % 1440, end % 1440
        if s <= e:
            return s <= m < e
        return m >= s or m < e

    # -- 主入口 -----------------------------------------------------------

    def decide(self, ctx: CallContext) -> CallDecision:
        """给出接听决策。"""
        c = self.cfg
        d = self._decide(ctx)
        log.info("来电决策 %s(%s) → %s  振铃 %.1fs  原因：%s",
                 ctx.peer_name or ctx.peer_id, STAGE_LABELS.get(ctx.stage, ctx.stage),
                 d.action, d.ring_delay, d.reason)
        BUS.publish("call", {
            "event": "decision", "bot_id": ctx.bot_id, "peer_id": ctx.peer_id,
            "action": d.action, "reason": d.reason, "ring_delay": d.ring_delay,
        })
        return d

    def _decide(self, ctx: CallContext) -> CallDecision:
        c = self.cfg

        # 1) 每日限额 —— 真人也接不了太多电话
        if ctx.calls_today >= c.daily_limit:
            return CallDecision(IGNORE, "今日通话已达上限 %d" % c.daily_limit, 0)

        # 2) 黑名单 —— 真人会拉黑
        if ctx.is_blacklisted:
            return CallDecision(REJECT, "在黑名单里", 0)

        # 3) 占线 —— 真人单线程
        if ctx.has_active_call:
            return CallDecision(BUSY, "正在通话中", 0)

        # 4) 睡着 —— 听不见，让铃自然响完
        if ctx.sleep_state == "ASLEEP":
            return CallDecision(IGNORE, "睡着了没听见", 0)

        # 5) 情绪 —— 还在生气就不接
        if ctx.annoyance >= c.max_annoyance:
            return CallDecision(
                REJECT, "不满度 %d（≥%d），还在生气" % (ctx.annoyance, c.max_annoyance), 0)

        # 6) 关系 —— 不熟的号不接
        rank = self.stage_rank(ctx.stage)
        if rank < self.stage_rank(c.min_relation):
            return CallDecision(
                IGNORE, "关系%s，低于门槛%s" % (
                    STAGE_LABELS.get(ctx.stage, ctx.stage),
                    STAGE_LABELS.get(c.min_relation, c.min_relation)), 0)

        # 7) 深夜 —— 只接亲密的人
        if self.is_night(ctx.minute_of_day, c.night_quiet_start, c.night_quiet_end):
            if rank < self.stage_rank(c.night_min_relation):
                return CallDecision(
                    IGNORE, "深夜且关系不够近（%s）" % STAGE_LABELS.get(ctx.stage, ""), 0)

        # 8) 冷却 —— 刚聊完不会马上又聊
        if ctx.last_call_ts and ctx.now_ts:
            gap = ctx.now_ts - ctx.last_call_ts
            if 0 <= gap < c.cooldown_sec:
                return CallDecision(
                    IGNORE, "距上次通话仅 %d 秒（冷却 %d 秒）" % (gap, c.cooldown_sec), 0)

        # 9) 基线随机 —— 留出"在忙/没听见/不想接"的余地
        if not dice(c.base_answer_prob):
            return CallDecision(
                IGNORE, "随机未接（基线接听率 %.0f%%）" % (c.base_answer_prob * 100), 0)

        # ---- 决定接听，下面算振铃延迟和状态标签 ----
        state, delay = self._ring(ctx, rank)
        return CallDecision(ANSWER, "接听", delay, state)

    # -- 振铃延迟 ---------------------------------------------------------

    def _ring(self, ctx: CallContext, rank: int) -> Any:
        """振铃延迟 + 接起时的状态。

        这是拟人化的灵魂：真人接电话要放下手上的事、走到手机旁，
        不可能第一声就接起来。
        """
        c = self.cfg
        stage = ctx.stage

        # 刚被吵醒：起得慢、声音也是哑的
        if ctx.sleep_state in ("DROWSY", "WAKING") or ctx.sleep_just_woke:
            lo = min(c.ring_delay_sleepy * 0.6, c.ring_delay_sleepy)
            return "sleepy", random.uniform(lo, c.ring_delay_sleepy)

        # 能量低 = 在忙/累 —— 接得慢一点，语气敷衍
        if ctx.energy < 0.35:
            return "busy", random.uniform(c.ring_delay_busy * 0.75, c.ring_delay_busy)

        # 有点烦：接得慢，开口是"干嘛"
        if ctx.annoyance >= c.max_annoyance * 0.5:
            return "annoyed", random.uniform(c.ring_delay_max, c.ring_delay_max * 1.4)

        # 关系越近，接得越快
        bias = rank / max(1, len(STAGE_ORDER) - 1)     # 0~1
        lo = c.ring_delay_min * (1 - bias * 0.35)
        hi = c.ring_delay_max * (1 - bias * 0.25)
        return "", random.uniform(lo, max(lo + 0.5, hi))

    # -- 辅助：给上层拼上下文 ---------------------------------------------

    def build_context(self, bot_id: int, peer_id: int, *,
                      stage: str, annoyance: int, energy: float,
                      sleep_state: str, minute_of_day: int,
                      has_active_call: bool, calls_today: int,
                      last_call_ts: int = 0, now_ts: int = 0,
                      peer_name: str = "", is_blacklisted: bool = False,
                      mode: str = "voice",
                      sleep_just_woke: bool = False) -> CallContext:
        return CallContext(
            bot_id=bot_id, peer_id=peer_id, peer_name=peer_name, stage=stage,
            annoyance=annoyance, energy=energy, sleep_state=sleep_state,
            minute_of_day=minute_of_day, has_active_call=has_active_call,
            calls_today=calls_today, last_call_ts=last_call_ts, now_ts=now_ts,
            is_blacklisted=is_blacklisted, mode=mode,
            sleep_just_woke=sleep_just_woke,
        )
