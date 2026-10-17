"""已读不回的自主决策。

真人不会每条消息都回——永远秒回、有问必答，比任何措辞都更像机器人。
这里的判定分三层：

  1) 硬性必回：被点名、问句、长消息、非文本、陌生人、对方情绪明显、
     连续未回已达上限
  2) 高概率不回：纯填充词（嗯/哦/好的/哈哈/在吗）、和上一条几乎重复、
     群里主动插嘴
  3) 概率抽样：在基础概率上，按关系阶段、不满度、能量、时段、
     距上次回复的间隔做修正

「不回」不是随机装死：连续不回有上限，重要消息一律回，
关系越近越不容易被冷落。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .utils import dice


@dataclass
class SkipDecision:
    """一次「回不回」的判定结果。"""

    skip: bool
    reason: str = ""
    prob: float = 0.0


# 纯填充词：对方只是应了一声，没有新信息
_FILLER_RE = re.compile(
    r"^[\s，。、！？~～,.!?]*"
    r"(嗯+|哦+|噢+|喔+|好+|好呀|好嘞|好的|好滴|好哒|行|可以|收到|了解|知道了|明白|"
    r"哈+|呵+|嘿+|嘿嘿|在吗|在不在|在么|早|早安|午安|晚安|睡了|"
    r"拜拜|白白|886|ok|okay|fine|嗯嗯|对对对|是啊|是吧|真的吗)"
    r"[\s，。、！？~～,.!?]*$",
    re.I,
)

# 问句：必须回，不然就是「没听见」
_QUESTION_RE = re.compile(
    r"[?？]|吗|呢|什么|怎么|为什么|为啥|哪[里儿个]|谁|几[点个岁]|"
    r"是不是|能不能|要不要|有没有|多少|多久|如何|咋"
)

# 情绪明显：对方在求助或崩溃，不能装死
_EMOTION_RE = re.compile(
    r"生气|难过|伤心|想哭|哭了|委屈|烦死|好烦|累死|崩溃|受不了|讨厌|"
    r"失望|害怕|焦虑|睡不着|呜呜|唉|救命|帮我|怎么办|陪我"
)

# 对方在收尾，要立刻接一句，否则显得被无视
_URGENT_RE = re.compile(r"算了|当我没说|没事了|不说了|随便吧|不用了")


def _norm(t: str) -> str:
    return re.sub(r"[\s\W_]+", "", (t or "").lower())


def _similar(a: str, b: str) -> float:
    """字符集合 Jaccard 相似度（够用且便宜）。"""
    sa, sb = set(_norm(a)), set(_norm(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class SkipDecider:
    """决定这条消息要不要回。"""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def decide(self, *, text: str, is_text: bool = True,
               at_me: bool = False, stage: str = "stranger",
               annoyance: float = 0, energy: float = 0.6,
               hour: int = 12, is_group: bool = False,
               proactive: bool = False, streak: int = 0,
               since_last_reply: float = 999,
               last_user_text: str = "") -> SkipDecision:
        c = self.cfg
        if not getattr(c, "skip_enabled", True):
            return SkipDecision(False, "功能关闭", 0.0)

        t = (text or "").strip()
        max_streak = int(getattr(c, "skip_max_streak", 2))

        # ---- 1) 硬性必回 ----
        if streak >= max_streak:
            return SkipDecision(False, "连发 %d 条没回了" % streak, 0.0)
        if not is_text:
            return SkipDecision(False, "图片/语音等", 0.0)
        if at_me:
            return SkipDecision(False, "被点名", 0.0)
        if _URGENT_RE.search(t):
            return SkipDecision(False, "对方在收尾", 0.0)
        if _QUESTION_RE.search(t):
            return SkipDecision(False, "问句", 0.0)
        if len(_norm(t)) >= 30:
            return SkipDecision(False, "长消息", 0.0)
        if _EMOTION_RE.search(t):
            return SkipDecision(False, "对方情绪明显", 0.0)
        if annoyance >= 60:
            return SkipDecision(False, "已到冷战阈值", 0.0)

        # ---- 2) 概率基线 ----
        p = float(getattr(c, "skip_base_prob", 0.12))
        reason = "普通消息"

        if _FILLER_RE.match(t):
            p = float(getattr(c, "skip_filler_prob", 0.65))
            reason = "只是应了一声"
        elif last_user_text and _similar(t, last_user_text) >= 0.8:
            p = float(getattr(c, "skip_repeat_prob", 0.5))
            reason = "跟上一条几乎一样"

        # ---- 3) 情境修正 ----
        p += {"stranger": -0.08, "acquaintance": -0.06, "familiar": 0.0,
              "close": -0.03, "dependent": -0.05}.get(stage, 0.0)
        if annoyance >= 30:
            p += float(getattr(c, "skip_annoy_boost", 0.25))
            reason += "·有点烦"
        if energy < 0.3:
            p += float(getattr(c, "skip_low_energy_boost", 0.1))
            reason += "·没精神"
        if hour >= 23 or hour < 7:
            p += float(getattr(c, "skip_night_boost", 0.1))
            reason += "·深夜"
        if since_last_reply < 15:
            p += float(getattr(c, "skip_quick_reply_boost", 0.1))
            reason += "·刚回过"
        if is_group and proactive:
            p += float(getattr(c, "skip_group_proactive_boost", 0.15))
            reason += "·群里插嘴"

        p = max(0.0, min(float(getattr(c, "skip_cap", 0.85)), p))
        return SkipDecision(dice(p), reason, p)
