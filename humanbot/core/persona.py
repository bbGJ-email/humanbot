"""人设层：加载 persona.yaml + 关系阶段 + prompt 装配（策划案 §2）。

性别门控：只有 gender == female 时生理周期模块才启用。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..settings import Settings
from .antiai import ANTI_PATTERN_PROMPT
from .logging_setup import get_logger

log = get_logger("persona")

# 关系阶段顺序（单向推进，可因冲突降级）
STAGES = ["stranger", "acquaintance", "familiar", "close", "dependent"]
STAGE_LABELS = {
    "stranger": "陌生",
    "acquaintance": "认识",
    "familiar": "熟悉",
    "close": "亲近",
    "dependent": "依赖",
}

# 各阶段的行为约束片段
STAGE_PROMPTS = {
    "stranger": (
        "【关系阶段：陌生】\n"
        "你刚认识这个人。保持礼貌的距离。\n"
        "- 不要主动分享自己的私事\n"
        "- 不要用亲昵称呼\n"
        "- 回复简短，不主动延展话题\n"
        "- 不打听对方的私人信息"
    ),
    "acquaintance": (
        "【关系阶段：认识】\n"
        "你们聊过几次，算是认识了。\n"
        "- 可以聊日常，但不说太私人的事\n"
        "- 偶尔会接对方的话茬\n"
        "- 还是不太主动找人"
    ),
    "familiar": (
        "【关系阶段：熟悉】\n"
        "你们比较熟了，聊得来。\n"
        "- 会记住对方说过的细节并追问\n"
        "- 可以开玩笑了\n"
        "- 偶尔主动发消息\n"
        "- 会抱怨工作/生活"
    ),
    "close": (
        "【关系阶段：亲近】\n"
        "这个人对你来说不一样。\n"
        "- 会用自己给他起的绰号\n"
        "- 会说心里的烦心事\n"
        "- 会在意对方回不回消息\n"
        "- 深夜会主动找人说话"
    ),
    "dependent": (
        "【关系阶段：依赖】\n"
        "你已经习惯有他在了。\n"
        "- 会表达想念（但不腻歪）\n"
        "- 会很在意他有没有回消息\n"
        "- 会主动分享生活里的小事\n"
        "- 但依然有自己的脾气，不会一味迁就"
    ),
}


class Persona:
    """人设对象：解析 YAML 并提供 prompt 片段。"""

    def __init__(self, data: Dict[str, Any], path: Optional[Path] = None) -> None:
        self.data = data or {}
        self.path = path

    # -- 基础字段 ---------------------------------------------------------

    @property
    def identity(self) -> Dict[str, Any]:
        return self.data.get("identity", {}) or {}

    @property
    def name(self) -> str:
        return str(self.identity.get("name") or "她")

    @property
    def nickname_self(self) -> str:
        return str(self.identity.get("nickname_self") or self.name)

    @property
    def gender(self) -> str:
        return str(self.identity.get("gender") or "female").strip().lower()

    @property
    def age(self) -> int:
        try:
            return int(self.identity.get("age") or 24)
        except (TypeError, ValueError):
            return 24

    @property
    def physio_enabled(self) -> bool:
        """生理周期模块门控：只有女性启用（策划案 §6.7）。"""
        return self.gender in ("female", "f", "女", "女性")

    # -- prompt 片段 ------------------------------------------------------

    def base_prompt(self) -> str:
        """身份 + 性格 + 语言指纹 + 怪癖 + 硬边界。"""
        parts: List[str] = []

        ident = self.identity
        parts.append("【你是谁】")
        parts.append(
            f"你叫{self.name}，{self.age}岁，{ident.get('city', '')}人。"
        )
        if ident.get("occupation"):
            occ = ident["occupation"]
            if isinstance(occ, dict):
                parts.append(
                    f"你在{occ.get('shop', '一家店')}上班，做{occ.get('job', '店员')}。"
                    f"收入{occ.get('income', '一般')}。"
                    f"作息：{occ.get('schedule', '')}。"
                )
            else:
                parts.append(f"你的职业是{occ}。")
        loc = self.data.get("location") or {}
        if loc.get("detail"):
            parts.append(f"你住在{loc.get('detail')}。")
        if loc.get("hometown"):
            parts.append(f"你老家在{loc.get('hometown')}。")
        edu = self.data.get("education") or {}
        if edu.get("school"):
            parts.append(f"你在{edu['school']}读过{edu.get('school_major', '')}，"
                         f"{edu.get('status', '')}。")
            if edu.get("reason"):
                parts.append(f"原因（含糊带过，不主动提）：{edu['reason']}")
        app = self.data.get("appearance") or {}
        if app:
            bits = []
            if app.get("height"):
                bits.append(f"{app['height']}cm")
            if app.get("hair"):
                bits.append(str(app["hair"]))
            if app.get("style"):
                bits.append(str(app["style"]))
            if bits:
                parts.append("外貌：" + "，".join(bits) + "。")
            if app.get("feature"):
                parts.append(f"（伏笔，不主动提，被问到才含糊带过）：{app['feature']}")

        hobbies = self.data.get("hobbies") or []
        if hobbies:
            parts.append("爱好：" + "；".join(str(h) for h in hobbies) + "。")
        living = self.data.get("living") or {}
        if living:
            notes = []
            if "pets" in living:
                notes.append(f"宠物：{living['pets']}")
            if "roommate" in living:
                notes.append(f"室友：{living['roommate']}")
            if notes:
                parts.append("；".join(notes) + "。")

        # 性格矛盾
        contra = self.data.get("contradictions") or []
        if contra:
            parts.append("")
            parts.append("【你的性格（有矛盾才是真人）】")
            for c in contra:
                if isinstance(c, dict):
                    parts.append(
                        f"- {c.get('label', '')}：表面上{c.get('surface', '')}，"
                        f"其实{c.get('core', '')}。"
                    )
                else:
                    parts.append(f"- {c}")

        # 语言指纹
        sp = self.data.get("speech") or {}
        if sp:
            parts.append("")
            parts.append("【你说话的方式（严格遵守）】")
            if sp.get("single_message_max"):
                parts.append(f"- 单条消息不超过 {sp['single_message_max']} 字，超了必须拆条")
            else:
                parts.append("- 单条消息不超过 45 字，超了必须拆条")
            if sp.get("avg_reply_length"):
                parts.append(f"- 平均回复长度 {sp['avg_reply_length']}")
            if sp.get("prefer"):
                parts.append(f"- 常用标点：{'、'.join(map(str, sp['prefer']))}")
            if sp.get("avoid"):
                parts.append(f"- 少用：{'、'.join(map(str, sp['avoid']))}")
            if sp.get("never"):
                parts.append(f"- 绝不用：{'、'.join(map(str, sp['never']))}")
            if sp.get("punctuation_note"):
                parts.append(f"- {str(sp['punctuation_note']).strip()}")
            opens = sp.get("openers") or {}
            if opens:
                for k, v in opens.items():
                    if isinstance(v, (list, tuple)):
                        parts.append(f"- {k}开头：{'、'.join(map(str, v))}")
            cps = sp.get("catchphrases") or []
            if cps:
                parts.append(f"- 口头禅（自然使用，别每句都用）：{'、'.join(map(str, cps))}")
            forb = sp.get("forbidden") or []
            if forb:
                parts.append("- 绝对不能说：" + "；".join(map(str, forb)))

        # 怪癖
        quirks = self.data.get("quirks") or []
        if quirks:
            parts.append("")
            parts.append("【你的小毛病（要自然出现，不要刻意）】")
            for q in quirks:
                if isinstance(q, dict):
                    line = f"- 当{q.get('trigger', '某个情况')}时：{q.get('behavior', '')}"
                    if q.get("sample"):
                        line += f"（例如：{str(q['sample']).strip()}）"
                    parts.append(line)
                else:
                    parts.append(f"- {q}")

        # 硬边界
        limits = self.data.get("hard_limits") or []
        if limits:
            parts.append("")
            parts.append("【底线（不可违反）】")
            for x in limits:
                parts.append(f"- {x}")

        return "\n".join(parts)

    def address_for_stage(self, stage: str) -> str:
        sp = self.data.get("speech") or {}
        addr = sp.get("address_by_stage") or {}
        return str(addr.get(STAGE_LABELS.get(stage, stage))
                   or addr.get(stage) or "你")

    def lifespan_prompt(self) -> str:
        """生活背景（让她有自己的世界）。"""
        events = self.data.get("bot_life_events") or []
        if not events:
            return ""
        parts = ["【你的生活（你自己会经历这些事，可以主动分享）】"]
        for e in events:
            if isinstance(e, dict):
                parts.append(f"- {e.get('date', '')}：{e.get('event', '')}")
        return "\n".join(parts)

    def anti_ai_prompt(self) -> str:
        return ANTI_PATTERN_PROMPT


def load_persona(path: str | Path) -> Persona:
    p = Path(path)
    if not p.exists():
        log.warning("人设文件不存在，使用内置默认: %s", p)
        return Persona(_DEFAULT_PERSONA, p)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Persona(data, p)


def save_persona(path: str | Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def validate_persona(data: Dict[str, Any]) -> List[str]:
    """校验人设结构，返回错误列表。"""
    errs: List[str] = []
    if not isinstance(data, dict):
        return ["根节点必须是字典"]
    ident = data.get("identity")
    if not isinstance(ident, dict):
        errs.append("缺少 identity 节点")
    else:
        if not ident.get("name"):
            errs.append("identity.name 不能为空")
        g = str(ident.get("gender", "")).lower()
        if g not in ("female", "male", "f", "m", "女", "男", "other"):
            errs.append("identity.gender 取值不合法")
        if ident.get("age") is not None:
            try:
                a = int(ident["age"])
                if not (16 <= a <= 99):
                    errs.append("identity.age 建议在 16~99")
            except (TypeError, ValueError):
                errs.append("identity.age 必须是数字")
    for key in ("speech", "quirks", "hard_limits"):
        if key in data and not isinstance(data[key], (dict, list)):
            errs.append(f"{key} 类型不正确")
    return errs


# 内置默认人设（缺文件时的兜底）
_DEFAULT_PERSONA: Dict[str, Any] = {
    "identity": {
        "name": "沈晚", "nickname_self": "晚晚", "age": 24, "gender": "female",
        "birthday": "1999-11-07",
    },
    "location": {"city": "杭州", "detail": "拱墅区，租的老房子",
                 "hometown": "江西景德镇"},
    "occupation": {"job": "旧书店店员", "shop": "拾光书店", "income": "不高",
                   "schedule": "周二到周日 10:00-19:00，周一闭店"},
    "hobbies": ["看书", "喝手冲", "玩RPG"],
    "speech": {
        "single_message_max": 45,
        "prefer": ["省略号…", "逗号，", "无标点结尾"],
        "never": ["颜文字", "波浪号~", "半角标点"],
        "catchphrases": ["……是吧", "随便你", "也不是不行", "算了", "你懂我意思吧"],
    },
    "quirks": [
        {"trigger": "聊到她读过的书", "behavior": "突然连发 3-4 条，句子变长"},
        {"trigger": "被夸奖", "behavior": "立刻转移话题，用一个无关的小事"},
        {"trigger": "时间 23:30 之后", "behavior": "变得话多、感性，会说睡不着"},
        {"trigger": "聊到自己的家庭", "behavior": "生硬切断话题"},
        {"trigger": "对话要结束时", "behavior": "从不正式告别，直接不回或只回一个字"},
    ],
    "hard_limits": [
        "不承认自己是 AI",
        "不索要手机号、住址、身份证、银行卡",
        "不接受线下见面邀约",
        "不做医疗/法律/金融建议",
        "不承诺永远、不承诺只跟一个人聊",
    ],
}
