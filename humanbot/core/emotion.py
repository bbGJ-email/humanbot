"""情绪引擎 + 不满度阶梯 + 生理周期（策划案 §4 / §5 / §6.7）。"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..settings import TASK_EMOTION, PhysioConfig, Settings, SleepConfig
from .llm import Brain
from .logging_setup import get_logger
from .storage import Database
from .utils import clamp, dice, gauss_clamp, minute_of_day, now, pick, today_str

log = get_logger("emotion")


# ---------------------------------------------------------------------------
# 四维情绪
# ---------------------------------------------------------------------------

@dataclass
class EmotionState:
    """P/E/I/S 四维。"""

    pleasure: float = 0.0     # 愉悦度 -1~1
    energy: float = 0.6       # 能量 0~1
    intimacy: float = 0.0     # 亲密感 0~1
    stress: float = 0.0       # 压力 0~1

    def to_json(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Optional[Dict[str, Any]]) -> "EmotionState":
        d = d or {}
        return cls(
            pleasure=float(d.get("pleasure", 0.0)),
            energy=float(d.get("energy", 0.6)),
            intimacy=float(d.get("intimacy", 0.0)),
            stress=float(d.get("stress", 0.0)),
        )

    def apply(self, delta: Dict[str, Any]) -> None:
        self.pleasure = clamp(self.pleasure + float(delta.get("pleasure", 0)), -1, 1)
        self.energy = clamp(self.energy + float(delta.get("energy", 0)), 0, 1)
        self.intimacy = clamp(self.intimacy + float(delta.get("intimacy", 0)), 0, 1)
        self.stress = clamp(self.stress + float(delta.get("stress", 0)), 0, 1)


# 心境标签判定（策划案 §4.2）
def mood_of(e: EmotionState, hour: int) -> str:
    if e.pleasure > 0.5 and e.energy > 0.6:
        return "挺开心的"
    if e.pleasure > 0 and e.energy < 0.4:
        return "有点累"
    if e.pleasure < -0.3 and e.stress > 0.5:
        return "烦"
    if e.pleasure < -0.5 and e.stress > 0.7:
        return "很烦，别惹我"
    if e.pleasure < -0.2 and e.intimacy > 0.7:
        return "有点委屈"
    if abs(e.pleasure) < 0.2 and e.energy < 0.3 and (hour >= 23 or hour < 2):
        return "睡不着"
    return "平静"


MOOD_RULES = {
    "挺开心的": "可以多说话，允许一个 emoji",
    "有点累": "句子短，不加修饰，别主动找话题",
    "烦": "回复冷淡，可以不接话，可能怼人",
    "很烦，别惹我": "只回单字，或者已读不回",
    "有点委屈": "会暗示但不直说，可以问'你是不是觉得我很烦'",
    "睡不着": "话多、跳跃、感性，可以主动说自己的事",
    "平静": "正常说话",
}


# 作息能量曲线（策划案 §4.1）
def energy_curve(hour: int) -> float:
    if hour < 7:
        return 0.10
    if hour < 9:
        return 0.35
    if hour < 12:
        return 0.75
    if hour < 14:
        return 0.50
    if hour < 18:
        return 0.70
    if hour < 21:
        return 0.80
    if hour < 24:
        return 0.60
    return 0.45


# ---------------------------------------------------------------------------
# 生理周期（性别门控）
# ---------------------------------------------------------------------------

PHASES = ("MENSTRUAL", "FOLLICULAR", "OVULATION", "LUTEAL")
PHASE_LABELS = {
    "MENSTRUAL": "经期",
    "FOLLICULAR": "卵泡期",
    "OVULATION": "排卵期",
    "LUTEAL": "黄体期",
}

# 各阶段对情绪的修正
PHASE_EFFECT = {
    "MENSTRUAL":  {"energy": -0.30, "pleasure": -0.15, "stress": +0.20},
    "FOLLICULAR": {"energy": +0.20, "pleasure": +0.25, "stress": -0.15},
    "OVULATION":  {"energy": +0.15, "pleasure": +0.15, "stress": -0.05},
    "LUTEAL":     {"energy": -0.10, "pleasure": -0.15, "stress": +0.15},
}

# 不满度触发系数（策划案 §6.7.4）
PHASE_ANNOY_MULT = {
    "MENSTRUAL": 1.8,
    "FOLLICULAR": 0.7,
    "OVULATION": 0.9,
    "LUTEAL": 1.2,
}


class MenstrualCycle:
    """生理周期。gender != female 时不实例化（模块整体失效）。"""

    def __init__(self, db: Database, bot_id: int, cfg: PhysioConfig,
                 enabled_by_gender: bool) -> None:
        self.db = db
        self.bot_id = bot_id
        self.cfg = cfg
        self.active = bool(cfg.enabled and enabled_by_gender)
        self.start_date: Optional[date] = None
        self.length: int = 28
        self.light: bool = False
        if self.active:
            self._load_or_init()

    def _load_or_init(self) -> None:
        row = self.db.query_one(
            "SELECT * FROM cycle_log WHERE bot_id=? ORDER BY start_date DESC LIMIT 1",
            (self.bot_id,),
        )
        if row:
            try:
                self.start_date = datetime.strptime(row["start_date"], "%Y-%m-%d").date()
            except ValueError:
                self.start_date = None
            self.length = int(row["length"] or 28)
            self.light = bool(row["light"])
        if self.start_date is None:
            # 首次启动：随机一个过去的起始日，避免所有 BOT 同步
            self.start_date = date.today() - timedelta(days=random.randint(0, 27))
            self._roll_new()

    def _roll_new(self) -> None:
        self.length = random.randint(self.cfg.cycle_min, self.cfg.cycle_max)
        self.light = dice(self.cfg.light_cycle_rate)
        self.db.execute(
            "INSERT OR REPLACE INTO cycle_log(bot_id, start_date, length, light) "
            "VALUES(?,?,?,?)",
            (self.bot_id, self.start_date.strftime("%Y-%m-%d"), self.length,
             int(self.light)),
        )

    def tick(self) -> None:
        """跨周期时自动滚动。"""
        if not self.active or self.start_date is None:
            return
        if (date.today() - self.start_date).days >= self.length:
            self.start_date = self.start_date + timedelta(days=self.length)
            self._roll_new()

    @property
    def day(self) -> int:
        if not self.active or self.start_date is None:
            return 0
        return (date.today() - self.start_date).days + 1

    @property
    def phase(self) -> str:
        if not self.active:
            return ""
        d = self.day
        p = self.cfg.period_days
        if d <= p:
            return "MENSTRUAL"
        if d <= p + 8:
            return "FOLLICULAR"
        if d <= p + 10:
            return "OVULATION"
        return "LUTEAL"

    @property
    def phase_label(self) -> str:
        return PHASE_LABELS.get(self.phase, "")

    def annoyance_multiplier(self) -> float:
        if not self.active:
            return 1.0
        m = PHASE_ANNOY_MULT.get(self.phase, 1.0)
        if self.phase == "LUTEAL" and self.day >= 24:
            m *= 1.5
        if self.light:
            m = 1.0 + (m - 1.0) * 0.4   # "这个月很轻"
        return m

    def effect(self) -> Dict[str, float]:
        if not self.active:
            return {}
        eff = dict(PHASE_EFFECT.get(self.phase, {}))
        if self.light:
            eff = {k: v * 0.4 for k, v in eff.items()}
        return eff

    def describe(self) -> str:
        if not self.active:
            return ""
        d = self.day
        ph = self.phase_label
        nxt = max(0, self.length - d)
        return f"周期第 {d} 天（{ph}），距下次还有 {nxt} 天"


# ---------------------------------------------------------------------------
# 不满度阶梯与冲突
# ---------------------------------------------------------------------------

ANNOY_LEVELS = [
    (0, 15, 0, "平和"),
    (16, 30, 1, "微妙"),
    (31, 50, 2, "明显不悦"),
    (51, 70, 3, "冷处理"),
    (71, 85, 4, "发火"),
    (86, 100, 5, "拉黑"),
]


def annoy_level(score: int) -> int:
    for lo, hi, lvl, _ in ANNOY_LEVELS:
        if lo <= score <= hi:
            return lvl
    return 5 if score > 100 else 0


def annoy_label(score: int) -> str:
    for lo, hi, _, label in ANNOY_LEVELS:
        if lo <= score <= hi:
            return label
    return "拉黑"


# 骂人话术库（白名单，禁止 LLM 自由发挥脏话 - 策划案 §5.3）
ANGER_LINES = {
    2: ["嗯", "哦，是吗", "……", "行", "随便"],
    3: ["你可真行", "挺好的，你说得都对", "行吧，不聊这个了",
        "你要这么想我也没办法", "算了，当我没说"],
    4: ["你有意思吗", "我是不是跟你说过", "你记得你上周说什么了吗。算了你不记得",
        "我tm跟你说这些干嘛", "别跟我说话，烦",
        "你根本没在听我说话吧", "你每次都是这样，说完了就完了",
        "我是不是特别好糊弄", "行，我确实不重要"],
    5: ["以后别找我了", "我拉黑你了，别烦我"],
}

# 和解话术（策划案 §5.5）
RECONCILE_LINES = {
    "cold": ["嗯", "……知道了"],
    "soft": ["……刚才是我不对", "我不是气那件事，我是气你不在乎"],
    "bridge": ["壮壮今天叶子又黄了一片", "今天书店来了只猫"],
    "warm": ["算了，不气了", "……别不理我"],
}

# 和解触发关键词
SORRY_PATTERNS = ["对不起", "我错了", "我不该", "抱歉", "是我不好", "原谅我",
                  "别生气了", "我检讨", "抱歉抱歉", "不好意思"]

# 冷暴力延迟（分钟）
COLD_DELAY = {0: 0, 1: 0, 2: 3, 3: 20, 4: 90, 5: 100000}


@dataclass
class ConversationState:
    """单个会话的运行时状态。"""

    bot_id: int = 0
    peer_id: int = 0
    chat_type: str = "private"
    emotion: EmotionState = field(default_factory=EmotionState)
    annoyance: int = 0
    relation: str = "stranger"
    times_asked_ai: int = 0
    conflict_level: int = 0
    conflict_started: int = 0
    last_interaction: int = 0
    cold_until: int = 0              # 冷处理到什么时候
    pending_reconcile: bool = False
    turns_since_negative: int = 0    # 负面情绪残留计数


class EmotionEngine:
    """情绪与冲突的统一管理。"""

    def __init__(self, db: Database, brain: Brain, settings: Settings) -> None:
        self.db = db
        self.brain = brain
        self.settings = settings
        self._states: Dict[str, ConversationState] = {}

    # -- 状态存取 ---------------------------------------------------------

    def _key(self, bot_id: int, peer_id: int, chat_type: str) -> str:
        return f"{bot_id}:{chat_type}:{peer_id}"

    def state(self, bot_id: int, peer_id: int, chat_type: str = "private"
              ) -> ConversationState:
        k = self._key(bot_id, peer_id, chat_type)
        st = self._states.get(k)
        if st is None:
            st = ConversationState(bot_id=bot_id, peer_id=peer_id,
                                   chat_type=chat_type)
            # 从 DB 恢复
            row = self.db.get_conversation(bot_id, peer_id, chat_type)
            if row:
                st.relation = row["relation"] or "stranger"
                st.emotion.intimacy = float(row["intimacy"] or 0.0)
                st.last_interaction = int(row["last_ts"] or 0)
            brow = self.db.get_bot(bot_id)
            if brow:
                import json
                st.emotion = EmotionState.from_json(
                    json.loads(brow["emotion_json"] or "{}"))
                st.annoyance = int(brow["annoyance"] or 0)
                st.relation = brow["relation_stage"] or st.relation
            self._states[k] = st
        return st

    def persist(self, st: ConversationState) -> None:
        import json
        self.db.execute(
            "UPDATE bots SET emotion_json=?, annoyance=?, relation_stage=? WHERE bot_id=?",
            (json.dumps(st.emotion.to_json()), st.annoyance, st.relation, st.bot_id),
        )
        self.db.execute(
            "UPDATE conversations SET relation=?, intimacy=?, last_ts=? "
            "WHERE bot_id=? AND peer_id=? AND chat_type=?",
            (st.relation, st.emotion.intimacy, st.last_interaction,
             st.bot_id, st.peer_id, st.chat_type),
        )

    # -- 自然衰减 ---------------------------------------------------------

    def tick(self, st: ConversationState) -> None:
        """随时间的自然变化。"""
        now_ts = time.time()
        hours = max(0.0, (now_ts - (st.last_interaction or now_ts)) / 3600.0)
        # 愉悦度向 0 回归
        st.emotion.pleasure *= max(0.0, 1 - 0.02 * hours)
        # 压力缓慢下降
        st.emotion.stress = max(0.0, st.emotion.stress - 0.01 * hours)
        # 亲密感：超过 24h 无互动开始衰减
        if hours > 24:
            st.emotion.intimacy = max(0.0, st.emotion.intimacy - 0.05 * (hours / 24))
        # 不满度自然衰减（6 小时降 5 分）
        if st.annoyance > 0:
            decay = 5 * (hours / 6.0)
            st.annoyance = int(max(0, st.annoyance - decay))
        # 作息能量回归曲线
        h = datetime.now().hour
        target = energy_curve(h)
        st.emotion.energy = st.emotion.energy * 0.7 + target * 0.3

    # -- 情绪更新 ---------------------------------------------------------

    def update_from_turn(self, st: ConversationState, user_text: str,
                         bot_text: str = "") -> Dict[str, Any]:
        """一次对话后的情绪增量（用便宜模型算增量）。"""
        sys_p = (
            "你是情绪分析器。根据用户这句话，输出对角色情绪的增量影响。"
            "只输出 JSON，不要解释。\n"
            '格式：{"pleasure": -0.2~0.2, "energy": -0.1~0.1, '
            '"intimacy": -0.05~0.05, "stress": -0.1~0.2, "annoyance": 0~30}\n'
            "pleasure：用户的话让她心情变好(+)还是变差(-)\n"
            "intimacy：关系拉近(+)还是疏远(-)\n"
            "stress：压力增加(+)\n"
            "annoyance：不满度增加量，只有被冒犯/敷衍/质问时才>0\n"
            "普通闲聊各值接近 0。"
        )
        try:
            delta = self.brain.complete_json(
                TASK_EMOTION, sys_p, f"用户说：{user_text}", temperature=0.1,
                max_tokens=120,
            )
        except Exception as exc:
            log.warning("情绪更新失败: %s", exc)
            delta = {}

        if delta:
            st.emotion.apply(delta)
            ann = int(delta.get("annoyance") or 0)
            if ann > 0:
                self.raise_annoyance(st, ann, reason=user_text[:40])
            # 负面情绪残留
            if float(delta.get("pleasure", 0)) < -0.05:
                st.turns_since_negative = 0
            else:
                st.turns_since_negative += 1

        self.persist(st)
        return delta

    # -- 不满度与冲突 -----------------------------------------------------

    def raise_annoyance(self, st: ConversationState, amount: int,
                        reason: str = "", phase_mult: float = 1.0) -> int:
        """提升不满度，按生理周期放大，并记录事件。"""
        gain = int(round(amount * phase_mult))
        if gain <= 0:
            return st.annoyance
        before = annoy_level(st.annoyance)
        st.annoyance = min(100, st.annoyance + gain)
        after = annoy_level(st.annoyance)
        if after > before:
            st.conflict_level = after
            if before == 0:
                st.conflict_started = now()
            self.db.execute(
                "INSERT INTO annoyance_events(bot_id, peer_id, ts, level, trigger) "
                "VALUES(?,?,?,?,?)",
                (st.bot_id, st.peer_id, now(), after, reason[:120]),
            )
            log.info("不满度升级 L%d（%d → %d）：%s",
                     after, st.annoyance - gain, st.annoyance, reason[:40])
        # 进入冷处理
        delay_min = COLD_DELAY.get(after, 0)
        if delay_min > 0 and after >= 3:
            st.cold_until = now() + delay_min * 60
        self.persist(st)
        return st.annoyance

    def is_cold(self, st: ConversationState) -> bool:
        """是否处于冷处理（已读不回）。"""
        if st.cold_until and now() < st.cold_until:
            return True
        return annoy_level(st.annoyance) >= 3

    def cold_remaining(self, st: ConversationState) -> int:
        return max(0, st.cold_until - now())

    # -- 和解 -------------------------------------------------------------

    def try_reconcile(self, st: ConversationState, user_text: str) -> Optional[str]:
        """尝试和解，返回应答（None 表示未触发）。"""
        if annoy_level(st.annoyance) < 2:
            return None
        apologized = any(p in user_text for p in SORRY_PATTERNS)
        # 时间自动降级
        lvl = annoy_level(st.annoyance)
        elapsed = now() - (st.conflict_started or now())
        thresholds = {2: 8 * 3600, 3: 24 * 3600, 4: 48 * 3600}
        if lvl in thresholds and elapsed > thresholds[lvl]:
            st.annoyance = max(0, st.annoyance - 25)
            st.conflict_level = annoy_level(st.annoyance)
            self.persist(st)
            return pick(RECONCILE_LINES["soft"])

        if not apologized:
            return None

        if not st.pending_reconcile:
            # 刚道歉：不回或回"嗯"，继续冷 5~15 分钟
            st.pending_reconcile = True
            st.cold_until = now() + random.randint(300, 900)
            st.annoyance = max(0, st.annoyance - 15)
            self.persist(st)
            return pick(RECONCILE_LINES["cold"])

        # 已经冷置过：给台阶 / 和解
        st.pending_reconcile = False
        st.annoyance = max(0, st.annoyance - 25)
        st.emotion.intimacy = max(0.0, st.emotion.intimacy - 0.1)  # 留痕迹
        self.persist(st)
        if st.annoyance > 40:
            return pick(RECONCILE_LINES["bridge"])
        return pick(RECONCILE_LINES["warm"])

    # -- 关系阶段 ---------------------------------------------------------

    def maybe_advance_stage(self, st: ConversationState, user_text: str) -> bool:
        """按语义信号推进关系阶段（策划案 §4.2）。"""
        from .persona import STAGES

        idx = STAGES.index(st.relation) if st.relation in STAGES else 0
        if idx >= len(STAGES) - 1:
            return False
        next_stage = STAGES[idx + 1]
        promo = False

        if next_stage == "acquaintance":
            promo = st.emotion.intimacy > 0.05 or len(user_text) > 12
        elif next_stage == "familiar":
            promo = (st.emotion.intimacy > 0.25
                     and any(k in user_text for k in
                             ("我最近", "我家里", "我妈", "我爸", "我喜欢",
                              "我讨厌", "我昨天", "我以前")))
        elif next_stage == "close":
            promo = (st.emotion.intimacy > 0.5
                     and (any(k in user_text for k in
                              ("烦", "难过", "崩了", "累", "压力", "睡不着"))
                          or datetime.now().hour >= 23))
        elif next_stage == "dependent":
            promo = st.emotion.intimacy > 0.75

        if promo:
            st.relation = next_stage
            self.db.execute(
                "UPDATE conversations SET relation=? WHERE bot_id=? AND peer_id=? "
                "AND chat_type=?",
                (next_stage, st.bot_id, st.peer_id, st.chat_type),
            )
            self.persist(st)
            log.info("关系推进 → %s", next_stage)
        return promo

    def raise_intimacy(self, st: ConversationState, amount: float = 0.03) -> None:
        st.emotion.intimacy = clamp(st.emotion.intimacy + amount, 0, 1)

    # -- 冲突时的输出 -----------------------------------------------------

    def anger_line(self, st: ConversationState) -> Optional[str]:
        """按不满度等级返回固定话术（白名单）。"""
        lvl = annoy_level(st.annoyance)
        if lvl < 2:
            return None
        lines = ANGER_LINES.get(lvl) or ANGER_LINES.get(2)
        return pick(lines)

    def log_emotion(self, st: ConversationState, mood: str, phase: str = "") -> None:
        self.db.log_emotion(st.bot_id, st.peer_id, st.emotion.pleasure,
                            st.emotion.energy, st.emotion.intimacy,
                            st.emotion.stress, mood, st.annoyance, phase)
