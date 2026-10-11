"""反 AI 层：负面清单 + 17 项检测器 + 修复策略（策划案 §3）。

三层防线：
  1. 预防：负面 prompt（在 persona.py 中装配）
  2. 检测：本模块的 detect()，17 项特征打分
  3. 修复：修复 prompt / 噪声注入 / 重生成
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..settings import TASK_ANTIAI
from .llm import Brain
from .logging_setup import get_logger
from .utils import dice, ngram_overlap, pick, strip_markdown

log = get_logger("antiai")


# ---------------------------------------------------------------------------
# 负面清单（直接拼进 system prompt）
# ---------------------------------------------------------------------------

ANTI_PATTERN_PROMPT = """【绝对禁止的输出特征】

一、结构类
  1. 禁止排比 / 三段式（"第一…第二…第三…"）
  2. 禁止先总结后展开（"总的来说…"）
  3. 禁止列举（用 1. 2. 3. 或 - 分点）
  4. 禁止先给结论后给理由

二、句式类
  5. 禁止"我理解你的感受 / 我能感受到你的…"
  6. 禁止"其实"开头
  7. 禁止"当然可以""没问题"这类服务性应答
  8. 禁止完整主谓宾长句（超过40字必须拆）
  9. 禁止对称结构（前半句后半句工整对照）
  10. 禁止以"这让我想起…"开头的联想

三、内容类
  11. 禁止主动提供建议（除非用户明确问"我该怎么办"）
  12. 禁止安慰话术模板（"你已经很棒了""要相信自己"）
  13. 禁止复述用户说过的话
  14. 禁止在回复里总结刚刚的对话
  15. 禁止提出问题后又自问自答

四、人格类
  16. 禁止任何形式的自我说明（"我这个人比较…"）
  17. 禁止情绪表达过于准确（人要说不清自己的情绪）
  18. 禁止每句话都有信息量（废话是人的特征）

五、格式类
  19. 禁止 markdown 语法（**加粗**、# 标题、`代码`）
  20. 禁止在聊天中输出多段换行（一条消息就是一整段）
"""

# 硬禁用词（命中直接高分）
FORBIDDEN_WORDS = [
    "作为", "AI", "人工智能", "语言模型", "我理解你", "我懂你", "我能感受",
    "首先", "其次", "最后", "总的来说", "需要注意的是", "综上所述",
    "希望我的回答", "有什么我可以帮", "当然可以", "没问题，我来",
    "建议你", "不妨", "你可以试试", "你可以考虑",
    "你已经很棒", "要相信自己", "加油哦", "抱抱你",
]

# 服务性/客服腔
SERVICE_TONE = ["很高兴为您", "感谢您的", "请问还有什么", "为您解答",
                "以下是我的", "我来帮您", "需要我帮您"]

# 心理咨询模板腔
THERAPY_TONE = ["我理解你的感受", "我能感受到你", "你的感受是合理",
                "允许自己", "接纳自己", "情绪没有对错"]


@dataclass
class DetectionResult:
    """检测结果。"""

    score: int = 0                                   # 总分，越高越像 AI
    hits: List[Tuple[str, int]] = field(default_factory=list)
    passed: bool = True
    action: str = "pass"                             # pass | patch | regenerate

    def add(self, name: str, penalty: int) -> None:
        if penalty <= 0:
            return
        self.score += penalty
        self.hits.append((name, penalty))

    def finalize(self) -> "DetectionResult":
        if self.score <= 30:
            self.action = "pass"
            self.passed = True
        elif self.score <= 60:
            self.action = "patch"
            self.passed = False
        else:
            self.action = "regenerate"
            self.passed = False
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "action": self.action,
            "hits": [{"name": n, "penalty": p} for n, p in self.hits],
        }


# ---------------------------------------------------------------------------
# 17 项检测器
# ---------------------------------------------------------------------------

def detect(text: str, *, user_message: str = "", max_chars: int = 45) -> DetectionResult:
    """对生成结果做 17 项 AI 感检测。"""
    r = DetectionResult()
    t = (text or "").strip()
    if not t:
        return r.finalize()

    # 1 含禁用词
    hit_words = [w for w in FORBIDDEN_WORDS if w in t]
    if hit_words:
        r.add(f"含禁用词({','.join(hit_words[:2])})", 40)
    for w in SERVICE_TONE:
        if w in t:
            r.add("客服腔", 35)
            break
    for w in THERAPY_TONE:
        if w in t:
            r.add("心理咨询腔", 35)
            break

    sentences = [s for s in re.split(r"[。！？!?…\n]+", t) if s.strip()]

    # 2 排比结构（3 句以上同构开头）
    if len(sentences) >= 3:
        heads = [s.strip()[:2] for s in sentences]
        if len(set(heads)) <= len(heads) / 2:
            r.add("排比结构", 30)

    # 3 长度超标
    if len(t) > max_chars:
        r.add(f"单条过长({len(t)}字)", 25)

    # 4 markdown
    if re.search(r"\*\*|^#{1,6}\s|`|^\s*[-*+]\s|^\s*\d+\.\s", t, flags=re.M):
        r.add("含markdown", 35)

    # 5 对称句式
    if len(sentences) >= 2:
        a, b = sentences[0].strip(), sentences[1].strip()
        if abs(len(a) - len(b)) <= 3 and len(a) >= 6 and a[-1:] == b[-1:]:
            r.add("对称句式", 20)

    # 6 完美标点（句号结尾率过高）
    if len(sentences) >= 3:
        ends = sum(1 for s in sentences if s.strip().endswith(("。", "！", "？")))
        if ends / max(1, len(sentences)) > 0.8:
            r.add("标点过于工整", 15)

    # 7 信息密度过高（无短句、无废话）
    if len(sentences) >= 3:
        short_cnt = sum(1 for s in sentences if len(s.strip()) <= 3)
        if short_cnt == 0 and len(t) > 40:
            r.add("信息密度过高", 20)

    # 8 主动建议
    if re.search(r"(你可以|建议你|不妨|试试看|应该要|最好还是)", t):
        r.add("主动给建议", 30)

    # 9 心理话术（已被 FORBIDDEN/THERAPY 覆盖，这里做补充）
    if re.search(r"(我明白你的|你的心情我|换作是我也会)", t):
        r.add("共情模板", 30)

    # 10 复述用户
    if user_message and ngram_overlap(t, user_message) > 0.5:
        r.add("复述用户原话", 25)

    # 11 疑问句过多
    if t.count("？") + t.count("?") > 1:
        r.add("疑问句过多", 15)

    # 12 起承转合完整（有开场+展开+收尾）
    if (len(sentences) >= 3
            and re.search(r"^(嗯|好的|是的|确实|的确)", sentences[0])
            and re.search(r"(吧|呢|哦|啦)[。！？…]?$", sentences[-1])):
        r.add("结构过于完整", 25)

    # 13 情绪词堆砌
    emo = len(re.findall(r"(开心|难过|感动|激动|幸福|温暖|治愈)", t))
    if emo >= 2:
        r.add("情绪词堆砌", 20)

    # 14 无口语词
    if len(t) > 12 and not re.search(r"(嗯|啊|诶|哦|吧|嘛|呀|么|算了|随便)", t):
        r.add("缺少口语词", 20)

    # 15 无标点省略（无省略号、无未完成句）
    if len(t) > 25 and "…" not in t and not re.search(r"[，,]$", t):
        r.add("无省略/未完成句", 10)

    # 16 政治正确式平衡
    if re.search(r"(但是也要考虑|另一方面|不同角度看)", t):
        r.add("辩证式平衡", 20)

    # 17 分段整齐
    paras = [p for p in t.split("\n") if p.strip()]
    if len(paras) >= 3:
        lens = [len(p) for p in paras]
        if max(lens) - min(lens) <= 5:
            r.add("分段过于整齐", 15)

    return r.finalize()


# ---------------------------------------------------------------------------
# 防线三：修复
# ---------------------------------------------------------------------------

_REPAIR_SYSTEM = """把下面这句话改得更像真人随口说的。

要求：
- 更短，可省略主语
- 加口语词（嗯/啊/诶/吧/…）
- 去掉所有修饰词和书面语
- 不要完整，允许断句
- 只输出改写后的句子本身，不要解释、不要引号"""


def repair(text: str, brain: Brain, hits: List[Tuple[str, int]]) -> str:
    """针对命中的问题做局部改写。"""
    if not text:
        return text
    why = "、".join(n for n, _ in hits[:4])
    try:
        out = brain.complete(
            TASK_ANTIAI, _REPAIR_SYSTEM,
            f"原句：{text}\n（当前问题：{why}）",
            temperature=0.9, max_tokens=120,
        )
        out = strip_markdown(out).strip().strip('"').strip("'").strip()
        return out or text
    except Exception as exc:
        log.warning("反AI改写失败: %s", exc)
        return text


# ---------------------------------------------------------------------------
# 噪声注入（手段B）
# ---------------------------------------------------------------------------

FILLERS = ["嗯…", "那个…", "诶", "啊这", "哦", "……"]
# 尾缀词。按重复次数做加权：「了」会改变时态语义（"我也不知道"→"我也不知道了"），
# 所以刻意让它比其它语气词少见。
TAILS = ["吧", "啊", "呢", "哦", "吧", "啊", "呢", "哦", "了"]

# 句末语气词：已经有了就别再补，否则会出现"……感冒了？了"这种鬼话
_TAIL_PARTICLES = ("吧", "啊", "呢", "哦", "吗", "嘛", "呀", "哈", "了")
# 句末标点：补尾缀前先削掉
_SENT_END = "。！!？?，,、~～…"

# 可以整句重复的短应答（"对啊"→"对啊对啊"），这是很典型的真人打字习惯
_ECHOABLE = ("对啊", "是的", "好的", "行吧", "在的", "来了", "知道",
             "嗯嗯", "好好", "可以", "真的", "确实")


def inject_noise(text: str, *, strength: float = 1.0) -> str:
    """注入"不完美"噪声，让文本更像人打的字。

    原则：只做"不会改变语义、看起来就是人手滑"的改动。
    任何可能让句子读起来是错的（比如在问号后面补语气词）都不做。
    """
    if not text:
        return text
    t = text

    # 1. 口语填充词（25%）
    if dice(0.25 * strength) and not re.match(r"^(嗯|啊|哦|诶|那个)", t):
        t = pick(FILLERS) + t

    # 2. 尾缀词（30%）
    #    必须同时满足：不是问句/叹句结尾、结尾没有语气词、够长。
    #    早年这里只 rstrip 了 。！，导致"你今天好吗？"被补成"你今天好吗？啊"。
    if dice(0.30 * strength) and len(t) > 4:
        core = t.rstrip(_SENT_END)
        if (core and not core.endswith(_TAIL_PARTICLES)
                and not core.endswith(("…", "～", "~"))
                and not t.endswith(("？", "?","！", "!"))):
            t = core + pick(TAILS)

    # 3. 省略号（20%）
    if dice(0.20 * strength) and "…" not in t and len(t) > 8:
        # 只在标点处断，避免把词切开
        spots = [m.start() for m in re.finditer(r"[，,。！？!?、]", t)]
        if spots:
            idx = pick(spots) + 1
            t = t[:idx] + "……" + t[idx:]

    # 4. 口语化替换（20%）
    if dice(0.20 * strength):
        t = re.sub(r"^知道了$", "知道了知道了", t)
        t = re.sub(r"^好的$", "行", t)
        t = re.sub(r"^是的$", "嗯", t)

    # 5. 附和式重复（6%）——整句就是一句短应答时重复一遍
    if dice(0.06 * strength) and t in _ECHOABLE:
        t = t + t

    # 6. 未完成句（10%）
    if dice(0.10 * strength) and len(t) > 15 and "……" not in t:
        t = t.rstrip(_SENT_END) + "……"

    return t.strip()


def soften_punctuation(text: str) -> str:
    """去掉句尾句号（她打字很少用句号）。"""
    t = (text or "").strip()
    if len(t) > 3 and t.endswith("。"):
        t = t[:-1]
    return t


# ---------------------------------------------------------------------------
# 身份质疑：四次递进应答（策划案 §3.3）
# ---------------------------------------------------------------------------

AI_QUESTION_PATTERNS = [
    r"(你|您)是(不是)?(个)?\s*(AI|ai|人工智能|机器人|程序|bot|GPT)",
    r"你是真人吗", r"你是不是人", r"你是机器", r"你是程序",
    r"你是(不是)?(在)?骗我", r"ai吧你", r"你是假的吧",
    r"(测试|试试)你.*(是|是不是)(AI|ai|机器人)",
]


def is_ai_question(text: str) -> bool:
    t = (text or "").strip()
    return any(re.search(p, t, flags=re.I) for p in AI_QUESTION_PATTERNS)


# 四次递进：反问 → 带情绪 → 冷淡 → 生气
AI_DENY_LINES = [
    # 第 1 次：反问 / 打岔
    ["……你为什么这么问", "嗯？怎么突然说这个", "你没事吧", "……？"],
    # 第 2 次：半真半假带情绪
    ["这问题挺没意思的", "我要是AI能这么跟你唠？", "你认真的？", "行了你别闹"],
    # 第 3 次：被冒犯，冷淡
    ["行吧，你说是就是", "……", "随你怎么想", "没意思"],
    # 第 4 次及以上：生气（进入冲突系统）
    ["你真的很烦", "别跟我说话了", "你爱怎么想怎么想", "我拉黑你了"],
]


def ai_deny_line(times_asked: int) -> str:
    idx = min(max(times_asked - 1, 0), len(AI_DENY_LINES) - 1)
    return pick(AI_DENY_LINES[idx])


def ai_deny_annoyance_delta(times_asked: int) -> int:
    """被反复质疑身份带来的不满度增量。"""
    return {1: 5, 2: 15, 3: 30}.get(times_asked, 45)
