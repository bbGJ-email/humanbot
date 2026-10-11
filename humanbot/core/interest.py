"""群聊兴趣模型（策划案 §7.2「主动参与感兴趣的话题」）。

要解决的问题：
现有实现只做了一次静态的关键词匹配——命中就给 0.25，没命中就给 0.05。
这有两个明显不像真人的地方：

1. **人设里的兴趣没被用起来**。一个开书店、爱看书喝手冲的人，
   在群里聊到小说和咖啡馆时应该比聊到股票更有反应。
2. **没有"聊到一半"的状态**。真人一旦在一个话题里开了口，
   接下来几条会更容易接话；反过来刚回过又马上回，会显得像刷屏。

所以这里做三件事：

- **兴趣画像**：从 persona.yaml 自动抽出关键词，不手写配置
- **话题热度**：群里反复出现的话题会升温（衰减），她参与过的升温更快
- **参与惯性 / 插话抑制**：刚说过话 → 加成；刚回过又回 → 降权

话题抽取不依赖分词库：中文短文本用「词表命中 + 高频 n-gram」就够了，
而且可控（不会因为分词器版本变化导致行为漂移）。
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .logging_setup import get_logger
from .storage import Database
from .utils import now

log = get_logger("interest")


# ---------------------------------------------------------------------------
# 通用话题词表
# ---------------------------------------------------------------------------
# 目的不是"覆盖所有中文词"，而是覆盖聊天里真正会被展开的话题。
# 每条 = (话题名, 触发词)。触发词之间是"或"关系。

_TOPIC_LEXICON: List[Tuple[str, Tuple[str, ...]]] = [
    ("游戏", ("游戏", "打游戏", "上分", "开黑", "王者", "原神", "lol", "LOL",
              "steam", "steam", "switch", "主机", "手游", "副本", "排位",
              "魔兽", "塞尔达", "魂系", "rpg", "RPG", "单机")),
    ("读书", ("书", "小说", "看书", "读完", "作者", "文学", "散文", "诗集",
              "书店", "图书馆", "kindle", "纸质书", "译本")),
    ("影视", ("电影", "看剧", "电视剧", "追剧", "番剧", "动漫", "综艺",
              "纪录片", "上映", "院线", "美剧", "日剧", "剧透")),
    ("音乐", ("音乐", "听歌", "歌单", "专辑", "乐队", "演唱会", "livehouse",
              "摇滚", "民谣", "说唱", "网易云")),
    ("咖啡", ("咖啡", "手冲", "拿铁", "美式", "瑞幸", "星巴克", "豆子",
              "探店", "咖啡馆")),
    ("吃喝", ("吃", "外卖", "奶茶", "火锅", "烧烤", "夜宵", "食堂", "做饭",
              "下厨", "好吃", "踩雷", "探店", "螺蛳粉", "麻辣烫")),
    ("睡眠", ("睡不着", "失眠", "熬夜", "早睡", "午睡", "做梦", "起床",
              "睡够", "困", "赖床")),
    ("上班", ("上班", "加班", "老板", "同事", "领导", "上班族",
              "通勤", "打工", "入职", "离职", "面试", "跳槽", "裁员",
              "年终奖", "绩效", "996", "摸鱼", "开早会", "开会")),
    ("钱", ("房租", "房贷", "存钱", "省钱", "花钱", "穷", "工资", "涨薪",
            "理财", "基金", "股票", "比特币", "负债", "花呗", "信用卡")),
    ("天气", ("天气", "下雨", "下雪", "台风", "好热", "好冷", "降温",
              "升温", "晒", "雾霾", "梅雨")),
    ("旅行", ("旅行", "出去玩", "旅游", "出差", "机票", "高铁", "订票",
              "酒店", "景点", "攻略", "自驾", "露营", "爬山", "海边")),
    ("感情", ("喜欢", "恋爱", "分手", "复合", "暗恋", "表白", "暧昧",
              "相亲", "前任", "男朋友", "女朋友", "对象", "结婚",
              "异地", "单身", "失恋")),
    ("家庭", ("我妈", "我爸", "爸妈", "家里", "父母", "亲戚", "催婚",
              "回老家", "过年", "长辈", "外婆", "奶奶")),
    ("健身", ("健身", "跑步", "减肥", "增肌", "撸铁", "游泳", "瑜伽",
              "运动", "体重", "卡路里", "步数")),
    ("学习", ("学习", "考试", "考研", "论文", "期末", "四六级", "雅思",
              "托福", "考证", "上课", "老师", "作业", "答辩")),
    ("宠物", ("猫", "狗", "宠物", "铲屎", "遛狗", "猫咪", "狗狗", "流浪猫",
              "喂猫")),
    ("健康", ("生病", "感冒", "发烧", "医院", "体检", "吃药", "头疼",
              "胃疼", "失眠", "焦虑", "抑郁", "心理咨询")),
    ("数码", ("手机", "电脑", "显卡", "配置", "装机", "bug", "BUG",
              "代码", "编程", "写代码", "程序员", "笔记本", "平板",
              "耳机", "相机")),
    ("穿搭", ("穿", "衣服", "穿搭", "买衣服", "鞋子", "包包", "化妆",
              "护肤", "口红", "发型", "剪头发")),
    ("日常", ("周末", "放假", "休息", "在家", "刷手机", "无聊", "发呆",
              "收拾房间", "洗衣服", "搬家")),
]

# 这些词出现在文本里不算话题（语气词 / 代词 / 单一动词）
_STOPWORDS: Set[str] = {
    "这个", "那个", "什么", "怎么", "为什么", "可以", "没有", "一个", "就是",
    "不是", "还是", "但是", "然后", "所以", "因为", "如果", "真的", "感觉",
    "知道", "觉得", "现在", "今天", "明天", "昨天", "我们", "你们", "他们",
    "哈哈", "嘿嘿", "嘻嘻", "emmm", "嗯嗯", "哦哦", "啊啊", "额额", "好的",
    "谢谢", "对不起", "没事", "好吧", "行吧", "算了", "其实", "应该", "可能",
    "自己", "别人", "时候", "东西", "事情", "问题", "情况", "地方",
}


class InterestModel:
    """话题抽取 + 兴趣打分 + 参与惯性。"""

    def __init__(self, db: Database, persona: Any, settings: Any = None) -> None:
        self.db = db
        self.persona = persona
        self.settings = settings
        self._base: Dict[int, Set[str]] = {}
        self._topic_names: Dict[int, Dict[str, str]] = {}   # bot -> {词: 话题}

    # -- 人设画像 ---------------------------------------------------------

    def base_keywords(self, bot_id: int) -> Set[str]:
        """从人设里抽出这个 BOT 天然感兴趣的话题词。

        取 persona.yaml 的 hobbies / occupation / bio / 生活描述，
        再映射到话题名。这样换人设不用改配置。
        """
        if bot_id in self._base:
            return self._base[bot_id]

        p = self.persona
        blob_parts: List[str] = []
        data = getattr(p, "data", {}) or {}

        for h in (data.get("hobbies") or []):
            blob_parts.append(str(h))
        occ = data.get("occupation") or {}
        if isinstance(occ, dict):
            blob_parts += [str(occ.get("job", "")), str(occ.get("shop", "")),
                           str(occ.get("schedule", ""))]
        else:
            blob_parts.append(str(occ))
        ident = data.get("identity") or {}
        blob_parts += [str(ident.get("job", "")), str(ident.get("city", "")),
                       str(ident.get("bio", ""))]
        blob_parts.append(str(data.get("background") or ""))
        living = data.get("living") or {}
        blob_parts += [str(living.get("pets", "")), str(living.get("roommate", ""))]
        speech = data.get("speech") or {}
        blob_parts += [str(x) for x in (speech.get("catchphrases") or [])]
        for q in (data.get("quirks") or []):
            if isinstance(q, dict):
                blob_parts += [str(q.get("trigger", "")), str(q.get("behavior", ""))]

        blob = " ".join(blob_parts)

        # 话题名集合
        topics: Set[str] = set()
        word_map: Dict[str, str] = {}
        for topic, keys in _TOPIC_LEXICON:
            for k in keys:
                if k and k in blob:
                    topics.add(topic)
                    break

        # 直接把人设里出现过的词登记为触发词（含 hobby 原词，如"手冲咖啡"）
        direct: Set[str] = set()
        for h in (data.get("hobbies") or []):
            h = str(h).strip()
            # 去掉括号里的说明，取主词
            h = re.sub(r"[（(].*?[)）]", "", h).strip()
            if 1 < len(h) <= 8:
                direct.add(h)
                for topic, keys in _TOPIC_LEXICON:
                    if any(k in h for k in keys):
                        word_map[h] = topic
                        break
                else:
                    word_map[h] = "兴趣"

        # 话题名本身也当触发词
        for t in topics:
            word_map.setdefault(t, t)

        self._base[bot_id] = direct | topics
        self._topic_names[bot_id] = word_map
        log.debug("兴趣画像(bot=%s): 话题=%s 直接词=%s",
                  bot_id, sorted(topics), sorted(direct))
        return self._base[bot_id]

    def interest_topics(self, bot_id: int) -> List[str]:
        """返回人设对应的话题名列表（用于控制台展示）。"""
        self.base_keywords(bot_id)
        names: Set[str] = set()
        for topic, keys in _TOPIC_LEXICON:
            for k in keys:
                if k in self._base.get(bot_id, set()):
                    names.add(topic)
                    break
        return sorted(names)

    # -- 话题抽取 ---------------------------------------------------------

    def extract(self, text: str, bot_id: int = 0,
                confident_only: bool = False) -> List[str]:
        """从一条消息里抽话题词。返回命中的词（可能空）。

        优先级（重要的在前，绝不因为凑数而挤掉真正的关键词）：
            1. 通用词表命中
            2. 人设兴趣词命中
            3. 只有在 1、2 都没命中时，才用 n-gram 兜底找"可能的话题"

        confident_only=True 时不做 n-gram 兜底。
        累积热度必须用这个模式——否则"隔壁部门老王调岗了"会被切成
        "隔壁部门""壁部门老"这种噪声存进热度表，几周后表里全是垃圾。
        """
        if not text:
            return []
        t = text.strip()
        if len(t) < 2:
            return []

        primary: List[str] = []
        low = t.lower()

        # 1) 通用词表命中（每个话题取一个代表词，避免同义堆叠）
        for _, keys in _TOPIC_LEXICON:
            for k in keys:
                if k and (k in t or k.lower() in low):
                    primary.append(k)
                    break

        # 2) 人设兴趣词命中
        if bot_id:
            for w in self.base_keywords(bot_id):
                if 2 <= len(w) <= 8 and w in t:
                    primary.append(w)

        if primary:
            uniq = sorted(set(primary), key=lambda x: (-len(x), x))
            return uniq[:6]

        if confident_only:
            return []

        # 3) 兜底：连续 3~4 个汉字，去掉含停用词的
        gram: List[str] = []
        for n in (4, 3):
            for i in range(len(t) - n + 1):
                g = t[i:i + n]
                if not re.fullmatch(r"[\u4e00-\u9fa5]{3,4}", g):
                    continue
                if any(s in g for s in _STOPWORDS):
                    continue
                if g in gram:
                    continue
                gram.append(g)
        return gram[:2]

    # -- 话题热度（按群累积 + 衰减）---------------------------------------

    def note(self, bot_id: int, group_id: int, text: str,
             engaged: bool = False) -> None:
        """记录一条群消息里的话题，累积热度。

        只记录「词表/人设命中」的确信话题，不用 n-gram 兜底结果，
        避免热度表被切错的词塞满。
        """
        words = self.extract(text, bot_id, confident_only=True)
        if not words:
            return
        base_gain = 0.6 if engaged else 0.35
        for w in words:
            try:
                self.db.execute(
                    "INSERT INTO group_topics(bot_id,group_id,keyword,heat,hits,last_ts) "
                    "VALUES(?,?,?,?,1,?) "
                    "ON CONFLICT(bot_id,group_id,keyword) DO UPDATE SET "
                    "heat = MIN(heat + ?, 6.0), hits = hits + 1, last_ts = ?",
                    (bot_id, group_id, w, base_gain, now(), base_gain, now()))
            except Exception as exc:
                log.debug("记录话题失败: %s", exc)

    def heat_of(self, bot_id: int, group_id: int, window: Optional[int] = None
                ) -> Dict[str, float]:
        """取该群当前的话题热度（已按时间衰减）。"""
        win = window if window is not None else 7 * 86400
        rows = self.db.query(
            "SELECT keyword, heat, last_ts FROM group_topics "
            "WHERE bot_id=? AND group_id=? AND last_ts>=?",
            (bot_id, group_id, now() - win))
        out: Dict[str, float] = {}
        t0 = now()
        for r in rows:
            age_h = max(0.0, (t0 - float(r["last_ts"] or t0)) / 3600.0)
            # 半衰期 6 小时
            decay = 0.5 ** (age_h / 6.0)
            h = float(r["heat"] or 0.0) * decay
            if h > 0.02:
                out[r["keyword"]] = h
        return out

    def decay(self, bot_id: int, group_id: int) -> None:
        """定期把过期的热度压下去，避免表无限膨胀。"""
        cutoff = now() - 30 * 86400
        try:
            self.db.execute(
                "UPDATE group_topics SET heat = heat * 0.5 WHERE last_ts < ?",
                (now() - 3 * 86400,))
            self.db.execute(
                "DELETE FROM group_topics WHERE last_ts < ? AND heat < 0.3",
                (cutoff,))
        except Exception as exc:
            log.debug("话题衰减失败: %s", exc)

    # -- 兴趣打分 ---------------------------------------------------------

    def score(self, bot_id: int, group_id: int, text: str) -> float:
        """0~1 的"这个话题她有多想插嘴"得分。

        组成：
          0.55 × 人设兴趣命中
          0.30 × 该群历史热度（她之前参与过的话题）
          0.15 × 话题词数量（聊得越具体越容易接）
        """
        if not text:
            return 0.0
        words = self.extract(text, bot_id)
        if not words:
            return 0.0

        base = self.base_keywords(bot_id)
        # 词 → 话题 的反查表：让"猫"能算作对"宠物"感兴趣
        topic_of = self._topic_names.setdefault(bot_id, {})
        if not topic_of:
            self.base_keywords(bot_id)
            topic_of = self._topic_names.get(bot_id, {})

        def is_interest(w: str) -> bool:
            if w in base:
                return True
            tp = topic_of.get(w)
            if tp and tp in base:
                return True
            # 词表里的触发词 → 话题 → 是否是她在意的话题
            for topic, keys in _TOPIC_LEXICON:
                if w in keys and topic in base:
                    return True
            return False

        base_hit = sum(1 for w in words if is_interest(w))
        base_score = min(1.0, base_hit / 2.0)

        heat = self.heat_of(bot_id, group_id)
        if heat:
            hmax = max(heat.values()) if heat else 0.0
            cur = max((heat.get(w, 0.0) for w in words), default=0.0)
            heat_score = min(1.0, cur / max(1.5, hmax))
        else:
            heat_score = 0.0

        breadth = min(1.0, len(words) / 4.0)

        s = base_score * 0.55 + heat_score * 0.30 + breadth * 0.15
        return round(min(1.0, s), 4)

    # -- 参与惯性 ---------------------------------------------------------

    def _engagement(self, group_id: int, bot_id: int) -> Dict[str, Any]:
        row = self.db.query_one(
            "SELECT * FROM group_engagement WHERE group_id=?", (group_id,))
        if row:
            return dict(row)
        try:
            self.db.execute(
                "INSERT OR IGNORE INTO group_engagement(group_id,bot_id) VALUES(?,?)",
                (group_id, bot_id))
        except Exception:
            pass
        row = self.db.query_one(
            "SELECT * FROM group_engagement WHERE group_id=?", (group_id,))
        return dict(row) if row else {}

    def momentum(self, group_id: int, bot_id: int, window: float = 600.0) -> float:
        """参与惯性 0~1：刚说过话 → 接近 1；超过窗口 → 0。"""
        st = self._engagement(group_id, bot_id)
        last = float(st.get("last_spoke_ts") or 0)
        if not last:
            return 0.0
        age = max(0.0, now() - last)
        if age >= window:
            return 0.0
        return round(1.0 - age / window, 4)

    def just_interjected(self, group_id: int, bot_id: int,
                         window: float = 90.0) -> bool:
        """刚才已经插过话（用于抑制连续刷屏）。"""
        st = self._engagement(group_id, bot_id)
        last = float(st.get("last_interject_ts") or 0)
        return bool(last) and (now() - last) < window

    def note_spoke(self, group_id: int, bot_id: int) -> None:
        try:
            self.db.execute(
                "INSERT INTO group_engagement(group_id,bot_id,last_spoke_ts,"
                "last_interject_ts,replies) VALUES(?,?,?,?,1) "
                "ON CONFLICT(group_id) DO UPDATE SET "
                "last_spoke_ts=excluded.last_spoke_ts, "
                "last_interject_ts=excluded.last_interject_ts, "
                "replies=replies+1, ignores=0",
                (group_id, bot_id, now(), now()))
        except Exception as exc:
            log.debug("记录群发言失败: %s", exc)

    def note_ignored(self, group_id: int, bot_id: int) -> None:
        """她开口了但没人理 → 记一次被无视。"""
        try:
            self.db.execute(
                "INSERT INTO group_engagement(group_id,bot_id,ignores) VALUES(?,?,1) "
                "ON CONFLICT(group_id) DO UPDATE SET ignores=ignores+1",
                (group_id, bot_id))
        except Exception as exc:
            log.debug("记录群无视失败: %s", exc)

    def ignores(self, group_id: int, bot_id: int) -> int:
        st = self._engagement(group_id, bot_id)
        return int(st.get("ignores") or 0)

    # -- 主动开口额度 -----------------------------------------------------

    def proactive_ok(self, group_id: int, bot_id: int, hourly_cap: int = 4) -> bool:
        """每群每小时的主动开口额度检查。"""
        st = self._engagement(group_id, bot_id)
        try:
            stamps = json.loads(st.get("proactive_ts") or "[]")
        except Exception:
            stamps = []
        t0 = now()
        stamps = [int(x) for x in stamps if t0 - int(x) < 3600]
        return len(stamps) < hourly_cap

    def note_proactive(self, group_id: int, bot_id: int) -> None:
        st = self._engagement(group_id, bot_id)
        try:
            stamps = json.loads(st.get("proactive_ts") or "[]")
        except Exception:
            stamps = []
        t0 = now()
        stamps = [int(x) for x in stamps if t0 - int(x) < 3600]
        stamps.append(t0)
        try:
            self.db.execute(
                "INSERT INTO group_engagement(group_id,bot_id,proactive_ts) "
                "VALUES(?,?,?) ON CONFLICT(group_id) DO UPDATE SET "
                "proactive_ts=excluded.proactive_ts",
                (group_id, bot_id, json.dumps(stamps)))
        except Exception as exc:
            log.debug("记录主动开口失败: %s", exc)

    # -- 控制台展示 -------------------------------------------------------

    def snapshot(self, bot_id: int, group_id: int, top: int = 12) -> Dict[str, Any]:
        heat = self.heat_of(bot_id, group_id)
        ranked = sorted(heat.items(), key=lambda x: -x[1])[:top]
        return {
            "interests": self.interest_topics(bot_id),
            "hot_topics": [{"keyword": k, "heat": round(v, 2)} for k, v in ranked],
            "momentum": self.momentum(group_id, bot_id),
            "ignores": self.ignores(group_id, bot_id),
        }
