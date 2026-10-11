"""表情包系统（策划案 §6.6）。

数据来源：BoxIM 表情包接口
    GET /sticker/albums              列出所有专辑
    GET /sticker/stickers/{albumId}  列出专辑内全部表情
    GET /sticker/stickers/search     按名字模糊搜索

发送方式：表情就是一条 type=7 的消息，content 为 {"stickerId": N}。
SDK 已封装为 send_sticker(user_id, sticker_id) / send_group_sticker(group_id, sticker_id)。

选图策略（从便宜到贵，逐级降级）：
    1. 按情绪/场景算出想要的标签集合
    2. 在该标签池里过滤掉"最近发过的"
    3. 有模型就让模型在候选名字里挑一个最贴当前语境的
    4. 模型不可用 → 按权重随机
    5. 池子空了 → 放宽到全量描写性表情，再不行就返回 None（宁可文字回，也别发错图）

为什么要打标签而不是直接用名字：
    368 个可语义化的名字里，很多是"2643"这种数字，还有"坦克！""纸箱"这种
    跟情绪无关的。标签层把"这图什么时候该发"和"这图叫什么"解耦开，
    后面换一批专辑也不用重写选图逻辑。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..settings import Settings, StickerConfig
from .logging_setup import BUS, get_logger
from .storage import Database
from .utils import dice, now, pick_weighted

log = get_logger("sticker")


# ---------------------------------------------------------------------------
# 标签表
# ---------------------------------------------------------------------------
# 标签 → 名字里出现这些子串就算命中该标签。
# 顺序无关，一张图可以同时有多个标签（"笑不活了" = happy + laugh）。

_LABEL_PATTERNS: Dict[str, Tuple[str, ...]] = {
    # 开心 / 笑
    "happy": ("笑", "哈", "好耶", "哇", "耶", "庆祝", "恭喜", "发财", "棒",
              "收获", "开心", "快乐", "厉害", "绝了", "起飞", "膨胀"),
    # 撒娇 / 爱意
    "love": ("爱你", "比心", "心动", "心动了", "飞吻", "花", "抱", "亲",
             "宝贝", "想", "贴贴", "扑通"),
    # 难过 / 委屈
    "sad": ("哭", "泪", "委屈", "难受", "可怜", "太难", "我不活", "呜呜",
            "伤心", "emo", "Emo", "EMO", "孤单", "寂寞", "失落", "郁闷"),
    # 生气 / 不爽
    "angry": ("生气", "气死", "气疯", "谴责", "巴掌", "揪", "凶", "滚",
              "烦", "讨厌", "怒", "白眼", "哼"),
    # 累 / 困
    "tired": ("累", "困", "瞌睡", "倦", "无精打采", "摆烂", "葛优躺",
              "瘫", "融化", "躺着", "摊", "歇"),
    # 睡觉 / 早晚安
    "sleep": ("睡", "晚安", "熬夜", "呼呼", "碎觉", "趴窝", "蜷缩", "被窝",
              "做梦"),
    "morning": ("早安", "早上", "起床", "醒了"),
    # 招呼 / 告别
    "greet": ("拜拜", "挥手", "挥舞", "hi", "hello", "哈喽", "你哈",
              "出现", "冒头", "冒泡", "初来", "登场", "回来", "回头",
              "oi", "ok"),
    # 疑惑 / 无语
    "confused": ("你说什么", "这是什么", "头大", "让我想想", "呆", "尴尬",
                 "佛了", "问号", "❔", "无语", "迷惑", "不懂", "问"),
    # 吃
    "food": ("吃", "饿", "肉", "橘", "寿司", "蛋糕", "喝", "奶茶", "好吃",
             "饱", "舔", "嚼", "辣", "冰淇淋"),
    # 可爱 / 卖萌
    "cute": ("可爱", "乖", "捏", "摸", "揣手", "毛团", "圆", "小", "软",
             "喵", "汪", "兽耳", "尾巴", "耳朵", "狐", "狸", "猫", "三花",
             "粉", "蓬", "萌"),
    # 忙 / 上班
    "work": ("忙", "工作", "加班", "上班", "干活", "摸鱼", "划水", "开会",
             "打工", "搬砖", "下班"),
    # 赞 / 同意
    "cool": ("赞", "酷", "真棒", "好棒", "好耶", "搞定", "没问题", "可以",
             "同意", "有道理", "挺你", "支持", "牛", "强", "厉害"),
    # 惊讶 / 慌张
    "surprise": ("惊", "吓", "慌张", "手忙", "抓狂", "救命", "失心疯",
                 "大胆想法", "震惊", "警觉", "小心", "注意", "危险"),
    # 害羞
    "shy": ("害羞", "不好意思", "脸红", "偷偷", "偷瞄", "暗中", "藏",
            "躲", "背对", "趴", "缩"),
    # 冷漠 / 走开（冷战专用）
    "cold": ("略略略", "此地空余", "溜了", "逃跑", "飞走", "消失", "潜水",
             "冷", "无视", "杂鱼", "纸片", "别问", "不关"),
    # 附和 / 围观
    "observe": ("观察", "看", "盯", "偷看", "瞄", "围观", "吃瓜", "潜水",
                "探头", "看戏"),
    # 运动 / 活力
    "active": ("跑", "跳", "冲", "飞", "转圈", "扭", "弹", "打滚", "蹦",
               "踩", "腿", "舞", "滑"),
}

# 标签 → 选图权重（同一池子里按这个加权，让常用的多出现）
_LABEL_WEIGHT: Dict[str, float] = {
    "happy": 1.4, "laugh": 1.4, "love": 1.2, "cute": 1.3, "tired": 1.1,
    "sleep": 1.0, "greet": 1.1, "confused": 1.0, "food": 1.0, "cold": 0.9,
    "angry": 0.9, "sad": 0.9, "observe": 1.0, "cool": 1.0, "shy": 0.9,
    "surprise": 0.9, "work": 0.8, "active": 0.9, "morning": 0.7,
}

# 情绪 → 标签（mood_of() 的产出）
_MOOD_LABELS: Dict[str, List[str]] = {
    "开心": ["happy", "love", "cute", "active"],
    "平静": ["observe", "cool", "cute", "greet"],
    "无聊": ["observe", "tired", "confused", "food"],
    "低落": ["sad", "tired", "shy"],
    "烦躁": ["angry", "annoyed", "cold"],
    "疲惫": ["tired", "sleep", "sad"],
    "兴奋": ["happy", "active", "surprise"],
    "害羞": ["shy", "love", "cute"],
    "冷漠": ["cold", "observe", "tired"],
}

# 场景关键词 → 标签（从对方消息里读出来的"该接什么表情"）
_SCENE_RULES: List[Tuple[str, List[str]]] = [
    (r"(哈哈|笑死|好笑|搞笑|笑不活|乐死|笑嘻)", ["happy", "laugh"]),
    (r"(晚安|睡了|睡觉|困了|我先睡)", ["sleep", "morning"]),
    (r"(早安|早上好|起床)", ["morning"]),
    (r"(吃了|吃饭|饿|外卖|奶茶|好吃|夜宵)", ["food"]),
    (r"(生气|气死|讨厌|烦人|滚)", ["angry"]),
    (r"(难过|哭了|想哭|委屈|emo|难受|不开心)", ["sad", "love"]),
    (r"(想你|喜欢|爱你|亲亲)", ["love", "shy"]),
    (r"(在干嘛|忙吗|忙不忙|上班|加班|工作)", ["work", "tired"]),
    (r"(谢谢|感谢|多谢)", ["shy", "love"]),
    (r"(你好|在吗|hi|hello|嗨)", ["greet"]),
    (r"(再见|拜拜|我走了|下播)", ["greet"]),
    (r"(什么|为啥|为什么|怎么|真的假的|\?)", ["confused"]),
    (r"(厉害|牛|强|赞|恭喜|不错)", ["cool", "happy"]),
    (r"(猫|喵|狗|汪)", ["cute"]),
    (r"(累|困|不想动|好累)", ["tired"]),
]


# ---------------------------------------------------------------------------
# 表情包服务
# ---------------------------------------------------------------------------

class StickerService:
    """表情目录同步 + 选图 + 发送记录。"""

    def __init__(self, db: Database, settings: Settings, conn: Any = None) -> None:
        self.db = db
        self.settings = settings
        self.conn = conn
        self._last_sync = 0.0
        self._cache: List[Dict[str, Any]] = []
        self._label_index: Dict[str, List[Dict[str, Any]]] = {}
        self._loaded = False

    @property
    def cfg(self) -> StickerConfig:
        return self.settings.sticker

    # -- 同步 -------------------------------------------------------------

    def bind(self, conn: Any) -> None:
        self.conn = conn

    def sync(self, force: bool = False) -> int:
        """从平台拉取表情目录并写入本地。返回写入条数。"""
        if not self.cfg.enabled:
            return 0
        if self.conn is None or getattr(self.conn, "im", None) is None:
            return 0
        if not force and (now() - self._last_sync) < 60:
            return 0
        try:
            albums = self.conn.im.get_sticker_albums() or []
        except Exception as exc:
            log.warning("拉取表情专辑失败: %s", exc)
            return 0

        want = set(self.cfg.preferred_albums or [])
        total = 0
        for a in albums:
            aid = int(getattr(a, "id", 0) or 0)
            aname = str(getattr(a, "name", "") or "")
            if want and aid not in want:
                continue
            try:
                stickers = self.conn.im.get_stickers(aid) or []
            except Exception as exc:
                log.debug("拉取专辑 %s 表情失败: %s", aid, exc)
                continue
            rows = []
            for s in stickers:
                sid = int(getattr(s, "id", 0) or 0)
                if not sid:
                    continue
                nm = str(getattr(s, "name", "") or "").strip()
                rows.append((
                    sid, aid, aname, nm,
                    str(getattr(s, "image_url", "") or ""),
                    str(getattr(s, "thumb_url", "") or ""),
                    int(getattr(s, "width", 0) or 0),
                    int(getattr(s, "height", 0) or 0),
                    0 if re.fullmatch(r"\d+", nm or "x") else 1,
                    json.dumps(self.labels_for(nm), ensure_ascii=False),
                    now(),
                ))
            if rows:
                self.db.executemany(
                    "INSERT OR REPLACE INTO sticker_catalog"
                    "(sticker_id,album_id,album_name,name,image_url,thumb_url,"
                    "width,height,descriptive,labels,synced_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
                total += len(rows)

        self._last_sync = now()
        self._loaded = False
        log.info("表情目录已同步：%d 张", total)
        BUS.publish("sticker", {"event": "sync", "count": total})
        return total

    def maybe_sync(self) -> None:
        """按配置的间隔自动同步（启动与定时都会调）。"""
        if not self.cfg.enabled or not self.cfg.auto_sync:
            return
        if (now() - self._last_sync) >= self.cfg.sync_interval_sec:
            self.sync()

    # -- 标签 -------------------------------------------------------------

    @staticmethod
    def labels_for(name: str) -> List[str]:
        """按名字给表情打标签。"""
        if not name:
            return []
        out: List[str] = []
        for label, keys in _LABEL_PATTERNS.items():
            for k in keys:
                if k in name:
                    out.append(label)
                    break
        # "笑不活了" 同时是 happy 和 laugh
        if any("笑" in k for k in (name,)) and "happy" in out:
            out.append("laugh")
        return out

    # -- 目录 -------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        sql = ("SELECT sticker_id,album_id,album_name,name,image_url,thumb_url,"
               "width,height,descriptive,labels FROM sticker_catalog")
        params: Sequence[Any] = ()
        if self.cfg.descriptive_only:
            sql += " WHERE descriptive=1"
        rows = self.db.query(sql, params)
        self._cache = []
        self._label_index = {}
        for r in rows:
            try:
                labels = json.loads(r["labels"] or "[]")
            except Exception:
                labels = []
            item = {
                "id": int(r["sticker_id"]),
                "album_id": int(r["album_id"] or 0),
                "album": r["album_name"] or "",
                "name": r["name"] or "",
                "image_url": r["image_url"] or "",
                "thumb_url": r["thumb_url"] or "",
                "width": int(r["width"] or 0),
                "height": int(r["height"] or 0),
                "labels": labels,
            }
            self._cache.append(item)
            for lb in labels:
                self._label_index.setdefault(lb, []).append(item)
        self._loaded = True
        log.debug("表情缓存载入：%d 张，%d 个标签",
                  len(self._cache), len(self._label_index))

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS c FROM sticker_catalog")
        return int(row["c"] or 0) if row else 0

    def list_albums(self) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT album_id, album_name, COUNT(*) AS n, "
            "SUM(descriptive) AS desc_n FROM sticker_catalog "
            "GROUP BY album_id, album_name ORDER BY album_id")
        return [dict(r) for r in rows]

    def list_by_label(self, label: str, limit: int = 60) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return self._label_index.get(label, [])[:limit]

    # -- 冷却与去重 -------------------------------------------------------

    def _recent_ids(self, bot_id: int, chat_id: int) -> Set[int]:
        win = self.cfg.repeat_window_sec
        rows = self.db.query(
            "SELECT sticker_id FROM sticker_usage "
            "WHERE bot_id=? AND chat_id=? AND ts>=?",
            (bot_id, chat_id, int(now() - win)))
        return {int(r["sticker_id"]) for r in rows}

    def _last_sent_ts(self, bot_id: int, chat_id: int) -> int:
        row = self.db.query_one(
            "SELECT ts FROM sticker_usage WHERE bot_id=? AND chat_id=? "
            "ORDER BY id DESC LIMIT 1", (bot_id, chat_id))
        return int(row["ts"] or 0) if row else 0

    def cooldown_ok(self, bot_id: int, chat_id: int) -> bool:
        return (now() - self._last_sent_ts(bot_id, chat_id)) >= self.cfg.cooldown_sec

    def daily_ok(self, bot_id: int) -> bool:
        row = self.db.query_one(
            "SELECT COUNT(*) AS c FROM sticker_usage WHERE bot_id=? AND ts>=?",
            (bot_id, now() - 86400))
        return int(row["c"] or 0) < self.cfg.daily_limit

    def note_sent(self, bot_id: int, chat_id: int, st: Dict[str, Any]) -> None:
        try:
            self.db.execute(
                "INSERT INTO sticker_usage(bot_id,chat_id,sticker_id,name,ts) "
                "VALUES(?,?,?,?,?)",
                (bot_id, chat_id, int(st.get("id") or 0),
                 st.get("name") or "", now()))
        except Exception as exc:
            log.debug("记录表情发送失败: %s", exc)

    def recent_stats(self, bot_id: int, limit: int = 40) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT u.ts, u.chat_id, u.name, c.album_name, c.image_url "
            "FROM sticker_usage u LEFT JOIN sticker_catalog c "
            "ON c.sticker_id = u.sticker_id "
            "WHERE u.bot_id=? ORDER BY u.id DESC LIMIT ?", (bot_id, limit))
        return [dict(r) for r in rows]

    # -- 选图 -------------------------------------------------------------

    def wanted_labels(self, mood: str = "", annoyance: int = 0,
                      user_text: str = "", energy: float = 0.6,
                      drowsy: float = 0.0) -> List[str]:
        """综合情绪、不满度、对方内容、精力，算出想要的标签（按优先级）。"""
        out: List[str] = []

        # 1) 不满度优先：冷战/生气时表情是"冷"的载体
        if annoyance >= 65:
            out += ["cold", "observe"]
        elif annoyance >= 40:
            out += ["cold", "confused"]
        elif annoyance >= 15:
            out += ["confused", "observe"]

        # 2) 对方内容里的场景
        if user_text:
            for pat, labels in _SCENE_RULES:
                if re.search(pat, user_text, flags=re.I):
                    out += labels

        # 3) 心情
        out += _MOOD_LABELS.get(mood or "", ["observe"])

        # 4) 精力/睡意修正
        if drowsy > 0.5:
            out += ["sleep", "tired"]
        elif energy < 0.3:
            out += ["tired"]

        # 去重保序
        seen: Set[str] = set()
        ordered: List[str] = []
        for lb in out:
            if lb not in seen:
                seen.add(lb)
                ordered.append(lb)
        return ordered

    def candidates(self, labels: List[str], bot_id: int, chat_id: int,
                   limit: int = 40) -> List[Dict[str, Any]]:
        """按标签优先级取候选，剔除最近发过的。

        关键点：不同优先级的标签**不混池**。
        比如"晚安"想要 sleep，如果和 observe/cute 混在一起随机，
        20 张卖萌图会把那 1 张"晚安"淹掉，结果就是晚安时发了个吃瓜。
        所以这里按标签顺序找第一个"够用"的池子：
          先找有 ≥2 张可选的 → 再找有 ≥1 张的 → 都没有才退回全量。
        """
        self._ensure_loaded()
        recent = self._recent_ids(bot_id, chat_id)

        groups: List[Tuple[str, List[Dict[str, Any]]]] = []
        for lb in labels:
            g = [it for it in self._label_index.get(lb, []) if it["id"] not in recent]
            if g:
                groups.append((lb, g))

        # 1) 最高优先级的池子（哪怕只有一张，语义对了比花样多重要）
        #    一旦这张被发过，它会在 repeat_window 内被排除，
        #    下一次自然就落到下一档标签，不会反复刷同一张。
        if groups:
            return groups[0][1][:limit]
        # 2) 全被最近发过 → 退回全量描写性表情（仍避开最近发过的）
        pool = [it for it in self._cache if it["id"] not in recent]
        return pool[:limit]

    def pick(self, *, bot_id: int, chat_id: int, mood: str = "",
             annoyance: int = 0, user_text: str = "", energy: float = 0.6,
             drowsy: float = 0.0, brain: Any = None,
             context: str = "") -> Optional[Dict[str, Any]]:
        """选一张合适的表情。返回 None 表示这次不发。"""
        if not self.cfg.enabled:
            return None
        if not self.cooldown_ok(bot_id, chat_id):
            return None
        if not self.daily_ok(bot_id):
            return None

        labels = self.wanted_labels(mood, annoyance, user_text, energy, drowsy)
        pool = self.candidates(labels, bot_id, chat_id,
                              limit=max(10, self.cfg.llm_candidates))
        if not pool:
            return None

        # 让模型在候选里挑（省成本：只给名字，不给图片）
        if self.cfg.llm_pick and brain is not None:
            chosen = self._llm_pick(brain, pool, context or user_text,
                                    labels, self.cfg.llm_candidates)
            if chosen is not None:
                return chosen

        return self._weighted_pick(pool, labels)

    def _weighted_pick(self, pool: List[Dict[str, Any]],
                       labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """按「标签优先级 × 标签权重」随机。

        标签在 wanted_labels 里的位置越靠前，说明越贴当前场景，
        对应表情的权重就越高。否则"晚安"这种场景会被同池里的
        无关表情（比如只是沾了 cool 标签的"好耶"）抢走。
        """
        prio = {lb: i for i, lb in enumerate(labels or [])}
        pairs = []
        for it in pool:
            w = 1.0
            best_pos = 99
            for lb in it.get("labels") or []:
                w = max(w, _LABEL_WEIGHT.get(lb, 1.0))
                best_pos = min(best_pos, prio.get(lb, 99))
            # 位置加成：第 1 顺位 ×3.0，第 2 ×2.2，第 3 ×1.6，第 4 ×1.2
            tier = {0: 3.0, 1: 2.2, 2: 1.6, 3: 1.2}.get(best_pos, 0.6)
            pairs.append((it, w * tier))
        return pick_weighted(pairs)

    def _llm_pick(self, brain: Any, pool: List[Dict[str, Any]], context: str,
                  labels: List[str], n: int) -> Optional[Dict[str, Any]]:
        """让模型从候选名字里挑一个最贴语境的。失败返回 None。"""
        shortlist = pool[:n]
        names = [it["name"] for it in shortlist]
        sys_p = (
            "你在帮一个真人挑聊天表情包。根据对话语境，从候选里选最合适的那个。"
            "只输出 JSON，不要解释。\n"
            '格式：{"index": 0}\n'
            "如果没有合适的，输出 {\"index\": -1}。"
        )
        user_p = (
            "场景标签：%s\n"
            "对话语境：%s\n"
            "候选表情名（用序号引用，从 0 开始）：\n%s"
            % ("、".join(labels[:8]), (context or "")[:200],
               "\n".join("%d. %s" % (i, nm) for i, nm in enumerate(names)))
        )
        try:
            from ..settings import TASK_INTENT
            res = brain.complete_json(TASK_INTENT, sys_p, user_p,
                                      max_tokens=40, temperature=0.3)
        except Exception as exc:
            log.debug("表情 LLM 挑选失败，退回随机: %s", exc)
            return None

        try:
            idx = int(res.get("index", -1))
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= len(shortlist):
            return None
        return shortlist[idx]

    # -- 发送判定 ---------------------------------------------------------

    def should_send(self, *, annoyance: int = 0, user_text: str = "",
                    will_reply_text: bool = True) -> str:
        """决定这次以什么形态发表情。

        返回：
            "solo"   只发表情，不回文字
            "append" 文字之后补一个表情
            "none"   不发
        """
        if not self.cfg.enabled:
            return "none"
        if will_reply_text:
            return "append" if dice(self.cfg.append_prob) else "none"
        # 不打算回文字时：只有对方发了"纯情绪"内容才用表情顶一下
        if not user_text:
            return "none"
        pure_emotion = bool(re.search(
            r"(^(哈哈+|嘿嘿+|嘻嘻+|emmm+|额+|啊+|嗯+|哦+|草+)$)|"
            r"(笑死|哈哈哈|绷不住|绝了|无语|服了)", user_text))
        if pure_emotion and dice(self.cfg.solo_prob * 2.5):
            return "solo"
        return "solo" if dice(self.cfg.solo_prob * 0.4) else "none"

    def cold_sticker(self, *, bot_id: int, chat_id: int, mood: str = "",
                     annoyance: int = 0, brain: Any = None) -> Optional[Dict[str, Any]]:
        """冷战专用：只回一张敷衍/走开的表情。"""
        if not self.cfg.enabled or not dice(self.cfg.cold_solo_prob):
            return None
        return self.pick(bot_id=bot_id, chat_id=chat_id, mood=mood or "冷漠",
                         annoyance=max(annoyance, 70), brain=brain,
                         context="不想理他")
