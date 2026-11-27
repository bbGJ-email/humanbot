"""Agent 工具层：给 BOT 一批"查资料"的真实能力。

设计：
- 每个工具都是独立函数，返回一段可以直接塞进 prompt 的中文短文本；
  失败时返回友好提示，绝不抛异常打断聊天流程。
- 全部走免费公开 API（不需要 Key）；VPS 在香港，维基/谷歌/DuckDuckGo
  都可以直连，国内部署时 wiki/search/translate 可能超时（走代理即可）。
- 是否调用由一个轻量 LLM 决策（TASK_TOOL），闲聊一律不查，防滥用。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

from .logging_setup import BUS, get_logger

log = get_logger("tools")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Open-Meteo weather_code → 中文
_WMO = {
    0: "晴", 1: "大致晴", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "毛毛雨", 55: "浓毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷暴冰雹",
}


@dataclass
class Tool:
    name: str
    desc: str
    args: str
    func: Callable[[Dict[str, Any]], str]


def _http_get(url: str, *, params: Dict[str, Any] = None,
              timeout: float = 10.0, headers: Dict[str, str] = None) -> Optional[Any]:
    """统一 GET：失败返回 None，绝不抛。202 也算成功（部分 API 反爬返回 202 但带正常数据）。"""
    try:
        resp = requests.get(url, params=params or {}, timeout=timeout,
                            headers={"User-Agent": _UA, **(headers or {})})
        if resp.status_code not in (200, 202):
            log.debug("工具请求 %s -> %s", url[:80], resp.status_code)
            return None
        if "json" in resp.headers.get("content-type", ""):
            return resp.json()
        return resp.text
    except Exception as exc:
        log.debug("工具请求异常 %s: %s", url[:80], exc)
        return None


def _clip(text: str, n: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:n] + ("…" if len(text) > n else "")


# =============================================================================
# 各工具实现
# =============================================================================

def t_weather(args: Dict[str, Any]) -> str:
    city = str(args.get("city") or args.get("location") or "").strip()
    if not city:
        return "查询失败：没给城市名"
    geo = _http_get("https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "language": "zh"})
    if not geo or not geo.get("results"):
        return f"没查到城市「{city}」"
    g = geo["results"][0]
    w = _http_get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": g["latitude"], "longitude": g["longitude"],
        "current": "temperature_2m,apparent_temperature,weather_code,"
                   "precipitation_probability,relative_humidity_2m,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,"
                 "precipitation_probability_max",
        "timezone": "auto", "forecast_days": 2,
    })
    if not w:
        return "天气查询失败"
    cur = w.get("current", {})
    daily = w.get("daily", {})
    name = g.get("name") or city
    desc = _WMO.get(cur.get("weather_code"), "未知天气")
    out = [f"{name}现在：{desc}，{cur.get('temperature_2m')}°C"
           f"（体感 {cur.get('apparent_temperature')}°C），"
           f"降水概率 {cur.get('precipitation_probability')}%，"
           f"风速 {cur.get('wind_speed_10m')} km/h"]
    dates = daily.get("time") or []
    for i, d in enumerate(dates[:2]):
        out.append(
            f"{'今天' if i == 0 else '明天'}{d}: "
            f"{_WMO.get((daily.get('weather_code') or [None])[i], '')} "
            f"{daily.get('temperature_2m_min', [None])[i]}~"
            f"{daily.get('temperature_2m_max', [None])[i]}°C, "
            f"降水 {daily.get('precipitation_probability_max', [None])[i]}%")
    return "；".join(out)


def t_search(args: Dict[str, Any]) -> str:
    """联网查资料：Google News RSS（新闻检索）→ DDG 即时答案 → 维基词条。"""
    q = str(args.get("query") or args.get("q") or "").strip()
    if not q:
        return "搜索失败：没给关键词"
    # 1) Google News RSS —— 最稳，新闻类问题首选
    g = _http_get("https://news.google.com/rss/search", params={
        "q": q, "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"}, timeout=12)
    if g:
        items = re.findall(r"<item>(.*?)</item>", g, re.S)
        out = []
        for it in items[:5]:
            m = re.search(r"<title>(.*?)</title>", it, re.S)
            title = re.sub(r"<[^>]+>", "", m.group(1) if m else "").strip()
            src = re.search(r"<source[^>]*>(.*?)</source>", it)
            if title:
                out.append(title + (f"（{src.group(1)}）" if src else ""))
        if out:
            return "相关的最新消息：" + "；".join(out)
    # 2) DDG 即时答案（实体类问题）
    d = _http_get("https://api.duckduckgo.com/", params={
        "q": q, "format": "json", "no_html": 1}, timeout=12)
    if d and d.get("AbstractText"):
        return f"查到：{_clip(d['AbstractText'], 300)}"
    # 3) 维基百科兜底
    r = t_wiki({"query": q})
    if "没查到" not in r and "失败" not in r:
        return r
    return f"「{q}」没搜到什么有用的"


def t_wiki(args: Dict[str, Any]) -> str:
    """维基百科词条摘要。走 api.wikimedia.org 官方网关——
    zh.wikipedia.org 对共享 IP 限流太狠（429），网关这边宽松得多。"""
    q = str(args.get("query") or args.get("term") or "").strip()
    if not q:
        return "百科查询失败：没给词条"
    d = _http_get("https://api.wikimedia.org/core/v1/wikipedia/zh/search/page",
                  params={"q": q, "limit": 1}, timeout=12,
                  headers={"User-Agent": "HumanBot/1.0 "
                          "(https://github.com/bbgj-email/humanbot)"})
    pages = (d or {}).get("pages") or []
    if not pages:
        return f"维基百科里没有「{q}」"
    pg = pages[0]
    excerpt = re.sub(r"<[^>]+>", "", pg.get("excerpt") or "").strip()
    return f"【{pg.get('title')}】{_clip(excerpt, 360)}"


_HOT_SOURCES = {"百度": "baidu", "微博": "baidu", "热搜": "baidu",
                "B站": "bilibili", "哔哩哔哩": "bilibili",
                "头条": "toutiao", "抖音": "toutiao"}


def t_hot(args: Dict[str, Any]) -> str:
    plat = str(args.get("platform") or args.get("type") or "百度").strip()
    src = _HOT_SOURCES.get(plat, "baidu")
    note = "" if src == _HOT_SOURCES.get(plat) else f"（{plat}榜源暂不可用，给相近的热榜）"
    items: List[str] = []

    if src == "baidu":
        d = _http_get("https://top.baidu.com/api/board/realtime",
                      params={"tab": "realtime"}, timeout=12)
        try:
            content = d["data"]["cards"][0]["content"]
            items = [str(c.get("word") or c.get("query") or "").strip()
                     for c in content][:8]
        except (KeyError, IndexError, TypeError):
            items = []
        label = "百度热搜"
    elif src == "bilibili":
        d = _http_get("https://api.bilibili.com/x/web-interface/popular",
                      params={"ps": 8}, timeout=12)
        try:
            items = [str(v.get("title") or "").strip()
                     for v in d["data"]["list"]][:8]
        except (KeyError, TypeError):
            items = []
        label = "B站热门"
    else:
        d = _http_get("https://www.toutiao.com/hot-event/hot-board/",
                      params={"origin": "toutiao_pc"}, timeout=12)
        try:
            items = [str(x.get("Title") or "").strip()
                     for x in d.get("data", [])][:8]
        except (KeyError, TypeError, AttributeError):
            items = []
        label = "头条热榜"

    items = [i for i in items if i]
    if not items:
        return f"{plat}热榜拉取失败"
    return (note + f"{label}：") + "；".join(items)


_60S_MIRRORS = ["https://60s.viki.moe/v2/60s", "https://60s.b23.run/v2/60s"]


def t_news(args: Dict[str, Any]) -> str:
    for url in _60S_MIRRORS:
        d = _http_get(url, timeout=12)
        data = (d or {}).get("data") or {}
        news = data.get("news") or []
        if news:
            tops = [re.sub(r"^\d+[、.．]\s*", "", str(n)) for n in news[:10]]
            return "今日新闻（60s读懂世界）：" + "；".join(t[:60] for t in tops)
    # 60s 全挂 → 用 Google News 头条兜底
    g = _http_get("https://news.google.com/rss", params={
        "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"}, timeout=12)
    items = re.findall(r"<title>(.*?)</title>", g or "", re.S)
    tops = [re.sub(r"<[^>]+>", "", t) for t in items[1:11]]
    if tops:
        return "今日要闻：" + "；".join(t[:60] for t in tops)
    return "今日新闻拉取失败"


def t_exchange(args: Dict[str, Any]) -> str:
    src = str(args.get("from") or args.get("base") or "USD").strip().upper()
    dst = str(args.get("to") or args.get("target") or "CNY").strip().upper()
    d = _http_get(f"https://open.er-api.com/v6/latest/{src}", timeout=12)
    rates = (d or {}).get("rates") or {}
    if dst not in rates:
        return f"汇率查询失败（{src}→{dst}）"
    r = rates[dst]
    return f"1 {src} ≈ {r:.4f} {dst}（来源：open.er-api.com，每日更新）"


_CRYPTO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "DOGE": "dogecoin",
    "ADA": "cardano", "TRX": "tron", "DOT": "polkadot",
}


def t_crypto(args: Dict[str, Any]) -> str:
    sym = str(args.get("coin") or args.get("symbol") or "BTC").strip().upper()
    cid = _CRYPTO_IDS.get(sym, sym.lower())
    d = _http_get("https://api.coingecko.com/api/v3/simple/price", params={
        "ids": cid, "vs_currencies": "usd,cny",
        "include_24hr_change": "true"}, timeout=12)
    row = (d or {}).get(cid)
    if not row:
        return f"没查到 {sym} 的价格"
    chg = row.get("usd_24h_change")
    trend = f"，24h{'+' if (chg or 0) >= 0 else ''}{chg:.1f}%" if chg is not None else ""
    return f"{sym} 现价 ≈ ${row.get('usd')}（约 ¥{row.get('cny')}）{trend}"


def t_translate(args: Dict[str, Any]) -> str:
    text = str(args.get("text") or args.get("q") or "").strip()
    tl = str(args.get("target") or args.get("to") or "").strip()
    if not text:
        return "翻译失败：没给内容"
    if not tl:
        # 没给目标语言时简单猜：含中文→英，否则→中
        tl = "en" if re.search(r"[\u4e00-\u9fff]", text) else "zh-CN"
    d = _http_get("https://translate.googleapis.com/translate_a/single", params={
        "client": "gtx", "sl": "auto", "tl": tl, "dt": "t", "q": text},
        timeout=12)
    if not d or not d[0]:
        return "翻译失败"
    out = "".join(seg[0] for seg in d[0] if seg and seg[0])
    return f"翻译（→{tl}）：{_clip(out, 400)}"


def t_holiday(args: Dict[str, Any]) -> str:
    date = str(args.get("date") or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        date = time.strftime("%Y-%m-%d")
    d = _http_get(f"https://timor.tech/api/holiday/info/{date}", timeout=12)
    data = (d or {}).get("type") or {}
    if not data:
        return "节假日查询失败"
    kinds = {0: "工作日", 1: "周末双休", 2: "节假日", 3: "调休补班"}
    name = (d or {}).get("holiday") or (d or {}).get("name") or ""
    week = data.get("week")
    out = f"{date} 是{kinds.get(data.get('type', '?'), '未知')}"
    if name:
        out += f"（{name}）"
    if week is not None:
        out += f"，星期{week}"
    after = (d or {}).get("after")
    if after:
        out += f"；下一个假期：{after.get('name')}（还有{after.get('rest')}天）"
    return out


def t_hitokoto(args: Dict[str, Any]) -> str:
    # c 必须用 list 传（requests 会展开成 c=a&c=b…），拼成一个逗号串会被拒
    d = _http_get("https://v1.hitokoto.cn/",
                  params={"c": ["a", "b", "d", "i", "k"]}, timeout=10)
    if not d or not d.get("hitokoto"):
        return "一言接口没响应"
    out = f"「{d['hitokoto']}」"
    if d.get("from"):
        out += f"——{d['from']}" + (f"·{d['from_who']}" if d.get("from_who") else "")
    return out


def t_ip_info(args: Dict[str, Any]) -> str:
    d = _http_get("http://ip-api.com/json/", params={"lang": "zh-CN"}, timeout=10)
    if not d or d.get("status") != "success":
        return "IP 查询失败"
    return (f"当前出口 IP：{d.get('query')}，"
            f"{d.get('country')}/{d.get('city')}，"
            f"时区 {d.get('timezone')}，运营商 {d.get('isp')}")


# =============================================================================
# 注册表
# =============================================================================

def default_tools() -> List[Tool]:
    return [
        Tool("weather", "查任意城市的实时天气和今明两天预报",
             '{"city":"上海"}', t_weather),
        Tool("search", "联网查资料：最新消息、新闻、不知道的事实",
             '{"query":"xxx的最新消息"}', t_search),
        Tool("wiki", "查维基百科词条简介，适合概念/人物/事物解释",
             '{"query":"相对论"}', t_wiki),
        Tool("hot", "查实时热榜（百度热搜/B站热门/头条热榜）",
             '{"platform":"百度"}', t_hot),
        Tool("news", "看今天的大新闻（每日60秒读懂世界）", "{}", t_news),
        Tool("exchange", "查汇率", '{"from":"USD","to":"CNY"}', t_exchange),
        Tool("crypto", "查加密货币现价（BTC/ETH/SOL/DOGE 等）",
             '{"coin":"BTC"}', t_crypto),
        Tool("translate", "翻译任意文本，target 用 en/zh-CN/ja 等",
             '{"text":"hello","target":"zh-CN"}', t_translate),
        Tool("holiday", "查某天是不是节假日/周末，以及下一个假期倒计时",
             '{"date":"2026-01-01"} 或 {}', t_holiday),
        Tool("hitokoto", "随机来一句文学/诗词/动画台词", "{}", t_hitokoto),
        Tool("ip_info", "查当前服务器出口 IP 和位置", "{}", t_ip_info),
    ]


class ToolKit:
    """工具集合：决策用 specs，执行用 run。"""

    def __init__(self, cfg: Any = None, media: Any = None) -> None:
        self.cfg = cfg
        self.media = media          # draw 用，BotRuntime 启动后注入
        self.tools: List[Tool] = default_tools()
        self._map = {t.name: t for t in self.tools}

    def specs_text(self) -> str:
        """给决策 LLM 看的工具清单。"""
        return "\n".join(
            f"- {t.name}：{t.desc} | 参数示例：{t.args}" for t in self.tools)

    def has(self, name: str) -> bool:
        return name in self._map

    def run(self, name: str, args: Dict[str, Any]) -> str:
        t = self._map.get(name)
        if not t:
            return f"未知工具：{name}"
        timeout = float(getattr(self.cfg, "tool_timeout", 12.0)) if self.cfg else 12.0
        # 单个工具内部有自己的超时；这里兜一层防卡死
        t0 = time.time()
        try:
            result = t.func(args or {})
        except Exception as exc:
            log.warning("工具 %s 异常: %s", name, exc)
            result = "查询失败"
        BUS.publish("tool", {
            "name": name, "args": args, "ms": int((time.time() - t0) * 1000),
            "preview": result[:80],
        })
        log.info("工具 %s %s -> %s", name, args, result[:60])
        return result
