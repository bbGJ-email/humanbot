"""图片系统：生成 + 检索（策划案 §6）。

实测结论（2026-10）：
- Pollinations.ai 文生图可用，但约 15 秒限流一次，必须带 UA + 重试
- Bing 图片检索可用（页面解析 murl 字段）
- 百度/Wikimedia/Openverse/Pixabay 均不可用或无响应
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from ..settings import ASSETS_DIR, MediaConfig
from .logging_setup import BUS, get_logger
from .storage import Database
from .utils import dice, now, text_hash

log = get_logger("media")

# 出图风格：假装是随手拍（避免"AI 精美感"）
PHOTO_STYLE = ("shot on iPhone, casual snapshot, slightly blurry, "
               "natural lighting, everyday life photo, amateur photography")
# 禁止出现的高级感词（会暴露AI生成）
NEGATIVE_STYLE = ["cinematic", "8k", "masterpiece", "ultra detailed",
                  "studio lighting", "professional"]

# assets/media 里最多保留多少张图（超出按时间裁掉旧的）
_MAX_CACHE_FILES = 200


class MediaService:
    """图片生成与检索。"""

    def __init__(self, db: Database, cfg: MediaConfig) -> None:
        self.db = db
        self.cfg = cfg
        self.out_dir = ASSETS_DIR / "media"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._last_gen = 0.0

    # -- 生成 -------------------------------------------------------------

    def build_prompt(self, scene: str, style: str = "casual") -> str:
        """把场景描述包装成"随手拍"的 prompt。"""
        s = (scene or "").strip()
        for bad in NEGATIVE_STYLE:
            s = re.sub(bad, "", s, flags=re.I)
        if style == "casual":
            return f"{s}, {PHOTO_STYLE}"
        return s

    def generate(self, prompt: str, *, seed: Optional[int] = None,
                 width: Optional[int] = None, height: Optional[int] = None
                 ) -> Optional[str]:
        """生成图片，返回本地路径。带缓存与限流重试。"""
        if not self.cfg.enabled:
            return None
        full = self.build_prompt(prompt)
        h = text_hash(full)
        cached = self._cache_get(h, "gen")
        if cached:
            log.info("图片命中缓存: %s", h)
            return cached

        # 限流保护：距上次生成不足间隔则等待
        gap = self.cfg.gen_rate_gap - (time.time() - self._last_gen)
        if gap > 0:
            log.info("等待限流间隔 %.1fs", gap)
            time.sleep(gap)

        url = self.cfg.gen_endpoint.format(prompt=quote(full, safe=""))
        params = {
            "width": width or self.cfg.gen_width,
            "height": height or self.cfg.gen_height,
            "nologo": "true",
            "model": self.cfg.gen_model,
        }
        if seed is not None:
            params["seed"] = seed

        for attempt in range(1, self.cfg.gen_retry + 1):
            try:
                resp = requests.get(
                    url, params=params, timeout=90,
                    headers={"User-Agent": self.cfg.user_agent},
                )
                ctype = resp.headers.get("content-type", "")
                if resp.status_code == 200 and ctype.startswith("image"):
                    path = self.out_dir / f"{h}.jpg"
                    path.write_bytes(resp.content)
                    self._last_gen = time.time()
                    self._cache_put(h, "gen", full, str(path))
                    log.info("生成图片成功 (%d bytes): %s", len(resp.content), path.name)
                    BUS.publish("media", {"kind": "gen", "path": str(path),
                                          "prompt": full})
                    return str(path)
                log.warning("生成失败 attempt=%d status=%s body=%s", attempt,
                            resp.status_code, resp.text[:120])
            except Exception as exc:
                log.warning("生成异常 attempt=%d: %s", attempt, exc)
            time.sleep(self.cfg.gen_rate_gap if attempt < self.cfg.gen_retry else 5)
        return None

    # -- 检索 -------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> List[str]:
        """图片检索（Bing 页面解析），返回图片直链。"""
        if not self.cfg.enabled or not query:
            return []
        try:
            resp = requests.get(
                "https://www.bing.com/images/search",
                params={"q": query}, timeout=20,
                headers={"User-Agent": self.cfg.user_agent},
            )
            urls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
            # 过滤明显非图片
            urls = [u for u in urls if u.startswith("http")
                    and not u.endswith((".svg", ".gif"))]
            seen, out = set(), []
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    out.append(u)
                if len(out) >= limit:
                    break
            log.info("检索到 %d 张图：%s", len(out), query)
            return out
        except Exception as exc:
            log.warning("图片检索失败: %s", exc)
            return []

    def search_and_download(self, query: str) -> Optional[str]:
        """检索并下载第一张可用图，返回本地路径。"""
        h = text_hash("search:" + query)
        cached = self._cache_get(h, "search")
        if cached:
            return cached
        for url in self.search(query, 6):
            try:
                resp = requests.get(url, timeout=20,
                                    headers={"User-Agent": self.cfg.user_agent})
                if resp.status_code == 200 and len(resp.content) > 4096:
                    path = self.out_dir / f"s_{h}.jpg"
                    path.write_bytes(resp.content)
                    self._cache_put(h, "search", query, str(path))
                    return str(path)
            except Exception:
                continue
        return None

    def manual_generate(self, prompt: str) -> Optional[str]:
        """控制台手动生成。"""
        return self.generate(prompt)

    # -- 缓存 -------------------------------------------------------------

    def _cache_get(self, h: str, kind: str) -> Optional[str]:
        if not self.cfg.cache_enabled:
            return None
        row = self.db.query_one(
            "SELECT path FROM media_cache WHERE hash=? AND kind=?", (h, kind))
        if row and row["path"] and Path(row["path"]).exists():
            self.db.execute("UPDATE media_cache SET used=used+1 WHERE hash=?", (h,))
            return str(row["path"])
        return None

    def _cache_put(self, h: str, kind: str, prompt: str, path: str) -> None:
        if not self.cfg.cache_enabled:
            return
        self.db.execute(
            "INSERT OR REPLACE INTO media_cache(hash, kind, prompt, path, ts, used) "
            "VALUES(?,?,?,?,?,1)",
            (h, kind, prompt, path, now()),
        )
        self._prune_cache()

    def _prune_cache(self) -> None:
        """裁掉过期的出图，避免 assets/media 无限膨胀。

        换头像和历史出图都会往这里写文件，常驻运行几个月就能堆到几百兆。
        只保留最近 _MAX_CACHE_FILES 张，被裁掉的文件和记录一起删。
        """
        try:
            rows = self.db.query(
                "SELECT hash, path FROM media_cache ORDER BY ts DESC LIMIT -1 OFFSET ?",
                (_MAX_CACHE_FILES,),
            )
            if not rows:
                return
            for r in rows:
                p = Path(str(r["path"]))
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
                self.db.execute("DELETE FROM media_cache WHERE hash=?", (r["hash"],))
            log.info("媒体缓存已清理 %d 个旧文件", len(rows))
        except Exception as exc:
            log.debug("媒体缓存清理失败: %s", exc)

    def list_cache(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT hash, kind, prompt, path, ts, used FROM media_cache "
            "ORDER BY ts DESC LIMIT ?", (limit,),
        )
        return [dict(r) for r in rows]

    # -- 触发决策 ---------------------------------------------------------

    def should_send_image(self, user_text: str, mood: str) -> Optional[Tuple[str, str]]:
        """判断是否该发图，返回 (kind, query/prompt)。"""
        if not self.cfg.enabled:
            return None
        t = (user_text or "").lower()
        if re.search(r"(在干嘛|在做什么|在干什么|忙什么)", t):
            return ("gen", "a desk in an old bookstore, afternoon light, a cup of coffee")
        if re.search(r"(发张|发个|拍张).*(照片|图)", t):
            return ("search", "old bookstore interior")
        if re.search(r"(喜欢你|我的猫|我的狗)", t):
            return ("search", "cat on a windowsill")
        if mood in ("烦", "很烦，别惹我") and dice(0.4):
            return ("gen", "rainy street at night, empty, window reflection")
        if mood == "睡不着" and dice(0.5):
            return ("gen", "night window view from inside a small apartment")
        return None
