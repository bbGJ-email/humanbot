"""日志与事件总线。"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..settings import LOG_DIR

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_configured = False


class ConsoleLogHandler(logging.Handler):
    """把日志送进事件总线，供控制台实时展示。"""

    def __init__(self, bus: "EventBus") -> None:
        super().__init__()
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bus.publish("log", {
                "ts": int(record.created * 1000),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            })
        except Exception:
            pass


def setup_logging(level: str = "INFO", bus: Optional["EventBus"] = None) -> None:
    """初始化日志：控制台 + 文件 + 可选事件总线。"""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("humanbot")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    fmt = logging.Formatter(_LOG_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # 文件日志必须带轮转：BOT 是 7×24 常驻进程，无轮转的 FileHandler
    # 会在几周内把磁盘吃满（尤其 VPS 上还并存着别的站点）。
    from logging.handlers import RotatingFileHandler

    fh = RotatingFileHandler(
        LOG_DIR / "humanbot.log", maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if bus is not None:
        ch = ConsoleLogHandler(bus)
        ch.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(ch)


class EventBus:
    """轻量进程内事件总线，附带环形缓冲供新客户端回放。"""

    def __init__(self, history: int = 500) -> None:
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._all: List[Callable[[Dict[str, Any]], None]] = []
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history)
        self._lock = threading.RLock()

    def subscribe(self, topic: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subs.setdefault(topic, []).append(fn)

    def subscribe_all(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._all.append(fn)

    def unsubscribe_all(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            if fn in self._all:
                self._all.remove(fn)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        evt = {"topic": topic, "ts": int(time.time() * 1000), "data": payload}
        with self._lock:
            self._history.append(evt)
            subs = list(self._subs.get(topic, [])) + list(self._all)
        for fn in subs:
            try:
                fn(evt)
            except Exception:
                pass

    def history(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._history)
        return items[-limit:]


# 全局总线（进程内单例）
BUS = EventBus()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"humanbot.{name}")
