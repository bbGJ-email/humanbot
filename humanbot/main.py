"""HumanBot 启动入口。

流程：
1. 加载配置 / 初始化数据库 / 装载 provider
2. 决定控制台端口（0 = 自动探测空闲端口）
3. 首次启动自动生成强密码并落库
4. 启动 BOT 管理器（后台线程跑 asyncio 事件循环）
5. 启动 FastAPI 控制台（绑定 0.0.0.0）
6. 打印启动横幅（含密码与访问地址）

用法：
    python -m humanbot.main
    python -m humanbot.main --port 8899
    python -m humanbot.main --no-bots      # 只跑控制台
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import threading
import time
from typing import Optional

from .settings import ROOT, Settings
from .core.crypto import gen_password
from .core.logging_setup import BUS, get_logger, setup_logging
from .core.storage import Database
from .console.server import (build_app, find_free_port, load_providers_from_db,
                             local_ip, public_ip)

log = get_logger("main")


# ---------------------------------------------------------------------------
# BOT 事件循环线程
# ---------------------------------------------------------------------------

class BotThread:
    """在独立线程中运行 asyncio 事件循环，托管所有 BOT。"""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.manager = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="humanbot-loop",
                                        daemon=True)
        self._thread.start()
        # 最多等 15 秒，避免 SDK 网络卡住时阻塞控制台启动
        self._ready.wait(timeout=15)

    def _run(self) -> None:
        from .layers.manager import BotManager

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.manager = BotManager(self.settings, self.db)
            # BotManager.start 是同步的（内部自己往 loop 上挂任务），
            # 不要 await / run_until_complete，否则这里会直接抛 TypeError
            self.manager.start(self.loop)
            log.info("BOT 管理器已启动，共 %d 个账号", len(self.manager.bots()))
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            log.exception("BOT 管理器启动失败: %s", exc)
        finally:
            self._ready.set()
        # 保持事件循环常驻
        try:
            self.loop.run_forever()
        except Exception as exc:  # noqa: BLE001
            log.debug("事件循环退出: %s", exc)

    def stop(self) -> None:
        if self.loop and self.manager:
            async def _shutdown() -> None:
                try:
                    await self.manager.stop()
                except Exception as exc:  # noqa: BLE001
                    log.debug("停止管理器异常: %s", exc)

            try:
                fut = asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)
                fut.result(timeout=8)
            except Exception as exc:  # noqa: BLE001
                log.debug("关闭 BOT 线程超时: %s", exc)
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)


# ---------------------------------------------------------------------------
# 密码
# ---------------------------------------------------------------------------

def ensure_password(settings: Settings, db: Database) -> tuple[str, bool]:
    """确保存在控制台密码。返回 (明文用于首次打印, 是否新建)。

    密码只以 PBKDF2 哈希 + 明文（供启动脚本打印）存在本地 data 目录，
    不会写入 config.yaml。
    """
    from .core.crypto import hash_password

    user = db.get_setting("console_user") or settings.console.username
    db.set_setting("console_user", user)

    hashed = db.get_setting("console_password_hash")
    if hashed:
        plain = db.get_setting("console_password_plain") or ""
        return plain, False

    plain = settings.console.password.strip() or gen_password(20)
    db.set_setting("console_password_hash", hash_password(plain))
    db.set_setting("console_password_plain", plain)
    return plain, True


# ---------------------------------------------------------------------------
# 横幅
# ---------------------------------------------------------------------------

def banner(settings: Settings, port: int, username: str, password: str,
           new_pwd: bool, n_bots: int, bot_err: Optional[str]) -> str:
    line = "=" * 62
    lan = local_ip()
    wan = public_ip()

    rows = [
        line,
        "  HumanBot - Super-human-like AI Bot System",
        line,
        f"  Console   : http://127.0.0.1:{port}",
        f"              http://{lan}:{port}          (LAN)",
    ]
    if wan:
        rows.append(f"              http://{wan}:{port}      (WAN)")
    rows += [
        f"  Username  : {username}",
        # 只有首次启动才打印明文密码；之后重启复用 DB 里的哈希，
        # 只提示去 data/console.txt 查，避免每次重启都像"重新生成"。
        f"  Password  : {password if new_pwd else '(已设置，见 data/console.txt)'}",
    ]
    if new_pwd:
        rows.append("              [NEW] 首次启动已随机生成强密码")
    rows += [
        f"  Bots      : {n_bots} account(s)",
        f"  Data dir  : {ROOT / 'data'}",
        f"  Log dir   : {ROOT / 'logs'}",
    ]
    if bot_err:
        rows.append(f"  !! Bot startup warning: {bot_err}")
    rows += [
        line,
        "  Keep this window open. Press Ctrl+C to stop.",
        line,
    ]
    return "\n".join(rows)


def write_credential_file(username: str, password: str, port: int) -> None:
    """把访问信息写到 data/console.txt，方便脚本读取。"""
    try:
        p = ROOT / "data" / "console.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        lan = local_ip()
        want_wan = os.environ.get("HUMANBOT_SKIP_IP", "") != "1"
        wan = public_ip() if want_wan else None
        urls = [f"http://127.0.0.1:{port}", f"http://{lan}:{port}"]
        if wan:
            urls.append(f"http://{wan}:{port}")
        body = [
            "HumanBot Console",
            "=" * 40,
            f"username: {username}",
            f"password: {password}",
            f"port    : {port}",
            "",
            "access urls:",
        ] + [f"  {u}" for u in urls] + [""]
        p.write_text("\n".join(body), encoding="utf-8")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    except Exception as exc:  # noqa: BLE001
        log.debug("写凭证文件失败: %s", exc)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HumanBot launcher")
    ap.add_argument("--port", type=int, default=None,
                    help="console port (0 or omit = auto detect)")
    ap.add_argument("--host", type=str, default=None,
                    help="console bind host (default 0.0.0.0)")
    ap.add_argument("--no-bots", action="store_true",
                    help="start console only, do not log in bots")
    ap.add_argument("--reset-password", action="store_true",
                    help="regenerate console password")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    settings = Settings.load()
    setup_logging(level=settings.log_level, bus=BUS)

    db = Database(settings.db_path)
    db.init_schema()

    if args.port is not None:
        settings.console.port = args.port
    if args.host:
        settings.console.host = args.host
    if args.reset_password:
        db.set_setting("console_password_hash", "")
        db.set_setting("console_password_plain", "")

    # provider 配置：DB 是唯一真相来源
    load_providers_from_db(db, settings)

    username = db.get_setting("console_user") or settings.console.username
    password, new_pwd = ensure_password(settings, db)

    # 端口：0 表示自动探测
    if not settings.console.port:
        port = find_free_port(settings.console.port_scan_from,
                              settings.console.port_scan_to,
                              settings.console.host)
    else:
        port = settings.console.port

    # BOT 线程
    bot_thread: Optional[BotThread] = None
    bot_err: Optional[str] = None
    n_bots = 0
    accounts = settings.boxim.account_list()
    if not args.no_bots and accounts:
        bot_thread = BotThread(settings, db)
        bot_thread.start()
        if bot_thread._error:
            bot_err = f"{type(bot_thread._error).__name__}: {bot_thread._error}"
        elif bot_thread.manager:
            n_bots = len(bot_thread.manager.bots())
    elif not args.no_bots:
        bot_err = "config.yaml 里没填 BoxIM 账号，只启动了控制台"

    # 控制台
    app = build_app(settings, db, bot_thread.manager if bot_thread else None,
                    port=port)

    write_credential_file(username, password, port)
    print(banner(settings, port, username, password, new_pwd, n_bots, bot_err),
          flush=True)

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(
        app, host=settings.console.host, port=port,
        log_level="warning", access_log=False,
    ))

    def _sigterm(signum, frame):  # noqa: ANN001, ARG001
        print("\n正在关闭 ...", flush=True)
        server.should_exit = True

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sigterm)
        except (ValueError, OSError):
            pass

    try:
        server.run()
    finally:
        if bot_thread:
            bot_thread.stop()
        BUS.publish("system", {"event": "shutdown"})
        time.sleep(0.2)
    print("已停止。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
