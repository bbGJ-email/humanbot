"""连接层：BoxIM 封装（策划案 §6.5.1 / §10）。

- 事件适配：把 SDK 的 WebSocket 事件转成统一的 InboundMessage
- 断线补偿：持久化每个会话游标，重连后按 min_id 拉离线消息
- 睡着时真正断开（stop_listening），醒来重连并补拉
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..settings import BoxIMConfig, Settings
from ..core.logging_setup import BUS, get_logger
from ..core.storage import Database
from ..core.utils import (CHAT_MESSAGE_TYPES, MSG_TYPE_IMAGE, MSG_TYPE_LOADING,
                          MSG_TYPE_ONLINE_STATUS, MSG_TYPE_STICKER, MSG_TYPE_TEXT,
                          http_ts_to_sec, now, parse_message)

log = get_logger("conn")

# 消息类型统一从 core.utils 引入（那里对齐了 SDK 的完整枚举），
# 这里只保留转发，避免两处定义漂移。


def _to_int(v: Any) -> int:
    """把消息 id / seq 安全转成 int（服务端偶尔给空串或非数字）。"""
    try:
        return int(v or 0)
    except (ValueError, TypeError):
        return 0


@dataclass
class InboundMessage:
    """标准化的入站消息。"""

    bot_id: int
    sender_id: int
    is_group: bool
    text: str
    msg_type: int
    chat_id: int                # 私聊=对方 uid，群聊=group_id
    msg_id: str = ""
    local_id: str = ""
    seq_no: int = 0
    ts: int = 0
    at_user_ids: List[int] = field(default_factory=list)
    quote_id: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def chat_type(self) -> str:
        return "group" if self.is_group else "private"

    @property
    def at_me(self) -> bool:
        return self.bot_id in (self.at_user_ids or [])


class BoxIMConnection:
    """单个 BOT 的连接管理。"""

    def __init__(self, bot_cfg: BoxIMConfig, settings: Settings,
                 db: Database, on_message: Callable[[InboundMessage], None],
                 on_event: Optional[Callable[[str, Any], None]] = None,
                 on_rtc: Optional[Callable[[int, Dict[str, Any]], None]] = None
                 ) -> None:
        self.cfg = bot_cfg
        self.settings = settings
        self.db = db
        self.on_message = on_message
        self.on_event = on_event or (lambda *a, **k: None)
        # 通话信令钩子（来电/挂断等）。RTC 消息不在聊天白名单里，
        # 如果不单独挑出来就会被当噪音丢掉，BOT 永远收不到来电。
        self.on_rtc = on_rtc

        self.im = None                # boxim 实例
        self.bot_id: int = 0
        self.nickname: str = ""
        self.connected: bool = False
        self.listening: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reconnect_count = 0
        self._last_online: Dict[int, int] = {}

    # -- 登录 -------------------------------------------------------------

    def login(self) -> bool:
        """同步登录，返回是否成功。"""
        try:
            import boxim
        except ImportError:
            log.error("未安装 boxim-sdk，请执行 pip install boxim-sdk")
            return False

        try:
            self.im = boxim.BoxIM(base_url=self.cfg.base_url)
            self.im.login(self.cfg.username, self.cfg.password)
            self._resolve_identity()
            self.connected = True
            log.info("BOT 登录成功: %s (id=%s)", self.nickname, self.bot_id)
            BUS.publish("conn", {"event": "login", "bot_id": self.bot_id,
                                 "nickname": self.nickname})
            return True
        except Exception as exc:
            log.error("BOT 登录失败: %s", exc)
            BUS.publish("conn", {"event": "login_failed", "error": str(exc)})
            return False

    def _resolve_identity(self) -> None:
        """从 SDK 多种可能的属性上取当前账号身份。

        BoxIM 各版本暴露位置不一致：可能是 `im.user`、`im.get_me()`，
        也可能直接挂在 `im` 上（uid / nickName）。这里按优先级兜底，
        任何一个失败都不影响登录本身成功。
        """
        if self.im is None:
            return

        cands: List[Any] = []
        obj = getattr(self.im, "user", None)
        if obj is not None:
            cands.append(obj)
        getme = getattr(self.im, "get_me", None)
        if callable(getme):
            try:
                got = getme()
                if got:
                    cands.append(got)
            except Exception as exc:  # noqa: BLE001
                log.debug("get_me() 不可用: %s", exc)
        cands.append(self.im)

        for src in cands:
            uid = self._pick(src, ("uid", "id", "userId", "user_id"))
            if uid:
                try:
                    self.bot_id = int(uid)
                except (TypeError, ValueError):
                    continue
                nick = self._pick(src, ("nickName", "nickname", "name"))
                self.nickname = str(nick or self.cfg.username)
                sex = self._pick(src, ("sex", "gender"))
                if sex is not None:
                    try:
                        self.sex = int(sex)
                    except (TypeError, ValueError):
                        pass
                return

        log.warning("无法确定账号 id，将退化为用户名标识")
        self.nickname = self.cfg.username

    @staticmethod
    def _pick(obj: Any, names: Any) -> Any:
        """从对象或字典里按候选键取第一个非空值。"""
        for n in names:
            v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
            if v not in (None, "", 0):
                return v
        return None

    # -- 事件注册 ---------------------------------------------------------

    def register_handlers(self) -> None:
        """注册 SDK 事件回调。"""
        if self.im is None:
            return
        ws = getattr(self.im, "_ws", None)
        if ws is None:
            log.warning("无法获取 WebSocket 传输层，事件注册跳过")
            return

        def _pm(msg: Dict[str, Any]) -> None:
            self._dispatch(msg, is_group=False)

        def _gm(msg: Dict[str, Any]) -> None:
            self._dispatch(msg, is_group=True)

        def _online(msg: Dict[str, Any]) -> None:
            try:
                uid = int(msg.get("sendId") or msg.get("userId") or 0)
                if uid:
                    self._last_online[uid] = now()
                    self.on_event("online_status", {"user_id": uid, "raw": msg})
            except Exception:
                pass

        def _force_offline(msg: Dict[str, Any]) -> None:
            log.warning("被强制下线")
            self.connected = False
            BUS.publish("conn", {"event": "force_offline", "bot_id": self.bot_id})
            self.on_event("force_offline", msg)

        ws.on("private_message", _pm)
        ws.on("group_message", _gm)
        ws.on("online_status", _online)
        ws.on("force_offline", _force_offline)
        log.info("事件监听已注册")

    def _dispatch(self, msg: Dict[str, Any], is_group: bool) -> None:
        """把 SDK 消息转成 InboundMessage 并交给上层。"""
        try:
            p = parse_message(msg)
            # RTC 通话信令（100~119 单人 / 200~229 群组）不在聊天白名单里，
            # 必须先挑出来交给通话层，否则来电会被当噪音丢掉。
            if self.on_rtc and (100 <= p["type"] < 120 or 200 <= p["type"] < 230):
                try:
                    self.on_rtc(p["type"], msg)
                except Exception as exc:
                    log.debug("RTC 事件处理异常: %s", exc)
                return
            # 只处理"人能说的话"。回执/已读/撤回/系统提示/好友群事件
            # 一律丢弃——之前不过滤，BOT 会去回复 {"id":19032170} 这种系统回执。
            if p["type"] not in CHAT_MESSAGE_TYPES:
                log.debug("忽略非聊天消息 type=%s", p["type"])
                return
            # 自己发的忽略
            if p["sender_id"] == self.bot_id:
                return
            chat_id = p["group_id"] if is_group else p["sender_id"]
            if not chat_id:
                return
            inbound = InboundMessage(
                bot_id=self.bot_id,
                sender_id=p["sender_id"],
                is_group=is_group,
                text=str(p["content"] or ""),
                msg_type=p["type"],
                chat_id=chat_id,
                msg_id=p["msg_id"],
                local_id=p["local_id"],
                seq_no=p["seq_no"],
                ts=http_ts_to_sec(p["send_time"]),
                at_user_ids=[int(x) for x in (p["at_user_ids"] or [])
                             if str(x).isdigit()],
                quote_id=p["quote_id"],
                raw=msg,
            )
            BUS.publish("inbound", {
                "bot_id": self.bot_id, "sender": inbound.sender_id,
                "group": inbound.is_group, "text": inbound.text[:120],
                "type": inbound.msg_type,
            })
            self.on_message(inbound)
        except Exception as exc:
            log.error("消息分发异常: %s", exc)

    # -- 监听循环 ---------------------------------------------------------

    def start_listening(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """在指定事件循环中启动监听（非阻塞）。"""
        if self.im is None or self.listening:
            return
        self._loop = loop or asyncio.get_event_loop()
        try:
            asyncio.run_coroutine_threadsafe(self._start_async(), self._loop)
            self.listening = True
            log.info("开始监听")
        except Exception as exc:
            log.error("启动监听失败: %s", exc)

    async def _start_async(self) -> None:
        try:
            await self.im.start_listening()
        except Exception as exc:
            log.error("异步监听启动失败: %s", exc)

    def stop_listening(self) -> None:
        """停止监听（睡着时调用，真正断开连接）。"""
        if self.im is None or not self.listening:
            return
        try:
            if self._loop and self._loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(self.im.stop_listening(),
                                                       self._loop)
                fut.result(timeout=15)
            else:
                self.im.stop_listening()
        except Exception as exc:
            log.warning("停止监听异常: %s", exc)
        self.listening = False
        self.connected = False
        log.info("已断开连接（休眠）")

    def reconnect(self) -> bool:
        """重连并补拉离线消息。"""
        try:
            if self.im is None:
                return self.login()
            self.im.refresh_token()
            self.start_listening(self._loop)
            self.connected = True
            self._reconnect_count += 1
            log.info("已重连（第 %d 次）", self._reconnect_count)
            return True
        except Exception as exc:
            log.error("重连失败: %s", exc)
            return False

    # -- 离线消息补偿 -----------------------------------------------------

    def sync_offline(self) -> int:
        """拉取离线消息并入库。返回新增条数。"""
        if self.im is None:
            return 0
        total = 0
        for conv in self.db.conversations_of(self.bot_id):
            cid = int(conv["id"])
            peer = int(conv["peer_id"])
            last_id = int(conv["last_msg_id"] or 0)
            try:
                if conv["chat_type"] == "private":
                    msgs = self.im.load_private_offline_message(last_id) or []
                else:
                    msgs = self.im.load_group_offline_message(last_id) or []
            except Exception as exc:
                log.warning("拉取离线消息失败 conv=%s: %s", cid, exc)
                continue

            for m in msgs:
                p = parse_message(m)
                sender = p["sender_id"]
                role = "bot" if sender == self.bot_id else "user"
                inserted = self.db.insert_message(
                    self.bot_id, cid, role, str(p["content"] or ""),
                    msg_id=p["msg_id"], local_id=p["local_id"],
                    seq_no=p["seq_no"], msg_type=p["type"],
                    ts=http_ts_to_sec(p["send_time"]),
                    at_user_ids=p["at_user_ids"], quote_id=p["quote_id"],
                )
                if inserted:
                    total += 1
                    # 离线消息也要走回复流程
                    if role == "user" and not self._is_self(p["content"]):
                        chat_id = p["group_id"] if conv["chat_type"] == "group" else sender
                        self.on_message(InboundMessage(
                            bot_id=self.bot_id, sender_id=sender,
                            is_group=conv["chat_type"] == "group",
                            text=str(p["content"] or ""), msg_type=p["type"],
                            chat_id=chat_id, msg_id=p["msg_id"],
                            local_id=p["local_id"], seq_no=p["seq_no"],
                            ts=http_ts_to_sec(p["send_time"]),
                            at_user_ids=p["at_user_ids"],
                            quote_id=p["quote_id"], raw=m,
                        ))
            # 更新游标：必须推进到「消息 ID」（loadOfflineMessage 的 minId 语义），
            # 之前误用 seq_no，导致 last_msg_id 永远停在 0、每次重启重拉全部历史。
            if msgs:
                parsed = [parse_message(x) for x in msgs]
                max_id = max((_to_int(p["msg_id"]) for p in parsed), default=0)
                max_seq = max((_to_int(p["seq_no"]) for p in parsed), default=0)
                self.db.save_cursor(cid, max_id, max_seq)
        if total:
            log.info("补拉离线消息 %d 条", total)
        return total

    @staticmethod
    def _is_self(content: str) -> bool:
        return False

    # -- 发送 -------------------------------------------------------------

    @staticmethod
    def _msg_id_of(res: Any) -> int:
        """从发送接口的返回里抠出真实消息 id。

        接口返回 {"code":200,"data":{"id":19032170,...}}，
        撤回时必须用这个 id；以前传 0 所以一直报"消息不存在"。
        """
        if not isinstance(res, dict):
            return 0
        for src in (res.get("data"), res):
            if isinstance(src, dict):
                for k in ("id", "messageId", "msgId"):
                    v = src.get(k)
                    if v:
                        try:
                            return int(v)
                        except (TypeError, ValueError):
                            continue
        return 0

    def send_text(self, chat_id: int, text: str, is_group: bool = False,
                  at_users: Optional[List[int]] = None,
                  quote_id: int = 0) -> Optional[int]:
        """同步发送文本。成功返回消息 id，失败返回 None。

        返回值仍是"真值即成功"，所以老的 `if ok:` 判断照常工作，
        但调用方现在能拿到 id 用于撤回。
        """
        if self.im is None or not text:
            return None
        try:
            if is_group:
                # 群聊要拿 id 就得走 raw（send_group_text 不返回响应）
                res = self.im.send_group_raw(
                    chat_id, text, MSG_TYPE_TEXT,
                    at_users=at_users or None)
            elif quote_id:
                res = self.im.send_private_raw(chat_id, text, MSG_TYPE_TEXT,
                                               quote_message_id=quote_id)
            else:
                res = self.im.send_private_raw(chat_id, text, MSG_TYPE_TEXT)
            mid = self._msg_id_of(res)
            return mid or 1        # 拿不到 id 也视为发送成功（真值）
        except Exception as exc:
            log.error("发送失败: %s", exc)
            return None

    def send_image(self, chat_id: int, path: str, is_group: bool = False) -> bool:
        if self.im is None or not path:
            return False
        try:
            if is_group:
                self.im.send_group_image(chat_id, path)
            else:
                self.im.send_image(chat_id, path)
            return True
        except Exception as exc:
            log.error("发送图片失败: %s", exc)
            return False

    def _sticker_content(self, sticker: Any) -> Optional[str]:
        """构造表情消息的 content。

        踩坑记录：SDK 的 MessageBuilder.sticker() 生成的是 {"stickerId": N}，
        但盒子IM 的 Web 客户端渲染表情时读的是 **content.imageUrl**
        （见 ChatView 的 STICKER 分支：<img :src="content.imageUrl">），
        所以用 SDK 那个格式发出去，对方只看到"[动画表情] undefined"。

        实际客户端发送时用的是整个表情对象：
            {id, albumId, name, imageUrl, thumbUrl, width, height}
        这里按客户端的行为来拼。
        """
        if isinstance(sticker, dict) and sticker.get("image_url"):
            obj = {
                "id": int(sticker.get("id") or 0),
                "albumId": int(sticker.get("album_id") or 0),
                "name": str(sticker.get("name") or ""),
                "imageUrl": str(sticker.get("image_url") or ""),
                "thumbUrl": str(sticker.get("thumb_url") or sticker.get("image_url") or ""),
                "width": int(sticker.get("width") or 0),
                "height": int(sticker.get("height") or 0),
            }
            return json.dumps(obj, ensure_ascii=False)

        # 传的是纯 id → 查本地表情库补全字段
        try:
            sid = int(sticker)
        except (TypeError, ValueError):
            return None
        if not sid or self.db is None:
            return None
        row = self.db.query_one(
            "SELECT sticker_id,album_id,album_name,name,image_url,thumb_url,"
            "width,height FROM sticker_catalog WHERE sticker_id=?", (sid,))
        if not row:
            log.warning("表情库中没有 id=%s，无法发送", sid)
            return None
        obj = {
            "id": int(row["sticker_id"]),
            "albumId": int(row["album_id"] or 0),
            "name": row["name"] or "",
            "imageUrl": row["image_url"] or "",
            "thumbUrl": row["thumb_url"] or row["image_url"] or "",
            "width": int(row["width"] or 0),
            "height": int(row["height"] or 0),
        }
        return json.dumps(obj, ensure_ascii=False)

    def send_sticker(self, chat_id: int, sticker: Any,
                     is_group: bool = False) -> Optional[int]:
        """发送表情。sticker 可以是完整表情 dict，也可以是 sticker_id。

        成功返回消息 id，失败返回 None。
        """
        if self.im is None:
            return None
        content = self._sticker_content(sticker)
        if not content:
            return None
        try:
            if is_group:
                res = self.im.send_group_raw(chat_id, content, MSG_TYPE_STICKER)
            else:
                res = self.im.send_private_raw(chat_id, content, MSG_TYPE_STICKER)
            mid = self._msg_id_of(res)
            log.info("表情已发送 id=%s (msg_id=%s)",
                     (sticker.get("name") if isinstance(sticker, dict)
                      else sticker), mid or "?")
            return mid or 1
        except Exception as exc:
            log.error("发送表情失败: %s", exc)
            return None

    def recall(self, msg_id: int, is_group: bool = False) -> bool:
        if self.im is None:
            return False
        try:
            if is_group:
                self.im.recall_group_message(msg_id)
            else:
                self.im.recall_private_message(msg_id)
            return True
        except Exception as exc:
            log.warning("撤回失败: %s", exc)
            return False

    def mark_read(self, peer_id: int, is_group: bool = False) -> bool:
        if self.im is None:
            return False
        try:
            if is_group:
                self.im.mark_group_read(peer_id)
            else:
                self.im.mark_private_read(peer_id)
            return True
        except Exception as exc:
            log.warning("标记已读失败: %s", exc)
            return False

    # -- 账号行为 ---------------------------------------------------------

    def update_profile(self, **kw: Any) -> bool:
        """更新个人资料。注意 SDK 是 get_me() + update + PUT 整体提交。"""
        if self.im is None:
            return False
        try:
            self.im.update_profile(**kw)
            log.info("资料已更新: %s", list(kw.keys()))
            return True
        except Exception as exc:
            log.error("更新资料失败: %s", exc)
            return False

    def set_signature(self, text: str) -> bool:
        return self.update_profile(signature=text)

    def set_head_image(self, url_or_path: str) -> bool:
        """换头像。支持 http 链接和本地文件。

        坑：SDK 的上传器挂在 `im.uploader` 上（不是 `im`），
        而且 UploadResult 把图片地址放在 `origin_url` / `thumb_url`，
        `url` 字段是空的——早期版本这里读 `url` 导致上传成功但换不上头像。
        """
        if not url_or_path:
            return False
        try:
            if url_or_path.startswith("http"):
                return self.update_profile(headImage=url_or_path)

            up = getattr(self.im, "uploader", None)
            fn = getattr(up, "upload_image", None) if up is not None else None
            if not callable(fn):
                # 兼容没有 uploader 的旧版 SDK
                fn = getattr(self.im, "upload_image", None)
            if not callable(fn):
                log.warning("当前 SDK 不支持图片上传，换头像跳过")
                return False

            res = fn(url_or_path)
            url = ""
            if res is not None:
                for attr in ("origin_url", "url", "thumb_url"):
                    v = getattr(res, attr, None)
                    if not v and isinstance(res, dict):
                        v = res.get(attr) or res.get(attr.replace("_", ""))
                    if v:
                        url = str(v)
                        break
            if not url:
                log.warning("上传成功但拿不到图片地址，换头像跳过")
                return False
            return self.update_profile(headImage=url)
        except Exception as exc:
            log.error("换头像失败: %s", exc)
            return False

    def get_online_friends(self) -> List[Dict[str, Any]]:
        if self.im is None:
            return []
        try:
            return self.im.get_online_friends() or []
        except Exception:
            return []

    # -- 其他 SDK 能力的直通 ---------------------------------------------

    def raw(self) -> Any:
        return self.im
