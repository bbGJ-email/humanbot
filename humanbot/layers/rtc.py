"""WebRTC 桥接层：aiortc ↔ BoxIM SDK。

BoxIM SDK 只做**信令**和**媒体流抽象**（`RTCCallSession` + `AudioStream` 协议），
不含 WebRTC 实现——它的依赖里没有 aiortc，源码里也没有 `RTCPeerConnection`。
真正的 SDP 协商、ICE 连通、音频编解码要在这一层补上。

分工：

    RTCCallSession   信令收发、状态机、心跳、设备状态
    RTCBridge        SDP 生成、ICE 交换、音频轨道收发   ← 本模块
    VoiceService     PCM ↔ 文字

音频全程 **16kHz / 单声道 / 16bit PCM**，与 SDK 协议一致，不做二次重采样。

SDP 流程（被叫 = BOT 接听）：

    收到 RTC_SETUP_VOICE(100)
        → session.accept()                  先接受（此时还没有 offer）
        → 等 'remote_sdp' 事件（主叫的 offer）
        → pc.setRemoteDescription(offer)
        → answer = pc.createAnswer()
        → session.send_answer(answer)
        → 双向往来 candidate
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

from ..core.logging_setup import BUS, get_logger

log = get_logger("rtc")

SAMPLE_RATE = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000      # 320

# 拉不到官方配置时的兜底
_FALLBACK_ICE = [{"urls": "stun:stun.l.google.com:19302"}]

# ---------------------------------------------------------------------------
# 公网 candidate 注入
#
# 谜团：双方 candidate 齐全（含 TURN relay），ICE 检查却全部失败。
# 排查结论：云安全组只阻断了 UDP 3478（STUN/TURN 控制端口），其余 UDP
# 端口畅通（NTP 123 实测通）。而浏览器的 NAT 对「探测 BOT relay」的源
# 端口映射与 srflx 端口不一致（symmetric NAT），TURN 的 channel bind
# 永远盖不住它，双向探测互相丢失。
#
# 解法：云服务器 EIP 是 1:1 全端口 NAT（公网任意端口 → 内网同端口）。
# 把 host candidate 复制一份换成公网 IP 注入 answer，浏览器即可直连
# BOT 的 UDP 端口，完全绕开 STUN/TURN。
# ---------------------------------------------------------------------------

_PUB_IP_CACHE: Optional[str] = None
_HOST_CAND_RE = re.compile(
    r"a=candidate:(\S+) 1 udp (\d+) (\d+\.\d+\.\d+\.\d+) (\d+) typ host")


def _public_ip() -> Optional[str]:
    """本机公网出口 IP（缓存一次；HTTP 查询，多服务兜底）。"""
    global _PUB_IP_CACHE
    if _PUB_IP_CACHE is not None:
        return _PUB_IP_CACHE or None
    ip = ""
    if os.environ.get("HUMANBOT_SKIP_IP", "") != "1":
        try:
            import requests
            for url in ("https://api.ipify.org", "https://ifconfig.me/ip",
                        "https://api.myip.com"):
                try:
                    r = requests.get(url, timeout=4)
                    txt = (r.text or "").strip() if r.status_code == 200 else ""
                    if re.match(r"^\d+\.\d+\.\d+\.\d+$", txt):
                        ip = txt
                        break
                except Exception:
                    continue
        except Exception:
            ip = ""
    _PUB_IP_CACHE = ip
    return ip or None


def _with_public_candidates(sdp: str) -> str:
    """给 SDP 里「云内网主网卡(10.x)」的 host candidate 追加公网映射版本。"""
    pub = _public_ip()
    if not pub or not sdp:
        return sdp
    out: List[str] = []
    added = 0
    for line in sdp.splitlines():
        out.append(line)
        m = _HOST_CAND_RE.match(line.strip())
        if m and m.group(3).startswith("10.") and m.group(3) != pub:
            # EIP 1:1 NAT：公网同端口直达内网。srflx 语义（raddr 指向内网），
            # priority 略低于原 host，让直连排在 relay 前面。
            out.append(
                "a=candidate:%spub 1 udp %d %s %s typ srflx raddr %s rport %s "
                "generation 0" % (m.group(1), max(1, int(m.group(2)) - 1),
                                  pub, m.group(4), m.group(3), m.group(4)))
            added += 1
    if added:
        log.info("已注入 %d 条公网 candidate（%s，浏览器可直连）", added, pub)
    return "\r\n".join(out) + "\r\n"


def _relay_only(sdp: str) -> str:
    """只保留 TURN relay candidate，剔除内网 host 与（不可达的）srflx。

    这台服务器的入站 UDP 被云安全组【全端口阻断】（外部发 UDP 到任意端口
    都收不到），出站 UDP 也只剩 DNS/NTP 等少数端口，STUN/TURN 的 3478 出不去。
    于是：
      - 内网 host candidate（10.x / docker 172.x）公网对端根本连不到
      - srflx 要么拿不到、要么是伪造的，同样连不到

    把它们留在 answer 里只会稀释对端的 ICE 检查、把对端引向死路。
    跨机对照实测：answer 只给 relay → 1 秒连通；给全量 → 180 个配对全失败。

    若整份 SDP 里没有任何 relay，则原样返回（没有 TURN 的环境不能自断退路）。
    """
    if not sdp:
        return sdp
    lines = sdp.splitlines()
    if not any(l.startswith("a=candidate:") and " typ relay" in l for l in lines):
        return sdp
    keep = []
    for l in lines:
        if l.startswith("a=candidate:") and " typ relay" not in l:
            continue
        # c= 行在 WebRTC 里必须是 0.0.0.0/::（连接地址一律由 ICE 候选决定）。
        # aiortc 会填真实网卡地址（如内网 10.x.x.x），部分客户端会把它当成
        # 「默认候选」去连内网地址，反而忽略了 relay 候选。
        if l.startswith("c=IN IP4 ") and not l.strip().endswith("0.0.0.0"):
            l = "c=IN IP4 0.0.0.0"
        elif l.startswith("c=IN IP6 ") and not l.strip().endswith("::"):
            l = "c=IN IP6 ::"
        keep.append(l)
    log.info("answer 收敛为 relay-only（剔除 %d 条不可达 candidate）",
             len(lines) - len(keep))
    return "\r\n".join(keep) + "\r\n"


def _pcm_to_frame(data: bytes, rate: int = SAMPLE_RATE):
    """16bit 单声道 PCM → aiortc AudioFrame。"""
    import av
    import numpy as np

    arr = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")
    frame.sample_rate = rate
    frame.time_base = Fraction(1, rate)
    return frame


def candidate_to_payload(cand: Any) -> Optional[str]:
    """aiortc 的 ICE candidate → 发给 BoxIM 的 JSON 字符串。

    aiortc 1.15 的 candidate 是字段化对象（没有 .candidate），
    要用官方工具 candidate_to_sdp 还原成 SDP 行。
    """
    if cand is None:
        return None
    try:
        from aiortc.sdp import candidate_to_sdp

        line = candidate_to_sdp(cand)
        if not line:
            return None
        if not line.startswith("candidate:"):
            line = "candidate:" + line
        idx = getattr(cand, "sdpMLineIndex", None)
        return json.dumps({
            "candidate": line,
            "sdpMid": getattr(cand, "sdpMid", None) or "0",
            "sdpMLineIndex": 0 if idx is None else int(idx),
        }, ensure_ascii=False)
    except Exception as exc:
        log.debug("candidate 序列化失败: %s", exc)
        return None


def payload_to_candidate(raw: str):
    """BoxIM 发来的 candidate → aiortc RTCIceCandidate。

    aiortc 1.15 的 RTCIceCandidate 要求**字段化**的对象（component /
    foundation / ip / port ...），不能直接塞一个 candidate 字符串，
    所以用官方工具 candidate_from_sdp 解析。

    SDK 只把 `candidate` 字段透传给我们（sdpMid/sdpMLineIndex 在解析时
    被丢掉了）。语音通话只有一条 m-line，用 0/0 补上——这也是浏览器的默认值。
    """
    try:
        from aiortc.sdp import candidate_from_sdp
    except ImportError:
        return None

    text, mid, idx = raw, "0", 0
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            text = parsed.get("candidate") or ""
            mid = parsed.get("sdpMid") or "0"
            idx = parsed.get("sdpMLineIndex")
            if idx is None:
                idx = 0
    except (ValueError, TypeError):
        pass

    if not text:
        return None
    line = text[len("candidate:"):] if text.startswith("candidate:") else text
    try:
        cand = candidate_from_sdp(line)
    except Exception as exc:
        log.debug("candidate 解析失败（%s）: %s", str(exc)[:60], line[:60])
        return None
    cand.sdpMid = mid
    try:
        cand.sdpMLineIndex = int(idx)
    except (TypeError, ValueError):
        cand.sdpMLineIndex = 0
    return cand


try:
    from aiortc import MediaStreamTrack as _MediaStreamTrack
except ImportError:                     # 未装 aiortc 时不影响模块导入
    class _MediaStreamTrack:            # type: ignore[no-redef]
        """占位：真正用到 WebRTC 时会由 open() 报错提示。"""


class OutboundAudioTrack(_MediaStreamTrack):
    """自定义发送轨道：把队列里的 PCM 变成音频帧发出去。

    不能用 SDK 的 AudioStream——那是给 SDK 自己的流处理用的，
    这里要直接对接 aiortc 的 MediaStreamTrack 协议。
    """

    kind = "audio"

    def __init__(self, rate: int = SAMPLE_RATE) -> None:
        super().__init__()
        self._rate = rate
        self._queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self._buf = bytearray()
        self._pts = 0

    async def recv(self):
        need = SAMPLES_PER_FRAME * 2
        # 尽量攒够一帧；不够就补静音，保持发送节奏稳定（否则会一卡一卡）
        deadline = time.time() + FRAME_MS / 1000 * 2.5
        while len(self._buf) < need and time.time() < deadline:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.01)
                self._buf.extend(chunk)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
        data = bytes(self._buf[:need])
        del self._buf[:need]
        if len(data) < need:
            data += b"\x00" * (need - len(data))

        frame = _pcm_to_frame(data, self._rate)
        frame.pts = self._pts
        self._pts += SAMPLES_PER_FRAME
        return frame

    def push(self, pcm: bytes) -> None:
        if pcm:
            self._queue.put_nowait(pcm)

    def clear(self) -> None:
        """清空待播音频（被打断时用）。"""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buf.clear()

    @property
    def pending_ms(self) -> int:
        """还有多少音频没播（打断判断用）。"""
        n = len(self._buf) + sum(len(c) for c in list(self._queue._queue))
        return int(n / (self._rate * 2) * 1000)


class RTCBridge:
    """把 aiortc 的 PeerConnection 和 BoxIM 的 RTCCallSession 接起来。"""

    def __init__(self, conn: Any, settings: Any) -> None:
        self.conn = conn
        self.settings = settings
        self.pc: Any = None
        self.session: Any = None
        self.peer_id: int = 0
        self.ice_servers: List[Dict[str, Any]] = []

        self._out: Optional[OutboundAudioTrack] = None
        self._in_track: Any = None
        self._in_queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=500)
        self._pump_task: Optional[asyncio.Task] = None
        self._closed = False
        self._connected = asyncio.Event()
        self._remote_sdp: Optional[str] = None

    # -- ICE --------------------------------------------------------------

    async def fetch_ice_servers(self) -> List[Dict[str, Any]]:
        """从 /system/config 拉 ICE 配置（官方提供 STUN + TURN）。"""
        try:
            cfg = await asyncio.to_thread(self.conn.im.get_system_config)
            webrtc = getattr(cfg, "webrtc", None)
            if isinstance(webrtc, dict):
                servers = webrtc.get("iceServers") or []
            else:
                servers = getattr(webrtc, "ice_servers", None) or [] if webrtc else []
            norm = []
            for s in servers:
                if isinstance(s, dict):
                    urls = s.get("urls")
                    if urls:
                        item = dict(s)
                        # 出站 UDP 3478 被云安全组阻断（DNS 53 通、STUN/TURN
                        # UDP 不通），只有内网 host candidate，对端连不上。
                        # TURN 改走 TCP 中继绕过 UDP 限制（实测能拿到 relay）。
                        if (isinstance(urls, str) and urls.startswith("turn:")
                                and "transport=" not in urls):
                            item["urls"] = urls + "?transport=tcp"
                        norm.append(item)
                else:
                    urls = getattr(s, "urls", None)
                    if urls:
                        if (isinstance(urls, str) and urls.startswith("turn:")
                                and "transport=" not in urls):
                            urls = urls + "?transport=tcp"
                        norm.append({
                            "urls": urls,
                            "username": getattr(s, "username", None),
                            "credential": getattr(s, "credential", None),
                        })
            if norm:
                self.ice_servers = norm
                log.info("ICE 配置已获取：%s", [n.get("urls") for n in norm])
                return norm
        except Exception as exc:
            log.warning("拉取 ICE 配置失败（用兜底 STUN）: %s", exc)
        self.ice_servers = list(_FALLBACK_ICE)
        return self.ice_servers

    # -- PeerConnection ---------------------------------------------------

    async def open(self, peer_id: int, session: Any) -> None:
        """建立 PeerConnection 并挂上音频轨道。"""
        from aiortc import (RTCConfiguration, RTCIceServer,
                            RTCPeerConnection)

        self.peer_id = peer_id
        self.session = session
        self._closed = False
        self._connected.clear()

        if not self.ice_servers:
            await self.fetch_ice_servers()

        servers = []
        for s in self.ice_servers:
            servers.append(RTCIceServer(
                urls=s.get("urls"),
                username=s.get("username") or None,
                credential=s.get("credential") or None,
            ))
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))

        # 打开 aioice 的 DEBUG 日志：ICE 配对/探测的每一步都会记到
        # logs/ice_debug.log，排查「双方 candidate 齐了却连不上」的关键证据。
        # 注意：aioice 在 __init__ 里挂了 NullHandler 会吞掉所有日志，须移除。
        try:
            import logging as _lg
            _ice = _lg.getLogger("aioice")
            for _h in list(_ice.handlers):
                if isinstance(_h, _lg.NullHandler):
                    _ice.removeHandler(_h)
            _ice.setLevel(_lg.DEBUG)
            if not _ice.handlers:
                _h = _lg.FileHandler("logs/ice_debug.log")
                _h.setFormatter(_lg.Formatter(
                    "%(asctime)s %(levelname)s %(name)s %(message)s"))
                _ice.addHandler(_h)
            _ice.propagate = False
        except Exception:
            pass

        self._out = OutboundAudioTrack()
        self.pc.addTrack(self._out)

        @self.pc.on("track")
        def _on_track(track):        # noqa: ANN001
            log.info("收到远端轨道: %s", track.kind)
            if track.kind == "audio" and self._pump_task is None:
                self._in_track = track
                self._pump_task = asyncio.ensure_future(self._pump_inbound())

        @self.pc.on("icecandidate")
        async def _on_ice(candidate):    # noqa: ANN001
            payload = candidate_to_payload(candidate)
            if payload:
                log.info("本地 ICE candidate: %s", payload[:160])
            if payload and self.session is not None:
                try:
                    await self.session.send_candidate(payload)
                    log.info("本地 candidate 已通过信令发送")
                except Exception as exc:
                    log.warning("发送 candidate 失败: %s", exc)

        @self.pc.on("connectionstatechange")
        async def _on_state():           # noqa: ANN001
            st = self.pc.connectionState if self.pc else "?"
            log.info("WebRTC 连接状态: %s", st)
            BUS.publish("call", {"event": "rtc_state", "peer_id": self.peer_id,
                                 "state": st})
            if st == "connected":
                self._connected.set()
            elif st in ("failed", "closed"):
                self._connected.clear()

        # 关键：注册 SDK 会话的信令回调。不注册就收不到对端的 offer/candidate，
        # 通话会在「等不到远端 SDP」处超时挂断（历史 bug：漏了这两行）。
        if self.session is not None:
            self.session.on_remote_sdp(self.on_remote_sdp)
            self.session.on_remote_candidate(self.add_candidate)
            log.info("已注册 SDK 信令回调（remote_sdp / remote_candidate）")
            # 提前注册 SDK 的 rtc 消息处理器。SDK 只在 accept() 内部注册，
            # 但主叫可能在 accept 之前就发 offer，晚注册会丢 offer。
            try:
                self.session._register_rtc_ws_handler()
                log.info("已提前注册 rtc 消息处理器（防 offer 早于 accept 丢失）")
            except Exception as exc:
                log.debug("提前注册 rtc 处理器失败: %s", exc)

        # 双保险：直接监听 SDK ws 的 rtc_message 事件，绕过 session 的私有
        # 转发链路。session 那套在真实环境里没生效（offer 到连接层却没到桥
        # 接层），这里自己解析 offer/candidate 兜底。
        ws = getattr(self.conn.im, "_ws", None)
        if ws is not None:

            async def _direct_rtc(msg_data: Dict[str, Any],
                                  is_group: bool = False) -> None:
                mt = int(msg_data.get("type") or 0)
                log.info("RTCBridge 直接收到 rtc_message type=%s", mt)
                content = msg_data.get("content") or ""
                if mt == 107:      # RTC_OFFER
                    sdp = content
                    # 现场取证：保存服务端推来的**完整** content（不只是 SDP）。
                    # 若 content 里除 sdp 外还带 type/mode 等字段，那 BOT 回
                    # answer 时也必须带上同样的字段，否则服务端/对端可能拒收。
                    try:
                        with open("/tmp/last_offer_raw.json", "w") as f:
                            f.write(content or "")
                    except OSError:
                        pass
                    keys: List[str] = []
                    try:
                        parsed = json.loads(content) if content else {}
                        if isinstance(parsed, dict):
                            keys = sorted(parsed.keys())
                            sdp = parsed.get("sdp") or content
                    except (ValueError, TypeError, AttributeError):
                        pass
                    log.info("offer content：总长 %d，字段 %s，提取 SDP %d 字节",
                             len(content or ""), keys, len(sdp or ""))
                    # 现场取证：保存真实 offer 供离线复现调试
                    try:
                        with open("/tmp/last_offer.sdp", "w") as f:
                            f.write(sdp or "")
                    except OSError:
                        pass
                    # 诊断：打印对端 offer 的 candidate 类型分布，确认对端
                    # 是 trickle 还是非 trickle、有没有公网 candidate。
                    cand_types = []
                    for ln in (sdp or "").splitlines():
                        if "a=candidate" in ln:
                            cand_types.append(
                                ln.strip().split(" typ ")[-1].split(" ")[0])
                    log.info("对端 offer 含 %d 个 candidate: %s",
                             len(cand_types), cand_types)
                    self.on_remote_sdp(sdp)
                elif mt == 109:    # RTC_CANDIDATE
                    cand = content
                    cand_keys: List[str] = []
                    try:
                        parsed = json.loads(content) if content else {}
                        if isinstance(parsed, dict):
                            cand_keys = sorted(parsed.keys())
                        cand = parsed.get("candidate", content)
                    except (ValueError, TypeError):
                        pass
                    # 只打印第一条：确认服务端约定的候选字段名（BOT 发送时
                    # 必须用同一套字段，否则对端解析不了）。
                    if not getattr(self, "_shown_cand", False):
                        self._shown_cand = True
                        log.info("对端 candidate 字段 %s，原始内容：%s",
                                 cand_keys, repr((content or "")[:160]))
                    # 现场取证：保存对端 candidate 的完整原始 content
                    try:
                        with open("/tmp/last_candidates.txt", "a") as f:
                            f.write(str(content) + "\n")
                    except OSError:
                        pass
                    await self.add_candidate(cand)

            ws.on("rtc_message", _direct_rtc)
            self._direct_rtc = _direct_rtc   # 保引用，防止被回收
            log.info("已直接监听 ws rtc_message（双保险）")

        log.info("PeerConnection 已建立（peer=%s，ICE %d 个）",
                 peer_id, len(servers))

    async def wait_connected(self, timeout: float = 20.0) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # -- SDP --------------------------------------------------------------

    def on_remote_sdp(self, sdp: str) -> None:
        """SDK 通知收到远端 SDP。"""
        self._remote_sdp = sdp
        log.info("收到远端 SDP（%d 字节）", len(sdp or ""))

    def set_remote_sdp(self, sdp: str) -> None:
        """直接注入远端 SDP。

        来电消息(100/101)的 content 里就带 offer 时，不必再等单独的 107。
        """
        self._remote_sdp = sdp or ""
        log.info("直接采用来电消息里的 offer（%d 字节）", len(self._remote_sdp))

    async def apply_remote_sdp(self, is_caller: bool) -> Optional[str]:
        """把远端 SDP 应用到 PC。

        - 被叫：远端是 offer → 生成 answer 并返回（调用方负责 send_answer）
        - 主叫：远端是 answer → 直接应用，返回 None
        """
        from aiortc import RTCSessionDescription

        if not (self._remote_sdp and self.pc):
            log.warning("没有远端 SDP 可应用")
            return None

        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=self._remote_sdp,
                                  type="answer" if is_caller else "offer"))
        log.info("远端 SDP 已应用（%s）", "answer" if is_caller else "offer")

        if is_caller:
            return None
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        # 等待 ICE gathering 完成再返回 answer。出站 UDP 被云安全组阻断，
        # 只有 TURN(TCP) 的 relay candidate 才能让对端连通，而 relay 分配
        # 需要几秒；不等待的话 answer 里只有内网 host，对端连不上。
        await self._wait_ice_gathering(timeout=10.0)
        # 服务器入站 UDP 全端口被阻断，内网 host / 伪造 srflx 都不可达，
        # 留着只会把对端的 ICE 检查引向死路。只给 TURN relay（实测 1 秒连通）。
        answer = self.pc.localDescription.sdp
        if getattr(getattr(self.settings, "call", None),
                   "ice_relay_only", True):
            answer = _relay_only(answer)
        # 候选必须等 answer 送达后再补发（见 flush_candidates）。之前是在这里
        # 就发，早于 send_answer——对端此时还没有 remote description，会把这批
        # 「无主候选」直接丢掉，于是它手里一个我们的地址都没有，自然不探测。
        return answer

    async def flush_candidates(self, sdp: str = "") -> int:
        """answer 之后再补发本地候选（aiortc 不触发 icecandidate 事件）。

        首轮之后隔 2 秒重发一轮：对端可能刚 setRemoteDescription、尚未
        建好 m-line，太早到达的候选仍会被丢。
        """
        first = await self._flush_candidates(sdp)
        if first:
            await asyncio.sleep(2.0)
            again = await self._flush_candidates(sdp)
            log.info("候选补发完成：首轮 %d 条，重发 %d 条", first, again)
        return first

    async def _send_raw(self, kind: str, payload: Any) -> Any:
        """绕过 session 封装直连 SDK 底层，好拿到服务端响应做诊断。

        session.send_candidate() 只返回 self，看不出服务端到底收没收。
        这里直接调 awebrtc_*，失败再退回封装方法。
        """
        sess = self.session
        sdk = getattr(sess, "_sdk", None)
        inner = getattr(sess, "_session", None)
        uid = getattr(inner, "user_id", None)
        if sdk is not None and uid is not None:
            # 注意 SDK 里候选的方法名是 awebrtc_send_candidate（不是
            # awebrtc_candidate），写错会静默回退、拿不到服务端响应。
            name = ("awebrtc_send_candidate" if kind == "candidate"
                    else "awebrtc_answer")
            fn = getattr(sdk, name, None)
            if fn is not None:
                return await fn(uid, payload)
        fn = getattr(sess, "send_%s" % kind)
        return await fn(payload)

    async def _flush_candidates(self, sdp: str = "") -> int:
        """把本地 SDP 里的 candidate 提取出来，通过信令单独发给对端。"""
        if not self.pc or not self.session or not sdp:
            return 0
        sent = 0
        mid = "0"        # 跟踪当前 m-line 的 mid（offer 可能有 audio+video 两条）
        seen = set()
        for line in sdp.splitlines():
            line = line.strip()
            if line.startswith("a=mid:"):
                mid = line.split(":", 1)[1].strip() or "0"
            elif line.startswith("a=candidate:"):
                cand_str = line[2:]          # 去掉 "a="，得 "candidate:..."
                if (mid, cand_str) in seen:
                    continue                 # bundle 时 audio/video 共用同一候选
                seen.add((mid, cand_str))
                payload = {
                    "candidate": cand_str,
                    "sdpMid": mid,
                    "sdpMLineIndex": int(mid) if mid.isdigit() else 0,
                }
                try:
                    res = await self._send_raw("candidate", payload)
                    sent += 1
                    log.info("候选已投递给对端 mid=%s %s → 服务端 %s",
                             mid, cand_str[:56], res)
                except Exception as exc:
                    log.warning("发送 candidate 失败: %s", exc)
        return sent

    async def send_answer(self, sdp: str) -> None:
        """以服务端期望的 JSON 对象格式发送 answer。

        SDK 的 send_answer(json_data=纯字符串) 会产生 "v=0..."（JSON 字符串），
        而对端 offer 到达时 content 是 {"sdp": "..."}（JSON 对象）——格式不对称，
        对端解析不了 answer 就不会进入 ICE 协商。这里对齐格式。
        """
        if self.session is None:
            return
        try:
            # 现场取证：保存 BOT 的 answer 供离线复现
            try:
                with open("/tmp/last_answer.sdp", "w") as f:
                    f.write(sdp or "")
            except OSError:
                pass
            # 关键：对端拿到 answer 后执行 new RTCSessionDescription(answer)，
            # 这要求 {type, sdp} 两个字段（客户端发的是 RTCSessionDescriptionInit）。
            # 只给 {sdp} 的话浏览器构造会抛异常 → setRemoteDescription 失败 →
            # 对端没有 remote description，就不会建 ICE 检查清单，于是它只管发
            # 自己的候选、从不探测 BOT（「对端零探测」的根因就在这里）。
            res = await self._send_raw(
                "answer", {"type": "answer", "sdp": sdp})
            log.info("answer 已发送（JSON 对象格式，%d 字节）→ 服务端 %s",
                     len(sdp), res)
        except Exception as exc:
            log.warning("发送 answer 失败: %s", exc)

    async def _wait_ice_gathering(self, timeout: float = 10.0) -> None:
        """等 ICE gathering 完成（拿齐 host/srflx/relay candidate）。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.pc and getattr(self.pc, "iceGatheringState", None) == "complete":
                return
            await asyncio.sleep(0.2)
        log.info("ICE gathering 未在 %.1fs 内完成（当前 %s）",
                 timeout, getattr(self.pc, "iceGatheringState", "?"))

    async def create_offer(self) -> str:
        """主叫：生成 offer SDP。"""
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        return self.pc.localDescription.sdp

    async def add_candidate(self, raw: str) -> None:
        if not self.pc:
            return
        cand = payload_to_candidate(raw)
        if cand is None:
            log.warning("忽略空 candidate: %s", str(raw)[:120])
            return
        log.info("收到对端 candidate 并加入 ICE: %s", str(raw)[:120])
        try:
            await self.pc.addIceCandidate(cand)
        except Exception as exc:
            log.warning("addIceCandidate 失败: %s", exc)

    # -- 音频 -------------------------------------------------------------

    async def _pump_inbound(self) -> None:
        """持续读远端音频 → 16k PCM → 队列。"""
        import av

        resampler = av.AudioResampler(format="s16", layout="mono",
                                      rate=SAMPLE_RATE)
        try:
            while not self._closed and self._in_track is not None:
                frame = await self._in_track.recv()
                for f in resampler.resample(frame):
                    try:
                        self._in_queue.put_nowait(f.to_ndarray().tobytes())
                    except asyncio.QueueFull:
                        # 消费端跟不上：丢最旧的，保证实时性
                        try:
                            self._in_queue.get_nowait()
                            self._in_queue.put_nowait(f.to_ndarray().tobytes())
                        except Exception:
                            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("接收音频循环结束: %s", exc)

    async def read_audio(self, timeout: float = 0.5) -> Optional[bytes]:
        """取一段远端音频（阻塞至多 timeout 秒）。"""
        try:
            return await asyncio.wait_for(self._in_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def send_audio(self, pcm: bytes) -> None:
        """把要播放的 PCM 推给发送轨道。"""
        if self._out is not None and pcm:
            self._out.push(pcm)

    def interrupt(self) -> None:
        """清空待播音频（对方抢话时用）。"""
        if self._out is not None:
            self._out.clear()

    @property
    def pending_ms(self) -> int:
        return self._out.pending_ms if self._out else 0

    @property
    def is_connected(self) -> bool:
        return bool(self.pc and self.pc.connectionState == "connected")

    # -- 关闭 -------------------------------------------------------------

    async def close(self) -> None:
        self._closed = True
        for task in (self._pump_task,):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._pump_task = None
        if self.pc is not None:
            try:
                await self.pc.close()
            except Exception:
                pass
        self.pc = None
        self._out = None
        self._in_track = None
        self._remote_sdp = None
        self._connected.clear()
        log.info("PeerConnection 已关闭")
