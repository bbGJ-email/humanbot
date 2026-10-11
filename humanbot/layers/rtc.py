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
                    if s.get("urls"):
                        norm.append(s)
                else:
                    urls = getattr(s, "urls", None)
                    if urls:
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
            if payload and self.session is not None:
                try:
                    await self.session.send_candidate(payload)
                except Exception as exc:
                    log.debug("发送 candidate 失败: %s", exc)

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
        return self.pc.localDescription.sdp

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
            log.debug("忽略空 candidate")
            return
        try:
            await self.pc.addIceCandidate(cand)
        except Exception as exc:
            log.debug("addIceCandidate 失败: %s", exc)

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
