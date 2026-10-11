"""语音活动检测（VAD）。

两个用途（策划案 §5.3）：

1. **断句** —— 对方说完一句（静音持续数百毫秒）就送去 STT。
   不能一直等，也不能一有停顿就切断。
2. **打断** —— BOT 正在说话时，一旦检测到对方开口就停下听。
   「自顾自把话说完」是很多 AI 通话产品的硬伤，真人不会这样。

用能量（RMS）+ 自适应噪声底实现，不引入 webrtcvad 之类的二进制依赖：
通话链路的瓶颈在 LLM，VAD 精度够用就行。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .logging_setup import get_logger
from .voice import pcm_rms

log = get_logger("vad")


@dataclass
class VadEvent:
    """一次 VAD 事件。"""

    kind: str                     # speech_start | speech_end
    pcm: bytes = b""              # speech_end 时携带完整语音段
    duration_ms: int = 0

    def __repr__(self) -> str:
        return "<VadEvent %s %dms>" % (self.kind, self.duration_ms)


class VAD:
    """流式语音活动检测。

    用法：不断把收到的 PCM 喂给 `push()`，返回事件列表。
    """

    def __init__(self, cfg: Any = None, sample_rate: int = 16000,
                 frame_ms: int = 20, channels: int = 1,
                 sample_width: int = 2,
                 energy_threshold: float = 0.016,
                 noise_ratio: float = 3.0) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.frame_ms = frame_ms
        # 每帧字节数
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * sample_width * channels

        self.silence_ms = int(getattr(cfg, "vad_silence_ms", 600) or 600)
        self.min_speech_ms = int(getattr(cfg, "vad_min_speech_ms", 250) or 250)
        self.energy_threshold = energy_threshold
        self.noise_ratio = noise_ratio

        self._buf = bytearray()
        self._speech = False
        self._silence_run = 0          # 连续静音帧数
        self._seg = bytearray()        # 当前语音段
        self._speech_frames = 0        # 语音帧数（用于最短长度判断）
        self._noise = 0.01             # 自适应噪声底

    # -- 状态 -------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """当前是否处于语音中（打断检测用）。"""
        return self._speech

    @property
    def noise_floor(self) -> float:
        return self._noise

    def reset(self) -> None:
        """清空缓冲（新一轮对话开始时调用）。"""
        self._buf.clear()
        self._seg.clear()
        self._speech = False
        self._silence_run = 0
        self._speech_frames = 0

    # -- 主入口 -----------------------------------------------------------

    def push(self, pcm: bytes) -> List[VadEvent]:
        """喂入音频，返回本次产生的事件。"""
        if not pcm:
            return []
        self._buf.extend(pcm)
        events: List[VadEvent] = []

        while len(self._buf) >= self.frame_bytes:
            frame = bytes(self._buf[:self.frame_bytes])
            del self._buf[:self.frame_bytes]
            ev = self._process_frame(frame)
            if ev is not None:
                events.append(ev)
        return events

    def flush(self) -> Optional[VadEvent]:
        """收尾：把还在缓冲里的语音段吐出来（对方挂断或被强制打断时用）。"""
        if self._speech and self._speech_frames * self.frame_ms >= self.min_speech_ms:
            seg = bytes(self._seg)
            dur = self._frames_to_ms(len(seg))
            self.reset()
            return VadEvent("speech_end", seg, dur)
        self.reset()
        return None

    # -- 内部 -------------------------------------------------------------

    def _process_frame(self, frame: bytes) -> Optional[VadEvent]:
        rms = pcm_rms(frame)

        if not self._speech:
            # 只在静音期更新噪声底，否则会把说话声当成背景噪音
            self._noise = self._noise * 0.95 + rms * 0.05

        thr = max(self.energy_threshold, self._noise * self.noise_ratio)
        voiced = rms > thr

        if voiced:
            started = not self._speech
            self._speech = True
            self._silence_run = 0
            self._seg.extend(frame)
            self._speech_frames += 1
            if started:
                return VadEvent("speech_start", b"", 0)
            return None

        # 静音帧
        if not self._speech:
            return None

        self._silence_run += 1
        # 尾部静音也留住一点，免得把字尾削掉
        self._seg.extend(frame)
        if self._silence_run * self.frame_ms >= self.silence_ms:
            return self._close_segment()
        return None

    def _close_segment(self) -> Optional[VadEvent]:
        speech_ms = self._speech_frames * self.frame_ms
        seg = bytes(self._seg)
        dur = self._frames_to_ms(len(seg))
        # 太短的多半是咳嗽/呼吸/环境音，丢掉
        too_short = speech_ms < self.min_speech_ms
        self._speech = False
        self._seg = bytearray()
        self._silence_run = 0
        self._speech_frames = 0
        if too_short:
            log.debug("VAD 丢弃过短语音段 %dms", speech_ms)
            return None
        log.debug("VAD 断句：语音 %dms / 音频 %dms", speech_ms, dur)
        return VadEvent("speech_end", seg, dur)

    def _frames_to_ms(self, n_bytes: int) -> int:
        per_frame = self.frame_bytes or 1
        return int(n_bytes / per_frame * self.frame_ms)
