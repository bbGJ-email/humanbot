"""语音服务：TTS（edge-tts）+ STT（模型站）+ 音频工具。

设计要点（策划案 §6）：

- 音频格式统一 **16kHz / 单声道 / 16bit PCM**，与 BoxIM WebRTC 一致，
  全程不做二次重采样
- TTS **分句合成**，句间插停顿、长句插换气——真人说话不是一口气念完
- 情绪影响语速/音调/音量。语调一成不变是 TTS 最容易露馅的地方
- STT 主备双模型，主模型失败自动切备用，不让一通电话因为一个接口挂掉
"""
from __future__ import annotations

import asyncio
import io
import random
import re
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

from .logging_setup import BUS, get_logger

log = get_logger("voice")


# ============================================================================
# 音频工具
# ============================================================================

def pcm_to_wav(pcm: bytes, rate: int = 16000, channels: int = 1,
               sample_width: int = 2) -> bytes:
    """16k PCM → WAV（STT 接口要文件，不认裸 PCM）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm(data: bytes) -> Tuple[bytes, int]:
    """WAV → (PCM, 采样率)。"""
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def silence(ms: int, rate: int = 16000, channels: int = 1,
            sample_width: int = 2) -> bytes:
    """生成静音 PCM。句间停顿、换气都用它。"""
    if ms <= 0:
        return b""
    frames = int(rate * ms / 1000)
    return b"\x00" * (frames * channels * sample_width)


def pcm_duration_ms(pcm: bytes, rate: int = 16000, channels: int = 1,
                    sample_width: int = 2) -> int:
    """PCM 时长（毫秒）。"""
    if not pcm:
        return 0
    return int(len(pcm) / (rate * channels * sample_width) * 1000)


def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """线性插值重采样（只在格式对不上时兜底用）。

    正常链路上两端都是 16k，走不到这里；留着是为了防止哪天
    某个 STT 接口换成 8k 时整条链路崩掉。
    """
    if src_rate == dst_rate or not pcm:
        return pcm
    import array
    src = array.array("h")
    src.frombytes(pcm[:len(pcm) // 2 * 2])
    n_dst = int(len(src) * dst_rate / src_rate)
    if n_dst <= 0:
        return b""
    out = array.array("h", [0] * n_dst)
    ratio = (len(src) - 1) / max(1, n_dst - 1)
    for i in range(n_dst):
        pos = i * ratio
        i0 = int(pos)
        i1 = min(i0 + 1, len(src) - 1)
        frac = pos - i0
        out[i] = int(src[i0] * (1 - frac) + src[i1] * frac)
    return out.tobytes()


def mp3_to_pcm(mp3: bytes, rate: int = 16000) -> bytes:
    """MP3 → 16k 单声道 PCM（走 av，不依赖系统 ffmpeg）。"""
    if not mp3:
        return b""
    try:
        import av
    except ImportError:
        log.error("缺少 av 库，无法解码 TTS 音频（pip install av）")
        return b""

    out = bytearray()
    with av.open(io.BytesIO(mp3)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                out.extend(rf.to_ndarray().tobytes())
        # 冲掉重采样器里残留的尾巴，否则每句结尾会少几毫秒
        for rf in resampler.resample(None):
            out.extend(rf.to_ndarray().tobytes())
    return bytes(out)


def pcm_rms(pcm: bytes) -> float:
    """PCM 音量（RMS，0~1）。VAD 用它判断有没有人在说话。"""
    if len(pcm) < 2:
        return 0.0
    import array
    a = array.array("h")
    a.frombytes(pcm[:len(pcm) // 2 * 2])
    if not a:
        return 0.0
    # 抽样计算，长音频不必全量遍历
    step = max(1, len(a) // 2000)
    s = 0
    n = 0
    for i in range(0, len(a), step):
        v = a[i] / 32768.0
        s += v * v
        n += 1
    return (s / n) ** 0.5 if n else 0.0


def split_for_speech(text: str, max_chars: int = 22) -> List[str]:
    """把回复切成适合逐句合成的小块。

    口语一次说不了太长；长句必须拆开，否则听着像念稿子。
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1) 先按句末标点切
    parts = re.split(r"(?<=[。！？!?…；;])\s*", text)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            out.append(p)
            continue
        # 2) 太长的再按逗号切，并尽量把小段拼回接近 max_chars
        buf = ""
        for s in re.split(r"(?<=[，,、])\s*", p):
            if not s:
                continue
            if len(buf) + len(s) <= max_chars:
                buf += s
            else:
                if buf:
                    out.append(buf)
                buf = s
        if buf:
            out.append(buf)
    return out or [text]


def add_offset(base: str, offset: str) -> str:
    """两个语音参数偏移相加，如 '+0%' + '+8%' → '+8%'。

    edge-tts 只接受单个相对值，而我们要"基准 + 情绪"叠加。
    """
    def num(s: str) -> int:
        m = re.match(r"\s*([+-]?\d+)", s or "0")
        return int(m.group(1)) if m else 0

    unit = "%"
    for u in ("%", "Hz"):
        if (base or "").endswith(u) or (offset or "").endswith(u):
            unit = u
            break
    return "%+d%s" % (num(base) + num(offset), unit)


# ============================================================================
# 语音服务
# ============================================================================

class VoiceService:
    """TTS + STT。所有方法都是 async（通话链路本身跑在事件循环里）。"""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.cfg = settings.voice
        self.call_cfg = settings.call
        # 合成结果缓存：同一句话（含情绪）短时间内重复说，直接复用
        self._tts_cache: Dict[str, bytes] = {}
        self._cache_order: List[str] = []
        self._cache_max = 40
        self._stt_fail_count = 0
        self._stt_last_fail = 0.0

    # -- 内部：供应商凭据 -------------------------------------------------

    def _stt_credentials(self) -> Tuple[str, str]:
        """取 STT 用的 base_url 与 api_key。

        语音模型挂在同一个 NewAPI 站上，所以直接复用配置里的供应商凭据。
        """
        for p in self.settings.model.providers:
            if p.enabled and p.base_url and p.api_key:
                return p.base_url.rstrip("/"), p.api_key
        return "", ""

    # -- TTS --------------------------------------------------------------

    def _voice_params(self, mood: str = "平静") -> Tuple[str, str, str]:
        """情绪 → (rate, pitch, volume)。"""
        c = self.cfg
        rate = add_offset(c.tts_rate, (c.mood_rate or {}).get(mood, "+0%"))
        pitch = add_offset(c.tts_pitch, (c.mood_pitch or {}).get(mood, "+0Hz"))
        vol = add_offset(c.tts_volume, (c.mood_volume or {}).get(mood, "+0%"))
        return rate, pitch, vol

    async def _tts_one(self, text: str, mood: str) -> bytes:
        """合成一句 → 16k PCM。"""
        if not text.strip():
            return b""
        try:
            import edge_tts
        except ImportError:
            log.error("缺少 edge-tts（pip install edge-tts）")
            return b""

        rate, pitch, vol = self._voice_params(mood)
        voice = self.cfg.tts_voice
        t0 = time.time()
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch,
                                        volume=vol)
            mp3 = bytearray()
            async for chunk in comm.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    mp3.extend(chunk["data"])
            if not mp3:
                log.warning("TTS 返回空音频: %s", text[:20])
                return b""
            pcm = mp3_to_pcm(bytes(mp3), self.cfg.sample_rate)
            log.debug("TTS %.2fs %d字 → %dms 音频", time.time() - t0,
                      len(text), pcm_duration_ms(pcm, self.cfg.sample_rate))
            BUS.publish("voice", {"event": "tts", "text": text[:30],
                                  "ms": int((time.time() - t0) * 1000),
                                  "audio_ms": pcm_duration_ms(pcm, self.cfg.sample_rate)})
            return pcm
        except Exception as exc:
            log.error("TTS 合成失败: %s", exc)
            return b""

    async def synthesize(self, text: str, mood: str = "平静") -> bytes:
        """整段合成（带缓存）。

        分句合成，句间插停顿、长句插换气——这是"像人在说"而不是
        "像在念"的关键。
        """
        text = (text or "").strip()
        if not text:
            return b""
        key = "%s|%s" % (mood, text)
        if key in self._tts_cache:
            return self._tts_cache[key]

        sentences = split_for_speech(text, self.call_cfg.reply_max_chars)
        gap_lo, gap_hi = (self.cfg.sentence_gap_ms + [200, 500])[:2]

        out = bytearray()
        for i, s in enumerate(sentences):
            pcm = await self._tts_one(s, mood)
            if not pcm:
                continue
            out.extend(pcm)
            if i < len(sentences) - 1:
                # 句间停顿：真人换气、想下一句都要时间
                out.extend(silence(random.randint(int(gap_lo), int(gap_hi)),
                                   self.cfg.sample_rate))
        data = bytes(out)
        if data:
            self._cache_put(key, data)
        return data

    def _cache_put(self, key: str, data: bytes) -> None:
        self._tts_cache[key] = data
        self._cache_order.append(key)
        while len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._tts_cache.pop(old, None)

    async def synthesize_stream(self, text: str, mood: str = "平静"):
        """逐句产出 PCM，边合成边播。

        通话场景用它能把"首字节延迟"从"整段合成完"压到"第一句合成完"。
        """
        sentences = split_for_speech(text, self.call_cfg.reply_max_chars)
        for i, s in enumerate(sentences):
            pcm = await self._tts_one(s, mood)
            if not pcm:
                continue
            yield pcm
            if i < len(sentences) - 1:
                lo, hi = (self.cfg.sentence_gap_ms + [200, 500])[:2]
                yield silence(random.randint(int(lo), int(hi)),
                              self.cfg.sample_rate)

    async def speak_filler(self, mood: str = "平静") -> bytes:
        """随机一句填充词（"嗯……"/"那个……"）。

        这是掩盖 LLM 延迟的核心手段：2 秒沉默很假，但"嗯……"之后的
        2 秒非常正常。

        注意：这里会逐个尝试直到有一句合成成功。edge-tts 对纯语气词
        （如"嗯……"）偶尔返回空音频，所以不能只试一个就放弃。
        """
        words = self.persona_fillers() or ["嗯", "那个什么", "我想想"]
        for w in random.sample(words, len(words)):
            pcm = await self.synthesize(w, mood)
            if pcm:
                return pcm
        # 全失败也要给一小段声音，不能返回空让调用方卡住
        return silence(400, self.cfg.sample_rate)

    # -- 预热 --------------------------------------------------------------

    def _all_fixed_phrases(self) -> List[str]:
        """所有"通话中固定要说"的短语。"""
        v = self._persona_voice or {}
        out: List[str] = []
        for key in ("fillers", "unclear_reply", "video_excuse"):
            out.extend(v.get(key) or [])
        for key in ("answer_style", "hangup_style"):
            out.extend((v.get(key) or {}).values())
        out.append(self.call_cfg.video_excuse)
        seen, uniq = set(), []
        for p in out:
            p = (p or "").strip()
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    async def prewarm(self, persona: Any = None) -> int:
        """预生成固定短语的音频（并发）。

        通话里"接起"和"填充词"都必须在一瞬间出声，现场合成要 2 秒，
        根本等不起。所以在启动时一次性生成好放进缓存。

        必须并发：30 多条短语串行要一两分钟，启动会卡住。
        并发度由 voice.tts_concurrency 控制（大了容易触发 edge-tts 限流）。

        返回成功预热的条数。
        """
        if persona is not None:
            self.bind_persona(persona)
        phrases = self._all_fixed_phrases()
        if not phrases:
            return 0

        sem = asyncio.Semaphore(max(1, int(self.cfg.tts_concurrency or 1)))

        async def _one(p: str) -> Tuple[str, bytes]:
            async with sem:
                return p, await self._tts_one(p, "平静")

        t0 = time.time()
        results = await asyncio.gather(*[_one(p) for p in phrases],
                                       return_exceptions=True)
        ok = 0
        for r in results:
            if isinstance(r, BaseException):
                log.debug("预热某条短语失败: %s", r)
                continue
            p, pcm = r
            if pcm:
                self._cache_put("平静|" + p, pcm)
                ok += 1
        log.info("语音预热完成：%d/%d 条固定短语，耗时 %.1fs",
                 ok, len(phrases), time.time() - t0)
        BUS.publish("voice", {"event": "prewarm", "ok": ok, "total": len(phrases)})
        return ok

    def is_prewarmed(self, text: str, mood: str = "平静") -> bool:
        """这句是否已在缓存里（用于判断能否瞬间出声）。"""
        return ("%s|%s" % (mood, text.strip())) in self._tts_cache

    def persona_fillers(self) -> List[str]:
        """从人设读填充词（由 bind_persona 注入）。"""
        return getattr(self, "_persona_fillers", []) or []

    def bind_persona(self, persona: Any) -> None:
        """注入人设里的语音话术（填充词、接起语等）。"""
        v = (persona.data.get("voice") if persona else None) or {}
        self._persona_fillers = list(v.get("fillers") or [])
        self._persona_voice = v

    # -- STT --------------------------------------------------------------

    async def transcribe(self, pcm: bytes) -> str:
        """16k PCM → 文字。主备模型依次尝试。"""
        if not pcm:
            return ""
        base, key = self._stt_credentials()
        if not base or not key:
            log.warning("未配置 STT 凭据（需要至少一个带 api_key 的供应商）")
            return ""

        wav = pcm_to_wav(pcm, self.cfg.sample_rate, self.cfg.channels,
                         self.cfg.sample_width)
        models = [self.cfg.stt_primary]
        if self.cfg.stt_fallback and self.cfg.stt_fallback != self.cfg.stt_primary:
            models.append(self.cfg.stt_fallback)

        last_err = ""
        for m in models:
            t0 = time.time()
            try:
                text = await asyncio.to_thread(self._stt_call, base, key, m, wav)
                if text:
                    log.info("STT[%s] %.2fs %dms音频 → %s", m, time.time() - t0,
                             pcm_duration_ms(pcm, self.cfg.sample_rate), text[:40])
                    BUS.publish("voice", {"event": "stt", "text": text[:40],
                                          "model": m,
                                          "ms": int((time.time() - t0) * 1000)})
                    self._stt_fail_count = 0
                    return text
                last_err = "返回空文本"
            except Exception as exc:
                last_err = str(exc)[:120]
                log.warning("STT[%s] 失败: %s", m, last_err)
        self._stt_fail_count += 1
        self._stt_last_fail = time.time()
        log.error("STT 全部失败: %s", last_err)
        return ""

    def _stt_call(self, base: str, key: str, model: str, wav: bytes) -> str:
        """同步调用 /audio/transcriptions（在线程池里跑）。"""
        import requests

        r = requests.post(
            f"{base}/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"model": model},
            timeout=self.cfg.stt_timeout,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        try:
            return (r.json() or {}).get("text", "").strip()
        except ValueError:
            raise RuntimeError(f"响应非 JSON: {r.text[:120]}")

    @property
    def stt_unhealthy(self) -> bool:
        """连续失败 3 次以上视为不可用（上层可以据此回一句"你刚说啥"）。"""
        return self._stt_fail_count >= 3

    # -- 合成"接起语/收尾语" ------------------------------------------------

    def answer_line(self, stage: str, state: str = "") -> str:
        """接起电话的第一句，按关系阶段 + 当前状态选。

        真人接电话，开口第一句就带着"是谁打的、我现在什么状态"的判断。
        """
        styles = (self._persona_voice or {}).get("answer_style") or {}
        if state and state in styles:
            return styles[state]
        pool = []
        # 关系阶段优先级由近到远，低阶段也保留少量口语变化
        order = ["dependent", "close", "familiar", "acquaintance", "stranger"]
        if stage in order:
            idx = order.index(stage)
            pool = [styles.get(s) for s in order[idx:idx + 2] if styles.get(s)]
        if not pool:
            pool = [styles.get("acquaintance") or "喂？"]
        return random.choice(pool)

    def hangup_line(self, kind: str = "soft") -> str:
        """挂断前的收尾语。直接断掉很假，真人总要说点什么。"""
        styles = (self._persona_voice or {}).get("hangup_style") or {}
        return styles.get(kind) or styles.get("soft") or "那先这样？"

    def unclear_line(self) -> str:
        """没听清时的话。"""
        pool = (self._persona_voice or {}).get("unclear_reply") or []
        return random.choice(pool) if pool else "喂？你那边听得到吗"

    def video_excuse_line(self) -> str:
        """挡视频通话的借口。"""
        pool = (self._persona_voice or {}).get("video_excuse") or []
        if pool:
            return random.choice(pool)
        return self.call_cfg.video_excuse or "我摄像头好像坏了，语音行吗"
