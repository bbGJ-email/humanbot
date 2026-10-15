"""通话状态机与对话循环（策划案 §5）。

一通电话的完整生命周期：

    来电 → 决策 → 振铃延迟 → 接听 → 对话循环 → 挂断 → 后处理

三个决定"像不像人"的设计点，都落在这个文件里：

1. **振铃延迟**（§5.2）——真人接电话要放下手上的事，绝不秒接
2. **填充词掩盖延迟**（§6.1）——LLM 超过 800ms 先播"嗯……"，
   2 秒沉默很假，但"嗯……"之后的 2 秒非常正常
3. **打断让步**（§5.3）——对方抢话就停下听，不像很多 AI 产品那样自顾自说完
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

from ..core.call_policy import ANSWER, CallPolicy
from ..core.logging_setup import BUS, get_logger
from ..core.utils import minute_of_day, now, pick
from ..core.vad import VAD
from ..core.voice import pcm_duration_ms
from ..settings import TASK_REPLY
from .rtc import RTCBridge

log = get_logger("call")

# 通话专用的口语 prompt。不能复用聊天 prompt：
# 聊天是给人看的，通话是要念出来的，要求完全不同。
_VOICE_RULES = """
【当前场景】你正在和对方打电话（语音通话）。
说话的规矩：
- 就像平时接电话一样，1~2 句，最多 25 个字
- 口语，可以用"嗯""诶""那个""就是"这类语气词
- 绝对不要用表情符号、颜文字、括号、列表、加粗
- 不要念标点，不要书面语，不要说教
- 说完就停，不要总结，不要连着反问好几个问题
- 如果没听清，直接说"啊？你说啥"
"""

_VOICE_RULES_ANGRY = """
【当前场景】你还在生气，对方打电话来了。
- 话要短、冷，可以敷衍，但不爆粗
- 别主动热络，等对方先给台阶
"""


class CallFlow:
    """一通电话的完整流程。"""

    def __init__(self, bot: Any, voice: Any, db: Any, settings: Any,
                 conn: Any) -> None:
        self.bot = bot                     # BotRuntime
        self.voice = voice                 # VoiceService
        self.db = db
        self.settings = settings
        self.call_cfg = settings.call
        self.conn = conn
        self.policy = CallPolicy(self.call_cfg)

        self.bridge: Optional[RTCBridge] = None
        self.call_id: int = 0
        self.peer_id: int = 0
        self.peer_name: str = ""
        self.mode: str = "voice"
        self.pending_offer: Optional[Dict[str, Any]] = None  # 来电消息里带的 offer

        self.talking = False
        self._interrupted = False
        self._speaking = False
        self._started_at: float = 0.0
        self._answered_at: float = 0.0
        self._turn = 0
        self._latencies: List[int] = []
        self._last_activity: float = 0.0

    # ==================================================================
    # 主入口
    # ==================================================================

    async def run_incoming(self, peer_id: int, peer_name: str = "",
                           mode: str = "voice",
                           extra: Optional[Dict[str, Any]] = None) -> None:
        """处理一通来电（由 BotRuntime 在收到 RTC_SETUP 时调用）。"""
        if not self.call_cfg.enabled:
            return
        if self.talking:
            # 真人单线程：已经在通话就不再接第二通
            log.info("已在通话中，忽略来自 %s 的来电", peer_id)
            return

        self.peer_id = peer_id
        self.peer_name = peer_name or str(peer_id)
        self.mode = mode
        # 协议兼容：来电消息(100/101)的 content 里通常就是 offer，客户端
        # 直接拿它 createAnswer，然后把 answer 随 accept 一起回给主叫。
        self.pending_offer = (extra or {}).get("offer")
        self._started_at = time.time()
        self._turn = 0
        self._latencies = []

        self.call_id = self.db.create_call(
            self.bot.bot_id, peer_id, peer_name=self.peer_name,
            direction="incoming", mode=mode)

        # ---- 决策 ----
        decision = self._decide()
        self.db.update_call(self.call_id, decision=decision.action,
                            decision_reason=decision.reason)

        if not decision.answered:
            self._event("decided", "%s: %s" % (decision.action, decision.reason))
            log.info("不接这通电话：%s（%s）", decision.action, decision.reason)
            self.db.update_call(self.call_id, ended_at=now(),
                                end_reason=decision.action)
            BUS.publish("call", {"event": "declined", "peer_id": peer_id,
                                 "action": decision.action,
                                 "reason": decision.reason})
            if decision.action == "reject":
                # 明确拒接：让对方知道"我不接"
                try:
                    session = self._make_session(is_caller=False)
                    await session.reject()
                except Exception as exc:
                    log.debug("reject 失败: %s", exc)
            # ignore：什么都不做，让铃声自然响完（显得"人不在"）
            return

        # ---- 接听 ----
        await self._answer(decision)

    def _decide(self):
        """收集上下文并做接听决策。"""
        st = self.bot.emotion.state(self.bot.bot_id, self.peer_id, "private")
        # 排除本次来电自己刚建的那条记录，否则"距上次通话"恒为 0 秒
        last = self.db.last_call_with(self.bot.bot_id, self.peer_id,
                                      exclude_id=self.call_id)
        return self.policy.decide(self.policy.build_context(
            bot_id=self.bot.bot_id,
            peer_id=self.peer_id,
            peer_name=self.peer_name,
            stage=st.relation,
            annoyance=int(st.annoyance or 0),
            energy=float(st.emotion.energy or 0.6),
            sleep_state=self.bot.sleep.state_of(self.bot.bot_id) if self.bot.sleep else "AWAKE",
            minute_of_day=minute_of_day(),
            has_active_call=False,
            calls_today=self.db.calls_today(self.bot.bot_id),
            last_call_ts=int(last["ended_at"] or last["started_at"]) if last else 0,
            now_ts=now(),
            is_blacklisted=False,
            mode=self.mode,
        ))

    # ==================================================================
    # 接听
    # ==================================================================

    async def _answer(self, decision: Any) -> None:
        # ---- 振铃延迟：拟人化的灵魂，绝不秒接 ----
        delay = max(0.5, float(decision.ring_delay or 3.0))
        self._event("ringing", "振铃 %.1fs" % delay)
        BUS.publish("call", {"event": "ringing", "peer_id": self.peer_id,
                             "delay": delay})
        log.info("振铃 %.1fs 后接听（%s）", delay, self.peer_name)
        await asyncio.sleep(delay)

        session = self._make_session(is_caller=False)
        self.bridge = RTCBridge(self.conn, self.settings)
        try:
            await self.bridge.open(self.peer_id, session)

            # 协议兼容（照客户端实现）：来电消息里已经带了 offer，此时要
            # setRemoteDescription + createAnswer，并把 answer 随 accept 一起
            # 提交——服务端把 answer 装进 ACCEPT(102) 信令回给主叫。旧流程
            # 「先 accept、再等 107、再用独立 /answer 接口发」主叫收不到，
            # 于是它永远不建 ICE 检查清单、从不探测 BOT。
            answer = None
            if self.pending_offer:
                self.bridge.set_remote_sdp(
                    str(self.pending_offer.get("sdp") or ""))
                answer = await self.bridge.apply_remote_sdp(is_caller=False)
            accept_answer = ({"type": "answer", "sdp": answer} if answer else None)
            if answer:
                log.info("已从来电消息生成 answer（%d 字节），随 accept 一并提交",
                         len(answer))

            # accept 走 HTTP，偶发网络抖动会抛 CancelledError——重试兜底。
            for attempt in range(3):
                try:
                    await session.accept(accept_answer)
                    break
                except asyncio.CancelledError:
                    if attempt == 2:
                        raise
                    log.warning("accept 请求被取消（第 %d 次），1s 后重试",
                                attempt + 1)
                    await asyncio.sleep(1)
            self._answered_at = time.time()
            self.db.update_call(
                self.call_id, answered_at=now(), ring_delay_ms=int(delay * 1000))
            self._event("answered", "接通")
            log.info("已接通：%s", self.peer_name)

            # 立刻上报设备（开麦）。原先这段在 wait_connected 之后，要等
            # 20 秒超时才发出——服务端/对端很可能在等这个信号才开始转发
            # 候选与媒体，晚发等于整通话干等。
            try:
                await session.update_device(camera=False, microphone=True,
                                            screen_share=False)
                log.info("已上报设备状态（麦克风开启）")
            except Exception as exc:
                log.warning("上报设备状态失败: %s", exc)

            if answer is None:
                # 回退：来电消息里没带 offer，才等主叫的 offer(107)
                if not await self._wait_remote_sdp(15.0):
                    log.warning("等不到远端 SDP，挂断")
                    await self._hangup("no_sdp")
                    return
                answer = await self.bridge.apply_remote_sdp(is_caller=False)

            if answer:
                # 独立 answer 接口也发一份（兼容商业版协议；重复投递无害）
                await self.bridge.send_answer(answer)
                # 候选必须在 answer 送达之后补发：早发的话对端还没 setRemote-
                # Description，会把它们当无主候选直接丢掉。
                await self.bridge.flush_candidates(answer)

            # 再等 ICE 连接（offer/answer 交换后 candidate 才开始协商）
            if not await self.bridge.wait_connected(20.0):
                log.warning("WebRTC 连接超时（20s）")

            # 视频通话：用"摄像头坏了"挡回去
            if self.mode == "video":
                excuse = self.voice.video_excuse_line()
                log.info("对方要求视频，用借口挡：%s", excuse)
                await self._speak(excuse, "平静")

            # 接起第一句
            greet = self.voice.answer_line(
                self.bot.emotion.state(self.bot.bot_id, self.peer_id,
                                       "private").relation,
                state=decision.state)
            await self._speak(greet, self._mood())
            self.db.add_transcript(self.call_id, "bot", greet)

            self.talking = True
            self._last_activity = time.time()
            await self._talk_loop()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("通话异常: %s", exc)
            self._event("error", str(exc)[:120])
        finally:
            await self._finalize()

    def _make_session(self, is_caller: bool):
        """通过 SDK 创建通话会话。"""
        from boxim import RTCMode

        mode = RTCMode.VOICE if self.mode != "video" else RTCMode.VIDEO
        return self.conn.im.create_incoming_call(
            user_id=self.peer_id, mode=mode)

    async def _wait_remote_sdp(self, timeout: float) -> bool:
        """等 主叫 的 SDP 到达。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.bridge and self.bridge._remote_sdp:
                return True
            await asyncio.sleep(0.1)
        return False

    # ==================================================================
    # 对话循环
    # ==================================================================

    async def _talk_loop(self) -> None:
        """听 → 想 → 说。"""
        vad = VAD(self.call_cfg)
        events: "asyncio.Queue[Any]" = asyncio.Queue()
        listen = asyncio.ensure_future(self._listen_loop(vad, events))
        try:
            while self.talking:
                # 超时检查
                if self._should_end():
                    await self._say_goodbye()
                    break
                try:
                    ev = await asyncio.wait_for(events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if ev.kind != "speech_end":
                    continue

                # ---- 对方说完一句 ----
                t0 = time.time()
                stt_ms = 0
                text = await self.voice.transcribe(ev.pcm)
                stt_ms = int((time.time() - t0) * 1000)
                if not text:
                    if self.voice.stt_unhealthy:
                        line = self.voice.unclear_line()
                        await self._speak(line, self._mood())
                    vad.reset()
                    continue
                if len(text) < 2:
                    vad.reset()
                    continue

                self._turn += 1
                self._last_activity = time.time()
                self.db.add_transcript(self.call_id, "peer", text, stt_ms=stt_ms)
                log.info("对方说：%s（STT %dms）", text, stt_ms)
                BUS.publish("call", {"event": "heard", "peer_id": self.peer_id,
                                     "text": text, "stt_ms": stt_ms})

                # ---- 生成 + 说 ----
                reply, llm_ms = await self._generate(text)
                if not reply:
                    vad.reset()
                    continue

                self.db.add_transcript(self.call_id, "bot", reply,
                                       latency_ms=int((time.time() - t0) * 1000),
                                       llm_ms=llm_ms)
                self._latencies.append(int((time.time() - t0) * 1000))
                ok = await self._speak(reply, self._mood())
                if not ok:
                    self._event("interrupted", "被打断")
                vad.reset()
        finally:
            listen.cancel()
            try:
                await listen
            except (asyncio.CancelledError, Exception):
                pass

    async def _listen_loop(self, vad: VAD, out: "asyncio.Queue") -> None:
        """持续读音频喂 VAD，并处理打断。"""
        while self.talking:
            pcm = await self.bridge.read_audio(timeout=0.3) if self.bridge else None
            if not pcm:
                continue
            for ev in vad.push(pcm):
                if ev.kind == "speech_start" and self._speaking:
                    # 对方抢话：停下听（真人都这么做）
                    if self.call_cfg.interrupt_enabled:
                        self._interrupted = True
                        self._event("interrupt", "对方抢话")
                await out.put(ev)

    # ==================================================================
    # 生成与发声
    # ==================================================================

    async def _generate(self, user_text: str) -> Any:
        """生成回复。LLM 慢就先播填充词掩盖。"""
        system, user = self._build_prompt(user_text)
        mood = self._mood()

        task = asyncio.ensure_future(
            asyncio.to_thread(self.bot.brain.complete, TASK_REPLY, system, user))
        t0 = time.time()
        threshold = max(0.3, self.call_cfg.filler_after_ms / 1000.0)

        done, _ = await asyncio.wait({task}, timeout=threshold)
        if not done:
            # 还没想好 → 先"嗯……"，等人声掩护真实的思考时间
            log.info("LLM 超过 %.1fs，先播填充词", threshold)
            self._event("filler", "%.0fms" % (threshold * 1000))
            try:
                filler = await self.voice.speak_filler(mood)
                await self._play(filler)
            except Exception as exc:
                log.debug("填充词播放失败: %s", exc)
            reply = await task
        else:
            reply = task.result()

        llm_ms = int((time.time() - t0) * 1000)
        text = self._clean(reply)
        log.info("她说：%s（LLM %dms）", text, llm_ms)
        return text, llm_ms

    def _build_prompt(self, user_text: str) -> Any:
        """通话专用 prompt：比文字更口语、更短、禁止表情符号。"""
        persona = self.bot.persona
        st = self.bot.emotion.state(self.bot.bot_id, self.peer_id, "private")
        parts = [persona.base_prompt(), _VOICE_RULES]

        if int(st.annoyance or 0) >= self.call_cfg.max_annoyance * 0.5:
            parts.append(_VOICE_RULES_ANGRY)

        mood = self._mood()
        parts.append("【你现在的状态】心情%s，和对方是%s关系。"
                     % (mood, st.relation))

        # 通话前的记忆（不塞太多，语音场景要快）
        try:
            profile, episodic = self.bot.memory.retrieve(
                self.bot.bot_id, self.peer_id, "private", user_text, top_k=3)
            if profile or episodic:
                mem = list(profile)[:3] + list(episodic)[:3]
                parts.append("【你记得的事】" + "；".join(str(m)[:40] for m in mem))
        except Exception:
            pass

        # 上次通话（如果有）
        last = self.db.last_call_with(self.bot.bot_id, self.peer_id)
        if last and int(last["id"] or 0) != self.call_id:
            dur = int(last["duration_sec"] or 0)
            if dur > 30:
                parts.append("【上次打电话】聊了 %d 分钟" % (dur // 60 or 1))

        return "\n\n".join(parts), user_text

    def _clean(self, text: str) -> str:
        """清洗成"能念出来"的口语。"""
        import re

        t = (text or "").strip()
        # 去掉 markdown / 括号说明 / 表情
        t = re.sub(r"\*\*|__|`|~~", "", t)
        t = re.sub(r"^[-*•]\s*", "", t, flags=re.M)
        t = re.sub(r"（[^）]*）|\([^)]*\)", "", t)
        t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", t)
        # 去掉乱码：西里尔等非中文的杂字符（模型偶尔会在句尾冒出 ефон 之类）
        t = re.sub(r"[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        # 太长就截断（口语本来就不长）
        limit = self.call_cfg.reply_max_chars
        if len(t) > limit * 2:
            cut = re.split(r"[。！？!?；;，,]", t[:limit * 2])
            t = (cut[0] if cut and cut[0] else t[:limit * 2]).strip()
        return t

    async def _speak(self, text: str, mood: str) -> bool:
        """合成并播放。返回 False 表示被打断。"""
        if not text:
            return True
        pcm = await self.voice.synthesize(text, mood)
        if not pcm:
            return True
        return await self._play(pcm)

    async def _play(self, pcm: bytes) -> bool:
        """推送音频并等它播完；期间对方开口就停下。"""
        if not self.bridge or not pcm:
            return True
        self._speaking = True
        self._interrupted = False
        try:
            self.bridge.send_audio(pcm)
            total = pcm_duration_ms(pcm)
            waited = 0
            step = 50
            while waited < total:
                if self._interrupted:
                    self.bridge.interrupt()
                    log.info("被打断，停止播放（已播 %d/%dms）", waited, total)
                    return False
                await asyncio.sleep(step / 1000.0)
                waited += step
            return True
        finally:
            self._speaking = False
            self._interrupted = False

    # ==================================================================
    # 挂断与后处理
    # ==================================================================

    def _should_end(self) -> bool:
        """该收尾了吗。"""
        dur = time.time() - (self._answered_at or self._started_at)
        c = self.call_cfg
        if dur >= c.max_duration_min * 60:
            return True
        # 长时间没人说话（>90s）也挂
        if self._last_activity and time.time() - self._last_activity > 90:
            return True
        return False

    async def _say_goodbye(self) -> None:
        """挂断前必须说点什么，直接断很假。"""
        dur = time.time() - (self._answered_at or self._started_at)
        c = self.call_cfg
        hour = time.localtime().tm_hour
        if hour >= 23 or hour < 5:
            kind = "night"
        elif dur > c.soft_duration_min * 60:
            kind = "tired"
        else:
            kind = random.choice(["soft", "warm"])
        line = self.voice.hangup_line(kind)
        log.info("主动收尾（%s）：%s", kind, line)
        await self._speak(line, self._mood())
        self._hangup_kind = kind

    async def _hangup(self, reason: str = "normal") -> None:
        self.talking = False
        try:
            if self.bridge and self.bridge.session:
                await self.bridge.session.hangup()
        except Exception as exc:
            log.debug("hangup 失败: %s", exc)

    async def _finalize(self) -> None:
        """通话结束后的收尾：落库 + 情绪 + 记忆 + 冷却。"""
        self.talking = False
        dur = int(time.time() - (self._answered_at or self._started_at))
        avg = int(sum(self._latencies) / len(self._latencies)) if self._latencies else 0

        if self.bridge:
            await self.bridge.close()

        try:
            self.db.update_call(
                self.call_id, ended_at=now(), duration_sec=dur,
                hangup_by=getattr(self, "_hangup_kind", "") and "bot" or "peer",
                end_reason="normal", turn_count=self._turn,
                avg_latency_ms=avg, mood=self._mood())
            self._event("ended", "通话 %ds，%d 轮，平均延迟 %dms" % (
                dur, self._turn, avg))
        except Exception as exc:
            log.debug("更新通话记录失败: %s", exc)

        # ---- 情绪结算：语音比文字影响大得多 ----
        try:
            st = self.bot.emotion.state(self.bot.bot_id, self.peer_id, "private")
            mins = dur / 60.0
            st.emotion.intimacy = min(1.0, float(st.emotion.intimacy or 0)
                                      + self.call_cfg.intimacy_gain_per_min * mins)
            st.emotion.energy = max(0.0, float(st.emotion.energy or 0.6)
                                    - self.call_cfg.energy_cost_per_min * mins)
            self.bot.emotion.persist(st)

            # 聊够 5 分钟推进关系阶段（通话是关系跃迁的强信号）
            if dur >= 300 and hasattr(self.bot.emotion, "maybe_advance_stage"):
                self.bot.emotion.maybe_advance_stage(st, delta=1)
        except Exception as exc:
            log.debug("通话情绪结算失败: %s", exc)

        # ---- 记忆沉淀：只存摘要，不存全文 ----
        try:
            if self._turn >= 2 and dur >= 60:
                self.bot.memory.add_manual(
                    self.bot.bot_id, self.peer_id, "private", "episode",
                    "%s 通了个电话，聊了 %d 分钟" % (
                        time.strftime("%m月%d日"), max(1, dur // 60)), 0.8)
        except Exception as exc:
            log.debug("通话记忆沉淀失败: %s", exc)

        BUS.publish("call", {"event": "ended", "peer_id": self.peer_id,
                             "duration": dur, "turns": self._turn,
                             "avg_latency_ms": avg})
        log.info("通话结束：%s，%ds，%d 轮，平均延迟 %dms",
                 self.peer_name, dur, self._turn, avg)

    # ==================================================================
    # 小工具
    # ==================================================================

    def _mood(self) -> str:
        try:
            st = self.bot.emotion.state(self.bot.bot_id, self.peer_id, "private")
            from ..core.emotion import mood_of
            return mood_of(st.emotion, time.localtime().tm_hour)
        except Exception:
            return "平静"

    def _event(self, kind: str, detail: str = "") -> None:
        try:
            self.db.add_call_event(self.call_id, kind, detail)
        except Exception:
            pass
        BUS.publish("call", {"event": kind, "peer_id": self.peer_id,
                             "detail": detail})

    @property
    def is_talking(self) -> bool:
        return self.talking
