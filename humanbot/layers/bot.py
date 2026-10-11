"""BOT 运行时：把所有层串成一条完整的处理链（策划案 §10）。

处理链：
  入站 → 落库 → 调度合并 → 状态检查(睡眠/冲突)
       → 记忆召回 → prompt 装配 → LLM 生成 → 反AI检测/修复
       → 噪声注入 → 输出计划(延迟/分条/错别字/撤回) → 发送
       → 情绪更新 → 记忆抽取 → 账号行为
"""
from __future__ import annotations

import random
import asyncio
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from ..core import antiai
from ..core.emotion import (EmotionEngine, MenstrualCycle, annoy_level,
                            mood_of, MOOD_RULES)
from ..core.interest import InterestModel
from ..core.llm import Brain, LLMError
from ..core.logging_setup import BUS, get_logger
from ..core.media import MediaService
from ..core.memory import MemoryStore
from ..core.persona import STAGE_PROMPTS, Persona, load_persona
from ..core.sleep import (ASLEEP, AWAKE, DROWSY, SleepScheduler, WAKING)
from ..core.stickers import StickerService
from ..core.storage import Database
from ..core.voice import VoiceService
from ..core.utils import (dice, fmt_clock, human_time, minute_of_day, now,
                          pick, today_str, truncate)
from ..settings import Settings, TASK_REPLY
from .account import AccountBehavior
from .call_flow import CallFlow
from .connection import BoxIMConnection, InboundMessage, MSG_TYPE_TEXT
from .output import OutputPlanner
from .scheduler import Scheduler

log = get_logger("bot")


class BotRuntime:
    """单个 BOT 的完整运行时。"""

    def __init__(self, settings: Settings, db: Database, pool: ThreadPoolExecutor,
                 boxim: Optional[Any] = None) -> None:
        self.settings = settings
        self.db = db
        self.pool = pool
        # 多账号时，每个 BOT 用自己的 BoxIMConfig；单账号时退回全局配置
        self.boxim = boxim or settings.boxim

        self.persona: Persona = load_persona(settings.persona_file)
        self.brain = Brain(settings, db)
        self.memory = MemoryStore(db, self.brain, settings)
        self.emotion = EmotionEngine(db, self.brain, settings)
        self.media = MediaService(db, settings.media)
        self.stickers = StickerService(db, settings)
        self.interest = InterestModel(db, self.persona, settings)
        self.account = AccountBehavior(db, self.persona, settings.humanize)
        self.output = OutputPlanner(settings.humanize, db)

        # 语音通话
        self.voice = VoiceService(settings)
        self.voice.bind_persona(self.persona)
        self.call_flow: Optional[CallFlow] = None

        self.conn: Optional[BoxIMConnection] = None
        self.scheduler: Optional[Scheduler] = None
        self.sleep: Optional[SleepScheduler] = None
        self.cycle: Optional[MenstrualCycle] = None

        self.bot_id: int = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False
        self._last_sent: float = 0.0
        self._wake_flush_running = False

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        self._loop = loop
        cfg = self.boxim
        if not cfg.username or not cfg.password:
            log.error("未配置 BoxIM 账号，请在 config.yaml 或控制台填写")
            return False

        self.conn = BoxIMConnection(cfg, self.settings, self.db,
                                    on_message=self.on_inbound,
                                    on_event=self.on_event,
                                    on_rtc=self._on_rtc)
        if not self.conn.login():
            return False
        self.bot_id = self.conn.bot_id

        # 落库 BOT 信息
        self.db.upsert_bot(
            self.bot_id, cfg.username,
            nickname=self.persona.name,
            gender=self.persona.gender,
            physio_enabled=int(self.persona.physio_enabled
                               and self.settings.physio.enabled),
            persona_yaml=json.dumps(self.persona.data, ensure_ascii=False),
        )
        self.account.bind(self.bot_id)

        # 通话流程：要在开始监听之前就位，否则第一批来电会漏掉
        self.call_flow = CallFlow(self, self.voice, self.db, self.settings,
                                  self.conn)
        # 预生成"接起语/填充词"音频。通话时现场合成要 2 秒，根本等不起。
        if self.settings.voice.enabled and self.settings.call.enabled:
            try:
                self._loop.create_task(self.voice.prewarm(self.persona))
                log.info("语音预热已排入队列（并发 %d）",
                         self.settings.voice.tts_concurrency)
            except Exception as exc:
                log.debug("语音预热调度失败: %s", exc)

        # 可选：把 IM 昵称改成和人设一致（默认关，避免动用户账号）
        if cfg.sync_nickname and self.persona.name:
            if (self.conn.nickname or "") != self.persona.name:
                if self.conn.update_profile(nickName=self.persona.name):
                    log.info("IM 昵称已同步为「%s」（原：%s）",
                             self.persona.name, self.conn.nickname)
                    self.conn.nickname = self.persona.name
                else:
                    log.warning("IM 昵称同步失败，继续用原昵称")
        else:
            log.info("别名列表：%s", " / ".join(self._aliases()))

        # 生理周期（gender 门控）
        self.cycle = MenstrualCycle(self.db, self.bot_id, self.settings.physio,
                                    self.persona.physio_enabled)

        # 睡眠
        self.sleep = SleepScheduler(self.db, self.settings.sleep)
        self.sleep.bind(self.bot_id, self._on_sleep, self._on_wake)
        self.sleep.set_state(self.bot_id, AWAKE)

        # 调度（带兴趣模型与群聊参数）
        self.scheduler = Scheduler(self.db, self.settings.humanize,
                                   self._handle_merged,
                                   interest=self.interest,
                                   group_cfg=self.settings.group)

        # 表情包服务：绑定连接并同步目录
        self.stickers.bind(self.conn)
        if self.settings.sticker.enabled and self.settings.sticker.auto_sync:
            self.pool.submit(self._safe_sync_stickers)

        # 事件注册
        self.conn.register_handlers()
        self.conn.start_listening(loop)

        # 恢复工作记忆
        for conv in self.db.conversations_of(self.bot_id):
            chat_type = conv["chat_type"]
            self.memory.load_working_from_db(
                self.bot_id, int(conv["peer_id"]), chat_type, int(conv["id"]))

        # 补拉离线消息（进程重启场景）
        self.pool.submit(self._safe_sync_offline)

        self._started = True
        log.info("BOT 运行时已启动：%s (%s)", self.persona.name, self.bot_id)
        BUS.publish("bot", {"event": "started", "bot_id": self.bot_id,
                            "name": self.persona.name})
        return True

    def _safe_sync_offline(self) -> None:
        try:
            if self.conn:
                self.conn.sync_offline()
        except Exception as exc:
            log.error("离线消息同步异常: %s", exc)

    def _safe_sync_stickers(self) -> None:
        """后台同步表情目录。失败不影响主流程。"""
        try:
            n = self.stickers.sync(force=True)
            if n:
                log.info("表情库就绪：%d 张可用", n)
        except Exception as exc:
            log.warning("表情目录同步异常: %s", exc)

    # ------------------------------------------------------------------
    # 通话信令
    # ------------------------------------------------------------------

    def _on_rtc(self, msg_type: int, raw: Dict[str, Any]) -> None:
        """通话信令入口（由连接层在 WS 线程中调用）。

        只拦"来电"（100 语音 / 101 视频）；offer / answer / candidate 这些
        在通话建立后由 RTCBridge 自行接管，不必在这里管。
        """
        if msg_type not in (100, 101):
            return
        if not self.settings.call.enabled or self.call_flow is None:
            return
        try:
            from ..core.utils import parse_message as _pm
            p = _pm(raw)
        except Exception as exc:
            log.debug("解析来电消息失败: %s", exc)
            return
        peer = int(p.get("sender_id") or 0)
        if not peer or peer == self.bot_id:
            return

        mode = "voice" if msg_type == 100 else "video"
        name = p.get("send_nick_name") or str(peer)
        log.info("收到%s来电：%s", "视频" if mode == "video" else "语音", name)
        BUS.publish("call", {"event": "incoming", "peer_id": peer,
                             "peer_name": name, "mode": mode})

        coro = self.call_flow.run_incoming(peer, name, mode)
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            # 不在事件循环线程里：丢回 BOT 的主循环执行
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(coro, self._loop)
            else:
                log.warning("没有可用事件循环，无法处理来电")

    # ------------------------------------------------------------------
    # 入站消息
    # ------------------------------------------------------------------

    def on_inbound(self, msg: InboundMessage) -> None:
        """收到消息（在 WS 线程中调用）。"""
        # 落库
        conv_id = self.db.ensure_conversation(msg.bot_id, msg.chat_id,
                                              msg.chat_type)
        inserted = self.db.insert_message(
            msg.bot_id, conv_id, "user", msg.text, msg_id=msg.msg_id,
            local_id=msg.local_id, seq_no=msg.seq_no, msg_type=msg.msg_type,
            ts=msg.ts, at_user_ids=msg.at_user_ids, quote_id=msg.quote_id,
        )
        # 游标要推进到「消息 ID」（服务端 id），不是 seq_no——否则断线补拉
        # 的 minId 永远停在 0，重启后会把全部历史重拉一遍。
        self.db.save_cursor(conv_id, msg.msg_id, int(msg.seq_no or 0))

        # 重复消息（重启补拉 / 推送重叠）：只更新游标，不重复回复
        if inserted is None:
            log.debug("跳过重复消息 msg_id=%s", msg.msg_id)
            return

        st = self.emotion.state(self.bot_id, msg.chat_id, msg.chat_type)
        st.last_interaction = now()
        self.account.note_replied(msg.chat_id)

        # 睡眠判断
        state = self.sleep.state_of(self.bot_id) if self.sleep else AWAKE
        if state == ASLEEP:
            log.info("睡眠中，消息已入库待醒来处理（来自 %s）", msg.sender_id)
            BUS.publish("inbound", {"bot_id": self.bot_id, "sleeping": True,
                                    "text": truncate(msg.text, 40)})
            return

        # 群聊：只有被 @ / 提到名字 / 权重够高才响应
        self.scheduler.submit(msg, aliases=self._aliases())
        # 兴趣关键词由 InterestModel 从人设自动生成，这里不再硬编码

    def _aliases(self) -> List[str]:
        """群聊里"叫到她"的所有可能名字。

        必须同时包含：
          - IM 账号实际显示的昵称（群里别人 @ 的就是这个）
          - 人设名 / 自称
        两者不一致时（比如账号叫 Samsara、人设叫沈晚），
        只认其中一个会导致她"被叫了却没反应"。
        """
        out: List[str] = []
        for n in (self.persona.name, self.persona.nickname_self,
                  (self.conn.nickname if self.conn else ""),
                  self.settings.boxim.username):
            n = (n or "").strip()
            if n and n not in out:
                out.append(n)
        return out

    def on_event(self, event: str, payload: Any) -> None:
        if event == "online_status":
            uid = payload.get("user_id") if isinstance(payload, dict) else None
            if uid:
                self._maybe_proactive(int(uid))
        elif event == "force_offline":
            BUS.publish("alert", {"level": "error",
                                  "message": "BOT 被强制下线，请检查账号"})

    # ------------------------------------------------------------------
    # 合并后的处理
    # ------------------------------------------------------------------

    def _handle_merged(self, msgs: List[InboundMessage], merged_text: str,
                       bucket: Any = None) -> None:
        """防抖窗口结束后，在后台线程处理。"""
        self.pool.submit(self._process, msgs, merged_text, bucket)

    def _process(self, msgs: List[InboundMessage], merged_text: str,
                 bucket: Any = None) -> None:
        msg = msgs[-1]
        chat_type = msg.chat_type
        peer_id = msg.chat_id
        # 这批消息是不是"她自己想插嘴"（群聊、没人叫她）
        proactive_group = bool(
            bucket is not None and msg.is_group and getattr(bucket, "proactive", False))
        trigger_weight = float(getattr(bucket, "weight", 1.0) or 1.0)

        try:
            st = self.emotion.state(self.bot_id, peer_id, chat_type)
            self.cycle.tick() if self.cycle else None
            self.emotion.tick(st)

            # 生理周期影响情绪
            if self.cycle and self.cycle.active:
                eff = self.cycle.effect()
                if eff:
                    st.emotion.apply({k: v * 0.3 for k, v in eff.items()})

            # 剧本事件影响
            impact = self.account.active_impact()
            if impact:
                st.emotion.apply({k: v * 0.2 for k, v in impact.items()})

            # --- 冷处理检查（冲突系统）---
            if self.emotion.is_cold(st):
                self.conn.mark_read(peer_id, msg.is_group)
                rec = self.emotion.try_reconcile(st, merged_text)
                if rec is None:
                    log.info("冷处理中，已读不回（L%d）", annoy_level(st.annoyance))
                    BUS.publish("conflict", {"event": "ignored", "peer_id": peer_id,
                                             "annoyance": st.annoyance})
                    # 冷战里偶尔甩一个表情过去——比装死不回更真实
                    self.send_sticker_only(peer_id, chat_type, mood="冷漠",
                                           annoyance=st.annoyance,
                                           user_text=merged_text)
                    return
                self._send_planned(peer_id, rec, chat_type, st,
                                   user_text=merged_text)
                return

            # --- 和解尝试 ---
            rec = self.emotion.try_reconcile(st, merged_text)
            if rec:
                self._send_planned(peer_id, rec, chat_type, st,
                                   user_text=merged_text)
                return

            # --- 身份质疑（四次递进）---
            if antiai.is_ai_question(merged_text):
                st.times_asked_ai += 1
                line = antiai.ai_deny_line(st.times_asked_ai)
                st.annoyance = min(100, st.annoyance
                                   + antiai.ai_deny_annoyance_delta(st.times_asked_ai))
                self.emotion.persist(st)
                if st.times_asked_ai >= 4:
                    self._enter_conflict(st, "反复质疑身份")
                self._send_planned(peer_id, line, chat_type, st,
                                   user_text=merged_text)
                return

            # --- 已读 ---
            self.conn.mark_read(peer_id, msg.is_group)

            # --- 阅读延迟 ---
            time.sleep(self.output.read_delay())

            # --- 纯情绪消息：有时只回一个表情就够 ---
            if self.settings.sticker.enabled and not proactive_group:
                form = self.stickers.should_send(
                    annoyance=st.annoyance, user_text=merged_text,
                    will_reply_text=False)
                if form == "solo" and dice(0.7):
                    mood0 = mood_of(st.emotion,
                                    __import__("datetime").datetime.now().hour)
                    if self.send_sticker_only(peer_id, chat_type, mood=mood0,
                                              user_text=merged_text):
                        log.info("只回了一个表情：%s", merged_text[:30])
                        self.memory.push_working(self.bot_id, peer_id,
                                                 chat_type, "user", merged_text)
                        return

            # --- 生成 ---
            mood = mood_of(st.emotion, __import__("datetime").datetime.now().hour)
            reply = self._generate(st, merged_text, msgs, mood, chat_type, peer_id)
            if not reply:
                return

            # --- 反 AI 检测与修复 ---
            reply = self._antiai_pipeline(reply, merged_text, st)

            # --- 发送 ---
            self._send_planned(peer_id, reply, chat_type, st, mood=mood,
                               user_text=merged_text)

            # --- 后处理 ---
            self._post_turn(st, merged_text, reply, mood, peer_id, chat_type,
                            proactive_group=proactive_group,
                            trigger_weight=trigger_weight)

        except Exception as exc:
            log.exception("处理消息异常: %s", exc)
            BUS.publish("alert", {"level": "error", "message": f"处理异常: {exc}"})

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def _build_prompt(self, st: Any, user_text: str, msgs: List[InboundMessage],
                      mood: str, chat_type: str, peer_id: int) -> str:
        """装配完整 system prompt（策划案 §3.6）。"""
        segs: List[str] = []
        segs.append(self.persona.base_prompt())
        segs.append("")
        segs.append(self.persona.lifespan_prompt())
        segs.append("")
        segs.append(STAGE_PROMPTS.get(st.relation, STAGE_PROMPTS["stranger"]))

        # 情绪
        m = mood_of(st.emotion, __import__("datetime").datetime.now().hour)
        segs.append("")
        segs.append("【你现在的状态】")
        segs.append(f"心情：{m}（{MOOD_RULES.get(m, '')}）")
        segs.append(f"能量：{st.emotion.energy:.2f}（越低越懒得说话、回得越短）")
        segs.append(f"压力：{st.emotion.stress:.2f}")
        if st.annoyance >= 16:
            segs.append(f"你对这个人有点不满（程度 {st.annoyance}/100），"
                        f"语气可以冷一点、短一点")

        # 生理周期
        if self.cycle and self.cycle.active:
            segs.append("")
            segs.append(f"【身体状态】{self.cycle.describe()}")
            ph = self.cycle.phase
            if ph == "MENSTRUAL":
                segs.append("这几天身体不舒服，容易累、容易烦，说话比平时少。"
                            "但不主动诉苦，被问到了可以承认。")
            elif ph == "LUTEAL" and self.cycle.day >= 24:
                segs.append("最近情绪不太好，说不出具体原因，可能突然烦躁。")
            elif ph == "FOLLICULAR":
                segs.append("这几天状态不错，愿意多聊。")

        # 睡眠/困意
        if self.sleep:
            drow = self.sleep.drowsiness(self.bot_id)
            if drow > 0.3:
                segs.append("")
                segs.append(f"【困意】{drow:.1f}（越高越迷糊），"
                            f"回复要更短、更慢、可以语无伦次")

        # 记忆
        profile, episodic = self.memory.retrieve(self.bot_id, peer_id,
                                                 chat_type, user_text)
        if profile or episodic:
            segs.append("")
            segs.append("【你记得的事】")
            for p in profile[:20]:
                segs.append(f"- {p}")
            if episodic:
                segs.append("【相关的往事】")
                for e in episodic:
                    segs.append(f"- {e}")

        # 场景
        segs.append("")
        segs.append("【此刻】")
        segs.append(f"时间：{human_time()}（{fmt_clock(minute_of_day())}）")
        if chat_type == "group":
            try:
                members = self.conn.im.get_group_members(peer_id) or []
                names = [m.get("nickName") or m.get("userName") or ""
                         for m in members[:12]]
                segs.append(f"这是群聊，群里有：{'、'.join(n for n in names if n)}")
                segs.append("群里说话要更随意，被@了必须回；不相关的话题可以只看着不说话")
            except Exception:
                segs.append("这是群聊，说话随意一点")
        else:
            segs.append("这是私聊")

        segs.append("")
        segs.append(self.persona.anti_ai_prompt())

        segs.append("")
        segs.append("【任务】")
        segs.append(f'对方刚说："{user_text}"')
        segs.append("以你的身份回一条消息。只输出消息本身，不要引号，不要任何说明。"
                    "不要复述对方的话，不要总结对话。")
        return "\n".join(segs)

    def _generate(self, st: Any, user_text: str, msgs: List[InboundMessage],
                  mood: str, chat_type: str, peer_id: int) -> str:
        system = self._build_prompt(st, user_text, msgs, mood, chat_type, peer_id)
        history = self.memory.working(self.bot_id, peer_id, chat_type)
        temperature = 0.92
        if mood in ("烦", "很烦，别惹我"):
            temperature = 0.85

        # 长回复先发 LOADING 状态
        if len(user_text) > 60 and self.conn:
            try:
                self.output.send_loading(self.conn, peer_id, chat_type == "group")
            except Exception:
                pass

        try:
            out = self.brain.complete(TASK_REPLY, system, user_text,
                                      temperature=temperature, max_tokens=180,
                                      history=history)
        except LLMError as exc:
            log.error("生成失败: %s", exc)
            BUS.publish("alert", {"level": "error", "message": f"生成失败: {exc}"})
            return ""
        return (out or "").strip().strip('"').strip("'").strip()

    # ------------------------------------------------------------------
    # 反 AI 流水线
    # ------------------------------------------------------------------

    def _antiai_pipeline(self, reply: str, user_text: str, st: Any) -> str:
        """检测 → 修复 → 噪声注入 → 软化标点。"""
        limit = self.settings.humanize.split_max_chars
        res = antiai.detect(reply, user_message=user_text, max_chars=limit)
        BUS.publish("antiai", {
            "text": truncate(reply, 60), **res.to_dict(),
        })

        if res.action == "regenerate":
            log.info("AI 感过高(%d)，重生成", res.score)
            for _ in range(2):
                try:
                    alt = self.brain.complete(
                        TASK_REPLY,
                        "把下面这句话改成真人聊天时会说的话。要短、口语、不完整。"
                        "只输出改写结果。",
                        reply, temperature=0.95, max_tokens=120,
                    )
                except Exception:
                    break
                alt = (alt or "").strip().strip('"')
                r2 = antiai.detect(alt, user_message=user_text, max_chars=limit)
                if r2.score < res.score:
                    reply, res = alt, r2
                if r2.action == "pass":
                    break
        elif res.action == "patch":
            log.info("AI 感中等(%d)，局部改写", res.score)
            reply = antiai.repair(reply, self.brain, res.hits)

        # 噪声注入：睡意越强注入越多
        drow = self.sleep.drowsiness(self.bot_id) if self.sleep else 0.0
        strength = 1.0 + drow
        if st.annoyance >= 50:
            strength *= 0.4      # 生气时不加填充词
        reply = antiai.inject_noise(reply, strength=strength)
        reply = antiai.soften_punctuation(reply)

        # 记录统计
        try:
            fin = antiai.detect(reply, user_message=user_text, max_chars=limit)
            self.db.execute(
                "INSERT INTO antiai_stats(ts, score, hits, text, passed) "
                "VALUES(?,?,?,?,?)",
                (now(), fin.score, json.dumps(fin.hits, ensure_ascii=False),
                 truncate(reply, 200), int(fin.passed)),
            )
        except Exception:
            pass
        return reply

    # ------------------------------------------------------------------
    # 发送
    # ------------------------------------------------------------------

    def _send_planned(self, peer_id: int, text: str, chat_type: str, st: Any,
                      mood: str = "", at_users: Optional[List[int]] = None,
                      quote_id: int = 0, user_text: str = "") -> int:
        """按拟人计划发送。

        user_text 用于判断要不要在文字后面补一个表情——对方说了什么，
        决定了配哪个表情才不突兀。
        """
        if not self.conn:
            return 0
        if not self.scheduler.daily_limit_ok(self.bot_id):
            return 0

        is_group = chat_type == "group"
        drow = self.sleep.drowsiness(self.bot_id) if self.sleep else 0.0
        conv_id = self.db.ensure_conversation(self.bot_id, peer_id, chat_type)

        # 要不要配表情：先掷骰子决定形态，再选图。选不到就不发。
        sticker = None
        if self.settings.sticker.enabled and text:
            form = self.stickers.should_send(annoyance=st.annoyance,
                                            user_text=user_text)
            if form == "append":
                sticker = self.stickers.pick(
                    bot_id=self.bot_id, chat_id=peer_id, mood=mood,
                    annoyance=st.annoyance, user_text=user_text,
                    energy=st.emotion.energy, drowsy=drow,
                    brain=self.brain, context=f"{user_text} / {text}")

        sent = 0
        if text:
            plan = self.output.plan(text, energy=st.emotion.energy,
                                    drowsiness=drow, mood=mood)
            if plan:
                # 发送在同步上下文中；这里已经处于线程池线程，可以安全 sleep
                sent += self.output.execute(
                    self.conn, peer_id, plan, is_group=is_group,
                    at_users=at_users, quote_id=quote_id,
                    bot_id=self.bot_id, db=self.db, conv_id=conv_id,
                )
                self.memory.push_working(self.bot_id, peer_id, chat_type,
                                         "bot", text)

        # 文字之后补一个表情（间隔 0.4~1.2 秒，像随手点了一下）
        if sticker is not None:
            time.sleep(random.uniform(0.4, 1.2))
            sent += self._send_sticker(peer_id, sticker, chat_type, conv_id)

        self._last_sent = time.time()
        return sent

    # ------------------------------------------------------------------
    # 表情包
    # ------------------------------------------------------------------

    def _send_sticker(self, peer_id: int, sticker: Dict[str, Any],
                      chat_type: str, conv_id: int = 0) -> int:
        """发一张表情。返回发送条数（0 或 1）。"""
        if not self.conn or not sticker:
            return 0
        if not int(sticker.get("id") or 0):
            return 0
        # 整个表情对象传下去：客户端渲染靠 content.imageUrl，
        # 只传 id 会被拼成 {"stickerId":N}，对方看到"undefined"
        try:
            mid = self.conn.send_sticker(peer_id, sticker,
                                         is_group=(chat_type == "group"))
        except Exception as exc:
            log.warning("发表情失败: %s", exc)
            return 0
        if not mid:
            return 0

        self.stickers.note_sent(self.bot_id, peer_id, sticker)
        self.memory.push_working(self.bot_id, peer_id, chat_type, "bot",
                                 "[表情] " + (sticker.get("name") or ""))
        if conv_id:
            try:
                self.db.insert_message(self.bot_id, conv_id, "bot",
                                       "[表情] " + (sticker.get("name") or ""),
                                       ts=now())
                self.db.execute(
                    "UPDATE conversations SET last_ts=? WHERE id=?",
                    (now(), conv_id))
            except Exception:
                pass
        BUS.publish("sticker", {
            "event": "sent", "bot_id": self.bot_id, "chat_id": peer_id,
            "name": sticker.get("name"), "group": chat_type == "group",
        })
        log.info("发出表情：%s", sticker.get("name"))
        self._last_sent = time.time()
        return 1

    def send_sticker_only(self, peer_id: int, chat_type: str,
                          mood: str = "", annoyance: int = 0,
                          user_text: str = "") -> int:
        """只发一张表情，不回文字（纯情绪回应 / 冷战敷衍）。"""
        if not self.settings.sticker.enabled or not self.conn:
            return 0
        if not self.scheduler.daily_limit_ok(self.bot_id):
            return 0
        st = self.emotion.state(self.bot_id, peer_id, chat_type)
        drow = self.sleep.drowsiness(self.bot_id) if self.sleep else 0.0
        sticker = self.stickers.pick(
            bot_id=self.bot_id, chat_id=peer_id, mood=mood,
            annoyance=annoyance or st.annoyance, user_text=user_text,
            energy=st.emotion.energy, drowsy=drow,
            brain=self.brain, context=user_text)
        if sticker is None:
            return 0
        conv_id = self.db.ensure_conversation(self.bot_id, peer_id, chat_type)
        time.sleep(random.uniform(0.5, 1.5))
        return self._send_sticker(peer_id, sticker, chat_type, conv_id)

    # ------------------------------------------------------------------
    # 回合后处理
    # ------------------------------------------------------------------

    def _post_turn(self, st: Any, user_text: str, reply: str, mood: str,
                   peer_id: int, chat_type: str,
                   proactive_group: bool = False,
                   trigger_weight: float = 1.0) -> None:
        self.memory.push_working(self.bot_id, peer_id, chat_type, "user", user_text)

        # 群聊参与状态：记下"刚说过话"（供下一轮的惯性加成）
        if chat_type == "group" and self.interest is not None:
            try:
                self.interest.note_spoke(peer_id, self.bot_id)
                if proactive_group:
                    self.interest.note_proactive(peer_id, self.bot_id)
            except Exception as exc:
                log.debug("记录群参与状态失败: %s", exc)

        # 情绪更新
        self.emotion.update_from_turn(st, user_text, reply)
        self.emotion.raise_intimacy(st, 0.02)
        self.emotion.maybe_advance_stage(st, user_text)

        new_mood = mood_of(st.emotion, __import__("datetime").datetime.now().hour)
        phase = self.cycle.phase if (self.cycle and self.cycle.active) else ""
        self.emotion.log_emotion(st, new_mood, phase)

        # 记忆抽取
        try:
            self.memory.extract_and_store(self.bot_id, peer_id, chat_type,
                                          user_text, reply)
        except Exception as exc:
            log.debug("记忆抽取异常: %s", exc)

        # 个签：情绪跃迁时换
        prev_sig = self.account.current_signature()
        if new_mood != mood:
            new_sig = self.account.maybe_update_signature(new_mood, mood)
            if new_sig and self.conn:
                self.conn.set_signature(new_sig)

        # 生活事件
        self.account.tick_life_events()
        self.account.tick_script_events()

        # 状态推进
        if self.sleep:
            self.sleep.check_transition(self.bot_id, st.emotion.energy,
                                        st.emotion.pleasure)

    def _enter_conflict(self, st: Any, reason: str) -> None:
        self.emotion.raise_annoyance(st, 30, reason)
        log.warning("进入冲突状态：%s", reason)
        BUS.publish("conflict", {"event": "enter", "peer_id": st.peer_id,
                                 "reason": reason, "annoyance": st.annoyance})

    # ------------------------------------------------------------------
    # 睡眠回调
    # ------------------------------------------------------------------

    def _on_sleep(self) -> None:
        """睡着：记录游标 → 真正断开 WS。"""
        log.info("进入睡眠，准备断开连接")
        try:
            for conv in self.db.conversations_of(self.bot_id):
                self.db.save_cursor(int(conv["id"]), int(conv["last_msg_id"] or 0),
                                    int(conv["last_seq_no"] or 0))
            if self.conn:
                self.conn.stop_listening()
        except Exception as exc:
            log.error("入睡处理异常: %s", exc)

    def _on_wake(self) -> None:
        """醒来：重连 → 补拉离线消息 → 按节奏投递。"""
        if self._wake_flush_running:
            return
        self._wake_flush_running = True
        log.info("醒来，准备重连")
        try:
            if self.conn:
                self.conn.reconnect()
            time.sleep(self.settings.sleep.waking_lead_min * 60)
            if self.conn:
                self.conn.sync_offline()
        except Exception as exc:
            log.error("醒来处理异常: %s", exc)
        finally:
            self._wake_flush_running = False

    # ------------------------------------------------------------------
    # 主动消息
    # ------------------------------------------------------------------

    def _maybe_proactive(self, peer_id: int) -> None:
        try:
            conv = self.db.get_conversation(self.bot_id, peer_id, "private")
            if conv is None:
                return
            last_ts = int(conv["last_ts"] or 0)
            # 从没聊过（last_ts=0）就不主动——否则会把新加的好友挨个骚扰一遍
            if last_ts <= 0:
                return
            st = self.emotion.state(self.bot_id, peer_id, "private")
            hours = (now() - last_ts) / 3600.0
            import datetime as _dt
            hour = _dt.datetime.now().hour
            ok, line = self.account.should_proactive(
                peer_id, is_online=True, hours_since=hours,
                relation=st.relation, hour=hour,
            )
            if ok and line:
                self._send_planned(peer_id, line, "private", st)
        except Exception as exc:
            log.debug("主动消息检查异常: %s", exc)

    # ------------------------------------------------------------------
    # 对外
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        st = self.emotion.state(self.bot_id, 0, "private") if self.bot_id else None
        sig = self.account.current_signature()

        # 睡眠面板需要的字段（控制台直接读）
        sleep_d = self.sleep.describe(self.bot_id) if self.sleep else {}
        plan = sleep_d.get("plan") or {}

        # 表情库概况
        sticker_n = 0
        try:
            sticker_n = self.stickers.count()
        except Exception:
            pass

        d = {
            "bot_id": self.bot_id,
            "name": self.persona.name,
            "nickname": self.conn.nickname if self.conn else "",
            "gender": self.persona.gender,
            "connected": bool(self.conn and self.conn.connected),
            "listening": bool(self.conn and self.conn.listening),
            "state": self.sleep.state_of(self.bot_id) if self.sleep else AWAKE,
            "sleep_state": sleep_d.get("state", AWAKE),
            "sleep_plan": ("%s → %s" % (plan.get("sleep", "--"), plan.get("wake", "--"))
                           if plan else "--"),
            "sleep_text": (self.sleep.last_sleep_text(self.bot_id)
                           if self.sleep else ""),
            "signature": sig,
            "relation": st.relation if st else "stranger",
            "stage_label": None,   # 由控制台补（避免这里依赖 persona 常量）
            "mood": mood_of(st.emotion, __import__("datetime").datetime.now().hour) if st else "",
            "annoyance": st.annoyance if st else 0,
            "emotion": st.emotion.to_json() if st else {},
            "physio_enabled": bool(self.cycle and self.cycle.active),
            "physio": self.cycle.describe() if (self.cycle and self.cycle.active) else "",
            "cycle_phase": (self.cycle.phase_label
                            if (self.cycle and self.cycle.active) else ""),
            "cycle_day": (self.cycle.day if (self.cycle and self.cycle.active) else None),
            "cycle_mult": (self.cycle.annoyance_multiplier()
                           if (self.cycle and self.cycle.active) else None),
            "cycle_effect": (", ".join("%s%+.2f" % (k, v)
                                       for k, v in (self.cycle.effect() or {}).items())
                             if (self.cycle and self.cycle.active) else ""),
            "sleep": sleep_d,
            "stickers": sticker_n,
            "interests": (self.interest.interest_topics(self.bot_id)
                          if self.interest else []),
            "cost_today": self.brain.cost_today(),
        }
        return d

    def stop(self) -> None:
        try:
            if self.conn:
                self.conn.stop_listening()
        except Exception:
            pass
        log.info("BOT 已停止")
