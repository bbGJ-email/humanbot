"""BOT 管理器：生命周期、多 BOT 编排、控制台动作（策划案 §10 L2）。"""
from __future__ import annotations

import asyncio
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ..core import antiai
from ..core.emotion import mood_of
from ..core.llm import Brain, LLMError
from ..core.logging_setup import BUS, get_logger
from ..core.media import MediaService
from ..core.memory import MemoryStore
from ..core.persona import STAGE_PROMPTS, load_persona
from ..core.storage import Database
from ..core.utils import now, truncate
from ..settings import Settings, TASK_REPLY
from .bot import BotRuntime
from .connection import BoxIMConnection
from .output import OutputPlanner
from .scheduler import Scheduler

log = get_logger("manager")


class BotManager:
    """管理一个或多个 BOT 的生命周期。"""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="humanbot")
        self._bots: List[BotRuntime] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ticker: Optional[threading.Thread] = None
        self._running = False
        self._is_group_ids: set = set()

        # 供控制台直接使用的共享组件（取第一个 BOT 的）
        self.brain: Optional[Brain] = None
        self.memory: Optional[MemoryStore] = None
        self.media: Optional[MediaService] = None
        self.stickers: Any = None
        self.interest: Any = None

    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        """按配置启动 1~N 个账号。

        单个账号失败不影响其他账号继续跑——多开时一个号掉线了
        不该把别的号也拖下水。
        """
        self._loop = loop

        accounts = self.settings.boxim.account_list()
        if not accounts:
            log.error("未配置任何 BoxIM 账号，BOT 不会启动")
            return False

        for acct in accounts:
            cfg = self.settings.boxim.for_account(acct)
            rt = BotRuntime(self.settings, self.db, self.pool, boxim=cfg)
            try:
                ok = rt.start(loop)
            except Exception as exc:  # noqa: BLE001
                log.exception("账号 %s 启动异常: %s", acct["username"], exc)
                ok = False
            if not ok:
                log.error("账号 %s 启动失败，跳过", acct["username"])
                continue
            self._bots.append(rt)

        if not self._bots:
            log.error("所有账号都启动失败")
            return False

        first = self._bots[0]
        self.brain = first.brain
        self.memory = first.memory
        self.media = first.media
        self.stickers = first.stickers
        self.interest = first.interest
        self._running = True
        self._start_ticker()
        log.info("已启动 %d 个 BOT：%s", len(self._bots),
                 ", ".join(str(b.bot_id) for b in self._bots))
        return True

    def _start_ticker(self) -> None:
        """后台心跳：状态推进、生活事件、主动消息。"""
        def _tick() -> None:
            last_life = 0.0
            last_sticker_sync = time.time()
            while self._running:
                time.sleep(20)
                try:
                    for rt in self._bots:
                        if not rt.bot_id:
                            continue
                        st = rt.emotion.state(rt.bot_id, 0, "private")
                        rt.emotion.tick(st)
                        if rt.sleep:
                            rt.sleep.check_transition(
                                rt.bot_id, st.emotion.energy, st.emotion.pleasure)
                            rt.cycle.tick() if rt.cycle else None
                        # 每 ~2 分钟跑一次生活事件
                        if time.time() - last_life > 120:
                            rt.account.tick_life_events()
                            rt.account.tick_script_events()
                            # 换头像（低频）
                            hd = rt.account.maybe_update_head(
                                mood_of(st.emotion, __import__("datetime").datetime.now().hour),
                                rt.media)
                            if hd and rt.conn:
                                rt.conn.set_head_image(hd)
                        # 主动消息
                        self._check_proactive(rt)
                    if time.time() - last_life > 120:
                        last_life = time.time()
                    # 表情目录按配置间隔刷新（默认 24 小时）
                    interval = max(600.0, float(
                        getattr(self.settings.sticker, "sync_interval_sec", 86400)))
                    if time.time() - last_sticker_sync > interval:
                        for rt in self._bots:
                            try:
                                if rt.stickers:
                                    rt.stickers.sync()
                            except Exception as exc:
                                log.debug("表情目录刷新失败: %s", exc)
                        last_sticker_sync = time.time()
                except Exception as exc:
                    log.debug("心跳异常: %s", exc)

        self._ticker = threading.Thread(target=_tick, daemon=True,
                                        name="humanbot-ticker")
        self._ticker.start()
        log.info("后台心跳已启动")

    def _check_proactive(self, rt: BotRuntime) -> None:
        """检查所有私聊会话，判断是否该主动找人。"""
        for conv in self.db.conversations_of(rt.bot_id):
            if conv["chat_type"] != "private":
                continue
            try:
                peer = int(conv["peer_id"])
                last_ts = int(conv["last_ts"] or 0)
                # 从没聊过的会话不主动（避免把好友列表挨个骚扰一遍）
                if last_ts <= 0:
                    continue
                hours = (now() - last_ts) / 3600.0
                # 只在对方在线时考虑
                online = {int(f.get("id") or 0) for f in rt.conn.get_online_friends()} \
                    if rt.conn else set()
                if peer not in online:
                    continue
                st = rt.emotion.state(rt.bot_id, peer, "private")
                import datetime as _dt
                ok, line = rt.account.should_proactive(
                    peer, is_online=True, hours_since=hours,
                    relation=st.relation, hour=_dt.datetime.now().hour,
                    ignored=rt.account._ignored_streak.get(peer, 0),
                )
                if ok and line:
                    rt._send_planned(peer, line, "private", st)
            except Exception as exc:
                log.debug("主动消息检查异常: %s", exc)

    # ------------------------------------------------------------------

    def bots(self) -> List[BotRuntime]:
        return self._bots

    def all_status(self) -> List[Dict[str, Any]]:
        out = []
        for b in self._bots:
            if not b.bot_id:
                continue
            d = b.status()
            # 关系阶段的展示名在 persona 里定义，这里补齐
            try:
                from ..core.persona import STAGE_LABELS
                d["stage_label"] = STAGE_LABELS.get(d.get("relation", ""), "")
            except Exception:
                d["stage_label"] = d.get("relation", "")
            out.append(d)
        return out

    def _first(self) -> Optional[BotRuntime]:
        for b in self._bots:
            if b.bot_id:
                return b
        return None

    def reload_settings(self) -> None:
        """重新加载配置（热更新）。"""
        from ..settings import Settings as _S
        new = _S.load()
        # 保留运行时账号密码
        if not new.boxim.password:
            new.boxim.password = self.settings.boxim.password
        new.boxim.username = new.boxim.username or self.settings.boxim.username
        self.settings = new
        for b in self._bots:
            b.settings = new
            b.brain.reload(new)
            b.memory.settings = new
            b.media.cfg = new.media
            b.output.cfg = new.humanize
            if b.scheduler:
                b.scheduler.cfg = new.humanize
            if b.sleep:
                b.sleep.cfg = new.sleep
            if b.cycle:
                b.cycle.cfg = new.physio
        if self.brain:
            self.brain.reload(new)
        log.info("配置已热更新")

    def reload_persona(self) -> None:
        p = load_persona(self.settings.persona_file)
        for b in self._bots:
            b.persona = p
            b.account.persona = p
            if b.cycle:
                b.cycle.active = bool(p.physio_enabled and self.settings.physio.enabled)
        log.info("人设已热更新（gender=%s, physio=%s）", p.gender, p.physio_enabled)
        BUS.publish("persona", {"event": "reloaded", "name": p.name,
                                "physio_enabled": p.physio_enabled})

    # ------------------------------------------------------------------
    # 控制台动作
    # ------------------------------------------------------------------

    def mute(self, bot_id: int, chat_id: int, is_group: bool) -> None:
        for b in self._bots:
            if b.bot_id == bot_id and b.scheduler:
                b.scheduler.mute(bot_id, chat_id, is_group)

    def unmute(self, bot_id: int, chat_id: int, is_group: bool) -> None:
        for b in self._bots:
            if b.bot_id == bot_id and b.scheduler:
                b.scheduler.unmute(bot_id, chat_id, is_group)

    def is_muted(self, bot_id: int, chat_id: int, is_group: bool) -> bool:
        for b in self._bots:
            if b.bot_id == bot_id and b.scheduler:
                return b.scheduler.is_muted(bot_id, chat_id, is_group)
        return False

    def manual_send(self, peer_id: int, text: str, chat_type: str = "private") -> bool:
        """控制台手动发消息（仍走拟人延迟）。"""
        b = self._first()
        if b is None:
            return False
        st = b.emotion.state(b.bot_id, peer_id, chat_type)
        n = b._send_planned(peer_id, text, chat_type, st)
        return n > 0

    def manual_send_image(self, peer_id: int, prompt: str,
                          chat_type: str = "private") -> Optional[str]:
        b = self._first()
        if b is None or not b.conn:
            return None
        path = b.media.generate(prompt)
        if not path:
            return None
        b.conn.send_image(peer_id, path, chat_type == "group")
        return path

    def action(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行控制台动作。"""
        b = self._first()
        if b is None:
            return {"ok": False, "error": "BOT 未启动"}
        try:
            if name == "set_signature":
                text = str(payload.get("text", ""))
                ok = b.conn.set_signature(text) if b.conn else False
                if ok:
                    b.db.set_setting(f"sig:{b.bot_id}", text)
                return {"ok": ok}
            if name == "set_head_image":
                path = str(payload.get("path", ""))
                ok = b.conn.set_head_image(path) if b.conn else False
                return {"ok": ok}
            if name == "force_sleep":
                if b.sleep:
                    b.sleep.set_state(b.bot_id, "ASLEEP")
                    b._on_sleep()
                return {"ok": True}
            if name == "force_wake":
                if b.sleep:
                    b.sleep.set_state(b.bot_id, "AWAKE")
                    b._on_wake()
                return {"ok": True}
            if name == "sync_offline":
                n = b.conn.sync_offline() if b.conn else 0
                return {"ok": True, "count": n}
            if name == "set_dnd":
                ok = b.account.set_dnd(b.conn, int(payload.get("peer_id", 0)),
                                       bool(payload.get("enabled", True)),
                                       bool(payload.get("is_group", False)))
                return {"ok": ok}
            if name == "add_friend":
                ok = b.account.add_friend(b.conn, int(payload.get("peer_id", 0)),
                                          str(payload.get("remark", "")))
                return {"ok": ok}
            if name == "blacklist":
                ok = b.account.blacklist(b.conn, int(payload.get("peer_id", 0)),
                                         bool(payload.get("enable", True)))
                return {"ok": ok}
            if name == "trigger_life_event":
                ev = b.account.tick_life_events()
                return {"ok": bool(ev), "event": ev}
            if name == "reset_annoyance":
                st = b.emotion.state(b.bot_id, int(payload.get("peer_id", 0)),
                                     payload.get("chat_type", "private"))
                st.annoyance = 0
                st.pending_reconcile = False
                st.cold_until = 0
                b.emotion.persist(st)
                return {"ok": True}
            if name == "test_provider":
                pid = str(payload.get("provider_id", ""))
                cfg = self.settings.model.get_provider(pid)
                if not cfg:
                    return {"ok": False, "error": "供应商不存在"}
                return self.brain.test_provider(cfg) if self.brain else {"ok": False}
            # ---------- 表情包 ----------
            if name == "sync_stickers":
                n = b.stickers.sync(force=True)
                return {"ok": True, "count": n}
            if name == "send_sticker":
                sid = int(payload.get("sticker_id", 0))
                peer_id = int(payload.get("peer_id", 0))
                chat_type = str(payload.get("chat_type", "private"))
                if not sid:
                    # 没指定就随机挑一张
                    st = b.emotion.state(b.bot_id, peer_id, chat_type)
                    picked = b.stickers.pick(
                        bot_id=b.bot_id, chat_id=peer_id,
                        mood=str(payload.get("mood", "")),
                        annoyance=st.annoyance, brain=b.brain)
                    if picked is None:
                        return {"ok": False, "error": "没有可用的表情"}
                    sid = int(picked["id"])
                ok = b.conn.send_sticker(peer_id, sid,
                                         is_group=(chat_type == "group")) \
                    if b.conn else False
                return {"ok": ok, "sticker_id": sid}
            if name == "random_sticker":
                st = b.emotion.state(b.bot_id, int(payload.get("peer_id", 0)),
                                     payload.get("chat_type", "private"))
                s = b.stickers.pick(
                    bot_id=b.bot_id, chat_id=int(payload.get("peer_id", 0)),
                    mood=str(payload.get("mood", "")), annoyance=st.annoyance,
                    brain=b.brain, context=str(payload.get("context", "")))
                return {"ok": bool(s), "sticker": s}
            # ---------- 群聊参与 ----------
            if name == "group_snapshot":
                gid = int(payload.get("group_id", 0))
                if b.interest is None:
                    return {"ok": False, "error": "兴趣模型未启用"}
                return {"ok": True, "data": b.interest.snapshot(b.bot_id, gid)}
            if name == "reset_group_engagement":
                gid = int(payload.get("group_id", 0))
                b.db.execute(
                    "DELETE FROM group_engagement WHERE group_id=?", (gid,))
                b.db.execute(
                    "UPDATE group_rules SET muted_until=0 WHERE group_id=?", (gid,))
                return {"ok": True}
            if name == "set_group_interest":
                gid = int(payload.get("group_id", 0))
                kws = payload.get("keywords") or []
                import json as _json
                b.db.execute(
                    "INSERT INTO group_rules(group_id,bot_id,interest_keywords) "
                    "VALUES(?,?,?) ON CONFLICT(group_id) DO UPDATE SET "
                    "interest_keywords=excluded.interest_keywords",
                    (gid, b.bot_id, _json.dumps(kws, ensure_ascii=False)))
                return {"ok": True}
        except Exception as exc:
            log.error("动作 %s 失败: %s", name, exc)
            return {"ok": False, "error": str(exc)}
        return {"ok": False, "error": f"未知动作: {name}"}

    # ------------------------------------------------------------------
    # 预览与沙盒
    # ------------------------------------------------------------------

    def preview(self, message: str, apply_emotion: bool = True,
                apply_antiai: bool = True) -> Dict[str, Any]:
        """人设编辑器的实时预览（带调试信息）。"""
        b = self._first()
        if b is None:
            return {"ok": False, "error": "BOT 未启动"}

        st = b.emotion.state(b.bot_id, 0, "private")
        import datetime as _dt
        mood = mood_of(st.emotion, _dt.datetime.now().hour)
        system = b._build_prompt(st, message, [], mood, "private", 0)
        history: List[Dict[str, str]] = []

        t0 = time.time()
        try:
            raw = b.brain.complete(TASK_REPLY, system, message,
                                   temperature=0.92, max_tokens=180)
        except LLMError as exc:
            return {"ok": False, "error": f"生成失败: {exc}"}
        gen_ms = int((time.time() - t0) * 1000)

        det = antiai.detect(raw, user_message=message,
                            max_chars=self.settings.humanize.split_max_chars)
        final = raw
        repaired = False
        if apply_antiai:
            if det.action == "patch":
                final = antiai.repair(final, b.brain, det.hits)
                repaired = True
            elif det.action == "regenerate":
                try:
                    final = b.brain.complete(
                        TASK_REPLY,
                        "把下面这句话改成真人聊天会说的，要短、口语、不完整。只输出结果。",
                        final, temperature=0.95, max_tokens=120)
                    repaired = True
                except Exception:
                    pass
            final = antiai.soften_punctuation(
                antiai.inject_noise(final, strength=1.0))
        det2 = antiai.detect(final, user_message=message,
                             max_chars=self.settings.humanize.split_max_chars)

        plan = b.output.plan(final, energy=st.emotion.energy,
                             drowsiness=b.sleep.drowsiness(b.bot_id) if b.sleep else 0,
                             mood=mood)
        return {
            "ok": True,
            "raw": raw,
            "final": final,
            "repaired": repaired,
            "gen_ms": gen_ms,
            "detect_before": det.to_dict(),
            "detect_after": det2.to_dict(),
            "plan": [{"text": p["text"], "delay": round(p["delay"], 2),
                      "typo": p.get("typo", False),
                      "recall": p.get("recall", False)} for p in plan],
            "prompt": system,
            "prompt_len": len(system),
            "mood": mood,
            "emotion": st.emotion.to_json(),
            "provider": self.settings.model.provider_for(TASK_REPLY).id
            if self.settings.model.provider_for(TASK_REPLY) else "",
        }

    def sandbox(self, message: str, rounds: int = 1) -> Dict[str, Any]:
        """沙盒：不连 IM，纯本地跑人格引擎。"""
        b = self._first()
        if b is None:
            return {"ok": False, "error": "BOT 未启动"}
        out: List[Dict[str, Any]] = []
        for i in range(max(1, min(rounds, 20))):
            res = self.preview(message + (f"（第{i + 1}次）" if rounds > 1 else ""),
                               True, True)
            out.append(res)
        # 统计多样性
        texts = [r.get("final", "") for r in out if r.get("ok")]
        uniq = len(set(texts))
        scores = [r["detect_after"]["score"] for r in out if r.get("ok")]
        return {
            "ok": True,
            "results": out,
            "diversity": round(uniq / max(1, len(texts)), 2),
            "avg_score": round(sum(scores) / max(1, len(scores)), 1),
        }

    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        for b in self._bots:
            b.stop()
        self.pool.shutdown(wait=False)
        log.info("管理器已停止")
