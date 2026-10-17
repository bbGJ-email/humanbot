"""全局配置：YAML 文件 + 环境变量覆盖。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# 项目根目录（humanbot/settings.py -> humanbot/ -> 项目根）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
ASSETS_DIR = ROOT / "assets"
CONFIG_FILE = ROOT / "config.yaml"

for _d in (DATA_DIR, LOG_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: Any = None) -> Any:
    """读取环境变量。"""
    return os.environ.get(key, default)


@dataclass
class BoxIMConfig:
    """BoxIM 账号与连接配置。"""

    username: str = ""
    password: str = ""
    base_url: str = "https://www.boximchat.com/"
    ws_url: str = "wss://www.boximchat.com/im"
    ssl_verify: bool = True
    # 睡着时是否真正断开连接（策划案 §6.5.1）
    disconnect_on_sleep: bool = True
    # 登录后是否把 IM 账号昵称改成 persona 的名字。
    # 默认关：改名会动到你的真实账号，而且 IM 昵称和人设名不一致并不影响使用
    #（别名列表会同时认两个名字）。想让账号看起来也叫"沈晚"再打开。
    sync_nickname: bool = False
    # 多账号：要同时跑几个 BOT 时填这里，每项 {username, password}
    # 留空则只用上面的 username/password 单账号。
    accounts: List[Dict[str, str]] = field(default_factory=list)

    def account_list(self) -> List[Dict[str, str]]:
        """归一化后的账号列表。

        - 配了 accounts 就用它
        - 否则退回单账号 username/password
        - 两项都空时返回空列表（调用方据此跳过登录）
        """
        out: List[Dict[str, str]] = []
        for a in (self.accounts or []):
            if not isinstance(a, dict):
                continue
            u = str(a.get("username") or "").strip()
            p = str(a.get("password") or "")
            if u and p:
                out.append({"username": u, "password": p})
        if not out and self.username and self.password:
            out.append({"username": self.username, "password": self.password})
        return out

    def for_account(self, acct: Dict[str, str]) -> "BoxIMConfig":
        """以当前配置为模板，派生某个账号的配置。"""
        return BoxIMConfig(
            username=acct.get("username", ""),
            password=acct.get("password", ""),
            base_url=self.base_url,
            ws_url=self.ws_url,
            ssl_verify=self.ssl_verify,
            disconnect_on_sleep=self.disconnect_on_sleep,
            sync_nickname=self.sync_nickname,
            accounts=[],
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoxIMConfig":
        raw_accounts = d.get("accounts") or []
        accounts: List[Dict[str, str]] = []
        if isinstance(raw_accounts, list):
            for a in raw_accounts:
                if isinstance(a, dict) and a.get("username"):
                    accounts.append({"username": str(a["username"]),
                                     "password": str(a.get("password", ""))})
        return cls(
            username=str(d.get("username") or _env("BOXIM_USERNAME", "")),
            password=str(d.get("password") or _env("BOXIM_PASSWORD", "")),
            base_url=d.get("base_url") or _env("BOXIM_BASE_URL", cls.base_url),
            ws_url=d.get("ws_url") or _env("BOXIM_WS_URL", cls.ws_url),
            ssl_verify=bool(d.get("ssl_verify", True)),
            disconnect_on_sleep=bool(d.get("disconnect_on_sleep", True)),
            sync_nickname=bool(d.get("sync_nickname", False)),
            accounts=accounts,
        )


@dataclass
class ProviderConfig:
    """单个 AI 供应商配置。"""

    id: str = ""
    name: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    temperature: float = 0.9
    max_tokens: int = 400
    timeout: int = 60
    # 是否支持 embeddings（用于记忆向量化）
    supports_embedding: bool = False
    embedding_model: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProviderConfig":
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", d.get("id", ""))),
            base_url=str(d.get("base_url", "")),
            api_key=str(d.get("api_key", "")),
            model=str(d.get("model", "")),
            enabled=bool(d.get("enabled", True)),
            temperature=float(d.get("temperature", 0.9)),
            max_tokens=int(d.get("max_tokens", 400)),
            timeout=int(d.get("timeout", 60)),
            supports_embedding=bool(d.get("supports_embedding", False)),
            embedding_model=str(d.get("embedding_model", "")),
        )

    def to_dict(self, mask_key: bool = False) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "enabled": self.enabled,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "supports_embedding": self.supports_embedding,
            "embedding_model": self.embedding_model,
        }
        d["api_key"] = mask_api_key(self.api_key) if mask_key else self.api_key
        return d


# 任务类型常量（路由用）
TASK_REPLY = "reply_generate"
TASK_PROBE = "probe_reply"
TASK_INTENT = "intent_rewrite"
TASK_MEMORY = "memory_extract"
TASK_EMOTION = "emotion_update"
TASK_ANTIAI = "anti_ai_fix"
TASK_VISION = "vision"
TASK_IMAGE_PROMPT = "image_prompt"
TASK_EMBEDDING = "embedding"

ALL_TASKS = [
    TASK_REPLY,
    TASK_PROBE,
    TASK_INTENT,
    TASK_MEMORY,
    TASK_EMOTION,
    TASK_ANTIAI,
    TASK_VISION,
    TASK_IMAGE_PROMPT,
    TASK_EMBEDDING,
]

TASK_LABELS = {
    TASK_REPLY: "正式回复生成",
    TASK_PROBE: "探测式短回复",
    TASK_INTENT: "意图改写",
    TASK_MEMORY: "记忆抽取",
    TASK_EMOTION: "情绪增量计算",
    TASK_ANTIAI: "反AI改写",
    TASK_VISION: "图片理解",
    TASK_IMAGE_PROMPT: "图生文提示词",
    TASK_EMBEDDING: "记忆向量嵌入",
}


@dataclass
class ModelConfig:
    """多供应商 + 任务路由。"""

    providers: List[ProviderConfig] = field(default_factory=list)
    routing: Dict[str, str] = field(default_factory=dict)
    # 故障转移链：主供应商报 429 / 5xx / 超时时，按顺序改试这里的供应商。
    # 免费模型站普遍有并发限流，单靠换一个模型治不了根，
    # 配一条备用链才能让回复不因为偶发限流而直接失败。
    failover: List[str] = field(default_factory=list)
    daily_budget_usd: float = 5.0
    budget_action: str = "degrade"  # degrade | stop

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        providers = [ProviderConfig.from_dict(p) for p in d.get("providers", [])]
        routing = dict(d.get("routing", {}))
        # 未配置的任务默认走第一个可用 provider
        if providers:
            default_id = providers[0].id
            for t in ALL_TASKS:
                routing.setdefault(t, default_id)
        return cls(
            providers=providers,
            routing=routing,
            failover=[str(x) for x in (d.get("failover") or []) if str(x).strip()],
            daily_budget_usd=float(d.get("daily_budget_usd", 5.0)),
            budget_action=str(d.get("budget_action", "degrade")),
        )

    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        for p in self.providers:
            if p.id == provider_id and p.enabled:
                return p
        return None

    def provider_for(self, task: str) -> Optional[ProviderConfig]:
        """按任务取供应商，找不到则回退第一个可用。"""
        pid = self.routing.get(task, "")
        p = self.get_provider(pid)
        if p:
            return p
        for cand in self.providers:
            if cand.enabled:
                return cand
        return None


@dataclass
class HumanizeConfig:
    """拟人化参数（控制台可热调节）。"""

    # 打字延迟：base + chars * per_char，封顶 cap（秒）
    typing_base: float = 0.8
    typing_per_char: float = 0.06
    typing_cap: float = 4.0
    # 分条
    split_max_chars: int = 45
    split_max_parts: int = 3
    split_gap_min: float = 0.5
    split_gap_max: float = 1.5
    # 阅读延迟
    read_delay_min: float = 0.5
    read_delay_max: float = 2.0
    # 错别字概率
    typo_rate: float = 0.05
    # 撤回重发概率
    recall_rate: float = 0.05
    # 消息合并窗口（秒）
    merge_window_min: float = 1.5
    merge_window_max: float = 3.0
    # 防封
    daily_send_limit: int = 300
    min_send_interval: float = 1.0
    # 群聊
    group_at_weight: float = 1.0
    group_mention_weight: float = 0.8
    group_context_weight: float = 0.6
    group_interest_weight: float = 0.25
    group_other_weight: float = 0.05
    group_burst_penalty: float = 0.5

    # ---- 已读不回（允许她看完不接话）----
    skip_enabled: bool = True
    skip_base_prob: float = 0.12          # 普通消息的基础跳过概率
    skip_filler_prob: float = 0.65        # 「嗯/哦/好的/哈哈/在吗」这类
    skip_repeat_prob: float = 0.5         # 和上一条几乎重复
    skip_max_streak: int = 2              # 连续不回上限，到数必回
    skip_annoy_boost: float = 0.25        # 有点烦时加成（annoyance>=30）
    skip_low_energy_boost: float = 0.1
    skip_night_boost: float = 0.1
    skip_quick_reply_boost: float = 0.1   # 距上次回复不足 15 秒
    skip_group_proactive_boost: float = 0.15
    skip_cap: float = 0.85                # 单条跳过概率上限

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HumanizeConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class SleepConfig:
    """睡眠与作息参数（策划案 §6.5.9）。"""

    enabled: bool = True
    base_sleep_minute: int = 23 * 60 + 50  # 基准入睡 23:50
    sleep_jitter_sigma: int = 25
    sleep_earliest: int = 22 * 60 + 30
    sleep_latest: int = 27 * 60 + 30  # 次日 03:30
    never_on_the_hour: bool = True
    duration_mean: float = 7.8
    duration_sigma: float = 0.9
    duration_min: float = 5.5
    duration_max: float = 11.0
    drowsy_lead_min: int = 45  # 入睡前多久进 DROWSY
    waking_lead_min: int = 10
    p_all_nighter: float = 0.07
    p_no_goodbye: float = 0.20
    p_sleep_back: float = 0.20
    p_oversleep: float = 0.08
    p_wake_early: float = 0.05

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SleepConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class PhysioConfig:
    """生理周期参数（策划案 §6.7）。"""

    enabled: bool = True  # 总开关，且仍受 gender 门控
    cycle_min: int = 23
    cycle_max: int = 33
    period_days: int = 5
    light_cycle_rate: float = 0.20  # 20% 概率这个月很轻

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysioConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class MemoryConfig:
    """记忆层参数。"""

    enabled: bool = True
    episodic_top_k: int = 5
    profile_max_items: int = 40
    working_turns: int = 12
    # 是否用 API 做向量嵌入（关闭则退化为关键词打分）
    embedding_enabled: bool = True
    embedding_provider: str = ""  # 空则用 routing 里的 embedding 任务
    embedding_dim: int = 1024

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class MediaConfig:
    """图片系统参数。"""

    enabled: bool = True
    # 生成
    gen_endpoint: str = "https://image.pollinations.ai/prompt/{prompt}"
    gen_model: str = "flux"
    gen_width: int = 768
    gen_height: int = 768
    gen_retry: int = 3
    gen_rate_gap: float = 15.0  # 实测限流间隔
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    # 检索
    search_engine: str = "bing"
    cache_enabled: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MediaConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class ConsoleConfig:
    """Web 控制台配置。"""

    enabled: bool = True
    host: str = "0.0.0.0"  # 绑定外网
    port: int = 0  # 0 = 自动探测空闲端口
    port_scan_from: int = 8800
    port_scan_to: int = 8900
    username: str = "admin"
    password: str = ""  # 空 = 启动时自动生成强密码
    session_hours: int = 24

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConsoleConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        env_port = _env("HUMANBOT_PORT")
        if env_port:
            try:
                inst.port = int(env_port)
            except ValueError:
                pass
        if _env("HUMANBOT_HOST"):
            inst.host = str(_env("HUMANBOT_HOST"))
        return inst


@dataclass
class StickerConfig:
    """表情包系统参数（策划案 §6.6）。"""

    enabled: bool = True
    # 文字后补一个表情的概率
    append_prob: float = 0.18
    # 只发一个表情、不回文字的概率（对方发了纯情绪内容时）
    solo_prob: float = 0.12
    # 生气/冷战状态下只回表情的概率（比文字更冷）
    cold_solo_prob: float = 0.55
    # 连续两条消息之间不重复发表情的冷却
    cooldown_sec: float = 25.0
    # 每天最多发多少张
    daily_limit: int = 60
    # 同一张表情在这个窗口（秒）内不重复发给同一个人
    repeat_window_sec: float = 7 * 86400.0
    # 是否用 LLM 在候选里挑（关闭则纯关键词匹配，省成本）
    llm_pick: bool = True
    # 候选数量（喂给 LLM 的名字条数）
    llm_candidates: int = 12
    # 启动时是否自动同步表情目录
    auto_sync: bool = True
    # 同步间隔（秒）
    sync_interval_sec: float = 24 * 3600.0
    # 只用名字有语义的专辑（数字命名的专辑不参与关键词匹配）
    descriptive_only: bool = True
    # 优先使用的专辑 ID（留空 = 全部）
    preferred_albums: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StickerConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                if k == "preferred_albums":
                    try:
                        inst.preferred_albums = [int(x) for x in (v or [])]
                    except (TypeError, ValueError):
                        pass
                    continue
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class GroupConfig:
    """群聊参与度参数（策划案 §7）。"""

    enabled: bool = True
    # 兴趣得分映射到的权重上限
    interest_max_weight: float = 0.72
    # 兴趣得分忽略阈值（低于此值不走兴趣通道）
    interest_min_score: float = 0.22
    # 参与惯性：最近发言过，下一条消息的加成
    momentum_bonus: float = 0.30
    # 惯性衰减窗口（秒）——超过这个时间没说话，惯性清零
    momentum_window_sec: float = 600.0
    # 话题连续性：同一话题在窗口内的累积加成上限
    topic_boost_max: float = 0.25
    topic_window_sec: float = 420.0
    # 连续插话惩罚：短时间内已经回过又回，降权
    interrupt_penalty: float = 0.45
    interrupt_window_sec: float = 90.0
    # 群里完全无关话题的最低下限
    floor_weight: float = 0.05
    # 主动发起（无人 @ 也开口）需要的最小权重
    proactive_min_weight: float = 0.42
    # 每群每小时主动开口上限
    proactive_hourly_cap: int = 4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GroupConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class VoiceConfig:
    """语音参数（通话用）。"""

    enabled: bool = True

    # ---- TTS（edge-tts）----
    tts_voice: str = "zh-CN-XiaoxiaoNeural"   # 晓晓（成熟音色）
    tts_rate: str = "+0%"                     # 基准语速
    tts_pitch: str = "+0Hz"                   # 基准音调
    tts_volume: str = "+0%"                   # 基准音量
    tts_timeout: int = 30
    tts_concurrency: int = 4                  # 预热并发度（大了容易触发限流）

    # 情绪 → 语音参数偏移（叠加在基准之上）。真人说话随情绪起伏，
    # 语调一成不变是 TTS 最容易露馅的地方。
    mood_rate: Dict[str, str] = field(default_factory=lambda: {
        "开心": "+8%", "平静": "+0%", "低落": "-10%",
        "烦躁": "+12%", "疲惫": "-14%", "刚醒": "-18%",
    })
    mood_pitch: Dict[str, str] = field(default_factory=lambda: {
        "开心": "+2Hz", "平静": "+0Hz", "低落": "-3Hz",
        "烦躁": "+1Hz", "疲惫": "-2Hz", "刚醒": "-4Hz",
    })
    mood_volume: Dict[str, str] = field(default_factory=lambda: {
        "开心": "+0%", "平静": "+0%", "低落": "-8%",
        "烦躁": "+5%", "疲惫": "-12%", "刚醒": "-15%",
    })

    # 口语节奏：句间停顿、长句换气（毫秒区间）
    sentence_gap_ms: List[int] = field(default_factory=lambda: [200, 500])
    breath_gap_ms: int = 150
    breath_every_chars: int = 18              # 每多少字插一次换气

    # ---- STT（模型站）----
    stt_primary: str = "telespeechasr"
    stt_fallback: str = "sensevoicesmall"
    stt_timeout: int = 30

    # ---- 音频格式（与 BoxIM WebRTC 一致，不要改）----
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VoiceConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class CallConfig:
    """通话行为参数（策划案 §5）。"""

    enabled: bool = True

    # ---- 视频通话 ----
    # BOT 没有摄像头。发静态图一动就露馅，所以统一用"摄像头坏了"挡掉。
    allow_video: bool = False
    video_excuse: str = "我摄像头好像坏了，语音行吗"
    # 只向对端提供 TURN relay candidate（剔除内网 host / srflx）。
    # 服务器入站 UDP 被云安全组全端口阻断时，内网地址对公网对端不可达，
    # 留在 answer 里只会稀释对端的 ICE 检查。有公网直连能力时可关掉。
    ice_relay_only: bool = True

    # ---- 接听策略：关系 + 情绪 + 时段 综合判断 ----
    base_answer_prob: float = 0.92        # 基线接听率（留随机性，永远秒接最假）
    min_relation: str = "acquaintance"    # 关系低于此档不接
    max_annoyance: int = 60               # 不满度超过此值不接（还在生气）
    night_quiet_start: int = 0            # 深夜静默起点 00:00
    night_quiet_end: int = 6 * 60         # ~06:00
    night_min_relation: str = "close"     # 深夜只有亲密关系才接
    cooldown_sec: int = 300               # 刚挂断后的冷却期

    # ---- 振铃延迟（秒）：拟人化的灵魂，绝不秒接 ----
    ring_delay_min: float = 2.5
    ring_delay_max: float = 7.0
    ring_delay_busy: float = 8.0          # 在忙
    ring_delay_sleepy: float = 12.0       # 刚睡醒
    ring_timeout: int = 45                # 响铃多久算未接

    # ---- 对话循环 ----
    max_duration_min: int = 30            # 单次通话硬上限
    soft_duration_min: int = 12           # 软上限，之后开始收尾
    vad_silence_ms: int = 600             # 静音多久判定"说完了"
    vad_min_speech_ms: int = 250          # 最短有效语音段
    filler_after_ms: int = 800            # LLM 超过这么久先播填充词
    reply_max_chars: int = 25             # 口语回复长度上限
    interrupt_enabled: bool = True        # 允许被打断（真人会停下听）

    # ---- 限额 ----
    daily_limit: int = 10                 # 每日通话次数
    unanswered_reply_after_wake: bool = True   # 醒来后回拨/回消息

    # ---- 通话后的情绪结算（语音比文字影响大）----
    intimacy_gain_per_min: float = 0.4
    energy_cost_per_min: float = 0.8
    anger_gain: int = 15                  # 通话里吵起来比文字更重

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CallConfig":
        inst = cls()
        for k, v in (d or {}).items():
            if hasattr(inst, k):
                try:
                    setattr(inst, k, type(getattr(inst, k))(v))
                except (TypeError, ValueError):
                    pass
        return inst


@dataclass
class Settings:
    """聚合配置。"""

    boxim: BoxIMConfig = field(default_factory=BoxIMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    humanize: HumanizeConfig = field(default_factory=HumanizeConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    physio: PhysioConfig = field(default_factory=PhysioConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    sticker: StickerConfig = field(default_factory=StickerConfig)
    group: GroupConfig = field(default_factory=GroupConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    call: CallConfig = field(default_factory=CallConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    db_path: str = str(DATA_DIR / "humanbot.db")
    log_level: str = "INFO"
    persona_file: str = str(ROOT / "persona.yaml")

    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        p = Path(path) if path else CONFIG_FILE
        raw: Dict[str, Any] = {}
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        s = cls(
            boxim=BoxIMConfig.from_dict(raw.get("boxim", {})),
            model=ModelConfig.from_dict(raw.get("model", {})),
            humanize=HumanizeConfig.from_dict(raw.get("humanize", {})),
            sleep=SleepConfig.from_dict(raw.get("sleep", {})),
            physio=PhysioConfig.from_dict(raw.get("physio", {})),
            memory=MemoryConfig.from_dict(raw.get("memory", {})),
            media=MediaConfig.from_dict(raw.get("media", {})),
            sticker=StickerConfig.from_dict(raw.get("sticker", {})),
            group=GroupConfig.from_dict(raw.get("group", {})),
            console=ConsoleConfig.from_dict(raw.get("console", {})),
            voice=VoiceConfig.from_dict(raw.get("voice", {})),
            call=CallConfig.from_dict(raw.get("call", {})),
            db_path=raw.get("db_path", str(DATA_DIR / "humanbot.db")),
            log_level=str(raw.get("log_level", "INFO")).upper(),
            persona_file=raw.get("persona_file", str(ROOT / "persona.yaml")),
        )
        s._raw = raw
        # 环境变量兜底
        if _env("HUMANBOT_DB"):
            s.db_path = str(_env("HUMANBOT_DB"))
        return s

    def save(self, path: Optional[Path] = None) -> None:
        """回写配置（保留未识别的字段）。"""
        p = Path(path) if path else CONFIG_FILE
        raw = dict(self._raw)
        raw["boxim"] = {
            "username": self.boxim.username,
            "password": self.boxim.password,
            "base_url": self.boxim.base_url,
            "ws_url": self.boxim.ws_url,
            "ssl_verify": self.boxim.ssl_verify,
            "disconnect_on_sleep": self.boxim.disconnect_on_sleep,
            "sync_nickname": self.boxim.sync_nickname,
            "accounts": self.boxim.accounts,
        }
        raw["model"] = {
            "providers": [p_.to_dict() for p_ in self.model.providers],
            "routing": self.model.routing,
            "failover": self.model.failover,
            "daily_budget_usd": self.model.daily_budget_usd,
            "budget_action": self.model.budget_action,
        }
        raw["humanize"] = self.humanize.__dict__
        raw["sleep"] = self.sleep.__dict__
        raw["physio"] = self.physio.__dict__
        raw["memory"] = self.memory.__dict__
        raw["media"] = self.media.__dict__
        raw["sticker"] = self.sticker.__dict__
        raw["group"] = self.group.__dict__
        raw["voice"] = self.voice.__dict__
        raw["call"] = self.call.__dict__
        raw["console"] = {
            "enabled": self.console.enabled,
            "host": self.console.host,
            "port": self.console.port,
            "port_scan_from": self.console.port_scan_from,
            "port_scan_to": self.console.port_scan_to,
            "username": self.console.username,
            # 密码只以 PBKDF2 哈希存在 DB 里；这里永远留空，
            # 避免把明文密码写进会被备份/分享的配置文件。
            "password": "",
            "session_hours": self.console.session_hours,
        }
        raw["db_path"] = self.db_path
        raw["log_level"] = self.log_level
        raw["persona_file"] = self.persona_file
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)


def mask_api_key(key: str) -> str:
    """API Key 掩码，用于前端展示。"""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * 8}{key[-4:]}"
