"""
键盘模型 AI — 键（Key）数据结构  v1.0

键是记忆网络的最小存储单元。v1.0 重构：键不再存储原文句子，
而是存储以概念原语（primitive）为核心的原子意义结构。

身份由它连接谁定义。无标签、无分区、无向量。
亮度 ∈ {0, 1, 2}：三段式开关，关/中/大。

v1.0 新增（六条路线）：
- primitive + slots（概念依赖 CD）
- evidence_freq + evidence_conf（NARS 证据量）
- prediction_state（HTM 预测）

English:
The minimal storage unit of the memory network: a key holds a primitive
plus slots (conceptual dependency), NARS evidence and HTM prediction state,
with three-level brightness; identity comes from what it connects to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class Key:
    """记忆网络中的一个键。v1.0 概念原语版。"""

    id: int
    content: str               # 原文（保留用于输出参考，不参与匹配）

    # ── v1.0: 概念原语（CD） ──
    primitive: str = ""        # 原语类型: "SEE", "GIVE", "IS", ""（旧键未升级）
    slots: dict[str, int] = field(default_factory=dict)  # 槽位 → 键ID

    # 亮度
    brightness: int = 0        # 当前亮度 ∈ {0, 1, 2}

    # 时间戳
    created_at: float = field(default_factory=time.time)
    last_activated_at: float = 0.0

    # 统计
    activation_count: int = 0
    peak_brightness: int = 0   # 历史最高亮度

    # 休眠
    is_dormant: bool = False
    dormant_since: float | None = None
    original_content: str | None = None  # 休眠压缩前的原始内容

    # 信任
    source: str = "external"   # "master" | "external" | "internal"

    # 孤种键（M2）：零证据、只靠"师父说过这个词"存在的弱键
    is_orphan: bool = False

    # 残留：对话结束后不立即归零，保持微弱激活一段时间
    residual_until: float = 0.0  # 此时间戳之前 decay 不低于 1

    # 遗忘曲线：历次激活的时间戳
    activation_history: list[float] = field(default_factory=list)

    # ── v1.0: NARS 证据量 ──
    evidence_freq: float = 1.0   # 正面证据频率 (0.0–1.0)
    evidence_conf: float = 0.0   # 证据置信度 / 总量 (0.0–1.0)

    # ── v1.0: HTM 预测状态 ──
    prediction_state: int = 0    # 0=休眠 1=预测中 2=确认

    # ── 属性 ──────────────────────────────────

    @property
    def is_active(self) -> bool:
        """键当前是否在意识中（亮度 > 0）。"""
        return self.brightness > 0

    @property
    def is_residual(self) -> bool:
        """键是否处于残留状态。"""
        return self.brightness > 0 and self.residual_until > time.time()

    # ── 激活 / 衰减 ────────────────────────────

    def activate(self, brightness: int = 1):
        """激活键。若从休眠唤醒，恢复原始内容。"""
        if self.is_dormant:
            self._awaken()

        old = self.brightness
        self.brightness = min(brightness, 2)

        if old == 0 and self.brightness > 0:
            now = time.time()
            self.last_activated_at = now
            self.activation_count += 1
            self.activation_history.append(now)

        if self.brightness > self.peak_brightness:
            self.peak_brightness = self.brightness

    def decay(self):
        """自然衰减一档。残留期间不低于 1。"""
        if self.brightness == 0:
            return
        # 残留保护：如果有残留标记且亮度为 1，不归零
        if self.brightness == 1 and self.residual_until > time.time():
            return
        self.brightness -= 1

    def _awaken(self):
        """从休眠中唤醒——恢复原始内容。"""
        if self.original_content is not None:
            self.content = self.original_content
            self.original_content = None
        self.is_dormant = False
        self.dormant_since = None

    # ── 序列化 ──────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "primitive": self.primitive,
            "slots": self.slots,
            "brightness": self.brightness,
            "created_at": self.created_at,
            "last_activated_at": self.last_activated_at,
            "activation_count": self.activation_count,
            "peak_brightness": self.peak_brightness,
            "is_dormant": self.is_dormant,
            "dormant_since": self.dormant_since,
            "original_content": self.original_content,
            "source": self.source,
            "is_orphan": self.is_orphan,
            "residual_until": self.residual_until,
            "activation_history": self.activation_history,
            "evidence_freq": self.evidence_freq,
            "evidence_conf": self.evidence_conf,
            "prediction_state": self.prediction_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Key":
        # 向后兼容：旧数据没有 v1.0 字段时给默认值
        d.setdefault("primitive", "")
        d.setdefault("slots", {})
        d.setdefault("evidence_freq", 1.0)
        d.setdefault("evidence_conf", 0.0)
        d.setdefault("prediction_state", 0)
        d.setdefault("is_orphan", False)
        return cls(**d)

    def __repr__(self) -> str:
        icon = {2: "●", 1: "○", 0: "·"}[self.brightness]
        dorm = "[休眠]" if self.is_dormant else ""
        prim = f"[{self.primitive}]" if self.primitive else ""
        preview = self.content[:35] + ("…" if len(self.content) > 35 else "")
        return f"Key#{self.id} {icon} {dorm} {prim} {preview}"
