"""
键盘模型 AI — 电线（Wire）数据结构  v1.0

键与键之间的关联边。有粗细权重。
粗细决定传导效率：线越粗，电流越容易通过。

Hebbian 规则：一起亮的键 → 电线加粗。
遗忘规则：长期不同时亮的键 → 电线缓慢变细（但永不断）。

v1.0 新增（Copycat + HTM）：
- relation_type + slippage_distance（概念滑移）
- context（时序上下文，HTM 预测）
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


# ── 常量 ──────────────────────────────────────

MIN_THICKNESS = 0.001   # 线永不断——一根丝连着
INITIAL_THICKNESS = 1.0  # 新线默认粗细

# ── v1.0: 电线关系类型 ──
RELATION_TYPES = (
    "same-as",       # 完全等同（同原语键之间）
    "similar-to",    # 同类相似
    "opposite-of",   # 对立
    "part-of",       # 部分-整体
    "cause-of",      # 因果
)

# ── v1.0: 关系类型 → 默认滑移距离 ──
SLIPPAGE_DISTANCE: dict[str, int] = {
    "same-as": 0,
    "similar-to": 1,
    "part-of": 1,
    "cause-of": 1,
    "opposite-of": 2,
}

# ── 传导极性（三进制）：关系类型 → 激活/抑制/无 ──
# +1 激活（相似/部分/因果/等同），-1 抑制（对立），0 无关联。
# 极性是关系类型的固定属性，不是随机数、不是阈值——
# 与"亮度"（动态连续值）和"厚度"（力度）是两回事。
# Wire 不加 polarity 字段：极性由 relation_type 查此表派生（避免冗余状态）。
RELATION_POLARITY: dict[str, int] = {
    "same-as": +1,
    "similar-to": +1,
    "part-of": +1,
    "cause-of": +1,
    "opposite-of": -1,
}


@dataclass
class Wire:
    """键与键之间的电线。v1.0 增加概念滑移和时序上下文。"""

    key_from: int              # 源键 ID
    key_to: int                # 目标键 ID
    thickness: float = INITIAL_THICKNESS

    # 时间戳
    created_at: float = field(default_factory=time.time)
    last_strengthened_at: float = field(default_factory=time.time)

    # 统计
    co_activation_count: int = 0  # 两键同时亮过的总次数

    # 双向传导
    is_bidirectional: bool = True

    # ── v1.0: Copycat 概念滑移 ──
    relation_type: str = ""           # same-as/similar-to/opposite-of/part-of/cause-of
    slippage_distance: int = 0        # 滑移距离 (0–2)，只在高温时启用

    # ── v1.0: HTM 时序上下文 ──
    context: dict[int, int] = field(default_factory=dict)  # {前置键ID: 共现次数}

    # ── 属性 ──────────────────────────────────

    @property
    def is_effective(self) -> bool:
        """电线是否还能有效传导（太细则几乎不能）。"""
        return self.thickness >= 0.01

    # ── 操作 ──────────────────────────────────

    def strengthen(self, delta: float):
        """加粗电线。"""
        self.thickness += delta
        self.co_activation_count += 1
        self.last_strengthened_at = time.time()

    def weaken(self, decay: float):
        """变细电线，但不低于 MIN_THICKNESS。"""
        self.thickness = max(self.thickness - decay, MIN_THICKNESS)

    # ── 序列化 ──────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "key_from": self.key_from,
            "key_to": self.key_to,
            "thickness": self.thickness,
            "created_at": self.created_at,
            "last_strengthened_at": self.last_strengthened_at,
            "co_activation_count": self.co_activation_count,
            "is_bidirectional": self.is_bidirectional,
            "relation_type": self.relation_type,
            "slippage_distance": self.slippage_distance,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Wire":
        # 向后兼容：旧数据没有 v1.0 字段时给默认值
        d.setdefault("relation_type", "")
        d.setdefault("slippage_distance", 0)
        d.setdefault("context", {})
        return cls(**d)

    def __repr__(self) -> str:
        bar = "█" * min(int(self.thickness * 2), 10)
        return f"Wire({self.key_from}↔{self.key_to}) {bar} {self.thickness:.3f}"
