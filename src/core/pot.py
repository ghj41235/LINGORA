"""
键盘模型 AI — 花盆（Pot）

花盆是键网络的软隔离容器。每个盆内有独立的 KeyNetwork 实例，
盆内传导规则与键盘模型完全一致。盆之间通过桥接线进行有限联想。

盆的生命周期：创建 → 生长 → 可能拆分 / 合并 / 休眠。
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network import KeyNetwork
    from .key import Key


# ── 盆参数 ──────────────────────────────────

POT_MAX_KEYS = 800
POT_MIN_KEYS = 50
POT_SPLIT_WIRE_THRESHOLD = 0.4
POT_SPLIT_MIN_RATIO = 0.20
POT_SIGNATURE_DIM = 64
POT_SIGNATURE_UPDATE_INTERVAL = 50
POT_MERGE_MIN_AGE_DAYS = 7
POT_MERGE_DECLINE_WINDOW_DAYS = 14
POT_MERGE_NO_GROWTH_HOURS = 48
POT_DORMANCY_DAYS = 30
POT_CREATE_TOPIC_DISTANCE = 0.7   # 新文档到盆的最小余弦距离
POT_CREATE_MIN_KEYS = 10          # 新建盆最少键数
POT_MERGE_RECHECK_COOLDOWN = 14 * 86400  # 合并拒绝后重新检查间隔（秒）


# ═══════════════════════════════════════════════
# Pot 数据结构
# ═══════════════════════════════════════════════

@dataclass
class Pot:
    """一个花盆：包含一个独立的键子网络。v1.0 增加注意力权重。"""

    id: str
    network: "KeyNetwork"

    # 身份
    source_type: str = "dialogue"       # "dialogue" | "ingest" | "internal"
    topic_signature: list[float] | None = None  # 64 维 bigram TF 向量
    topic_label: str = ""

    # 元数据
    created_at: float = 0.0
    last_activated_at: float = 0.0
    key_count_history: list[tuple[float, int]] = field(default_factory=list)
    _keys_since_signature_update: int = 0

    # 状态
    is_dormant: bool = False
    merge_protected_until: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if not self.topic_label:
            self.topic_label = self.source_type

    # ── 主题签名 ─────────────────────────────

    def calculate_signature(self) -> list[float]:
        """计算盆的主题签名（64 维 bigram TF 向量，L2 归一化）。"""
        all_text = ""
        for key in self.network.all_keys():
            if not key.is_dormant:
                all_text += key.content

        if not all_text:
            self.topic_signature = [0.0] * POT_SIGNATURE_DIM
            return self.topic_signature

        # 提取 bigram
        bigrams: list[str] = []
        for i in range(len(all_text) - 1):
            bigrams.append(all_text[i:i + 2])

        counter = Counter(bigrams)
        total = sum(counter.values())

        # 取 top-64 bigram → 固定维度向量
        top = [bg for bg, _ in counter.most_common(POT_SIGNATURE_DIM)]

        vec = [0.0] * POT_SIGNATURE_DIM
        for i, bg in enumerate(top):
            vec[i] = counter[bg] / total

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        self.topic_signature = vec
        self._keys_since_signature_update = 0
        return vec

    def maybe_update_signature(self):
        """如果需要（新增键数到达阈值），更新主题签名。"""
        self._keys_since_signature_update += 1
        if self._keys_since_signature_update >= POT_SIGNATURE_UPDATE_INTERVAL:
            self.calculate_signature()

    def topic_distance_to(self, other: "Pot") -> float:
        """与另一个盆的主题距离（1 - 余弦相似度）。[0, 2]，越小越近。"""
        a = self.topic_signature
        b = other.topic_signature
        if a is None:
            a = self.calculate_signature()
        if b is None:
            b = other.calculate_signature()
        if a is None or b is None:
            return 1.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 1.0

        cosine = dot / (norm_a * norm_b)
        return 1.0 - cosine

    # ── 拆分 ─────────────────────────────────

    def needs_split(self) -> bool:
        """检查盆是否需要拆分。"""
        return self.network.key_count > POT_MAX_KEYS

    def split(self) -> list["Pot"] | None:
        """
        将盆拆分为多个子盆。返回子盆列表；如果无法拆分则返回 None。

        算法：
        1. 只保留粗度 ≥ POT_SPLIT_WIRE_THRESHOLD 的电线
        2. 在子图上用 BFS 找连通分量
        3. 过滤太小的分量 → 合并到最近的
        4. 每个分量 → 一个子盆
        """
        nw = self.network
        total = nw.key_count

        # 构建强电线邻接表
        adj: dict[int, list[int]] = {k.id: [] for k in nw.all_keys()}
        for (a, b), wire in nw.wires.items():
            if wire.thickness >= POT_SPLIT_WIRE_THRESHOLD:
                adj[a].append(b)
                adj[b].append(a)

        # BFS 找连通分量
        visited: set[int] = set()
        components: list[list[int]] = []

        for kid in adj:
            if kid not in visited:
                comp: list[int] = []
                queue = [kid]
                visited.add(kid)
                while queue:
                    node = queue.pop(0)
                    comp.append(node)
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                components.append(comp)

        # 处理孤立键（无强电线的键）
        all_ids = set(k.id for k in nw.all_keys())
        assigned = set(sum(components, []))
        unassigned = all_ids - assigned

        if unassigned:
            if components:
                for uid in unassigned:
                    # 找到与这个键有最强电线的组件
                    best_comp: list[int] | None = None
                    best_strength = 0.0
                    for comp in components:
                        for cid in comp:
                            w = nw.get_wire(uid, cid)
                            if w and w.thickness > best_strength:
                                best_strength = w.thickness
                                best_comp = comp
                    if best_comp is not None:
                        best_comp.append(uid)
                    else:
                        min(components, key=len).append(uid)
            else:
                # 没有任何强电线 → 无法拆分
                return None

        if not components:
            return None

        # 过滤太小的组件
        min_size = max(2, int(total * POT_SPLIT_MIN_RATIO))
        viable = [c for c in components if len(c) >= min_size]
        too_small = [c for c in components if len(c) < min_size]

        for small in too_small:
            if not viable:
                viable.append(small)
                continue
            # 合并到键重叠最多的组件
            best = max(viable, key=lambda vc:
                sum(1 for sid in small for vid in vc
                    if nw.get_wire(sid, vid) is not None))
            best.extend(small)

        if len(viable) <= 1:
            return None  # 只有一个组件，不需要拆分

        # 创建子盆
        sub_pots = []
        for i, comp in enumerate(viable):
            from .network import KeyNetwork

            sub = Pot(
                id=f"{self.id}#{i + 1}",
                network=KeyNetwork(),
                source_type=self.source_type,
                topic_label=f"{self.topic_label}-{i + 1}",
                merge_protected_until=time.time() + POT_MERGE_MIN_AGE_DAYS * 86400,
            )

            # 迁移键（保持 ID 不变，因为 KeyNetwork 独立）
            id_map: dict[int, int] = {}  # old_id → new_id
            for kid in comp:
                old_key = nw.get_key(kid)
                if old_key is None:
                    continue
                new_key = sub.network.create_key(
                    content=old_key.content,
                    source=old_key.source,
                    brightness=0,
                )
                # 复制元数据
                new_key.activation_count = old_key.activation_count
                new_key.activation_history = old_key.activation_history.copy()
                new_key.last_activated_at = old_key.last_activated_at
                new_key.peak_brightness = old_key.peak_brightness
                # v1.0 字段
                new_key.primitive = old_key.primitive
                new_key.slots = dict(old_key.slots)
                new_key.evidence_freq = old_key.evidence_freq
                new_key.evidence_conf = old_key.evidence_conf
                new_key.prediction_state = old_key.prediction_state
                id_map[kid] = new_key.id

            # 迁移盆内电线（含 v1.0 新字段）
            for kid in comp:
                new_id = id_map.get(kid)
                if new_id is None:
                    continue
                for (a, b), wire in nw.wires.items():
                    if a == kid and b in comp:
                        other_new = id_map.get(b)
                        if other_new is not None:
                            w = sub.network.set_wire(new_id, other_new, thickness=wire.thickness)
                            _copy_wire_fields(w, wire)
                    elif b == kid and a in comp:
                        other_new = id_map.get(a)
                        if other_new is not None:
                            w = sub.network.set_wire(other_new, new_id, thickness=wire.thickness)
                            _copy_wire_fields(w, wire)

            # 存储 ID 映射表（供 Garden 更新桥接）
            sub._split_id_map = id_map

            sub.calculate_signature()
            sub_pots.append(sub)

        return sub_pots

    # ── 合并条件检查 ──────────────────────────

    def can_be_merged(self) -> bool:
        """检查盆是否具备合并资格（三个条件）。"""
        now = time.time()

        # ① 保护期已过
        if now < self.merge_protected_until:
            return False

        # ② 键数 < 50 或在过去 14 天内持续下降
        if self.network.key_count >= POT_MIN_KEYS:
            # 检查是否在持续下降
            if len(self.key_count_history) < 2:
                return False
            recent = [(t, c) for t, c in self.key_count_history
                      if now - t <= POT_MERGE_DECLINE_WINDOW_DAYS * 86400]
            if len(recent) < 2:
                return False
            # 最近的键数均小于之前的 → 持续下降
            counts = [c for _, c in recent]
            if not all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)):
                return False

        # ③ 过去 48 小时内没有增长
        recent_growth = [(t, c) for t, c in self.key_count_history
                         if now - t <= POT_MERGE_NO_GROWTH_HOURS * 3600]
        if recent_growth and recent_growth[-1][1] > recent_growth[0][1]:
            return False

        return True

    # ── 激活 ─────────────────────────────────

    def activate(self):
        """标记盆为活跃状态。若休眠则唤醒。"""
        self.last_activated_at = time.time()
        if self.is_dormant:
            self.is_dormant = False

    # ── 快照 ─────────────────────────────────

    @property
    def key_count(self) -> int:
        return self.network.key_count

    def record_key_count(self):
        """记录当前键数到历史快照。"""
        self.key_count_history.append((time.time(), self.network.key_count))
        # 只保留近 60 天的记录
        cutoff = time.time() - 60 * 86400
        self.key_count_history = [(t, c) for t, c in self.key_count_history if t > cutoff]

    def snapshot(self) -> dict:
        """盆的可序列化快照。"""
        return {
            "id": self.id,
            "network": self.network.snapshot(),
            "source_type": self.source_type,
            "topic_signature": self.topic_signature,
            "topic_label": self.topic_label,
            "created_at": self.created_at,
            "last_activated_at": self.last_activated_at,
            "key_count_history": self.key_count_history,
            "_keys_since_signature_update": self._keys_since_signature_update,
            "is_dormant": self.is_dormant,
            "merge_protected_until": self.merge_protected_until,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "Pot":
        """从快照恢复盆。"""
        from .network import KeyNetwork

        pot = cls(
            id=data["id"],
            network=KeyNetwork.from_snapshot(data["network"]),
            source_type=data.get("source_type", "dialogue"),
            topic_label=data.get("topic_label", ""),
            created_at=data.get("created_at", time.time()),
            last_activated_at=data.get("last_activated_at", 0.0),
            is_dormant=data.get("is_dormant", False),
            merge_protected_until=data.get("merge_protected_until", 0.0),
        )
        pot.key_count_history = data.get("key_count_history", [])
        pot._keys_since_signature_update = data.get("_keys_since_signature_update", 0)
        pot.topic_signature = data.get("topic_signature")

        return pot

    def __repr__(self) -> str:
        status = "💤" if self.is_dormant else "🌱"
        return (f"Pot({self.id} {status} {self.source_type}/{self.topic_label} "
                f"keys={self.key_count})")


# ═══════════════════════════════════════════════
# 文档主题签名（独立函数，用于投喂路由）
# ═══════════════════════════════════════════════

def compute_document_signature(fragments: list[str]) -> list[float]:
    """
    计算一批投喂碎片的主题签名向量（64 维 bigram TF）。

    用于与已有盆比较，决定投喂去向。
    """
    all_text = "".join(fragments)
    if not all_text:
        return [0.0] * POT_SIGNATURE_DIM

    bigrams = []
    for i in range(len(all_text) - 1):
        bigrams.append(all_text[i:i + 2])

    counter = Counter(bigrams)
    total = sum(counter.values())

    top = [bg for bg, _ in counter.most_common(POT_SIGNATURE_DIM)]

    vec = [0.0] * POT_SIGNATURE_DIM
    for i, bg in enumerate(top):
        vec[i] = counter[bg] / total

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """两个等长向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════
# v1.0: 电线字段复制（用于键迁移时保留 v1.0 信息）
# ═══════════════════════════════════════════════

def _copy_wire_fields(new_wire, old_wire):
    """将旧电线的关系类型和时序上下文复制到新电线。"""
    if old_wire.relation_type:
        new_wire.relation_type = old_wire.relation_type
    # 显式复制 slippage_distance（包括 0 值——same-as 的合法距离）
    new_wire.slippage_distance = old_wire.slippage_distance
    if old_wire.context:
        new_wire.context = dict(old_wire.context)
    new_wire.co_activation_count = old_wire.co_activation_count
