"""
键盘模型 AI — 花园（Garden）

花园是盆栽的集合 + 跨盆桥接表。它是系统的顶层容器，
在键盘模型外面加了一层三维空间结构。

主要职责：
- 管理盆的生命周期（创建 / 拆分 / 合并 / 休眠）
- 管理桥接线（创建 / 触发 / 消化 / 衰减）
- 跨盆传导（三轮流程：盆内传导 → 桥接评估 → 浅跳）
- 投喂路由（按主题签名选择目标盆）
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network import KeyNetwork
    from .key import Key
    from .conduction import ConductionRequest, ConductionResult

from .pot import (
    Pot, compute_document_signature, cosine_similarity,
    POT_MAX_KEYS,
    POT_CREATE_TOPIC_DISTANCE, POT_CREATE_MIN_KEYS,
    POT_MERGE_MIN_AGE_DAYS, POT_DORMANCY_DAYS,
    POT_MERGE_RECHECK_COOLDOWN,
    POT_SIGNATURE_DIM,
)


# ── 桥接参数 ──────────────────────────────────

BRIDGE_CREATE_SIMILARITY = 0.7
BRIDGE_INITIAL_THICKNESS = 0.25
BRIDGE_MAX_PER_KEY = 3
BRIDGE_SRC_BRIGHTNESS = 2
BRIDGE_THICKNESS_MIN = 0.5
BRIDGE_SPILLOVER_BRIGHTNESS = 1
BRIDGE_MAX_HOP = 1
BRIDGE_HEBBIAN_DELTA = 0.02
BRIDGE_DECAY_RATE = 0.001
BRIDGE_DECAY_MIN = 0.05

# 盆合并分层判定阈值
MERGE_BRIDGE_THRESHOLD_LOW = 3.0
MERGE_BRIDGE_THRESHOLD_MID = 5.0
MERGE_BRIDGE_THRESHOLD_HIGH = 8.0
MERGE_MAX_KEYS_FOR_MID = 400
MERGE_SIMILARITY_FOR_HIGH = 0.75

# 盆创建参数（从 pot.py 导入，此处不再重复定义）
# 合并拒绝冷却（从 pot.py 导入，此处不再重复定义）


# ═══════════════════════════════════════════════
# BridgeEntry 数据结构
# ═══════════════════════════════════════════════

@dataclass
class BridgeEntry:
    """一条跨盆桥接线。"""

    src_key_id: int
    src_pot_id: str
    dst_key_id: int
    dst_pot_id: str
    thickness: float = BRIDGE_INITIAL_THICKNESS
    created_at: float = 0.0
    last_triggered_at: float = 0.0
    trigger_count: int = 0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def strengthen(self, delta: float = BRIDGE_HEBBIAN_DELTA):
        self.thickness += delta
        self.trigger_count += 1
        self.last_triggered_at = time.time()

    def weaken(self, decay: float):
        self.thickness -= decay

    def snapshot(self) -> dict:
        return {
            "src_key_id": self.src_key_id,
            "src_pot_id": self.src_pot_id,
            "dst_key_id": self.dst_key_id,
            "dst_pot_id": self.dst_pot_id,
            "thickness": self.thickness,
            "created_at": self.created_at,
            "last_triggered_at": self.last_triggered_at,
            "trigger_count": self.trigger_count,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "BridgeEntry":
        return cls(**data)

    def __repr__(self) -> str:
        return (f"Bridge({self.src_pot_id}:#{self.src_key_id} "
                f"→ {self.dst_pot_id}:#{self.dst_key_id} "
                f"t={self.thickness:.3f} ×{self.trigger_count})")


# ═══════════════════════════════════════════════
# GardenConductionResult
# ═══════════════════════════════════════════════

@dataclass
class GardenConductionResult:
    """跨盆传导的完整结果。"""

    # 第一轮：各盆独立传导结果 {pot_id: ConductionResult}
    round1: dict[str, "ConductionResult"] = field(default_factory=dict)

    # 第二轮：触发的桥接列表
    bridges_triggered: list[BridgeEntry] = field(default_factory=list)

    # 第三轮：被桥接盆的浅传导结果 {pot_id: ConductionResult}
    round3: dict[str, "ConductionResult"] = field(default_factory=dict)

    # 快捷访问：所有被激活键的传导路径（聚合自各盆）
    all_activation_paths: dict = field(default_factory=dict)

    @property
    def total_pots_activated(self) -> int:
        return len(self.round1) + len(self.round3)

    @property
    def has_bridge_spillover(self) -> bool:
        return len(self.bridges_triggered) > 0


# ═══════════════════════════════════════════════
# Garden
# ═══════════════════════════════════════════════

class Garden:
    """花园：盆的集合 + 跨盆桥接表。v1.0 增加温度、工作空间、案例库。"""

    def __init__(self):
        self.pots: dict[str, Pot] = {}
        # bridge_table: key_id → [BridgeEntry, ...]
        # 双向索引：src 和 dst 都可以查到同一条桥接
        self.bridge_table: dict[int, list[BridgeEntry]] = defaultdict(list)

        self.internal_pot_id: str = ""
        self._next_pot_num: int = 1

        # 合并拒绝缓存：{(pot_a, pot_b): 上次拒绝时间}
        self._merge_rejected: dict[tuple[str, str], float] = {}

        self.created_at: float = time.time()

        # ── v1.0: Copycat 全局温度 ──
        self.global_temperature: float = 0.5  # 初始中温 (0.0–1.0)

        # ── v1.0: Copycat 工作空间 ──
        self.workspace: dict[int, float] = {}  # {键ID: 残留痕迹强度}

        # ── v1.0: CBR 案例库 ──
        from .case import Case
        self.case_library: dict[int, Case] = {}  # {案例ID: Case对象}

        # ── v1.0: 语法键网络 ──
        self.grammar_library: dict[int, "GrammarPattern"] = {}  # {语法模式ID: GrammarPattern}
        self.grammar_pot_id: str = ""  # 语法盆 ID

        # ── v1.0: NARS 推理 ──
        from .inference import InferenceEnergyBudget
        self.inference_energy = InferenceEnergyBudget()
        self.rule_key_ids: dict[str, int] = {}  # {"deduction": kid, ...}

        # ── v1.0: HTM 注意力 ──
        self.attention_mode: str = "dialogue"       # "dialogue" | "solitude"
        self.last_master_input_at: float = 0.0      # 师父最后一次输入的时间戳
        self._predicted_keys: dict[int, float] = {}  # {key_id: 预测时间戳}，用于 TTL 管理

        # ── v1.0: 产生式系统 ──
        self.situation_key_ids: dict[str, int] = {}   # {"master_casual": kid, ...}
        self.meta_rule_key_ids: dict[str, int] = {}   # {"busy_short": kid, ...}
        self.last_proactive_at: float = 0.0            # 上次主动搭话的时间戳

    # ── 盆管理 ─────────────────────────────────

    def create_pot(
        self,
        source_type: str = "dialogue",
        topic_label: str = "",
        is_internal: bool = False,
    ) -> Pot:
        """创建一个新盆。"""
        from .network import KeyNetwork

        pot_id = f"pot-{self._next_pot_num}"
        self._next_pot_num += 1

        pot = Pot(
            id=pot_id,
            network=KeyNetwork(),
            source_type=source_type,
            topic_label=topic_label or source_type,
            merge_protected_until=time.time() + POT_MERGE_MIN_AGE_DAYS * 86400,
        )
        self.pots[pot_id] = pot

        if is_internal:
            self.internal_pot_id = pot_id

        return pot

    def remove_pot(self, pot_id: str):
        """删除一个盆及其所有关联桥接。"""
        if pot_id not in self.pots:
            return

        pot = self.pots[pot_id]
        # 删除该盆所有键的桥接
        for key in pot.network.all_keys():
            if key.id in self.bridge_table:
                # 同时清理对端索引中的桥接
                for bridge in list(self.bridge_table[key.id]):
                    self._remove_bridge_from_index(bridge)
                del self.bridge_table[key.id]

        del self.pots[pot_id]

    def get_active_pots(self) -> list[Pot]:
        """所有非休眠盆。"""
        return [p for p in self.pots.values() if not p.is_dormant]

    def get_pot_for_key(self, key_id: int) -> Pot | None:
        """根据键 ID 找到所属盆。"""
        for pot in self.pots.values():
            if pot.network.get_key(key_id) is not None:
                return pot
        return None

    # ── 桥接管理 ───────────────────────────────

    def create_bridge(
        self,
        src_key_id: int,
        src_pot_id: str,
        dst_key_id: int,
        dst_pot_id: str,
        thickness: float = BRIDGE_INITIAL_THICKNESS,
    ) -> BridgeEntry | None:
        """创建一条桥接线。如果已存在或超限则返回 None。"""
        if src_pot_id == dst_pot_id:
            return None  # 同盆不建桥接

        # 检查是否已存在
        existing = self.bridge_table.get(src_key_id, [])
        for b in existing:
            if b.dst_key_id == dst_key_id and b.dst_pot_id == dst_pot_id:
                return None  # 已存在

        # 检查单键桥接数上限
        if len(existing) >= BRIDGE_MAX_PER_KEY:
            # 只保留最粗的 BRIDGE_MAX_PER_KEY 条
            # 新桥接粗度太低则不创建
            if thickness <= min(b.thickness for b in existing):
                return None
            # 替换最细的
            weakest = min(existing, key=lambda b: b.thickness)
            self._remove_bridge_from_index(weakest)

        bridge = BridgeEntry(
            src_key_id=src_key_id,
            src_pot_id=src_pot_id,
            dst_key_id=dst_key_id,
            dst_pot_id=dst_pot_id,
            thickness=thickness,
        )

        # 双向索引
        self.bridge_table[src_key_id].append(bridge)
        self.bridge_table[dst_key_id].append(bridge)

        return bridge

    def _remove_bridge_from_index(self, bridge: BridgeEntry):
        """从桥接表中移除一条桥接（清理双向索引）。"""
        for kid in (bridge.src_key_id, bridge.dst_key_id):
            if kid in self.bridge_table:
                self.bridge_table[kid] = [
                    b for b in self.bridge_table[kid]
                    if not (b.src_key_id == bridge.src_key_id
                            and b.dst_key_id == bridge.dst_key_id)
                ]
                if not self.bridge_table[kid]:
                    del self.bridge_table[kid]

    def find_bridges_for_key(self, key_id: int) -> list[BridgeEntry]:
        """查找某个键持有的所有桥接线。"""
        return self.bridge_table.get(key_id, [])

    def auto_create_bridges_for_new_keys(
        self,
        new_keys: list["Key"],
        pot_id: str,
    ):
        """
        新键创建后，自动检查与其他盆键的内容相似度，
        满足条件则建桥。
        """
        src_pot = self.pots.get(pot_id)
        if src_pot is None:
            return

        for nk in new_keys:
            for other_pot in self.pots.values():
                if other_pot.id == pot_id or other_pot.is_dormant:
                    continue
                for other_key in other_pot.network.all_keys():
                    if other_key.is_dormant:
                        continue
                    sim = _bigram_jaccard(nk.content, other_key.content)
                    if sim >= BRIDGE_CREATE_SIMILARITY:
                        self.create_bridge(
                            src_key_id=nk.id,
                            src_pot_id=pot_id,
                            dst_key_id=other_key.id,
                            dst_pot_id=other_pot.id,
                        )

    # ── 跨盆传导 ───────────────────────────────

    def garden_conduct(
        self,
        seeds: list[tuple[int, int, str]],  # [(key_id, brightness, pot_id), ...]
        max_hops: int = -1,
        timeout_ms: int = 10_000,
        source: str = "user_input",
        temperature: float | None = None,
        workspace_preheat: dict[int, float] | None = None,
        relation_boost: dict[str, float] | None = None,  # M4: 探照灯
    ) -> GardenConductionResult:
        """
        执行跨盆传导（三轮流程）。

        第一轮：各盆独立传导（并行逻辑）
        第二轮：桥接评估（亮度=2 的键 → 检查桥接 → 触发）
        第三轮：目标盆浅传导（BRIDGE_MAX_HOP=1）

        v1.0：temperature 控制搜索广度，workspace_preheat 预热已知键。
        """
        from .conduction import conduct, ConductionRequest

        if temperature is None:
            temperature = self.global_temperature

        result = GardenConductionResult()
        all_paths: dict = {}

        # === 第一轮：盆内传导 ===
        seeds_by_pot: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for kid, brightness, pid in seeds:
            seeds_by_pot[pid].append((kid, brightness))

        for pot_id, pot_seeds in seeds_by_pot.items():
            pot = self.pots.get(pot_id)
            if pot is None or pot.is_dormant:
                continue
            pot.activate()
            request = ConductionRequest(
                seed_keys=pot_seeds,
                max_hops=max_hops,
                timeout_ms=timeout_ms,
                source=source,
                temperature=temperature,
                workspace_preheat=workspace_preheat,
                relation_boost=relation_boost,
            )
            pot_result = conduct(pot.network, request)
            result.round1[pot_id] = pot_result
            # 收集路径（标注盆 ID）
            for dst_id, path in pot_result.activation_paths.items():
                all_paths[(pot_id, dst_id)] = path

        # === 第二轮：桥接评估 ===
        spillover_seeds: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for pot_id, pot_result in result.round1.items():
            for key_id, brightness in pot_result.final_pattern.items():
                # M4: 峰值桥接——键「到过」满亮度即可触发桥，不必停在满亮度
                fire = max(brightness, pot_result.peak_pattern.get(key_id, 0))
                if fire < BRIDGE_SRC_BRIGHTNESS:
                    continue
                bridges = self.find_bridges_for_key(key_id)
                for bridge in bridges:
                    if bridge.thickness < BRIDGE_THICKNESS_MIN:
                        continue
                    # 桥接触发！
                    bridge.strengthen()
                    result.bridges_triggered.append(bridge)
                    spillover_seeds[bridge.dst_pot_id].append(
                        (bridge.dst_key_id, BRIDGE_SPILLOVER_BRIGHTNESS)
                    )
                    # 唤醒目标盆
                    dst_pot = self.pots.get(bridge.dst_pot_id)
                    if dst_pot is not None:
                        dst_pot.activate()

        # === 第三轮：目标盆浅传导 ===
        for pot_id, spill_seeds in spillover_seeds.items():
            pot = self.pots.get(pot_id)
            if pot is None or pot.is_dormant:
                continue
            # 去重
            unique_seeds: dict[int, int] = {}
            for kid, b in spill_seeds:
                unique_seeds[kid] = max(unique_seeds.get(kid, 0), b)
            deduped = list(unique_seeds.items())

            request = ConductionRequest(
                seed_keys=deduped,
                max_hops=BRIDGE_MAX_HOP,
                timeout_ms=timeout_ms,
                source="bridge_spillover",
                temperature=temperature,
                workspace_preheat=workspace_preheat,
            )
            spill_result = conduct(pot.network, request)
            result.round3[pot_id] = spill_result
            # 收集路径（标注桥接来源）
            for dst_id, path in spill_result.activation_paths.items():
                all_paths[(pot_id, dst_id)] = path

        result.all_activation_paths = all_paths
        return result

    # ── 投喂路由 ───────────────────────────────

    def route_ingest(
        self,
        fragments: list[str],
        source: str = "external",
    ) -> tuple[Pot, bool]:
        """
        决定投喂文档的目标盆。

        返回 (target_pot, is_new_pot)。

        规则：
        1. 算文档签名
        2. 与所有已有盆比较
        3. 最近盆的余弦距离 > 0.7 → 建新盆
        4. 否则归入最近盆
        """
        doc_sig = compute_document_signature(fragments)

        if not self.pots:
            # 没有任何盆 → 创建第一个
            pot = self.create_pot(source_type="ingest", topic_label="外部知识")
            return pot, True

        # 找最近的盆
        best_pot: Pot | None = None
        best_sim = -1.0
        for pot in self.pots.values():
            if pot.is_dormant:
                continue
            sim = cosine_similarity(doc_sig, pot.topic_signature or [0.0] * POT_SIGNATURE_DIM)
            if sim > best_sim:
                best_sim = sim
                best_pot = pot

        # 判断
        if best_sim < (1.0 - POT_CREATE_TOPIC_DISTANCE):  # 相似度 < 0.3
            # 距离太远 → 建新盆
            if len(fragments) >= POT_CREATE_MIN_KEYS:
                pot = self.create_pot(
                    source_type="ingest",
                    topic_label=f"外部知识-{self._next_pot_num}",
                )
                return pot, True
            # 碎片太少 → 归入最近盆（即使距离远）
            if best_pot is not None:
                return best_pot, False

        if best_pot is not None:
            return best_pot, False

        # 兜底
        pot = self.create_pot(source_type="ingest", topic_label="外部知识")
        return pot, True

    # ── 内容聚类拆分（强制分盆） ────────────────

    def split_by_content(self, pot_id: str, num_clusters: int = 3) -> list[Pot]:
        """
        基于内容相似度强制拆分一个盆（即使键数未达 POT_MAX_KEYS）。

        用于迁移：单盆混了太多不同主题的内容时，手动拆分。
        算法：用 bigram 签名聚类键 → 每个簇一个子盆。
        """
        pot = self.pots.get(pot_id)
        if pot is None or pot.key_count < 10:
            return []

        from .network import KeyNetwork

        keys = [k for k in pot.network.all_keys() if not k.is_dormant]
        if len(keys) < 10:
            return []

        # 为每个键计算 bigram 签名（8维，取top bigrams）
        def key_sig(k: "Key") -> list[float]:
            from collections import Counter
            content = k.content
            if len(content) < 2:
                return [0.0] * 8
            bigrams = [content[i:i+2] for i in range(len(content)-1)]
            counter = Counter(bigrams)
            top8 = [bg for bg, _ in counter.most_common(8)]
            total = sum(counter.values())
            vec = [counter.get(bg, 0) / total for bg in top8]
            # pad to 8
            while len(vec) < 8:
                vec.append(0.0)
            return vec[:8]

        # 简单 k-means 风格聚类
        import random as _random
        _random.seed(42)

        key_vecs = [(k, key_sig(k)) for k in keys]

        # 随机初始化聚类中心
        centers = []
        indices = list(range(len(key_vecs)))
        _random.shuffle(indices)
        for i in range(min(num_clusters, len(key_vecs))):
            centers.append(key_vecs[indices[i]][1])

        # 迭代分配
        clusters: list[list] = [[] for _ in centers]
        for _ in range(10):  # 最多10轮
            # 分配
            clusters = [[] for _ in centers]
            for k, vec in key_vecs:
                best_c = 0
                best_d = float('inf')
                for ci, cv in enumerate(centers):
                    d = sum((a-b)**2 for a, b in zip(vec, cv))
                    if d < best_d:
                        best_d = d
                        best_c = ci
                clusters[best_c].append((k, vec))

            # 更新中心
            for ci in range(len(centers)):
                if clusters[ci]:
                    dim = len(centers[ci])
                    new_center = [0.0] * dim
                    for _, vec in clusters[ci]:
                        for d in range(dim):
                            new_center[d] += vec[d]
                    for d in range(dim):
                        new_center[d] /= len(clusters[ci])
                    centers[ci] = new_center

        # 过滤太小的簇（至少2个键）
        viable = [c for c in clusters if len(c) >= 2]
        if len(viable) <= 1:
            return []

        # 创建子盆
        sub_pots = []
        for i, cluster in enumerate(viable):
            sub = Pot(
                id=f"{pot_id}-{i+1}",
                network=KeyNetwork(),
                source_type=pot.source_type,
                topic_label=f"{pot.topic_label}-{i+1}",
                merge_protected_until=time.time() + POT_MERGE_MIN_AGE_DAYS * 86400,
            )

            id_map: dict[int, int] = {}
            for old_key, _ in cluster:
                new_key = sub.network.create_key(
                    content=old_key.content,
                    source=old_key.source,
                    brightness=0,
                )
                new_key.activation_count = old_key.activation_count
                new_key.activation_history = old_key.activation_history.copy()
                new_key.last_activated_at = old_key.last_activated_at
                new_key.peak_brightness = old_key.peak_brightness
                # v1.0
                new_key.primitive = old_key.primitive
                new_key.slots = dict(old_key.slots)
                new_key.evidence_freq = old_key.evidence_freq
                new_key.evidence_conf = old_key.evidence_conf
                new_key.prediction_state = old_key.prediction_state
                id_map[old_key.id] = new_key.id

            # 迁移盆内电线（含 v1.0 新字段）
            for old_key, _ in cluster:
                new_id = id_map.get(old_key.id)
                if new_id is None:
                    continue
                for (a, b), wire in pot.network.wires.items():
                    if a == old_key.id:
                        other_new = id_map.get(b)
                        if other_new is not None:
                            w = sub.network.set_wire(new_id, other_new, thickness=wire.thickness)
                            from .pot import _copy_wire_fields
                            _copy_wire_fields(w, wire)
                    elif b == old_key.id:
                        other_new = id_map.get(a)
                        if other_new is not None:
                            w = sub.network.set_wire(other_new, new_id, thickness=wire.thickness)
                            from .pot import _copy_wire_fields
                            _copy_wire_fields(w, wire)

            sub._split_id_map = id_map
            sub.calculate_signature()
            sub_pots.append(sub)

        # 更新花园
        for sub in sub_pots:
            self.pots[sub.id] = sub

        # 迁移桥接表
        for sub in sub_pots:
            id_map = getattr(sub, '_split_id_map', {})
            for old_id, new_id in id_map.items():
                if old_id in self.bridge_table:
                    for bridge in self.bridge_table[old_id]:
                        if bridge.src_key_id == old_id:
                            bridge.src_key_id = new_id
                            bridge.src_pot_id = sub.id
                        if bridge.dst_key_id == old_id:
                            bridge.dst_key_id = new_id
                            bridge.dst_pot_id = sub.id
                    if new_id != old_id:
                        self.bridge_table[new_id] = self.bridge_table.pop(old_id)

        # 子盆之间建立桥接
        if len(sub_pots) >= 2:
            for i in range(len(sub_pots)):
                for j in range(i + 1, len(sub_pots)):
                    for key_i in sub_pots[i].network.all_keys():
                        for key_j in sub_pots[j].network.all_keys():
                            sim = _bigram_jaccard(key_i.content, key_j.content)
                            if sim > 0.4:
                                self.create_bridge(
                                    src_key_id=key_i.id,
                                    src_pot_id=sub_pots[i].id,
                                    dst_key_id=key_j.id,
                                    dst_pot_id=sub_pots[j].id,
                                    thickness=0.3,
                                )

        # 删除原盆
        self.remove_pot(pot_id)

        return sub_pots

    # ── 盆生命周期维护 ──────────────────────────

    def check_and_split(self, pot_id: str) -> list[Pot]:
        """检查盆是否需要拆分，并执行。返回新盆列表。"""
        pot = self.pots.get(pot_id)
        if pot is None or not pot.needs_split():
            return []

        sub_pots = pot.split()
        if sub_pots is None or len(sub_pots) <= 1:
            return []

        # 更新花园
        for sub in sub_pots:
            self.pots[sub.id] = sub

        # 迁移桥接表：原盆的桥接 → 按 id_map 更新到子盆
        for sub in sub_pots:
            id_map = getattr(sub, '_split_id_map', {})
            for old_id, new_id in id_map.items():
                if old_id in self.bridge_table:
                    # 更新桥接中的键 ID
                    for bridge in self.bridge_table[old_id]:
                        if bridge.src_key_id == old_id:
                            bridge.src_key_id = new_id
                            bridge.src_pot_id = sub.id
                        if bridge.dst_key_id == old_id:
                            bridge.dst_key_id = new_id
                            bridge.dst_pot_id = sub.id
                    # 将桥接移到新键 ID
                    if new_id != old_id:
                        self.bridge_table[new_id] = self.bridge_table.pop(old_id)

        # 子盆之间建立桥接（原盆内的线变为桥接）
        if len(sub_pots) >= 2:
            for i in range(len(sub_pots)):
                for j in range(i + 1, len(sub_pots)):
                    for key_i in sub_pots[i].network.all_keys():
                        for key_j in sub_pots[j].network.all_keys():
                            self.create_bridge(
                                src_key_id=key_i.id,
                                src_pot_id=sub_pots[i].id,
                                dst_key_id=key_j.id,
                                dst_pot_id=sub_pots[j].id,
                                thickness=0.3,  # 较弱桥接（曾是盆内线）
                            )

        # 删除原盆
        self.remove_pot(pot_id)

        return sub_pots

    def check_and_merge(self):
        """检查所有盆对是否需要合并。"""
        pot_ids = list(self.pots.keys())
        now = time.time()

        for i in range(len(pot_ids)):
            for j in range(i + 1, len(pot_ids)):
                pid_a, pid_b = pot_ids[i], pot_ids[j]
                pot_a = self.pots.get(pid_a)
                pot_b = self.pots.get(pid_b)
                if pot_a is None or pot_b is None:
                    continue
                if pot_a.is_dormant or pot_b.is_dormant:
                    continue

                # 检查合并拒绝冷却
                pair_key = (pid_a, pid_b) if pid_a < pid_b else (pid_b, pid_a)
                rejected_at = self._merge_rejected.get(pair_key)
                if rejected_at and (now - rejected_at) < POT_MERGE_RECHECK_COOLDOWN:
                    continue

                # 计算两盆之间的桥接总粗度
                bridge_sum = self._bridge_sum_between(pid_a, pid_b)

                # 分层判定
                should_merge = False
                if bridge_sum >= MERGE_BRIDGE_THRESHOLD_HIGH:
                    should_merge = True
                elif bridge_sum >= MERGE_BRIDGE_THRESHOLD_MID:
                    sim = 1.0 - pot_a.topic_distance_to(pot_b)
                    if sim >= MERGE_SIMILARITY_FOR_HIGH:
                        should_merge = True
                elif bridge_sum >= MERGE_BRIDGE_THRESHOLD_LOW:
                    merged_count = pot_a.key_count + pot_b.key_count
                    if merged_count <= MERGE_MAX_KEYS_FOR_MID:
                        should_merge = True

                if should_merge:
                    self._merge_pots(pid_a, pid_b)
                else:
                    self._merge_rejected[pair_key] = now

    def _bridge_sum_between(self, pot_id_a: str, pot_id_b: str) -> float:
        """计算两个盆之间所有桥接线的总粗度。"""
        total = 0.0
        seen: set[tuple[int, int]] = set()
        for bridges in self.bridge_table.values():
            for b in bridges:
                pair = (b.src_pot_id, b.dst_pot_id)
                if (pair == (pot_id_a, pot_id_b) or pair == (pot_id_b, pot_id_a)):
                    sig = (b.src_key_id, b.dst_key_id)
                    if sig not in seen:
                        seen.add(sig)
                        total += b.thickness
        return total

    def _merge_pots(self, pot_id_a: str, pot_id_b: str):
        """合并两个盆。pot_id_b 的内容并入 pot_id_a。"""
        pot_a = self.pots.get(pot_id_a)
        pot_b = self.pots.get(pot_id_b)
        if pot_a is None or pot_b is None:
            return

        # 如果合并后超过 800 键，标记为——合并后立即触发拆分
        will_oversize = (pot_a.key_count + pot_b.key_count) > POT_MAX_KEYS  # 使用 pot.py 的 POT_MAX_KEYS

        # 迁移 pot_b 的键到 pot_a
        id_map: dict[int, int] = {}  # old_b_id → new_in_a_id
        for key_b in pot_b.network.all_keys():
            new_key = pot_a.network.create_key(
                content=key_b.content,
                source=key_b.source,
                brightness=0,
            )
            new_key.activation_count = key_b.activation_count
            new_key.activation_history = key_b.activation_history.copy()
            new_key.last_activated_at = key_b.last_activated_at
            # v1.0
            new_key.primitive = key_b.primitive
            new_key.slots = dict(key_b.slots)
            new_key.evidence_freq = key_b.evidence_freq
            new_key.evidence_conf = key_b.evidence_conf
            new_key.prediction_state = key_b.prediction_state
            id_map[key_b.id] = new_key.id

        # 迁移 pot_b 的盆内电线到 pot_a（含 v1.0 字段）
        for (a, b), wire in pot_b.network.wires.items():
            new_a = id_map.get(a)
            new_b = id_map.get(b)
            if new_a is not None and new_b is not None:
                w = pot_a.network.set_wire(new_a, new_b, thickness=wire.thickness)
                from .pot import _copy_wire_fields
                _copy_wire_fields(w, wire)

        # 更新桥接表：pot_b 的桥接 → 指向 pot_a
        for key_b_id, bridges in list(self.bridge_table.items()):
            new_id = id_map.get(key_b_id)
            if new_id is None:
                continue
            for bridge in bridges:
                if bridge.src_key_id == key_b_id:
                    bridge.src_key_id = new_id
                    bridge.src_pot_id = pot_id_a
                if bridge.dst_key_id == key_b_id:
                    bridge.dst_key_id = new_id
                    bridge.dst_pot_id = pot_id_a
            if new_id in self.bridge_table:
                self.bridge_table[new_id].extend(bridges)
            else:
                self.bridge_table[new_id] = bridges

        # 删除 pot_b
        self.remove_pot(pot_id_b)

        # 更新 pot_a 的主题签名
        pot_a.calculate_signature()
        pot_a.topic_label = f"{pot_a.topic_label}+{pot_b.topic_label}"

        # 如果超限 → 立即拆分
        if will_oversize:
            self.check_and_split(pot_id_a)

        # 清除合并拒绝记录
        to_clear = [
            k for k in self._merge_rejected
            if pot_id_a in k or pot_id_b in k
        ]
        for k in to_clear:
            del self._merge_rejected[k]

    def check_dormancy(self):
        """检查盆是否需要休眠。"""
        now = time.time()
        for pot in self.pots.values():
            if pot.is_dormant:
                continue
            if pot.id == self.internal_pot_id:
                continue  # 内部盆永不休眠
            if pot.last_activated_at == 0:
                idle = now - pot.created_at
            else:
                idle = now - pot.last_activated_at
            if idle > POT_DORMANCY_DAYS * 86400:
                pot.is_dormant = True

    # ── 桥接衰减 ──────────────────────────────

    def decay_bridges(self):
        """对所有桥接线执行每日衰减。"""
        now = time.time()
        to_delete: list[BridgeEntry] = []

        for bridges in list(self.bridge_table.values()):
            for bridge in bridges:
                idle_days = (now - bridge.last_triggered_at) / 86400
                if idle_days >= 1.0:
                    decay = BRIDGE_DECAY_RATE * idle_days
                    bridge.weaken(decay)
                    if bridge.thickness <= BRIDGE_DECAY_MIN:
                        to_delete.append(bridge)

        for bridge in to_delete:
            self._remove_bridge_from_index(bridge)

    # ── 维护 ──────────────────────────────────

    def maintenance(self):
        """周期性维护：盆生命周期 + 桥接衰减。"""
        # 记录键数历史
        for pot in self.pots.values():
            pot.record_key_count()

        # 拆分检查
        for pot_id in list(self.pots.keys()):
            self.check_and_split(pot_id)

        # 合并检查
        self.check_and_merge()

        # 休眠检查
        self.check_dormancy()

        # 桥接衰减
        self.decay_bridges()

        # ── v1.0: 证据衰减 ──
        from .evidence import decay_all_evidence
        decay_all_evidence(self)

        # ── v1.0: 推理能量恢复 ──
        from .inference import INFERENCE_RECHARGE_PER_MAINTENANCE
        self.inference_energy.recharge(INFERENCE_RECHARGE_PER_MAINTENANCE)

        # ── v1.0: HTM 预测衰减（安全网——正常在每次对话后清理）──
        from .temporal import decay_predictions
        decay_predictions(self)

        # ── v1.0: CBR 案例维护 ──
        from .case import decay_cases
        decay_cases(self)

        # ── v1.0: 语法模式维护 ──
        from .grammar import decay_grammar_patterns
        decay_grammar_patterns(self)

    # ── 初始化 ─────────────────────────────────

    def init_default_garden(self):
        """
        首次启动：创建默认的盆结构。
        - 内部-系统盆（体感时间键等）
        - 对话-默认盆（日常对话）
        """
        # 内部盆
        internal = self.create_pot(
            source_type="internal",
            topic_label="系统",
            is_internal=True,
        )
        self.internal_pot_id = internal.id

        # 默认对话盆
        self.create_pot(
            source_type="dialogue",
            topic_label="日常对话",
        )

        # ── v1.0: 初始化推理规则键 ──
        from .inference import init_inference_rules
        init_inference_rules(self)

        # ── v1.0: 初始化产生式系统 ──
        from .production import init_situation_keys, init_meta_rules
        init_situation_keys(self)
        init_meta_rules(self)

    # ── 快照 ──────────────────────────────────

    def snapshot(self) -> dict:
        """花园的完整可序列化快照。"""
        return {
            "pots": {pid: pot.snapshot() for pid, pot in self.pots.items()},
            "bridge_table": {
                str(kid): [b.snapshot() for b in bridges]
                for kid, bridges in self.bridge_table.items()
            },
            "internal_pot_id": self.internal_pot_id,
            "_next_pot_num": self._next_pot_num,
            "_merge_rejected": {
                f"{a}|{b}": t for (a, b), t in self._merge_rejected.items()
            },
            "created_at": self.created_at,
            "global_temperature": self.global_temperature,
            "workspace": self.workspace,
            "case_library": {
                str(cid): case.snapshot()
                for cid, case in self.case_library.items()
            },
            "inference_energy": self.inference_energy.snapshot(),
            "rule_key_ids": self.rule_key_ids,
            # ── v1.0: HTM 注意力 ──
            "attention_mode": self.attention_mode,
            "last_master_input_at": self.last_master_input_at,
            # _predicted_keys 不持久化——重启后预测清空

            # ── v1.0: 产生式系统 ──
            "situation_key_ids": self.situation_key_ids,
            "meta_rule_key_ids": self.meta_rule_key_ids,
            "last_proactive_at": self.last_proactive_at,

            # ── v1.0: 语法键网络 ──
            "grammar_pot_id": self.grammar_pot_id,
            "grammar_library": {
                str(gid): gp.snapshot()
                for gid, gp in self.grammar_library.items()
            },
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "Garden":
        """从快照恢复花园。"""
        garden = cls()
        garden.internal_pot_id = data.get("internal_pot_id", "")
        garden._next_pot_num = data.get("_next_pot_num", 1)
        garden.created_at = data.get("created_at", time.time())
        garden.global_temperature = data.get("global_temperature", 0.5)
        garden.workspace = {int(k): v for k, v in data.get("workspace", {}).items()}

        # ── v1.0: CBR 案例库 ──
        from .case import Case
        for cid_str, cdata in data.get("case_library", {}).items():
            garden.case_library[int(cid_str)] = Case.from_snapshot(cdata)

        # ── v1.0: HTM 注意力 ──
        garden.attention_mode = data.get("attention_mode", "dialogue")
        garden.last_master_input_at = data.get("last_master_input_at", time.time())
        # _predicted_keys 不持久化，重启后预测清空

        # ── v1.0: 产生式系统 ──
        garden.situation_key_ids = {
            str(k): int(v)
            for k, v in data.get("situation_key_ids", {}).items()
        }
        garden.meta_rule_key_ids = {
            str(k): int(v)
            for k, v in data.get("meta_rule_key_ids", {}).items()
        }
        garden.last_proactive_at = data.get("last_proactive_at", 0.0)

        # ── v1.0: 语法键网络 ──
        garden.grammar_pot_id = data.get("grammar_pot_id", "")
        from .grammar import GrammarPattern
        for gid_str, gdata in data.get("grammar_library", {}).items():
            garden.grammar_library[int(gid_str)] = GrammarPattern.from_snapshot(gdata)

        # ── v1.0: 恢复推理能量 ──
        from .inference import InferenceEnergyBudget
        garden.inference_energy = InferenceEnergyBudget.from_snapshot(
            data.get("inference_energy", {})
        )
        garden.rule_key_ids = data.get("rule_key_ids", {})
        # 确保 rule_key_ids 的 key 是 str → 可能从 JSON 恢复时需要转换
        if garden.rule_key_ids:
            garden.rule_key_ids = {
                str(k): int(v) for k, v in garden.rule_key_ids.items()
            }

        for pid, pdata in data.get("pots", {}).items():
            garden.pots[pid] = Pot.from_snapshot(pdata)

        for kid_str, bridges_data in data.get("bridge_table", {}).items():
            kid = int(kid_str)
            garden.bridge_table[kid] = [
                BridgeEntry.from_snapshot(bd) for bd in bridges_data
            ]

        mr = data.get("_merge_rejected", {})
        for key, val in mr.items():
            a, b = key.split("|")
            garden._merge_rejected[(a, b)] = val

        # ── v1.0: 确保推理规则键存在 ──
        # 加载旧快照（Phase 3 之前）时 rule_key_ids 为空，需要补建
        if not garden.rule_key_ids and garden.pots:
            from .inference import init_inference_rules
            init_inference_rules(garden)

        # ── v1.0: 确保产生式系统键存在 ──
        # 加载旧快照（Phase 6 之前）时 situation_key_ids 为空，需要补建
        if (not garden.situation_key_ids or not garden.meta_rule_key_ids) and garden.pots:
            from .production import init_situation_keys, init_meta_rules
            init_situation_keys(garden)
            init_meta_rules(garden)

        return garden

    # ── 统计 ──────────────────────────────────

    @property
    def total_keys(self) -> int:
        return sum(p.key_count for p in self.pots.values())

    @property
    def total_wires(self) -> int:
        return sum(p.network.wire_count for p in self.pots.values())

    @property
    def total_bridges(self) -> int:
        seen: set[tuple[int, int]] = set()
        count = 0
        for bridges in self.bridge_table.values():
            for b in bridges:
                sig = (b.src_key_id, b.dst_key_id)
                if sig not in seen:
                    seen.add(sig)
                    count += 1
        return count

    @property
    def active_pot_count(self) -> int:
        return len(self.get_active_pots())

    @property
    def dormant_pot_count(self) -> int:
        return sum(1 for p in self.pots.values() if p.is_dormant)

    def __repr__(self) -> str:
        return (f"Garden(pots={len(self.pots)}, active={self.active_pot_count}, "
                f"dormant={self.dormant_pot_count}, keys={self.total_keys}, "
                f"bridges={self.total_bridges})")


# ═══════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════

def _bigram_jaccard(a: str, b: str) -> float:
    """字符 bigram Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    set_a = set(a[i:i + 2] for i in range(len(a) - 1))
    set_b = set(b[i:i + 2] for i in range(len(b) - 1))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
