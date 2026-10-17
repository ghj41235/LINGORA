"""
键盘模型 AI — NARS 推理规则键  v1.0

推理规则本身是 RULE 类型的键，存储在内部盆中。
推理不是独立的引擎——推理规则键在传导后由模式匹配触发，
执行动作（创建新键或修改已有键的 evidence）。

三种推理：
1. 演绎 (Deduction)：A→B ∧ B→C ⇒ A→C（传递链）
2. 归纳 (Induction)：同类实例 → 泛化概念
3. 类比 (Analogy)：A:B :: C:? → C:D（完成配对）

资源约束：每次推理消耗能量（InferenceEnergyBudget），
防止无限创建键。能量随对话恢复。

English:
NARS inference rules as RULE-type keys: deduction, induction and analogy
are pattern-matched after conduction and execute actions (creating keys or
revising evidence), bounded by an energy budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .wire import SLIPPAGE_DISTANCE

if TYPE_CHECKING:
    from .garden import Garden
    from .key import Key
    from .network import KeyNetwork


# ═══════════════════════════════════════════════
# 能量预算
# ═══════════════════════════════════════════════

INFERENCE_MAX_ENERGY = 100.0
INFERENCE_RECHARGE_PER_CONVO = 5.0       # 每次对话恢复
INFERENCE_RECHARGE_PER_MAINTENANCE = 2.0  # 每次维护恢复

ENERGY_COST_DEDUCTION = 10.0
ENERGY_COST_INDUCTION = 15.0
ENERGY_COST_ANALOGY = 20.0

INFERENCE_MIN_EVIDENCE_CONF = 0.10       # 参与推理的最低 evidence_conf
INFERENCE_MIN_WIRE_THICKNESS = 0.25      # 用于演绎的电线最低粗度

MAX_NEW_KEYS_PER_CYCLE = 10              # 每次推理周期最多创建新键数


@dataclass
class InferenceEnergyBudget:
    """推理能量预算。每次推理消耗能量，对话间隙恢复。"""
    current: float = INFERENCE_MAX_ENERGY
    max_energy: float = INFERENCE_MAX_ENERGY
    total_spent: float = 0.0
    total_fires: int = 0

    def can_afford(self, cost: float) -> bool:
        return self.current >= cost

    def spend(self, cost: float) -> bool:
        if self.current < cost:
            return False
        self.current -= cost
        self.total_spent += cost
        self.total_fires += 1
        return True

    def recharge(self, amount: float):
        self.current = min(self.max_energy, self.current + amount)

    def snapshot(self) -> dict:
        return {
            "current": self.current,
            "max_energy": self.max_energy,
            "total_spent": self.total_spent,
            "total_fires": self.total_fires,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "InferenceEnergyBudget":
        return cls(**{k: data.get(k, v) for k, v in {
            "current": INFERENCE_MAX_ENERGY,
            "max_energy": INFERENCE_MAX_ENERGY,
            "total_spent": 0.0,
            "total_fires": 0,
        }.items()})


# ═══════════════════════════════════════════════
# 推理动作数据类
# ═══════════════════════════════════════════════

@dataclass
class DeductionAction:
    """演绎推理结果：A→B→C ⇒ A→C"""
    key_a_id: int
    key_b_id: int
    key_c_id: int
    relation_type: str
    combined_freq: float
    combined_conf: float
    pot_id: str


@dataclass
class InductionAction:
    """归纳推理结果：多个同原语实例 → 泛化键"""
    primitive: str
    slot_names: tuple[str, ...]
    source_key_ids: list[int]
    combined_freq: float
    combined_conf: float
    pot_id: str


@dataclass
class AnalogyAction:
    """类比推理结果：A:B :: C:? → C:D"""
    key_a_id: int
    key_b_id: int
    key_c_id: int
    new_primitive: str
    new_slots: dict[str, int]
    combined_freq: float
    combined_conf: float
    pot_id: str


@dataclass
class InferenceCycleResult:
    """一次推理周期的结果。"""
    new_keys: list = field(default_factory=list)  # list[Key]
    rule_fires: dict[str, bool] = field(default_factory=dict)  # {rule_type: fired}
    energy_remaining: float = INFERENCE_MAX_ENERGY
    actions_taken: int = 0


# ═══════════════════════════════════════════════
# 推理规则键管理
# ═══════════════════════════════════════════════

def init_inference_rules(garden: "Garden") -> dict[str, int]:
    """
    在内部盆中创建推理规则键。
    返回 {rule_type: key_id} 映射。

    创建 3 个 RULE 键：演绎、归纳、类比。
    这些键本身在传导中被激活时，会触发对应的模式匹配逻辑。
    """
    internal_id = garden.internal_pot_id
    if not internal_id or internal_id not in garden.pots:
        return {}

    internal = garden.pots[internal_id]
    net = internal.network

    rule_defs = {
        "deduction": "inference_rule:deduction:transitive_chain",
        "induction": "inference_rule:induction:generalize",
        "analogy": "inference_rule:analogy:complete_pair",
    }

    rule_ids: dict[str, int] = {}

    for rule_type, content in rule_defs.items():
        key = net.create_key(content=content, source="internal")
        key.primitive = "RULE"
        # condition 和 action 槽位在推理周期中动态解析，此处设为 0（占位）
        key.slots = {"condition": 0, "action": 0}
        key.evidence_freq = 1.0
        key.evidence_conf = 0.5  # 推理规则以中等置信度启动
        key.activation_count = 9999  # 标记为系统键
        rule_ids[rule_type] = key.id

    garden.rule_key_ids = rule_ids
    return rule_ids


def _find_rule_key(garden: "Garden", rule_type: str) -> "Key | None":
    """在内部盆中查找指定类型的推理规则键。"""
    kid = garden.rule_key_ids.get(rule_type)
    if kid is None:
        return None
    internal = garden.pots.get(garden.internal_pot_id)
    if internal is None:
        return None
    return internal.network.get_key(kid)


# ═══════════════════════════════════════════════
# 模式匹配
# ═══════════════════════════════════════════════

def _collect_activated_keys(garden: "Garden", result) -> dict[int, str]:
    """
    从传导结果中收集所有亮度 ≥ 1 的键。
    返回 {key_id: pot_id}。
    """
    activated: dict[int, str] = {}
    for pot_id, cr in result.round1.items():
        for kid in cr.final_pattern:
            activated[kid] = pot_id
    for pot_id, cr in result.round3.items():
        for kid in cr.final_pattern:
            if kid not in activated:
                activated[kid] = pot_id
    return activated


def match_deduction(
    garden: "Garden",
    activated_keys: dict[int, str],
) -> list[DeductionAction]:
    """
    演绎模式匹配：A --[rel]--> B --[rel]--> C ⇒ A --[rel]--> C

    条件：
    - A、B、C 都在本轮被激活
    - A→B 和 B→C 的电线有相同的 relation_type
    - 两线的 thickness ≥ INFERENCE_MIN_WIRE_THICKNESS
    - A、B、C 的 evidence_conf ≥ INFERENCE_MIN_EVIDENCE_CONF
    - A 和 C 之间还没有同类型的电线（避免重复推导）
    """
    actions: list[DeductionAction] = []

    for a_id, a_pot in activated_keys.items():
        pot_a = garden.pots.get(a_pot)
        if pot_a is None:
            continue
        key_a = pot_a.network.get_key(a_id)
        if key_a is None or key_a.evidence_conf < INFERENCE_MIN_EVIDENCE_CONF:
            continue

        neighbors_b = pot_a.network.get_neighbors(a_id)
        for b_id, ab_thickness in neighbors_b:
            if b_id not in activated_keys:
                continue
            key_b = pot_a.network.get_key(b_id)
            if key_b is None or key_b.evidence_conf < INFERENCE_MIN_EVIDENCE_CONF:
                continue

            ab_wire = pot_a.network.get_wire(a_id, b_id)
            if ab_wire is None:
                continue
            rel_type = ab_wire.relation_type
            if not rel_type or ab_thickness < INFERENCE_MIN_WIRE_THICKNESS:
                continue

            # 找 B 的邻居 C
            for c_id, bc_thickness in pot_a.network.get_neighbors(b_id):
                if c_id == a_id or c_id not in activated_keys:
                    continue
                key_c = pot_a.network.get_key(c_id)
                if key_c is None or key_c.evidence_conf < INFERENCE_MIN_EVIDENCE_CONF:
                    continue

                bc_wire = pot_a.network.get_wire(b_id, c_id)
                if bc_wire is None:
                    continue
                if bc_wire.relation_type != rel_type:
                    continue
                if bc_thickness < INFERENCE_MIN_WIRE_THICKNESS:
                    continue

                # 避免重复：A 和 C 之间已经有同类电线
                ac_wire = pot_a.network.get_wire(a_id, c_id)
                if ac_wire is not None:
                    continue

                # 计算合并证据
                combined_freq = (_weighted_sum([
                    (key_a.evidence_freq, key_a.evidence_conf),
                    (key_b.evidence_freq, key_b.evidence_conf),
                    (key_c.evidence_freq, key_c.evidence_conf),
                ]))
                combined_conf = min(
                    key_a.evidence_conf, key_b.evidence_conf, key_c.evidence_conf,
                ) * 0.7  # 推导知识置信度打折

                actions.append(DeductionAction(
                    key_a_id=a_id,
                    key_b_id=b_id,
                    key_c_id=c_id,
                    relation_type=rel_type,
                    combined_freq=combined_freq,
                    combined_conf=combined_conf,
                    pot_id=a_pot,
                ))

    return actions


def match_induction(
    garden: "Garden",
    activated_keys: dict[int, str],
) -> list[InductionAction]:
    """
    归纳模式匹配：多个同原语键 → 泛化键。

    条件：
    - ≥ 2 个键有相同的 primitive（且不是 WORD/RULE/PATTERN/CASE）
    - 它们共享至少一个槽位名称
    - 所有键的 evidence_conf ≥ INFERENCE_MIN_EVIDENCE_CONF
    """
    # 按 primitive 分组
    by_primitive: dict[str, list[tuple[int, str, "Key"]]] = {}
    excluded_primitives = {"WORD", "PATTERN", "RULE", "CASE", ""}

    for kid, pid in activated_keys.items():
        pot = garden.pots.get(pid)
        if pot is None:
            continue
        key = pot.network.get_key(kid)
        if key is None:
            continue
        prim = key.primitive
        if prim in excluded_primitives:
            continue
        if key.evidence_conf < INFERENCE_MIN_EVIDENCE_CONF:
            continue
        if not key.slots:
            continue  # 没有槽位的键无法归纳

        by_primitive.setdefault(prim, []).append((kid, pid, key))

    actions: list[InductionAction] = []

    for prim, entries in by_primitive.items():
        if len(entries) < 2:
            continue

        # 找出共享的槽位名称
        slot_sets = [set(e[2].slots.keys()) for e in entries]
        common_slots = slot_sets[0]
        for ss in slot_sets[1:]:
            common_slots = common_slots & ss

        if not common_slots:
            continue  # 没有共享槽位，无法归纳

        # 收集证据
        source_ids = [e[0] for e in entries]
        freqs = [e[2].evidence_freq for e in entries]
        confs = [e[2].evidence_conf for e in entries]

        combined_freq = sum(freqs) / len(freqs)
        combined_conf = min(confs) * 0.6  # 归纳置信度打折

        # 取第一个键的所在盆
        pot_id = entries[0][1]

        actions.append(InductionAction(
            primitive=prim,
            slot_names=tuple(sorted(common_slots)),
            source_key_ids=source_ids,
            combined_freq=combined_freq,
            combined_conf=combined_conf,
            pot_id=pot_id,
        ))

    return actions


def match_analogy(
    garden: "Garden",
    activated_keys: dict[int, str],
) -> list[AnalogyAction]:
    """
    类比模式匹配：A:B :: C:? → C:D。

    简化实现：
    - 找两对键 (A, B)，A 和 B 有相同的 primitive
    - A 和 C 之间有 similar-to 电线
    - C 的 primitive 与 A/B 不同
    - 推断 D：与 C 同 primitive，槽位从 A→B 的映射应用到 C

    条件：A、B、C 的 evidence_conf ≥ INFERENCE_MIN_EVIDENCE_CONF
    """
    actions: list[AnalogyAction] = []

    activated_list = list(activated_keys.items())
    for i in range(len(activated_list)):
        a_id, a_pot_id = activated_list[i]
        pot = garden.pots.get(a_pot_id)
        if pot is None:
            continue
        key_a = pot.network.get_key(a_id)
        if key_a is None or key_a.evidence_conf < INFERENCE_MIN_EVIDENCE_CONF:
            continue
        if not key_a.primitive or key_a.primitive in ("WORD", "PATTERN", "RULE", "CASE"):
            continue

        # 找 B：与 A 同 primitive 且在 A 的邻居中（similar-to 关系）
        for b_id, thickness in pot.network.get_neighbors(a_id):
            if b_id not in activated_keys:
                continue
            key_b = pot.network.get_key(b_id)
            if key_b is None or key_b.evidence_conf < INFERENCE_MIN_EVIDENCE_CONF:
                continue
            if key_b.primitive != key_a.primitive:
                continue

            ab_wire = pot.network.get_wire(a_id, b_id)
            if ab_wire is None or ab_wire.relation_type != "similar-to":
                continue

            # 找 C：与 A 有 similar-to 关系但 primitive 不同
            for c_id, _c_thickness in pot.network.get_neighbors(a_id):
                if c_id == b_id or c_id not in activated_keys:
                    continue
                key_c = pot.network.get_key(c_id)
                if key_c is None or key_c.evidence_conf < INFERENCE_MIN_EVIDENCE_CONF:
                    continue
                if key_c.primitive == key_a.primitive:
                    continue
                if not key_c.primitive or key_c.primitive in ("WORD", "PATTERN", "RULE", "CASE"):
                    continue

                ac_wire = pot.network.get_wire(a_id, c_id)
                if ac_wire is None or ac_wire.relation_type != "similar-to":
                    continue

                # 已经有 C→D 的关联？检查 C 是否已有 same-primitive 的邻居
                has_pair = False
                for nid, _ in pot.network.get_neighbors(c_id):
                    nk = pot.network.get_key(nid)
                    if nk is not None and nk.primitive == key_c.primitive:
                        has_pair = True
                        break
                if has_pair:
                    continue  # C 已经有配对了，不需要推理

                # 计算 A→B 的槽位映射，应用到 C
                new_slots: dict[str, int] = {}
                for sname, sval in key_c.slots.items():
                    new_slots[sname] = sval
                # 如果 A 和 B 有共享槽位名但不同值，把 C 的对应槽位替换
                for sname in key_a.slots:
                    if sname in key_b.slots:
                        if key_a.slots[sname] != key_b.slots[sname]:
                            # A 在这个槽位上变了 → C 在对应槽位上也变
                            if sname in key_c.slots:
                                new_slots[sname] = key_b.slots[sname]

                combined_freq = _weighted_sum([
                    (key_a.evidence_freq, key_a.evidence_conf),
                    (key_b.evidence_freq, key_b.evidence_conf),
                    (key_c.evidence_freq, key_c.evidence_conf),
                ])
                combined_conf = min(
                    key_a.evidence_conf, key_b.evidence_conf, key_c.evidence_conf,
                ) * 0.4  # 类比最不可靠

                actions.append(AnalogyAction(
                    key_a_id=a_id,
                    key_b_id=b_id,
                    key_c_id=c_id,
                    new_primitive=key_c.primitive,
                    new_slots=new_slots,
                    combined_freq=combined_freq,
                    combined_conf=combined_conf,
                    pot_id=a_pot_id,
                ))

    return actions


# ═══════════════════════════════════════════════
# 推理动作执行
# ═══════════════════════════════════════════════

def _execute_deduction(garden: "Garden", action: DeductionAction) -> "Key | None":
    """执行演绎动作：创建 A→C 的概念键。"""
    pot = garden.pots.get(action.pot_id)
    if pot is None:
        return None

    key_a = pot.network.get_key(action.key_a_id)
    key_b = pot.network.get_key(action.key_b_id)
    key_c = pot.network.get_key(action.key_c_id)
    if key_a is None or key_b is None or key_c is None:
        return None

    # 检查是否已存在（去重：用内容字符串唯一标识 A→C 推导）
    deduced_content = f"推导:{{{key_a.content[:20]}}}→{{{key_c.content[:20]}}}"
    for existing in pot.network.all_keys():
        if existing.content == deduced_content and not existing.is_dormant:
            return None  # 已存在

    # 创建新键
    new_key = pot.network.create_key(content=deduced_content, source="internal")
    new_key.primitive = key_b.primitive  # 继承中间键的原语
    # 槽位：用 source/target 通用名记录 A→C 的推导关系
    # （不复制 key_b.slots——slot 值域是概念 ID，与 key_a/key_c 的 ID 不重合）
    new_key.slots = {"source": action.key_a_id, "target": action.key_c_id}
    new_key.evidence_freq = action.combined_freq
    new_key.evidence_conf = action.combined_conf

    # 创建电线：新键 ↔ A, B, C（细线，全部标记 relation_type）
    for target_id, thickness in [
        (action.key_a_id, 0.15),
        (action.key_b_id, 0.10),
        (action.key_c_id, 0.15),
    ]:
        pot.network.set_wire(new_key.id, target_id, thickness=thickness)
        w = pot.network.get_wire(new_key.id, target_id)
        if w:
            w.relation_type = action.relation_type
            w.slippage_distance = SLIPPAGE_DISTANCE.get(action.relation_type, 0)

    return new_key


def _execute_induction(garden: "Garden", action: InductionAction) -> "Key | None":
    """执行归纳动作：创建泛化概念键。"""
    pot = garden.pots.get(action.pot_id)
    if pot is None:
        return None

    # 收集源键
    source_keys = []
    for sid in action.source_key_ids:
        k = pot.network.get_key(sid)
        if k is not None:
            source_keys.append(k)
    if not source_keys:
        return None

    # 检查是否已存在泛化键（排除源键自身）
    common_slot_names = set(action.slot_names)
    source_id_set = set(action.source_key_ids)
    for existing in pot.network.all_keys():
        if existing.id in source_id_set:
            continue  # 排除源键自身
        if existing.primitive == action.primitive and not existing.is_dormant:
            existing_slots = set(existing.slots.keys())
            if existing_slots == common_slot_names:
                # 已存在 → 不重复创建，改为增强 evidence
                existing.evidence_freq = (existing.evidence_freq + action.combined_freq) / 2
                existing.evidence_conf = max(existing.evidence_conf, action.combined_conf)
                return None

    # 创建泛化键：共享的槽位值保留，不同的设为 0（未绑定）
    new_key = pot.network.create_key(
        content=f"泛化:{action.primitive}",
        source="internal",
    )
    new_key.primitive = action.primitive

    # 取第一个源键的槽位结构，共享值保留，不共享的设为 0
    ref = source_keys[0]
    new_slots: dict[str, int] = {}
    for sname in action.slot_names:
        values = [k.slots.get(sname, 0) for k in source_keys]
        if len(set(values)) == 1 and values[0] != 0:
            new_slots[sname] = values[0]  # 所有实例共享的槽位值
        else:
            new_slots[sname] = 0  # 槽位值不同 → 泛化（未绑定）
    new_key.slots = new_slots

    new_key.evidence_freq = action.combined_freq
    new_key.evidence_conf = action.combined_conf

    # 创建电线：新键 ↔ 所有源键
    for sid in action.source_key_ids:
        pot.network.set_wire(new_key.id, sid, thickness=0.12)
        w = pot.network.get_wire(new_key.id, sid)
        if w:
            w.relation_type = "part-of"
            w.slippage_distance = SLIPPAGE_DISTANCE.get("part-of", 0)

    return new_key


def _execute_analogy(garden: "Garden", action: AnalogyAction) -> "Key | None":
    """执行类比动作：完成 C→D 的配对。"""
    pot = garden.pots.get(action.pot_id)
    if pot is None:
        return None

    key_c = pot.network.get_key(action.key_c_id)
    if key_c is None:
        return None

    # 检查去重
    for existing in pot.network.all_keys():
        if existing.primitive == action.new_primitive and not existing.is_dormant:
            if existing.slots == action.new_slots:
                return None  # 已存在

    content = f"类比:{key_c.content[:20]}→?"
    new_key = pot.network.create_key(content=content, source="internal")
    new_key.primitive = action.new_primitive
    new_key.slots = action.new_slots
    new_key.evidence_freq = action.combined_freq
    new_key.evidence_conf = action.combined_conf

    # 创建电线
    pot.network.set_wire(new_key.id, action.key_a_id, thickness=0.10)
    pot.network.set_wire(new_key.id, action.key_b_id, thickness=0.10)
    pot.network.set_wire(new_key.id, action.key_c_id, thickness=0.15)
    w = pot.network.get_wire(new_key.id, action.key_c_id)
    if w:
        w.relation_type = "similar-to"
        w.slippage_distance = SLIPPAGE_DISTANCE.get("similar-to", 0)

    return new_key


# ═══════════════════════════════════════════════
# 推理周期主入口
# ═══════════════════════════════════════════════

def run_inference_cycle(
    garden: "Garden",
    result,  # GardenConductionResult
    energy_budget: InferenceEnergyBudget,
) -> InferenceCycleResult:
    """
    传导后执行推理周期。

    1. 收集所有激活键
    2. 模式匹配：演绎 → 归纳 → 类比
    3. 按置信度排序，优先执行高置信度推理
    4. 消耗能量，创建新键
    5. 能量耗尽或达到上限 → 停止

    返回 InferenceCycleResult。
    """
    cycle_result = InferenceCycleResult(
        energy_remaining=energy_budget.current,
    )

    # 1. 收集激活键
    activated = _collect_activated_keys(garden, result)
    if not activated:
        return cycle_result

    # 2. 模式匹配
    deduction_actions = match_deduction(garden, activated)
    induction_actions = match_induction(garden, activated)
    analogy_actions = match_analogy(garden, activated)

    # 3. 按置信度排序（高 → 低）
    all_actions: list[tuple[str, float, object]] = []
    for a in deduction_actions:
        all_actions.append(("deduction", a.combined_conf, a))
    for a in induction_actions:
        all_actions.append(("induction", a.combined_conf, a))
    for a in analogy_actions:
        all_actions.append(("analogy", a.combined_conf, a))

    all_actions.sort(key=lambda x: x[1], reverse=True)

    # 4. 执行
    cost_map = {
        "deduction": ENERGY_COST_DEDUCTION,
        "induction": ENERGY_COST_INDUCTION,
        "analogy": ENERGY_COST_ANALOGY,
    }

    for rule_type, _conf, action in all_actions:
        if len(cycle_result.new_keys) >= MAX_NEW_KEYS_PER_CYCLE:
            break

        cost = cost_map[rule_type]
        if not energy_budget.can_afford(cost):
            break

        energy_budget.spend(cost)
        cycle_result.rule_fires[rule_type] = True
        cycle_result.actions_taken += 1

        # 执行 + 激活对应的规则键
        if rule_type == "deduction":
            new_key = _execute_deduction(garden, action)
        elif rule_type == "induction":
            new_key = _execute_induction(garden, action)
        else:
            new_key = _execute_analogy(garden, action)

        if new_key is not None:
            cycle_result.new_keys.append(new_key)

        # 激活推理规则键（标记为被触发了）
        rule_key = _find_rule_key(garden, rule_type)
        if rule_key is not None:
            rule_key.activate(1)

    cycle_result.energy_remaining = energy_budget.current
    return cycle_result


# ═══════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════

def _weighted_sum(evidence_pairs: list[tuple[float, float]]) -> float:
    """
    加权求 frequency。
    evidence_pairs: [(freq, conf), ...]
    返回加权平均 freq。
    """
    total_weight = sum(c for _, c in evidence_pairs)
    if total_weight == 0:
        return 0.5
    return sum(f * c for f, c in evidence_pairs) / total_weight
