"""
键盘模型 AI — CBR 案例推理（节能模式） v1.0

基于案例推理 (Case-Based Reasoning) 映射到键网络：

1. 案例 = 完整的激活模式快照
   - trigger_pattern: 触发这个案例的种子键 ID 序列
   - activation_snapshot: 传导后亮起的键及其亮度
   - response_pattern: 输出时使用的键 ID 序列

2. 快速路径（CBR 匹配）
   - 当前种子键模式 → 匹配已有案例的 trigger_pattern
   - 匹配成功 + 温度低 → 直接加载 activation_snapshot（跳过传导！）
   - 适配：比较当前种子和案例种子的差异，微调激活模式

3. 案例创建与演化
   - 每次正常传导完成后，如果是新模式 → 自动创建案例快照
   - 案例被成功复用时 → 频率+1，evidence 增强
   - 案例复用时适配失败 → 进入"待修正"状态
   - 长期不用的案例 → 休眠

一切皆键+线：案例是激活模式的记忆，不是独立的推理引擎。
设计边界：见 src/core/__init__.py 完整清单（名称+概念+代码）。

English:
Case-Based Reasoning mapped onto the key network: a case is an activation
snapshot; matching a trigger pattern at low temperature loads the snapshot
directly (skipping conduction) for fast, energy-saving responses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .garden import Garden, GardenConductionResult


# ═══════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════

# CBR 快速路径温度阈值：低于此值优先走 CBR
CBR_TEMPERATURE_THRESHOLD = 0.45

# 案例匹配的最小 Jaccard 相似度
CBR_MIN_MATCH_SIMILARITY = 0.3

# 案例匹配的最小种子键数（太少不值得缓存）
CBR_MIN_SEEDS_FOR_CASE = 2

# 案例创建的最小激活键数（太少不值得记录）
CBR_MIN_ACTIVATION_FOR_CREATE = 5

# 案例休眠阈值（天）
CBR_DORMANCY_DAYS = 14

# 案例最大数量（防止无限增长）
CBR_MAX_CASES = 200

# 案例失败容忍度：连续失败次数超过此值 → 标记为待修正
CBR_MAX_CONSECUTIVE_FAILS = 3

# 案例适配时，当前种子中不在 trigger_pattern 中的键的预热加成
CBR_ADAPT_BONUS = 0.15

# 案例应用时 activation_snapshot 键的基础预热强度
CBR_SNAPSHOT_PREHEAT = 0.4


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class Case:
    """一个案例：完整的激活模式快照。"""

    id: int
    trigger_pattern: list[int] = field(default_factory=list)
    # 触发这个案例的种子键 ID 序列

    activation_snapshot: dict[int, float] = field(default_factory=dict)
    # {key_id: brightness} — 传导后亮起的键

    response_keys: list[int] = field(default_factory=list)
    # 输出时使用的键 ID 序列（按重要性排序）

    # 涉及的花盆 ID 列表（用于快速判断案例是否覆盖当前活跃盆）
    pot_ids: list[str] = field(default_factory=list)

    # 统计
    frequency: int = 0              # 被成功复用的次数
    success_count: int = 0          # 复用后输出被接受的次数
    fail_count: int = 0             # 复用后输出不佳的次数
    consecutive_fails: int = 0      # 连续失败次数

    # 能耗
    energy_saved: float = 0.0       # 累计节省的传导迭代数（估算）

    # 时间戳
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    last_modified: float = 0.0

    # 状态
    is_dormant: bool = False
    needs_revision: bool = False    # 标记为待修正

    def __post_init__(self):
        if self.last_modified == 0.0:
            self.last_modified = self.created_at

    @property
    def reliability(self) -> float:
        """案例的可靠性分数 (0.0–1.0)。"""
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5  # 新案例，中性
        return self.success_count / total

    @property
    def age_days(self) -> float:
        """案例创建以来的天数。"""
        return (time.time() - self.created_at) / 86400

    @property
    def idle_days(self) -> float:
        """案例上次使用以来的天数。"""
        if self.last_used == 0:
            return self.age_days
        return (time.time() - self.last_used) / 86400

    def record_success(self, iterations_saved: int = 0):
        """记录一次成功的复用。"""
        self.frequency += 1
        self.success_count += 1
        self.consecutive_fails = 0
        self.last_used = time.time()
        self.energy_saved += iterations_saved
        self.needs_revision = False

    def record_failure(self):
        """记录一次失败的复用。"""
        self.frequency += 1
        self.fail_count += 1
        self.consecutive_fails += 1
        self.last_used = time.time()
        if self.consecutive_fails >= CBR_MAX_CONSECUTIVE_FAILS:
            self.needs_revision = True

    def snapshot(self) -> dict:
        """可序列化快照。"""
        return {
            "id": self.id,
            "trigger_pattern": self.trigger_pattern,
            "activation_snapshot": self.activation_snapshot,
            "response_keys": self.response_keys,
            "pot_ids": self.pot_ids,
            "frequency": self.frequency,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "consecutive_fails": self.consecutive_fails,
            "energy_saved": self.energy_saved,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "last_modified": self.last_modified,
            "is_dormant": self.is_dormant,
            "needs_revision": self.needs_revision,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "Case":
        """从快照恢复案例。"""
        # 确保 trigger_pattern 中的键是 int（JSON 序列化后可能是 str）
        trigger = [int(x) for x in data.get("trigger_pattern", [])]
        # 确保 activation_snapshot 的键是 int
        snapshot_dict = {
            int(k): float(v)
            for k, v in data.get("activation_snapshot", {}).items()
        }
        response = [int(x) for x in data.get("response_keys", [])]

        return cls(
            id=data["id"],
            trigger_pattern=trigger,
            activation_snapshot=snapshot_dict,
            response_keys=response,
            pot_ids=data.get("pot_ids", []),
            frequency=data.get("frequency", 0),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            consecutive_fails=data.get("consecutive_fails", 0),
            energy_saved=data.get("energy_saved", 0.0),
            created_at=data.get("created_at", time.time()),
            last_used=data.get("last_used", 0.0),
            last_modified=data.get("last_modified", 0.0),
            is_dormant=data.get("is_dormant", False),
            needs_revision=data.get("needs_revision", False),
        )

    def __repr__(self) -> str:
        status = "💤" if self.is_dormant else ("⚠" if self.needs_revision else "✓")
        return (f"Case#{self.id} {status} triggers={len(self.trigger_pattern)} "
                f"snap={len(self.activation_snapshot)}keys "
                f"rel={self.reliability:.2f} ×{self.frequency}")


# ═══════════════════════════════════════════════
# 1. 案例匹配
# ═══════════════════════════════════════════════

def match_case(
    garden: "Garden",
    seed_ids: list[int],
    temperature: float,
) -> Case | None:
    """
    在案例库中寻找与当前种子模式最匹配的案例。

    算法：
    1. 过滤：只考虑非休眠、非待修正的案例
    2. 评分：Jaccard 相似度 × reliability × 温度折扣
    3. 温度门控：高温时降低匹配分数（鼓励探索而非复用）
    4. 返回最佳匹配（如果分数 ≥ CBR_MIN_MATCH_SIMILARITY）

    温度门控逻辑：
    - 低温 (< 0.35)：匹配分数 × 1.2（鼓励复用）
    - 中温 (0.35–0.55)：匹配分数 × 1.0
    - 高温 (> 0.55)：匹配分数 × 0.6（鼓励探索，不走快速路径）
    """
    if len(seed_ids) < CBR_MIN_SEEDS_FOR_CASE:
        return None

    if not garden.case_library:
        return None

    seed_set = set(seed_ids)
    best_case: Case | None = None
    best_score = 0.0

    for case in garden.case_library.values():
        if case.is_dormant:
            continue
        if case.needs_revision:
            continue
        if not case.trigger_pattern:
            continue

        trigger_set = set(case.trigger_pattern)

        # Jaccard 相似度
        intersection = len(seed_set & trigger_set)
        union = len(seed_set | trigger_set)
        if union == 0:
            continue
        jaccard = intersection / union

        # 综合评分：Jaccard × reliability
        score = jaccard * case.reliability

        # 温度门控
        if temperature < 0.35:
            score *= 1.2   # 低温：鼓励复用
        elif temperature > 0.55:
            score *= 0.6   # 高温：抑制复用，鼓励探索

        if score > best_score:
            best_score = score
            best_case = case

    if best_score >= CBR_MIN_MATCH_SIMILARITY:
        return best_case
    return None


# ═══════════════════════════════════════════════
# 2. 案例应用（快速路径）
# ═══════════════════════════════════════════════

def apply_case(
    garden: "Garden",
    case: Case,
    seed_ids: list[int],
) -> dict[int, float]:
    """
    应用案例的激活模式快照，返回预热字典用于跳过传导。

    适配逻辑：
    1. 案例的 activation_snapshot 键 → 基础预热强度 CBR_SNAPSHOT_PREHEAT
    2. 当前种子中不在案例 trigger_pattern 中的键 → 额外预热 CBR_ADAPT_BONUS
       （这些是新出现的键，需要额外的激活支持）
    3. 案例 trigger_pattern 中不在当前种子中的键 → 预热折扣 0.5
       （这些键不再相关，降低其影响）

    返回的预热字典可以直接传入 conduction 的 workspace_preheat，
    也可以被主循环用来跳过传导（亮度≥1 的键直接视为"已激活"）。
    """
    preheat: dict[int, float] = {}

    case_trigger_set = set(case.trigger_pattern)
    seed_set = set(seed_ids)

    # 1. 加载案例激活快照
    for kid, brightness in case.activation_snapshot.items():
        # 基础预热强度与原始亮度成正比
        preheat[kid] = CBR_SNAPSHOT_PREHEAT * (0.5 + 0.5 * min(brightness / 2.0, 1.0))

    # 2. 当前种子中的新键（不在案例 trigger 中）→ 额外预热
    new_seeds = seed_set - case_trigger_set
    for kid in new_seeds:
        if kid in preheat:
            preheat[kid] += CBR_ADAPT_BONUS
        else:
            preheat[kid] = CBR_ADAPT_BONUS

    # 3. 案例 trigger 中但不在当前种子中的键 → 折减
    missing_triggers = case_trigger_set - seed_set
    for kid in missing_triggers:
        if kid in preheat:
            preheat[kid] *= 0.5

    # 4. 裁剪：只保留强度 ≥ 0.1 的预热
    preheat = {k: min(v, 1.0) for k, v in preheat.items() if v >= 0.1}

    return preheat


# ═══════════════════════════════════════════════
# 3. 案例创建
# ═══════════════════════════════════════════════

def create_case_from_conduction(
    garden: "Garden",
    seed_ids: list[int],
    seed_pot_ids: list[str],
    conduction_result: "GardenConductionResult",
    response_key_ids: list[int],
) -> Case | None:
    """
    从一次成功的传导创建案例快照。

    条件：
    - 激活键总数 ≥ CBR_MIN_ACTIVATION_FOR_CREATE
    - 案例库未满（< CBR_MAX_CASES）
    - 种子键数 ≥ CBR_MIN_SEEDS_FOR_CASE

    如果案例库满了，休眠最老的案例腾空间。
    """
    if len(seed_ids) < CBR_MIN_SEEDS_FOR_CASE:
        return None

    # 收集所有激活键
    activation: dict[int, float] = {}
    for pot_id, cr in conduction_result.round1.items():
        for kid, brightness in cr.final_pattern.items():
            activation[kid] = float(brightness)
    for pot_id, cr in conduction_result.round3.items():
        for kid, brightness in cr.final_pattern.items():
            if kid not in activation:
                activation[kid] = float(brightness)
            else:
                activation[kid] = max(activation[kid], float(brightness))

    if len(activation) < CBR_MIN_ACTIVATION_FOR_CREATE:
        return None

    # 去重检查：是否已有非常相似的案例（Jaccard ≥ 0.8）
    seed_set = set(seed_ids)
    for existing in garden.case_library.values():
        if existing.is_dormant:
            continue
        existing_set = set(existing.trigger_pattern)
        intersection = len(seed_set & existing_set)
        union = len(seed_set | existing_set)
        if union > 0 and intersection / union >= 0.8:
            # 太相似 → 不创建新案例，更新已有案例
            _update_existing_case(existing, activation, response_key_ids, seed_pot_ids)
            return None

    # 案例库满了 → 清理
    if len(garden.case_library) >= CBR_MAX_CASES:
        _evict_oldest_case(garden)

    # 分配 ID
    case_id = _next_case_id(garden)

    case = Case(
        id=case_id,
        trigger_pattern=list(seed_ids),
        activation_snapshot=activation,
        response_keys=response_key_ids,
        pot_ids=list(seed_pot_ids),
    )

    garden.case_library[case_id] = case
    return case


def _update_existing_case(
    case: Case,
    new_activation: dict[int, float],
    response_keys: list[int],
    pot_ids: list[str],
):
    """用新的传导结果更新已有案例的快照。"""
    # 合并激活快照：保留重叠键，加入新键
    merged: dict[int, float] = {}
    all_keys = set(case.activation_snapshot.keys()) | set(new_activation.keys())
    for kid in all_keys:
        old_val = case.activation_snapshot.get(kid, 0.0)
        new_val = new_activation.get(kid, 0.0)
        # 指数移动平均
        merged[kid] = old_val * 0.7 + new_val * 0.3

    case.activation_snapshot = merged
    case.response_keys = response_keys
    case.pot_ids = list(set(case.pot_ids) | set(pot_ids))
    case.last_modified = time.time()
    # 新鲜数据进入 → 清除待修正标记
    if case.needs_revision:
        case.needs_revision = False
        case.consecutive_fails = 0


def _next_case_id(garden: "Garden") -> int:
    """分配下一个案例 ID。"""
    if not garden.case_library:
        return 1
    return max(garden.case_library.keys()) + 1


def _evict_oldest_case(garden: "Garden"):
    """驱逐最老的案例（优先休眠的，其次最久未用的）。"""
    if not garden.case_library:
        return

    # 优先驱逐休眠案例
    dormant = [
        (cid, c) for cid, c in garden.case_library.items() if c.is_dormant
    ]
    if dormant:
        oldest = min(dormant, key=lambda x: x[1].last_used)
        del garden.case_library[oldest[0]]
        return

    # 否则驱逐最久未用的
    oldest = min(garden.case_library.items(), key=lambda x: x[1].last_used)
    del garden.case_library[oldest[0]]


# ═══════════════════════════════════════════════
# 4. 案例演化
# ═══════════════════════════════════════════════

def record_case_outcome(
    case: Case,
    success: bool,
    iterations_saved: int = 0,
):
    """
    记录案例复用结果。

    - 成功：success_count++, consecutive_fails=0, energy_saved += iterations
    - 失败：fail_count++, consecutive_fails++, 可能标记 needs_revision
    """
    if success:
        case.record_success(iterations_saved)
    else:
        case.record_failure()


def revise_case(
    case: Case,
    new_activation: dict[int, float],
    new_response: list[int],
):
    """
    修正一个标记为 needs_revision 的案例。

    用新的传导结果更新快照。
    """
    case.activation_snapshot = {
        k: v for k, v in new_activation.items()
    }
    case.response_keys = list(new_response)
    case.needs_revision = False
    case.consecutive_fails = 0
    case.last_modified = time.time()


# ═══════════════════════════════════════════════
# 5. 案例维护
# ═══════════════════════════════════════════════

def decay_cases(garden: "Garden"):
    """
    案例库维护（在 garden.maintenance 中调用）。

    1. 长期未用的案例 → 休眠
    2. 可靠性过低的休眠案例 → 删除
    3. 待修正但长期未修正的案例 → 删除
    """
    now = time.time()
    to_delete: list[int] = []

    for cid, case in list(garden.case_library.items()):
        idle_days = (now - case.last_used) / 86400 if case.last_used > 0 else case.age_days

        # 超过 DORMANCY_DAYS 天未用 → 休眠
        if idle_days >= CBR_DORMANCY_DAYS and not case.is_dormant:
            case.is_dormant = True

        # 休眠超过 30 天 → 删除
        if case.is_dormant and idle_days >= 30:
            to_delete.append(cid)
            continue

        # 待修正超过 7 天未修正 → 删除
        if case.needs_revision and (now - case.last_modified) > 7 * 86400:
            to_delete.append(cid)
            continue

        # 可靠性 < 0.2 且被用过 ≥ 5 次 → 删除（烂案例）
        if case.reliability < 0.2 and case.frequency >= 5:
            to_delete.append(cid)
            continue

    for cid in to_delete:
        del garden.case_library[cid]


def get_case_stats(garden: "Garden") -> dict:
    """获取案例库统计信息。"""
    cases = list(garden.case_library.values())
    if not cases:
        return {
            "total": 0,
            "active": 0,
            "dormant": 0,
            "needs_revision": 0,
            "avg_reliability": 0.0,
            "total_energy_saved": 0.0,
        }

    active = [c for c in cases if not c.is_dormant]
    return {
        "total": len(cases),
        "active": len(active),
        "dormant": sum(1 for c in cases if c.is_dormant),
        "needs_revision": sum(1 for c in cases if c.needs_revision),
        "avg_reliability": (
            sum(c.reliability for c in active) / len(active) if active else 0.0
        ),
        "total_energy_saved": sum(c.energy_saved for c in cases),
    }
