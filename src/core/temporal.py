"""
键盘模型 AI — HTM 时序记忆 + 预测 + 注意力聚焦  v1.0

基于 HTM (Hierarchical Temporal Memory) 的三个核心机制映射到键网络：

1. 时序记忆：电线的 context 字典记录"前置键→后置键"的共现模式
   效果："刚提到小鱼干 → 现在说'盒子' → 优先激活'装小鱼干的铁盒'"

2. 预测状态：键的 prediction_state 三步状态机
   - 休眠(0) → 预测中(1) → 确认(2)
   预测由电线 context 驱动：当前激活键 → 查询出边 context → 预测后续键

3. 注意力聚焦：对话模式 vs 独处模式
   - 对话模式：外部种子，聚焦种子盆
   - 独处模式：自发传导（workspace 残留为种子），注意力锁防内耗

一切皆键+线：时序记忆存在电线里，预测由电线驱动，注意力由模式切换驱动。
设计边界：见 src/core/__init__.py 完整清单（名称+概念+代码）。

English:
HTM temporal memory, prediction and attention: wire contexts record
co-occurrence, prediction_state is a three-step state machine driven by
those contexts, and attention focus switches between dialogue and solitude.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .garden import Garden, GardenConductionResult
    from .key import Key
    from .wire import Wire


# ═══════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════

# 独处检测
SOLITUDE_THRESHOLD = 300           # 5 分钟无师父输入 → 进入独处模式

# 时序上下文
CONTEXT_BOOST_FACTOR = 0.3         # 上下文匹配时的额外传导权重（+30%）
CONTEXT_MIN_COOCCUR = 2            # 触发预测/加成的最小共现次数

# 预测
PREDICTION_MAX_KEYS = 5            # 每轮最多预测的键数
PREDICTION_STATE_TTL = 60          # 预测状态有效期（秒），超时回退
PREDICTION_CONFIRMED_TTL = 300     # 确认预测保留期（秒），超时回退到休眠
PREDICTION_EVIDENCE_BOOST = 0.05   # 预测命中 → 正面证据增量
PREDICTION_EVIDENCE_PENALTY = 0.03 # 预测未命中 → 负面证据增量

# 注意力锁
ATTENTION_LOCK_CONF_MIN = 0.3      # 注意力锁所需的最小 evidence_conf

# 独处传导
SOLITUDE_MAX_HOPS = 2              # 独处模式传导跳数
SOLITUDE_TEMP_BOOST = 0.15         # 独处模式温度额外升高


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class PredictionResult:
    """一次预测周期的结果。"""
    predicted_ids: list[int] = field(default_factory=list)    # 本轮新设为"预测中"的键 ID
    confirmed_ids: list[int] = field(default_factory=list)    # 本轮确认的键 ID（预测命中）
    failed_ids: list[int] = field(default_factory=list)       # 本轮失败的键 ID（预测过期未命中）
    attention_mode: str = "dialogue"                          # "dialogue" | "solitude"
    solitude_keys: list[int] = field(default_factory=list)    # 独处模式自发激活的键 ID


# ═══════════════════════════════════════════════
# 1. 时序记忆 — 电线上下文更新
# ═══════════════════════════════════════════════

def update_wire_context(
    garden: "Garden",
    conduction_result: "GardenConductionResult",
):
    """
    传导后更新电线时序上下文。

    从传导路径中提取"前置键→后置键"的序列对，
    递增 wire.context[prev_key_id]。

    电线成为序列记忆的载体——记录"当 prev_key_id 激活时，
    这条线曾传导过电流"的次数。
    """
    # 收集所有传导路径：{(pot_id, dst_key_id): (src_id, ...)}
    all_paths = conduction_result.all_activation_paths

    for (pot_id, dst_id), path_info in all_paths.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue

        # path_info = (src_id, src_content, dst_content, wire_thickness, push_amount)
        src_id = path_info[0]

        wire = pot.network.get_wire(src_id, dst_id)
        if wire is None:
            continue

        # 递增时序上下文：src_id 作为前置键，这条线被用过 1 次
        wire.context[src_id] = wire.context.get(src_id, 0) + 1


# ═══════════════════════════════════════════════
# 2. 上下文加成 — 传导时使用
# ═══════════════════════════════════════════════

def compute_context_boost(wire: "Wire", prev_key_id: int) -> float:
    """
    传导时计算上下文加成。

    如果 prev_key_id 在 wire.context 中出现过足够多次，
    返回 CONTEXT_BOOST_FACTOR（0.3），使这条线的传导权重 +30%。

    由 conduction._spread_activation 在每个传导步骤中调用。
    """
    ctx_count = wire.context.get(prev_key_id, 0)
    if ctx_count >= CONTEXT_MIN_COOCCUR:
        return CONTEXT_BOOST_FACTOR
    return 0.0


# ═══════════════════════════════════════════════
# 3. 预测状态机
# ═══════════════════════════════════════════════

def predict_next_keys(
    garden: "Garden",
    activated_key_ids: set[int],
) -> list[int]:
    """
    基于当前激活键和电线上下文，预测哪些键可能接下来被需要。

    算法：
    1. 收集"最近激活键"= workspace 中强度 >= 0.3 的键
    2. 对每个最近激活键，查找其所在 pot 的出边电线
    3. 如果 wire.context[recent_key] >= MIN_COOCCUR → 目标键加入候选
    4. 按 (context_count × evidence_expectation) 排序，取 top N
    5. 排除本轮已激活的键和已在预测中的键
    """
    from .evidence import truth_expectation

    candidates: dict[int, float] = {}  # {target_key_id: score}

    # 收集最近激活键（来自 workspace）
    recent_keys = [
        kid for kid, strength in garden.workspace.items()
        if strength >= 0.3
    ]

    for recent_id in recent_keys:
        # 找到这个键所在的 pot
        for pot in garden.pots.values():
            if pot.is_dormant:
                continue
            key = pot.network.get_key(recent_id)
            if key is None:
                continue

            # 遍历出边电线
            for neighbor_id, _thickness in pot.network.get_neighbors(recent_id):
                # 跳过本轮已激活的键
                if neighbor_id in activated_key_ids:
                    continue
                # 跳过已在预测中或已确认的键
                neighbor_key = pot.network.get_key(neighbor_id)
                if neighbor_key is None:
                    continue
                if neighbor_key.prediction_state in (1, 2):
                    continue

                wire = pot.network.get_wire(recent_id, neighbor_id)
                if wire is None:
                    continue

                ctx_count = wire.context.get(recent_id, 0)
                if ctx_count < CONTEXT_MIN_COOCCUR:
                    continue

                # 评分：共现次数 × 证据期望值
                exp = truth_expectation(
                    neighbor_key.evidence_freq,
                    neighbor_key.evidence_conf,
                )
                score = ctx_count * exp
                if score > candidates.get(neighbor_id, 0.0):
                    candidates[neighbor_id] = score

            break  # 键只属于一个盆，找到后无需继续遍历

    # 按分数排序，取 top N
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [kid for kid, _score in sorted_candidates[:PREDICTION_MAX_KEYS]]


def apply_predictions(garden: "Garden", predicted_ids: list[int]):
    """
    将预测键设为 prediction_state=1（预测中）。

    预测中的键：亮度不变，但激活阈值降低——
    在下一轮传导中更容易被激活（由传导中的 prediction preheat 实现）。

    记录预测时间戳到 garden._predicted_keys，用于 TTL 过期检查。
    """
    now = time.time()
    for kid in predicted_ids:
        for pot in garden.pots.values():
            if pot.is_dormant:
                continue
            key = pot.network.get_key(kid)
            if key is not None and not key.is_dormant:
                # 不覆盖已确认的预测（state=2）
                if key.prediction_state == 2:
                    break
                key.prediction_state = 1
                garden._predicted_keys[kid] = now
                break


def confirm_predictions(
    garden: "Garden",
    activated_ids: set[int],
) -> list[int]:
    """
    检查本轮实际激活的键中是否有 prediction_state=1 的键。

    命中 → prediction_state=2（确认），evidence 正面增强。
    返回确认的键 ID 列表。
    """
    from .evidence import revise_evidence

    confirmed: list[int] = []
    for kid in activated_ids:
        for pot in garden.pots.values():
            if pot.is_dormant:
                continue
            key = pot.network.get_key(kid)
            if key is None:
                continue
            if key.prediction_state == 1:
                # 预测命中！
                key.prediction_state = 2
                # 正面证据增强
                new_f, new_c = revise_evidence(
                    key.evidence_freq, key.evidence_conf,
                    1.0, PREDICTION_EVIDENCE_BOOST,
                )
                key.evidence_freq = new_f
                key.evidence_conf = new_c
                confirmed.append(kid)
                # 更新确认时间戳（用于后续 cooldown 回收）
                garden._predicted_keys[kid] = time.time()
                break

    return confirmed


def decay_predictions(garden: "Garden"):
    """
    过期预测清理（每次对话后调用）。

    - prediction_state=1 且超过 TTL 未确认 → 负面证据惩罚 → 回退到 0
    - prediction_state=2 且超过 CONFIRMED_TTL → 回退到 0（不再需要锁定）
    - 清理 _predicted_keys 中的过期条目
    """
    from .evidence import revise_evidence

    now = time.time()
    expired_ids: list[int] = []

    for kid, pred_time in list(garden._predicted_keys.items()):
        # 确定该键当前的状态和对应 TTL
        key_state = None
        for pot in garden.pots.values():
            if pot.is_dormant:
                continue
            k = pot.network.get_key(kid)
            if k is not None:
                key_state = k.prediction_state
                break

        if key_state == 1:
            if now - pred_time > PREDICTION_STATE_TTL:
                expired_ids.append(kid)
        elif key_state == 2:
            if now - pred_time > PREDICTION_CONFIRMED_TTL:
                expired_ids.append(kid)
        else:
            # 状态已变更（可能被其他路径修改），清理跟踪
            expired_ids.append(kid)

    for kid in expired_ids:
        garden._predicted_keys.pop(kid, None)
        for pot in garden.pots.values():
            if pot.is_dormant:
                continue
            key = pot.network.get_key(kid)
            if key is None:
                continue
            if key.prediction_state == 1:
                # 预测未命中 → 负面证据
                new_f, new_c = revise_evidence(
                    key.evidence_freq, key.evidence_conf,
                    0.0, PREDICTION_EVIDENCE_PENALTY,
                )
                key.evidence_freq = new_f
                key.evidence_conf = new_c
            # state=2 超时回退（不惩罚——预测曾正确）
            if key.prediction_state in (1, 2):
                key.prediction_state = 0
            break

    return expired_ids


# ═══════════════════════════════════════════════
# 4. 注意力聚焦 — 对话模式 vs 独处模式
# ═══════════════════════════════════════════════

def detect_attention_mode(garden: "Garden", now: float | None = None) -> str:
    """
    检测当前注意力模式。

    对话模式 ("dialogue")：师父在 SOLITUDE_THRESHOLD 秒内说过话
    独处模式 ("solitude")：师父超过 SOLITUDE_THRESHOLD 秒未说话
    """
    if now is None:
        now = time.time()
    elapsed = now - garden.last_master_input_at
    if garden.last_master_input_at > 0 and elapsed >= SOLITUDE_THRESHOLD:
        return "solitude"
    return "dialogue"


def run_solitude_cycle(garden: "Garden") -> list[tuple[int, str]]:
    """
    独处模式下的自发思考。

    1. 收集 workspace 中残留痕迹最强的键 → 作为自发种子
    2. 注意力锁：如果有高 evidence_conf 的预测中键，优先在它们附近活动
    3. 返回 [(key_id, pot_id), ...] 作为自发种子列表，供主循环当 all_seeds 使用

    返回空列表表示"完全没有想法，跳过本轮"。
    """
    seeds: list[tuple[int, str]] = []

    # 1. 注意力锁检查：是否有高置信度的预测键可以提供方向
    locked_keys: list[tuple[int, str, float]] = []  # [(key_id, pot_id, exp)]
    for pot in garden.pots.values():
        if pot.is_dormant:
            continue
        for key in pot.network.all_keys():
            if key.is_dormant:
                continue
            if key.prediction_state == 2 and key.evidence_conf >= ATTENTION_LOCK_CONF_MIN:
                from .evidence import truth_expectation
                exp = truth_expectation(key.evidence_freq, key.evidence_conf)
                locked_keys.append((key.id, pot.id, exp))

    if locked_keys:
        # 注意力锁存在 → 以预测确认键为锚点做小范围联想
        locked_keys.sort(key=lambda x: x[2], reverse=True)
        for kid, pid, _exp in locked_keys[:3]:
            seeds.append((kid, pid))
    else:
        # 无注意力锁 → 以 workspace 残留痕迹为种子
        if garden.workspace:
            sorted_ws = sorted(
                garden.workspace.items(),
                key=lambda x: x[1], reverse=True,
            )
            for kid, strength in sorted_ws[:5]:
                if strength < 0.2:
                    break
                # 找到键所在的 pot
                for pot in garden.pots.values():
                    if pot.is_dormant:
                        continue
                    if pot.network.get_key(kid) is not None:
                        seeds.append((kid, pot.id))
                        break

    return seeds


# ═══════════════════════════════════════════════
# 5. 预测周期主入口
# ═══════════════════════════════════════════════

def run_prediction_cycle(
    garden: "Garden",
    conduction_result: "GardenConductionResult",
) -> PredictionResult:
    """
    预测周期主入口（传导后调用）。

    执行顺序：
    1. update_wire_context()     — 记录本次传导的序列模式
    2. decay_predictions()       — 清理过期预测
    3. confirm_predictions()     — 检查本轮激活键中是否有预测命中
    4. predict_next_keys()       — 基于当前激活键预测下一轮
    5. apply_predictions()       — 设置新预测
    6. detect_attention_mode()   — 更新注意力模式

    返回 PredictionResult。
    """
    result = PredictionResult()

    # 1. 时序记忆更新
    update_wire_context(garden, conduction_result)

    # 2. 清理过期预测
    result.failed_ids = decay_predictions(garden)

    # 3. 确认本轮预测
    # 收集本轮所有激活键
    activated_ids: set[int] = set()
    for pot_id, cr in conduction_result.round1.items():
        activated_ids.update(cr.final_pattern.keys())
    for pot_id, cr in conduction_result.round3.items():
        activated_ids.update(cr.final_pattern.keys())

    result.confirmed_ids = confirm_predictions(garden, activated_ids)

    # 4. 预测下一轮
    result.predicted_ids = predict_next_keys(garden, activated_ids)

    # 5. 应用预测
    apply_predictions(garden, result.predicted_ids)

    # 6. 注意力模式
    result.attention_mode = detect_attention_mode(garden)
    garden.attention_mode = result.attention_mode

    return result
