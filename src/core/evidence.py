"""
键盘模型 AI — NARS 证据量模块  v1.0

基于 NARS (Non-Axiomatic Reasoning System) 的证据量真值理论：
每个键的真实性不由 content 字符串定义，而由它和其他键的全部连线历史定义。

truth = (frequency, confidence)
- frequency ∈ [0.0, 1.0]: 正面证据占比
- confidence ∈ [0.0, 1.0]: 总证据量 / (总证据量 + K)

期望值 expectation = confidence × (frequency - 0.5) + 0.5
  → 高 confidence + 高 frequency = 强信念
  → 低 confidence = 不确定（无论 frequency 多少）

证据来源：
- 每次被激活 → 正面证据
- 种子键（师父提到的）→ 强正面证据
- 桥接触发的键 → 跨域正面证据
- 犹豫/方向切换 → 负面证据
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .garden import Garden
    from .key import Key


# ═══════════════════════════════════════════════
# NARS 参数
# ═══════════════════════════════════════════════

# 每次证据事件的 confidence 增量
EVIDENCE_SEED_CONF = 0.15            # 种子键（师父提到的）→ 强正面
EVIDENCE_BRIGHT2_CONF = 0.10         # 亮度 = 2 的激活键
EVIDENCE_BRIGHT1_CONF = 0.05         # 亮度 = 1 的激活键
EVIDENCE_BRIDGE_CONF = 0.12          # 桥接触发的跨域键
EVIDENCE_COACT_CONF = 0.06           # 共激活关联键
EVIDENCE_HESITATION_NEG = 0.08       # 犹豫中的参与键（负面）
EVIDENCE_DIRECTION_NEG = 0.05        # 方向切换的参与键（负面）

# 衰减
EVIDENCE_DECAY_INTERVAL = 86400      # 1 天（秒）
EVIDENCE_CONF_DECAY_RATE = 0.001     # 每天 confidence 衰减量
EVIDENCE_CONF_MIN = 0.01             # confidence 最低值


# ═══════════════════════════════════════════════
# 核心公式
# ═══════════════════════════════════════════════

def truth_expectation(frequency: float, confidence: float) -> float:
    """
    NARS 真值期望值。
    公式：expectation = confidence × (frequency - 0.5) + 0.5

    含义：
    - (0.9, 0.8) → 0.82  "很可能是真的，而且我有不少证据"
    - (0.9, 0.1) → 0.54  "看起来像真的，但只有很少证据"
    - (0.1, 0.8) → 0.18  "很可能是假的，证据也不少"
    - (0.1, 0.1) → 0.46  "看起来像假的，但证据也不多"
    - (0.5, _)   → 0.50  "正反各半，不好判断"
    """
    return confidence * (frequency - 0.5) + 0.5


def _clamp_freq(value: float) -> float:
    """frequency 必须落在 [0.0, 1.0]，越界即夹回。"""
    return min(1.0, max(0.0, value))


def _clamp_conf(value: float) -> float:
    """confidence 必须落在 [EVIDENCE_CONF_MIN, 0.99]，越界即夹回。"""
    return min(0.99, max(EVIDENCE_CONF_MIN, value))


def revise_evidence(
    old_freq: float,
    old_conf: float,
    event_freq: float,
    event_conf: float,
) -> tuple[float, float]:
    """
    NARS 证据修正规则。
    将新证据 (event_freq, event_conf) 合并到旧真值 (old_freq, old_conf)。

    event_freq 取值：
    - 1.0 = 正面证据（这个知识被确认/用到了）
    - 0.0 = 负面证据（这个知识可能是错的）
    - 0.5 = 中性目标值（把 frequency 拉向 0.5；只有 frequency 已是 0.5 时才不变）

    公式：
    w1 = old_conf (旧证据权重)
    w2 = event_conf (新证据权重)
    new_freq = (old_freq * w1 + event_freq * w2) / (w1 + w2)
    """
    if old_conf + event_conf == 0.0:
        # 短路分支也必须统一夹紧：不能绕开 confidence 下限 / frequency 范围。
        return _clamp_freq(old_freq), _clamp_conf(old_conf)

    new_freq = (old_freq * old_conf + event_freq * event_conf) / (old_conf + event_conf)
    new_freq = _clamp_freq(new_freq)
    # confidence 用简化的增量模型（原版 NARS 用更复杂的公式）
    # 这里采用：每次证据事件增加 confidence，但趋近 1.0
    new_conf = old_conf + event_conf * (1.0 - old_conf)
    new_conf = _clamp_conf(new_conf)

    return new_freq, new_conf


# ═══════════════════════════════════════════════
# 证据更新（传导后）
# ═══════════════════════════════════════════════

def update_evidence_after_conduction(
    garden: "Garden",
    seeds_by_pot: dict[str, list[tuple[int, int]]],
    result,  # GardenConductionResult
) -> dict[int, float]:
    """
    传导后批量更新所有参与键的证据。

    按优先级处理：
    1. 种子键 → 强正面（外部确认）
    2. 亮度 ≥ 2 的激活键 → 正面
    3. 亮度 = 1 的激活键 → 弱正面
    4. 桥接触发键 → 跨域正面
    5. 犹豫/方向切换 → 负面证据

    返回 {key_id: total_delta}（用于调试日志）。
    """
    deltas: dict[int, float] = {}

    # ── 1. 种子键：师父直接提到的 → 强正面 ──
    for pot_id, seed_list in seeds_by_pot.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue
        for key_id, _brightness in seed_list:
            key = pot.network.get_key(key_id)
            if key is not None:
                new_f, new_c = revise_evidence(
                    key.evidence_freq, key.evidence_conf,
                    1.0, EVIDENCE_SEED_CONF,
                )
                delta = abs(key.evidence_freq - new_f) + abs(key.evidence_conf - new_c)
                key.evidence_freq = new_f
                key.evidence_conf = new_c
                deltas[key_id] = deltas.get(key_id, 0.0) + delta

    # ── 2-3. 传导激活键：按亮度分级 ──
    for pot_id, cr in result.round1.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue
        for key_id, brightness in cr.final_pattern.items():
            key = pot.network.get_key(key_id)
            if key is None:
                continue
            conf = EVIDENCE_BRIGHT2_CONF if brightness >= 2 else EVIDENCE_BRIGHT1_CONF
            new_f, new_c = revise_evidence(
                key.evidence_freq, key.evidence_conf,
                1.0, conf,
            )
            delta = abs(key.evidence_freq - new_f) + abs(key.evidence_conf - new_c)
            key.evidence_freq = new_f
            key.evidence_conf = new_c
            deltas[key_id] = deltas.get(key_id, 0.0) + delta

        # ── 3b. 共激活：同一盆内同时亮的键 → 关联证据 ──
        active_ids = list(cr.final_pattern.keys())
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                ki, kj = active_ids[i], active_ids[j]
                wire = pot.network.get_wire(ki, kj)
                if wire is not None:
                    # 两键有电线且同时亮 → 关联证据
                    for kid in (ki, kj):
                        k = pot.network.get_key(kid)
                        if k is not None:
                            new_f, new_c = revise_evidence(
                                k.evidence_freq, k.evidence_conf,
                                0.8, EVIDENCE_COACT_CONF,
                            )
                            delta = abs(k.evidence_freq - new_f) + abs(k.evidence_conf - new_c)
                            k.evidence_freq = new_f
                            k.evidence_conf = new_c
                            deltas[kid] = deltas.get(kid, 0.0) + delta

    # ── 4. 桥接触发键 → 跨域正面 ──
    for pot_id, cr in result.round3.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue
        for key_id in cr.final_pattern:
            key = pot.network.get_key(key_id)
            if key is not None:
                new_f, new_c = revise_evidence(
                    key.evidence_freq, key.evidence_conf,
                    0.7, EVIDENCE_BRIDGE_CONF,
                )
                delta = abs(key.evidence_freq - new_f) + abs(key.evidence_conf - new_c)
                key.evidence_freq = new_f
                key.evidence_conf = new_c
                deltas[key_id] = deltas.get(key_id, 0.0) + delta

    # ── 5. 犹豫/方向切换 → 负面证据 ──
    for pot_id, cr in result.round1.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue
        if cr.is_hesitating:
            for key_id in cr.final_pattern:
                key = pot.network.get_key(key_id)
                if key is not None:
                    new_f, new_c = revise_evidence(
                        key.evidence_freq, key.evidence_conf,
                        0.0, EVIDENCE_HESITATION_NEG,
                    )
                    delta = abs(key.evidence_freq - new_f) + abs(key.evidence_conf - new_c)
                    key.evidence_freq = new_f
                    key.evidence_conf = new_c
                    deltas[key_id] = deltas.get(key_id, 0.0) + delta
        if cr.direction_changed:
            for key_id in cr.final_pattern:
                key = pot.network.get_key(key_id)
                if key is not None:
                    new_f, new_c = revise_evidence(
                        key.evidence_freq, key.evidence_conf,
                        0.0, EVIDENCE_DIRECTION_NEG,
                    )
                    delta = abs(key.evidence_freq - new_f) + abs(key.evidence_conf - new_c)
                    key.evidence_freq = new_f
                    key.evidence_conf = new_c
                    deltas[key_id] = deltas.get(key_id, 0.0) + delta

    return deltas


# ═══════════════════════════════════════════════
# 证据衰减
# ═══════════════════════════════════════════════

def decay_all_evidence(garden: "Garden"):
    """
    对所有活跃盆中的键执行证据衰减。

    长期未激活的键，confidence 缓慢下降。
    永不低于 EVIDENCE_CONF_MIN (0.01)。
    在 maintenance() 中周期性调用。
    """
    now = time.time()
    for pot in garden.pots.values():
        if pot.is_dormant:
            continue
        for key in pot.network.all_keys():
            last_active = max(key.last_activated_at, key.created_at)
            idle = now - last_active
            if idle > EVIDENCE_DECAY_INTERVAL:
                # 处女键（从未被赋予证据，conf=0.0）没有证据可衰减，跳过——
                # 否则 max(EVIDENCE_CONF_MIN, 0.0 - 正数) 会把 confidence 反向抬升到 0.01。
                if key.evidence_conf <= 0.0:
                    continue
                days = idle / 86400.0
                key.evidence_conf = max(
                    EVIDENCE_CONF_MIN,
                    key.evidence_conf - EVIDENCE_CONF_DECAY_RATE * days,
                )


# ═══════════════════════════════════════════════
# 诊断
# ═══════════════════════════════════════════════

def evidence_label(freq: float, conf: float) -> str:
    """返回证据的人类可读标签。"""
    exp = truth_expectation(freq, conf)
    if conf < 0.1:
        return "未知"
    if exp > 0.75:
        return "确信"
    if exp > 0.6:
        return "比较确定"
    if exp > 0.4:
        return "不确定"
    if exp > 0.25:
        return "怀疑"
    return "否定"
