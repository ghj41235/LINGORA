"""
键盘模型 AI — Copycat 温度 + 工作空间  v1.0

全局温度反映系统对当前理解的信心，控制传导的搜索广度：
- 高温（> 0.7）：不确定，广撒网——激活扩散跳数多、阈值低、滑移启用
- 中温（0.3–0.7）：有一定方向——定向搜索
- 低温（< 0.3）：很确定——收敛到最佳答案，激活范围收窄

工作空间：当前对话的临时结构。激活过的键在工作空间中留下痕迹，
下轮对话时预热这些键（降低激活阈值），话题转换时衰减。

参考：Hofstadter & Mitchell (1995) Copycat
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .garden import Garden
    from .conduction import ConductionResult


# ═══════════════════════════════════════════════
# 温度参数
# ═══════════════════════════════════════════════

TEMPERATURE_BASELINE = 0.5       # 默认中温
TEMPERATURE_MIN = 0.05           # 最低温度（不完全冻结）
TEMPERATURE_MAX = 0.95           # 最高温度（不完全随机）

# 温度调节因子权重
TEMP_SEED_SCARCITY_WEIGHT = 0.20    # 种子稀缺时升温幅度
TEMP_POT_SCATTER_WEIGHT = 0.15      # 种子跨盆分散时升温幅度
TEMP_FOLLOWUP_COOLING = 0.15        # 师父追问时降温幅度
TEMP_HESITATION_HEATING = 0.12      # 上轮犹豫时升温幅度
TEMP_RETURN_RATE = 0.05             # 每次对话后温度向基线回归的速率

# 滑移温度阈值
SLIPPAGE_SIMILAR_THRESHOLD = 0.3    # similar-to/part-of/cause-of 在此温度以上启用
SLIPPAGE_OPPOSITE_THRESHOLD = 0.7   # opposite-of 在此温度以上启用

# ═══════════════════════════════════════════════
# 工作空间参数
# ═══════════════════════════════════════════════

WORKSPACE_DECAY_RATE = 0.85         # 每轮衰减倍率
WORKSPACE_PREHEAT_THRESHOLD = 0.3   # 痕迹强度高于此值 → 预热
WORKSPACE_PREHEAT_BOOST = 0.15      # 预热降低的跳档阈值（绝对值）
WORKSPACE_BRIGHT2_STRENGTH = 1.0    # 亮度 2 键的工作空间痕迹强度
WORKSPACE_BRIGHT1_STRENGTH = 0.5    # 亮度 1 键的工作空间痕迹强度


# ═══════════════════════════════════════════════
# 温度计算
# ═══════════════════════════════════════════════

def calculate_temperature(
    garden: "Garden",
    seeds_by_pot: dict[str, list[tuple[int, int]]],
    is_followup: bool = False,
    prev_hesitated: bool = False,
) -> float:
    """
    根据当前上下文计算全局温度。

    输入：
    - garden: 花园（读取/写入 global_temperature）
    - seeds_by_pot: {pot_id: [(key_id, brightness), ...]}
    - is_followup: 师父是否在追问（短问题紧跟长回答之后）
    - prev_hesitated: 上一轮是否发生了犹豫

    返回：新的全局温度 (0.05–0.95)

    温度由四个因素决定：
    1. 种子稀缺度：种子越少 → 温度越高（信息不够，广撒网）
    2. 盆分散度：种子跨越多盆 → 温度越高（方向不明确）
    3. 追问冷却：师父追问 → 温度降低（聚焦当前话题）
    4. 犹豫加热：上轮犹豫 → 温度升高（换方向探索）
    """
    # 从基线开始
    temp = garden.global_temperature

    # 统计种子
    total_seeds = sum(len(v) for v in seeds_by_pot.values())
    pot_count_with_seeds = len(seeds_by_pot)

    # ── 因素 1：种子稀缺度 ──
    if total_seeds <= 2:
        temp += TEMP_SEED_SCARCITY_WEIGHT        # +0.20
    elif total_seeds <= 5:
        temp += TEMP_SEED_SCARCITY_WEIGHT * 0.5  # +0.10
    elif total_seeds >= 12:
        temp -= TEMP_SEED_SCARCITY_WEIGHT * 0.5  # -0.10（种子多 → 聚焦）

    # ── 因素 2：盆分散度 ──
    if pot_count_with_seeds >= 5:
        temp += TEMP_POT_SCATTER_WEIGHT          # +0.15
    elif pot_count_with_seeds >= 3:
        temp += TEMP_POT_SCATTER_WEIGHT * 0.5    # +0.075
    # 只有一个盆 → 不加（方向明确）

    # ── 因素 3：师父追问 → 降温聚焦 ──
    if is_followup:
        temp -= TEMP_FOLLOWUP_COOLING            # -0.15

    # ── 因素 4：上轮犹豫 → 升温探索 ──
    if prev_hesitated:
        temp += TEMP_HESITATION_HEATING          # +0.12

    # ── 向基线回归 ──
    # 每次对话后温度微微向 0.5 回归（防止极端值持续太久）
    if temp > TEMPERATURE_BASELINE:
        temp -= TEMP_RETURN_RATE
    elif temp < TEMPERATURE_BASELINE:
        temp += TEMP_RETURN_RATE

    # ── 钳制 ──
    temp = max(TEMPERATURE_MIN, min(TEMPERATURE_MAX, temp))

    # 写入花园
    garden.global_temperature = temp

    return temp


def is_followup_question(text: str, prev_response_len: int, time_since_last: float) -> bool:
    """
    判断当前输入是否为对上一轮回答的追问。

    追问特征：
    - 输入很短（< 15 字）
    - 距上次对话不到 60 秒
    - 上一轮回答较长（> 50 字）→ 师父在深入讨论
    """
    if len(text) > 15:
        return False
    if time_since_last > 60:
        return False
    if prev_response_len < 50:
        return False
    return True


# ═══════════════════════════════════════════════
# 滑移判断
# ═══════════════════════════════════════════════

def slippage_allowed(slippage_distance: int, temperature: float) -> bool:
    """
    判断在给定温度下，给定滑移距离的电线是否应该被遍历。

    滑移距离 0 (same-as)：始终允许
    滑移距离 1 (similar-to / part-of / cause-of)：温度 >= 0.3 时允许
    滑移距离 2 (opposite-of)：温度 >= 0.7 时允许
    """
    if slippage_distance == 0:
        return True
    if slippage_distance == 1:
        return temperature >= SLIPPAGE_SIMILAR_THRESHOLD
    if slippage_distance >= 2:
        return temperature >= SLIPPAGE_OPPOSITE_THRESHOLD
    return True


def slippage_discount(slippage_distance: int, temperature: float) -> float:
    """
    计算滑移电线的推力折扣。

    折扣公式：1.0 / (1.0 + distance * (1.0 - temperature))

    含义：
    - 高温时折扣小（几乎和直接连线一样有效）
    - 低温时折扣大（滑移几乎不传导）

    示例：
    - distance=1, temp=0.8: 1/(1+0.2) = 0.833
    - distance=1, temp=0.4: 1/(1+0.6) = 0.625
    - distance=2, temp=0.8: 1/(1+0.4) = 0.714
    - distance=0: 1/(1+0) = 1.0（无折扣）
    """
    if slippage_distance == 0:
        return 1.0
    return 1.0 / (1.0 + slippage_distance * (1.0 - temperature))


# ═══════════════════════════════════════════════
# 温度 → 传导参数映射
# ═══════════════════════════════════════════════

def effective_jump_threshold(temperature: float, base_threshold: float = 1.0) -> float:
    """
    温度调整后的跳档阈值。

    高温 → 阈值降低（更容易跳档 → 搜索更广）
    低温 → 阈值升高（更难跳档 → 搜索更窄）

    公式：base * (1.5 - temperature)
    - temp=0.5: 1.0 * 1.0 = 1.0（不变）
    - temp=0.8: 1.0 * 0.7 = 0.7（更易跳）
    - temp=0.2: 1.0 * 1.3 = 1.3（更难跳）
    """
    return base_threshold * (1.5 - temperature)


def effective_hop_decay(temperature: float, base_decay: float = 0.65) -> float:
    """
    温度调整后的每跳衰减系数。

    高温 → 衰减减少（传得更远）
    低温 → 衰减增加（就近停止）

    公式：base * (1.2 - temperature * 0.4)
    - temp=0.5: 0.65 * 1.0 = 0.65（不变——与原版传导一致）
    - temp=0.8: 0.65 * 0.88 = 0.572（少衰减）
    - temp=0.2: 0.65 * 1.12 = 0.728（多衰减）
    """
    return base_decay * (1.2 - temperature * 0.4)


# ═══════════════════════════════════════════════
# 工作空间管理
# ═══════════════════════════════════════════════

def update_workspace(
    garden: "Garden",
    bright_keys_by_pot: dict[str, list[tuple[int, int]]],
):
    """
    传导结束后更新工作空间。

    亮过的键在工作空间中留下痕迹：
    - 亮度 2 → 痕迹强度 1.0
    - 亮度 1 → 痕迹强度 0.5
    - 已有痕迹的键 → 取最大值（不叠加）
    """
    for pot_id, key_brightnesses in bright_keys_by_pot.items():
        for key_id, brightness in key_brightnesses:
            strength = WORKSPACE_BRIGHT2_STRENGTH if brightness >= 2 else WORKSPACE_BRIGHT1_STRENGTH
            # 取最大值
            current = garden.workspace.get(key_id, 0.0)
            garden.workspace[key_id] = max(current, strength)


def decay_workspace(garden: "Garden"):
    """
    衰减工作空间中的所有痕迹。

    每次对话后调用。痕迹乘以 WORKSPACE_DECAY_RATE (0.85)。
    痕迹低于 0.05 的键从工作空间中移除。
    """
    to_remove = []
    for key_id in list(garden.workspace.keys()):
        garden.workspace[key_id] *= WORKSPACE_DECAY_RATE
        if garden.workspace[key_id] < 0.05:
            to_remove.append(key_id)

    for key_id in to_remove:
        del garden.workspace[key_id]


def get_workspace_preheat(garden: "Garden") -> dict[int, float]:
    """
    获取工作空间中应被预热的键及其预热强度。

    返回 {key_id: threshold_reduction}。
    预热降低这些键的激活阈值，使它们在下轮对话中更容易被推亮。

    只在痕迹强度 > WORKSPACE_PREHEAT_THRESHOLD 时生效。
    """
    preheat: dict[int, float] = {}
    for key_id, strength in garden.workspace.items():
        if strength >= WORKSPACE_PREHEAT_THRESHOLD:
            # 痕迹越强，预热越大（最多降低 0.15 的阈值）
            boost = WORKSPACE_PREHEAT_BOOST * (strength / WORKSPACE_BRIGHT2_STRENGTH)
            preheat[key_id] = min(boost, WORKSPACE_PREHEAT_BOOST)
    return preheat


def clear_workspace(garden: "Garden"):
    """
    清空工作空间（话题转换时调用）。
    目前由 decay_workspace 自然清空，此函数保留用于显式重置。
    """
    garden.workspace.clear()


# ═══════════════════════════════════════════════
# 温度诊断（调试用）
# ═══════════════════════════════════════════════

def temperature_label(temp: float) -> str:
    """返回温度的人类可读标签。"""
    if temp > 0.7:
        return "高温·广撒网"
    elif temp > 0.3:
        return "中温·定向搜索"
    else:
        return "低温·聚焦收敛"


def temperature_report(garden: "Garden") -> str:
    """生成温度诊断报告。"""
    t = garden.global_temperature
    ws_count = len(garden.workspace)
    active_ws = sum(
        1 for s in garden.workspace.values()
        if s >= WORKSPACE_PREHEAT_THRESHOLD
    )
    bar = "█" * int(t * 20) + "░" * (20 - int(t * 20))
    return (
        f"温度 [{bar}] {t:.3f} ({temperature_label(t)}) | "
        f"工作空间 {ws_count} 键（{active_ws} 预热中）"
    )
