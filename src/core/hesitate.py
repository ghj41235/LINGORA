"""
键盘模型 AI — 犹豫检测算法

犹豫 = 亮键格局在多个方向之间来回摆动，振幅不收敛。
不是"还在想"——还在想意味着往收敛走。
犹豫是"两个方向势均力敌，选不了"。

罗小黑模式下：脑回路简单，不太容易犹豫。
真犹豫时表现为"单纯困惑"，不纠结分析。

English:
Hesitation detection: when the lit-key pattern oscillates between directions
and amplitude fails to converge, the system reports hesitation rather than
continuing to think.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network import KeyNetwork
    from .key import Key

# ── 参数 ──────────────────────────────────────

HESITATION_MIN_TIME = 10_000   # 10 秒后才开始判定犹豫（毫秒）
HESITATION_MAX_TIME = 20_000   # 20 秒未收敛 → 判定犹豫（毫秒）
WINDOW_SIZE = 50               # 振幅历史窗口（轮数）
CONVERGENCE_THRESHOLD = -0.05  # 振幅下降超过此值 → 在收敛
MIN_AMPLITUDE = 0.08           # 低于此值 → 基本稳定，不判犹豫
                               # （比传导终止阈值 0.05 宽松，防止误判）


# ── 检测函数 ────────────────────────────────────

def detect_hesitation(
    network: KeyNetwork,
    amplitude_history: list[float],
    elapsed_ms: float,
) -> tuple[bool, str | None]:
    """
    判断当前是否处于"犹豫"状态。

    返回: (是否犹豫, 犹豫描述或 None)
    """

    # 条件 1：还没到判定窗口
    if elapsed_ms < HESITATION_MIN_TIME:
        return False, None

    # 取最近 N 轮振幅
    recent = amplitude_history[-min(len(amplitude_history), WINDOW_SIZE):]
    if len(recent) < 10:
        return False, None  # 数据不够

    # 分析趋势：前半段 vs 后半段平均振幅
    mid = len(recent) // 2
    first_half = sum(recent[:mid]) / mid
    second_half = sum(recent[mid:]) / (len(recent) - mid)
    trend = second_half - first_half

    # 条件 2：振幅在下降？→ 正在收敛
    if trend < CONVERGENCE_THRESHOLD:
        return False, None

    # 条件 3：平均振幅已经很低 → 其实快稳了
    avg = sum(recent) / len(recent)
    if avg < MIN_AMPLITUDE:
        return False, None

    # 条件 4：超过 20 秒 → 不管趋势，判犹豫
    if elapsed_ms > HESITATION_MAX_TIME:
        return True, _format_hesitation(network)

    # 条件 5：10-20 秒之间，振幅持平或上升
    if trend >= -0.05:
        return True, _format_hesitation(network)

    return False, None


# ── 犹豫表达 ────────────────────────────────────

def _format_hesitation(network: KeyNetwork) -> str:
    """
    分析当前亮键格局，生成罗小黑风格的犹豫表达。
    罗小黑不分析自己为什么纠结——只是单纯地困惑。
    """

    # 找竞争集群
    cluster_a, cluster_b = _find_competing_clusters(network)

    if cluster_a and cluster_b:
        topic_a = _pick_topic(cluster_a)
        topic_b = _pick_topic(cluster_b)
        return (
            f"师父，我在想……{topic_a}和{topic_b}之间，"
            f"好像有点矛盾，我不太懂。"
        )

    # 散射状
    bright = network.get_bright_keys()
    if len(bright) >= 3:
        preview = bright[0].content[:15]
        return f"唔……想到{preview}这些，但有点乱，让我再想想。"

    return "喵？这个问题好难……"


def _find_competing_clusters(
    network: KeyNetwork,
) -> tuple[list[Key] | None, list[Key] | None]:
    """
    找亮键中亮度最高的两个互不直接连接的集群。
    它们之间的"张力"就是犹豫的来源。
    """
    bright = [k for k in network.get_bright_keys() if k.brightness >= 1]
    if len(bright) < 4:
        return None, None

    # 按亮度降序，取靠前的
    bright.sort(key=lambda k: k.brightness, reverse=True)

    # 分成两组：彼此之间电线细的分为不同组
    group_a, group_b = [], []
    for k in bright:
        score_a = _avg_wire_to_group(network, k, group_a)
        score_b = _avg_wire_to_group(network, k, group_b)
        if score_a >= score_b:
            group_a.append(k)
        else:
            group_b.append(k)

    if len(group_a) >= 2 and len(group_b) >= 2:
        return group_a, group_b
    return None, None


def _avg_wire_to_group(
    network: KeyNetwork,
    key: Key,
    group: list[Key],
) -> float:
    """计算键到组内所有键的平均电线粗细。"""
    if not group:
        return 0.0
    total = 0.0
    for member in group:
        wire = network.get_wire(key.id, member.id)
        if wire:
            total += wire.thickness
    return total / len(group)


def _pick_topic(cluster: list[Key]) -> str:
    """从键集群中选一个代表性的话题词。"""
    if not cluster:
        return "…"
    # 选内容最短的——通常是核心实体
    cluster.sort(key=lambda k: len(k.content))
    preview = cluster[0].content[:20]
    return preview if preview else "…"
