"""
键盘模型 AI — 三段传导算法

"思考"的物理过程：
种子键亮了 → 电流沿电线扩散 → 邻居键跳档 → 旧键衰减 →
格局变化 → 继续传导 → 振幅收敛到稳定态 → 重播种 →
新一轮传导 → ... → 跨轮次收敛 → 思考结束。

外层：思考轮次（cycles）——每次将上一轮的结论重新播种
内层：传导迭代（iterations）——每轮内的扩散+衰减+跳档

不搜索、不匹配、不预测——只是电流顺着电线跑，跑到跑不动为止。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network import KeyNetwork
    from .key import Key


# ── 传导参数 ──────────────────────────────────

HOP_DECAY = 0.65           # 每跳推力衰减系数
JUMP_PUSH = 1.0            # 推力 ≥ 此值 → 跳一档 (0→1 或 1→2)
WEAK_PUSH = 0.3            # 推力 ≥ 此值但 < JUMP_PUSH → 不跳（累积用）
STABILITY_THRESHOLD = 0.05 # 振幅低于此值 → 本传导轮次稳定

# 循环参数
MAX_CYCLES = 5             # 最多重播种轮次
RE_SEED_TOP_K = 8          # 每轮结束后取最亮的 K 个键重播种
CYCLE_CONVERGENCE = 0.03   # 两轮之间的格局差异低于此值 → 真正稳定
DIRECTION_CHANGE = 0.30    # 两轮之间的格局差异高于此值 → 方向切换

# 注意力参数
FOCUS_TOP_K = 5            # 每轮取最亮的 K 个键作为焦点
FOCUS_BOOST = 1.5          # 焦点键扩散推力倍率
NON_FOCUS_DAMPEN = 0.3     # 非焦点键扩散推力倍率


# ── 请求 / 结果 ────────────────────────────────

@dataclass
class ConductionRequest:
    """传给传导引擎的请求。"""
    seed_keys: list[tuple[int, int]]  # [(key_id, initial_brightness), ...]
    max_hops: int = -1                # -1 = 不限，1-2 = 日常，3-5 = 复杂
    timeout_ms: int = 10_000          # 超时（毫秒）
    source: str = "user_input"        # "user_input" | "master_input" | "internal"
    temperature: float = 0.5          # v1.0: 全局温度 (0.0–1.0)，控制搜索广度
    workspace_preheat: dict[int, float] | None = None  # v1.0: {key_id: threshold_reduction}
    relation_boost: dict[str, float] | None = None  # M4: 探照灯，按 relation_type 调导电率


@dataclass
class ConductionResult:
    """传导引擎返回的结果。"""
    iterations: int = 0
    elapsed_ms: float = 0.0
    final_pattern: dict[int, int] = field(default_factory=dict)   # {key_id: brightness}
    is_stable: bool = False
    is_hesitating: bool = False
    hesitation_detail: str | None = None
    top_keys: list = field(default_factory=list)                   # list[Key]
    amplitude_history: list[float] = field(default_factory=list)   # 每轮振幅
    # 循环追踪
    cycles: int = 0                           # 完成的思考轮次数
    cycle_patterns: list[dict] = field(default_factory=list)  # 每轮结束时的格局
    direction_changed: bool = False           # 是否发生了方向切换
    # 传导路径：记录每个被推亮的键的"来源"
    # {dst_id: (src_id, src_content, dst_content, wire_thickness, push_amount)}
    activation_paths: dict[int, tuple[int, str, str, float, float]] = field(default_factory=dict)
    # M4: 峰值格局——每键本轮到过的最高亮度（"闪过 2 就算喊过一嗓子"）
    peak_pattern: dict[int, int] = field(default_factory=dict)


# ── 传导主函数 ─────────────────────────────────

def conduct(network: KeyNetwork, request: ConductionRequest) -> ConductionResult:
    """
    对网络执行传导——即一次完整的"思考"。

    外层循环（思考轮次）：
    1. 种子键设亮度
    2. 内层循环：扩散 → 衰减 → 算振幅 → 判断稳定
    3. 稳定后：记录格局，与前一轮比较
    4. 未收敛：取最亮的键重播种 → 下一轮
    5. 收敛或方向切换或超时 → 返回

    v1.0：温度影响搜索广度（高温 → 低阈值 + 滑移），
          工作空间预热降低已知相关键的激活门槛。
    """
    start_time = time.time()
    all_amp_history: list[float] = []
    cycle_patterns: list[dict[int, int]] = []
    total_iterations = 0
    direction_changed = False
    all_paths: dict[int, tuple[int, str, str, float, float]] = {}  # 跨轮累积传导路径

    # ── v1.0: 工作空间预热 ──────────────────────
    preheat = request.workspace_preheat or {}

    # ── M4: 探照灯 + 峰值 ──────────────────────
    rel_boost = request.relation_boost or {}
    peaks: dict[int, int] = {}

    # ── 1. 初始化种子键 ────────────────────────
    fresh_keys: set[int] = set()
    for key_id, init_brightness in request.seed_keys:
        key = network.get_key(key_id)
        if key is None:
            continue
        key.activate(init_brightness)
        peaks[key_id] = max(peaks.get(key_id, 0), init_brightness)
        fresh_keys.add(key_id)

    # ── 2. 外层：思考轮次循环 ──────────────────
    for cycle in range(MAX_CYCLES):
        # 2a. 内层：传导迭代（扩散→衰减→稳定）
        inner_iters, inner_amps, cycle_paths = _run_conduction_pass(
            network, fresh_keys, request.max_hops, start_time,
            request.temperature, preheat, rel_boost,
        )
        total_iterations += inner_iters
        all_amp_history.extend(inner_amps)

        # M4: 记峰值（每键本轮到过的最高亮度）
        for k in network.get_bright_keys():
            peaks[k.id] = max(peaks.get(k.id, 0), k.brightness)

        # 合并本轮路径（后来的不覆盖先来的——第一批路径最贴近原始种子）
        for dst_id, path_info in cycle_paths.items():
            if dst_id not in all_paths:
                all_paths[dst_id] = path_info

        # 2b. 记录本轮结束时的格局
        current_pattern = _snapshot_brightness(network)
        cycle_patterns.append(current_pattern)

        # 2c. 与上一轮比较
        if len(cycle_patterns) >= 2:
            delta = _calc_amplitude(cycle_patterns[-2], cycle_patterns[-1])

            if delta < CYCLE_CONVERGENCE:
                # 格局几乎不变 → 真正收敛
                elapsed = (time.time() - start_time) * 1000
                return _build_cycle_result(
                    network, total_iterations, elapsed, True, False,
                    all_amp_history, cycle + 1, cycle_patterns, False,
                    all_paths, peaks,
                )

            if delta > DIRECTION_CHANGE:
                # 格局大幅变化 → 思考换了方向
                direction_changed = True
                break

        # 2d. 检查超时
        elapsed = (time.time() - start_time) * 1000
        if elapsed > request.timeout_ms:
            return _build_cycle_result(
                network, total_iterations, elapsed, False, False,
                all_amp_history, cycle + 1, cycle_patterns, direction_changed,
                all_paths, peaks,
            )

        # 2e. 重播种：本轮最亮的键成为下一轮的种子
        fresh_keys = _reseeding(network)

        # 如果没有可重播种的键 → 停止
        if not fresh_keys:
            break

    # ── 3. 退出：超时 / 方向切换 / 达到最大轮次 ──
    elapsed = (time.time() - start_time) * 1000
    # 注意：direction_changed 只代表格局变化大（发散），不等于犹豫。
    # 犹豫由 hesitate.py 的 detect_hesitation() 根据时间/振幅统一判定。
    return _build_cycle_result(
        network, total_iterations, elapsed,
        not direction_changed, False,
        all_amp_history, min(cycle + 1, MAX_CYCLES), cycle_patterns,
        direction_changed, all_paths, peaks,
    )


# ── 内层传导轮次 ────────────────────────────────

def _run_conduction_pass(
    network: KeyNetwork,
    fresh_keys: set[int],
    max_hops: int,
    start_time: float,
    temperature: float = 0.5,
    preheat: dict[int, float] | None = None,
    rel_boost: dict[str, float] | None = None,
) -> tuple[int, list[float], dict[int, tuple[int, str, str, float, float]]]:
    """
    运行一次传导轮次（内层循环）。
    扩散 → 衰减 → 振幅 → 判断稳定/超时/跳数限制。

    v1.0：温度调整跳档阈值和每跳衰减；工作空间预热降低已知键的激活门槛。

    返回：(迭代数, 振幅历史列表, 激活路径)
    激活路径：{dst_id: (src_id, src_content, wire_thickness, push_amount)}
    只记录本轮首次被激活（0→>0）的键的来源。
    """
    amp_history: list[float] = []
    iteration = 0
    prev_pattern = _snapshot_brightness(network)
    all_paths: dict[int, tuple[int, str, str, float, float]] = {}

    while True:
        iteration += 1
        network.total_iterations += 1

        # 选焦点：当前最亮的 K 个键
        bright = network.get_bright_keys()
        bright.sort(key=lambda k: k.brightness, reverse=True)
        focus_ids = {k.id for k in bright[:FOCUS_TOP_K]}

        # 扩散（焦点加权 + 温度 + 滑移 + 探照灯）→ 收集激活路径
        newly_activated, paths = _spread_activation(
            network, focus_ids, temperature, preheat, rel_boost,
        )

        # 合并本轮路径（新激活的键才记录路径）
        all_paths.update(paths)

        # 衰减（本轮种子 + 新激活的键不衰减）
        _apply_decay(network, fresh_keys | newly_activated)
        fresh_keys.clear()

        # 振幅
        curr_pattern = _snapshot_brightness(network)
        amplitude = _calc_amplitude(prev_pattern, curr_pattern)
        amp_history.append(amplitude)

        # 判断
        elapsed = (time.time() - start_time) * 1000

        if amplitude < STABILITY_THRESHOLD:
            return iteration, amp_history, all_paths

        if max_hops > 0 and iteration >= max_hops:
            return iteration, amp_history, all_paths

        if elapsed > 10_000:  # 内层有自己独立的超时吗？不，由外层统一管理
            # 这里给一个合理的上限，实际超时由外层检测
            if iteration >= 20:
                return iteration, amp_history, all_paths

        prev_pattern = curr_pattern


# ── 重播种 ──────────────────────────────────────

def _reseeding(network: KeyNetwork) -> set[int]:
    """
    取当前最亮的 K 个键，清空所有亮度，将它们设为下一轮的种子。

    返回新种子键的 ID 集合。
    """
    top = network.get_top_k_keys(RE_SEED_TOP_K)
    # 筛选亮度 ≥ 1 的
    candidates = [(k.id, k.brightness) for k in top if k.brightness >= 1]

    if not candidates:
        return set()

    # 清空所有亮度
    for key in network.get_bright_keys():
        key.brightness = 0

    # 重新激活候选键
    fresh: set[int] = set()
    for kid, brightness in candidates:
        key = network.get_key(kid)
        if key is None:
            continue
        key.activate(brightness)  # 恢复原来的亮度
        fresh.add(kid)

    return fresh


# ── 扩散算法 ────────────────────────────────────

def _spread_activation(
    network: KeyNetwork,
    focus_ids: set[int] | None = None,
    temperature: float = 0.5,
    preheat: dict[int, float] | None = None,
    rel_boost: dict[str, float] | None = None,
) -> tuple[set[int], dict[int, tuple[int, str, str, float, float]]]:
    """
    每个亮键向邻居扩散推力。
    推力 = source_brightness × wire_thickness × hop_decay × weight × slippage_discount

    v1.0: 温度影响搜索广度：
    - 温度调整跳档阈值（高温 → 低阈值 → 更容易跳档）
    - 温度调整每跳衰减（高温 → 少衰减 → 传得更远）
    - 滑移折扣：根据电线的 relation_type/slippage_distance 调整推力

    焦点键（focus_ids）推力 × FOCUS_BOOST，非焦点 × NON_FOCUS_DAMPEN。
    工作空间预热降低已知键的跳档阈值。
    多个键同时推同一个邻居时，推力累积。
    累积推力 ≥ effective_jump → 邻居亮度 +1（不超过 2）。

    返回：
    - activated: 本轮被激活（亮度从 0 变为非 0）的键 ID 集合
    - paths: {dst_id: (src_id, src_content, dst_content, wire_thickness, push_amount)}
      记录每个被激活键的主要来源（贡献推力最大的源键）
    """
    from .temperature import (
        effective_jump_threshold,
        effective_hop_decay,
        slippage_allowed,
        slippage_discount,
    )

    bright_keys = network.get_bright_keys()
    focus_ids = focus_ids or set()
    preheat = preheat or {}
    activated: set[int] = set()

    # ── 温度调整后的参数 ──
    eff_jump = effective_jump_threshold(temperature, JUMP_PUSH)
    eff_decay = effective_hop_decay(temperature, HOP_DECAY)

    # 第一步：累积每个邻居从所有源键收到的推力（注意力加权 + 温度 + 滑移）
    # 同时追踪每个邻居的"最佳来源"
    accumulated: dict[int, float] = {}
    best_source: dict[int, tuple[int, str, str, float, float]] = {}  # neighbor → (src_id, src_content, dst_content, thickness, push)

    for src in bright_keys:
        # 焦点加权
        weight = FOCUS_BOOST if src.id in focus_ids else NON_FOCUS_DAMPEN

        neighbors = network.get_neighbors(src.id)
        for neighbor_id, thickness in neighbors:
            dst = network.get_key(neighbor_id)
            if dst is None or dst.is_dormant:
                continue

            # ── v1.0: 获取电线，检查滑移 ──
            wire = network.get_wire(src.id, neighbor_id)
            slippage_dist = wire.slippage_distance if wire else 0

            # 滑移允许性检查
            if not slippage_allowed(slippage_dist, temperature):
                continue  # 温度太低，这个滑移距离的电线不被遍历

            # 滑移折扣
            slip_discount = slippage_discount(slippage_dist, temperature)

            # ── v1.0 HTM: 时序上下文加成 ──
            # 如果源键最近在工作空间中被激活过，且这条电线记录了它作为
            # 前置键的历史 → 传导权重 +30%（"这条路我们走过，优先走"）
            ctx_boost = 0.0
            if wire is not None and src.id in preheat:
                from .temporal import compute_context_boost
                ctx_boost = compute_context_boost(wire, src.id)

            # ── M4: 探照灯（按电线关系类型调导电率）──
            rel_boost_mult = 1.0
            if wire is not None and rel_boost:
                rel_boost_mult = rel_boost.get(wire.relation_type, 1.0)

            push = src.brightness * thickness * eff_decay * weight * slip_discount * (1.0 + ctx_boost) * rel_boost_mult
            accumulated[neighbor_id] = accumulated.get(neighbor_id, 0.0) + push

            # 追踪贡献最大的源键（比较折扣后的 push）
            prev_best = best_source.get(neighbor_id)
            if prev_best is None or push > prev_best[4]:
                best_source[neighbor_id] = (src.id, src.content, dst.content, thickness, push)

    # 第二步：累积推力 ≥ 有效跳档阈值 → 激活
    # 工作空间预热 + HTM 预测状态 → 降低跳档阈值
    paths: dict[int, tuple[int, str, str, float, float]] = {}
    for neighbor_id, total_push in accumulated.items():
        dst = network.get_key(neighbor_id)
        if dst is None:
            continue
        # 工作空间预热：降低该键的跳档阈值
        preheat_boost = preheat.get(neighbor_id, 0.0)
        # HTM 预测：prediction_state=1 的键阈值降低 0.15
        prediction_boost = 0.15 if dst.prediction_state == 1 else 0.0
        threshold = max(0.3, eff_jump - preheat_boost - prediction_boost)

        if total_push >= threshold:
            old = dst.brightness
            dst.activate(old + 1)  # activate 内部 cap 到 2
            if old == 0 and dst.brightness > 0:
                activated.add(neighbor_id)
                # 记录传导路径（被谁推亮的）
                if neighbor_id in best_source:
                    paths[neighbor_id] = best_source[neighbor_id]

    return activated, paths


# ── 衰减算法 ────────────────────────────────────

def _apply_decay(network: KeyNetwork, fresh_keys: set[int] | None = None):
    """
    所有非新鲜键亮度 -1。
    新鲜键 = 本轮刚被外部设为亮度的种子键（由 fresh_keys 指定）。
    """
    fresh = fresh_keys or set()
    for key in network.get_bright_keys():
        if key.id in fresh:
            continue
        key.decay()


# ── 振幅计算 ────────────────────────────────────

def _calc_amplitude(
    prev: dict[int, int],
    curr: dict[int, int],
) -> float:
    """
    振幅 = 亮度发生变化的键的加权变化 / 总键数。
    归一化到 [0, 1]。
    """
    all_ids = set(prev.keys()) | set(curr.keys())
    if not all_ids:
        return 0.0

    changes = 0
    for kid in all_ids:
        a = prev.get(kid, 0)
        b = curr.get(kid, 0)
        if a != b:
            changes += abs(b - a)

    return changes / len(all_ids)


# ── 工具 ──────────────────────────────────────

def _snapshot_brightness(network: KeyNetwork) -> dict[int, int]:
    """拍摄当前网络中所有非零亮度的键。"""
    return {k.id: k.brightness for k in network.get_bright_keys()}


def _build_cycle_result(
    network: KeyNetwork,
    iterations: int,
    elapsed_ms: float,
    is_stable: bool,
    is_hesitating: bool,
    amp_history: list[float],
    cycles: int,
    cycle_patterns: list[dict],
    direction_changed: bool,
    activation_paths: dict[int, tuple[int, str, str, float, float]] | None = None,
    peak_pattern: dict[int, int] | None = None,
) -> ConductionResult:
    """构建含循环信息和传导路径的结果。"""
    top = network.get_top_k_keys(10)
    return ConductionResult(
        iterations=iterations,
        elapsed_ms=elapsed_ms,
        final_pattern=_snapshot_brightness(network),
        is_stable=is_stable,
        is_hesitating=is_hesitating,
        top_keys=top,
        amplitude_history=amp_history,
        cycles=cycles,
        cycle_patterns=cycle_patterns,
        direction_changed=direction_changed,
        activation_paths=activation_paths or {},
        peak_pattern=peak_pattern or _snapshot_brightness(network),
    )
