"""
键盘模型 AI — 消化与遗忘算法

消化 = Hebbian 学习：一起亮的键之间电线自动加粗。
遗忘 = 电线衰减 + 键休眠：长期不用的键压缩内容但不删除。

消化让关联变强，遗忘防止无限膨胀。两者对抗，网络自然生长。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .wire import SLIPPAGE_DISTANCE

if TYPE_CHECKING:
    from .network import KeyNetwork
    from .key import Key


# ── 消化参数 ──────────────────────────────────

BASE_DELTA = 0.1             # 基础加粗量
BRIGHTNESS_MULTIPLIER = 0.05 # 亮度乘积系数

# ── 遗忘参数 ──────────────────────────────────

DECAY_START = 3600            # 1 小时内的不衰减（秒）
DECAY_RATE = 0.01             # 每天基础衰减 0.01
MIN_THICKNESS = 0.001         # 线永不断
REPETITION_BONUS = 0.5        # 遗忘曲线：重复衰减因子的系数

# ── 休眠参数 ──────────────────────────────────

DORMANCY_THRESHOLD = 30 * 86400  # 30 天未激活 → 休眠（秒）


# ═══════════════════════════════════════════════
# 消化
# ═══════════════════════════════════════════════

def digest_after_conduction(network: KeyNetwork):
    """
    传导结束后立即消化。
    所有本轮亮度 > 0 的键两两之间 → 电线加粗。

    delta = BASE_DELTA + (brightness_a × brightness_b × BRIGHTNESS_MULTIPLIER)

    两个都亮度=2：delta = 0.1 + 0.2 = 0.3（最大）
    一个=1 一个=2：delta = 0.1 + 0.1 = 0.2
    两个都=1：    delta = 0.1 + 0.05 = 0.15
    """
    active = [k for k in network.get_bright_keys()]

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            delta = BASE_DELTA + (a.brightness * b.brightness * BRIGHTNESS_MULTIPLIER)
            network.strengthen_wire(a.id, b.id, delta)


def digest_new_key(
    network: KeyNetwork,
    new_key: Key,
    context_keys: list[Key],
):
    """
    新键与上下文键建立初始弱电线（粗度 0.05）。
    如果已有电线（比如同批新键之间已经互连了更粗的线）→ 不覆盖。
    后续对话中若同时亮，会自然加粗。
    """
    INITIAL_WEAK = 0.05
    for ctx in context_keys:
        if ctx.id == new_key.id:
            continue
        existing = network.get_wire(new_key.id, ctx.id)
        if existing is None:
            network.set_wire(new_key.id, ctx.id, thickness=INITIAL_WEAK)


def _text_bigram_sim(a: str, b: str) -> float:
    """快速 bigram 相似度（用于连线排序）。"""
    if not a or not b:
        return 0.0
    set_a = set(a[i:i + 2] for i in range(len(a) - 1))
    set_b = set(b[i:i + 2] for i in range(len(b) - 1))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def digest_batch_new_keys(
    network: KeyNetwork,
    new_keys: list[Key],
    context_keys: list[Key],
    interconnect_thickness: float = 0.3,
    max_interconnect: int = 4,
    max_context_connect: int = 5,
):
    """
    一批新键的批量消化：
    - 新键之间互连（限制每个键的连接数，不完全图）
    - 新键与上下文键建立弱连线（只连最相似的，不全连）

    用于文档投喂后。
    电线密度控制：避免 O(n²) 全连接导致噪音。
    """
    # ── 新键互连：只连最相似的邻居（滑动窗口 + 相似度排序） ──
    for i in range(len(new_keys)):
        nk = new_keys[i]
        # 与后续键计算相似度
        scored = []
        for j in range(len(new_keys)):
            if i == j:
                continue
            sim = _text_bigram_sim(nk.content, new_keys[j].content)
            scored.append((new_keys[j], sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        # 只连最相似的 top-N
        for other, sim in scored[:max_interconnect]:
            existing = network.get_wire(nk.id, other.id)
            if existing is None:
                network.set_wire(nk.id, other.id, thickness=interconnect_thickness * (0.5 + 0.5 * sim))
                w = network.get_wire(nk.id, other.id)
                if w:
                    rel = "similar-to" if sim > 0.5 else "part-of"
                    w.relation_type = rel
                    w.slippage_distance = SLIPPAGE_DISTANCE.get(rel, 0)

    # ── 新键与上下文键：只连最相似的 ──
    INITIAL_WEAK = 0.05
    for nk in new_keys:
        scored = []
        for ck in context_keys:
            if ck.id == nk.id:
                continue
            sim = _text_bigram_sim(nk.content, ck.content)
            scored.append((ck, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        # 只连最相似的 + 相似度达标（≥ 0.15）的
        connected = 0
        for ck, sim in scored:
            if connected >= max_context_connect:
                break
            if sim < 0.15:
                break
            existing = network.get_wire(nk.id, ck.id)
            if existing is None:
                network.set_wire(nk.id, ck.id, thickness=INITIAL_WEAK)
                w = network.get_wire(nk.id, ck.id)
                if w:
                    rel = "similar-to" if sim > 0.4 else "part-of"
                    w.relation_type = rel
                    w.slippage_distance = SLIPPAGE_DISTANCE.get(rel, 0)
            connected += 1


# ═══════════════════════════════════════════════
# 遗忘（电线衰减）
# ═══════════════════════════════════════════════

def decay_wires(network: KeyNetwork):
    """
    对所有电线执行衰减。

    遗忘曲线：衰减速率 = 基础衰减率 × 重复衰减因子。

    重复衰减因子 = 1.0 / (1.0 + REPETITION_BONUS × (co_activation_count - 1))

    激活 1 次：因子 1.0（全速衰减）
    激活 3 次：因子 0.5（减半）
    激活 10 次：因子 0.18（极慢）
    """
    now = time.time()

    for wire in network.all_wires():
        idle = now - wire.last_strengthened_at

        if idle <= DECAY_START:
            continue

        # 基础衰减（按天）
        base = DECAY_RATE * (idle / 86400)

        # 遗忘曲线
        rep_factor = 1.0 / (1.0 + REPETITION_BONUS * (wire.co_activation_count - 1))
        actual = base * rep_factor

        wire.weaken(actual)


# ═══════════════════════════════════════════════
# 休眠
# ═══════════════════════════════════════════════

def check_dormancy(network: KeyNetwork):
    """
    检查是否有键超过 30 天未被激活 → 进入休眠。
    休眠：内容压缩到 ≤ 30% 长度，但可唤醒。
    """
    now = time.time()

    for key in network.all_keys():
        if key.is_dormant:
            continue
        if key.source == "internal":
            continue  # 系统键（如体感时间键）永不休眠
        if key.last_activated_at == 0:
            # 从未被激活过，从创建时间算
            idle = now - key.created_at
        else:
            idle = now - key.last_activated_at

        if idle > DORMANCY_THRESHOLD:
            _enter_dormancy(key)


def _enter_dormancy(key: Key):
    """将键标记为休眠，压缩内容。"""
    key.original_content = key.content
    key.content = _compress(key.content)
    key.is_dormant = True
    key.dormant_since = time.time()


def _compress(content: str) -> str:
    """初版简单压缩：保留前 20 个字符 + …。"""
    if len(content) <= 20:
        return content
    # 保留开头的关键信息
    return content[:18] + "…"


# ═══════════════════════════════════════════════
# 花园级消化（v0.3 新增）
# ═══════════════════════════════════════════════

def digest_after_garden_conduction(garden):
    """
    花园级传导结束后执行消化：
    - 各盆独立执行盆内消化（同时亮的键对 → 电线加粗）
    - 跨盆同时亮的键对 → 桥接加粗（不新建桥接）
    """
    from .garden import BRIDGE_HEBBIAN_DELTA

    # 各盆内消化
    for pot in garden.pots.values():
        if pot.is_dormant:
            continue
        digest_after_conduction(pot.network)

    # 跨盆消化：同时亮的跨盆键对 → 已有桥接加粗
    bright_by_pot: dict[str, set[int]] = {}
    for pot in garden.pots.values():
        if pot.is_dormant:
            continue
        bright = {k.id for k in pot.network.get_bright_keys()}
        if bright:
            bright_by_pot[pot.id] = bright

    pot_ids = list(bright_by_pot.keys())
    for i in range(len(pot_ids)):
        for j in range(i + 1, len(pot_ids)):
            pid_a, pid_b = pot_ids[i], pot_ids[j]
            for kid_a in bright_by_pot[pid_a]:
                bridges = garden.find_bridges_for_key(kid_a)
                for bridge in bridges:
                    if bridge.dst_pot_id == pid_b and bridge.dst_key_id in bright_by_pot[pid_b]:
                        # 双方都亮 → 桥接加粗
                        bridge.strengthen(BRIDGE_HEBBIAN_DELTA)

def background_maintenance(network: KeyNetwork):
    """
    空闲时调用（每次处理一小批）。
    包含：电线衰减 + 休眠检查。
    """
    decay_wires(network)
    check_dormancy(network)
