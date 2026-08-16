"""
键盘模型 AI — CD 概念拆解引擎  v1.0

将输入文本拆解为概念键簇（原语 + 槽位）。
消化发生在键创建时——这是系统"理解"文本的入口。

流程：
1. 文本 → 切句
2. 每句 → jieba 分词
3. 分词 → 匹配 CD 模式（primitives.py 中的 CD_PATTERNS）
4. 模式匹配 → 槽位填充（提取 X/Y/Z 等成分）
5. 槽位填充 → 创建概念键簇（原语键 + 词键 + 连接线）

不依赖任何 LLM、句法分析器、外部模型。禁区见 src/core/__init__.py。
拆解模式本身以 PATTERN 键的形式存储在语言盆中。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jieba

from src.core.primitives import (
    Primitive,
    CDPattern,
    CD_PATTERNS,
    find_pattern_for_sentence,
)
from src.core.key import Key
from src.core.wire import SLIPPAGE_DISTANCE

if TYPE_CHECKING:
    from src.core.network import KeyNetwork


# ═══════════════════════════════════════════════
# 句子切分
# ═══════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    """
    将文本切分为句子。
    按句号、问号、感叹号、换行符切分。
    短于 5 字的片段合并到前句。
    """
    import re
    # 按句子终止符切分
    raw = re.split(r'[。！？\n]+', text)
    sentences = [s.strip() for s in raw if s.strip()]

    # 合并过短的片段
    merged = []
    for s in sentences:
        if len(s) < 5 and merged:
            merged[-1] = merged[-1] + "，" + s
        else:
            merged.append(s)

    return merged


# ═══════════════════════════════════════════════
# 槽位填充：从句子的分词结果中提取槽位值
# ═══════════════════════════════════════════════

def _extract_slot_fillers(
    text: str,
    tokens: list[str],
    pattern: CDPattern,
) -> dict[str, str]:
    """
    使用启发式方法从句子的分词结果中提取槽位填充值。

    策略（按优先级）：
    1. 触发词分割法：用 triggers 将句子分成前后段
    2. 位置法：第一个名词短语 → 第一个槽位，最后一个名词短语 → 最后一个槽位
    3. 直接映射：如果只有两个槽位且两个名词，按顺序填充

    返回 {slot_name: text_fragment}
    """
    slot_names = list(pattern.slot_map.keys())  # X, Y, Z 等
    slot_targets = list(pattern.slot_map.values())  # subject, category 等
    fillers: dict[str, str] = {}

    # ── 策略 1：用触发词分割 ──
    trigger_positions = []
    for trigger in pattern.triggers:
        pos = text.find(trigger)
        if pos >= 0:
            trigger_positions.append((pos, trigger))

    if trigger_positions:
        trigger_positions.sort()
        first_trigger_pos, first_trigger = trigger_positions[0]
        trigger_len = len(first_trigger)

        # 触发词前 → 第一个槽位
        before = text[:first_trigger_pos].strip()
        # 触发词后 → 后续槽位
        after = text[first_trigger_pos + trigger_len:].strip()

        # 如果有逗号分隔，拆分 after
        if before and slot_names:
            fillers[slot_names[0]] = before

        if after and len(slot_names) >= 2:
            # 检查 after 中是否还有触发词（多段）
            parts = []
            remaining = after
            for t in pattern.triggers[1:]:
                p = remaining.find(t)
                if p >= 0:
                    parts.append(remaining[:p].strip())
                    remaining = remaining[p + len(t):].strip()
            parts.append(remaining)
            parts = [p for p in parts if p]

            for i, part in enumerate(parts):
                if i + 1 < len(slot_names):
                    fillers[slot_names[i + 1]] = part
    else:
        # ── 策略 2：位置法 ──
        # 过滤出有意义的名词/代词（长度 > 0 且非纯标点）
        meaningful = [t for t in tokens if len(t) >= 1]

        if len(meaningful) >= len(slot_names):
            # 按顺序分配
            for i, sn in enumerate(slot_names):
                fillers[sn] = meaningful[i]
        elif len(meaningful) == len(slot_names) - 1 and slot_names:
            # "X 是 Y" 模式：第一个 token 是 X，最后一个非触发词是 Y
            non_trigger = [t for t in meaningful if t not in pattern.triggers]
            if non_trigger:
                if len(slot_names) >= 1:
                    fillers[slot_names[0]] = non_trigger[0]
                if len(slot_names) >= 2:
                    fillers[slot_names[1]] = non_trigger[-1]

    # ── 策略 3：维度词提取（用于比较句 BIGGER/SMALLER）──
    DIMENSION_WORDS = {"大", "小", "高", "低", "长", "短", "快", "慢",
                       "重", "轻", "多", "少", "宽", "窄", "粗", "细",
                       "深", "浅", "厚", "薄", "远", "近", "热", "冷"}
    if "dimension" in slot_targets:
        for t in tokens:
            if t in DIMENSION_WORDS:
                for var_name, target in pattern.slot_map.items():
                    if target == "dimension":
                        fillers[var_name] = t
                        break
                break

    # ── 确保所有槽位至少有空值 ──
    for sn in slot_names:
        fillers.setdefault(sn, "")

    return fillers


# ═══════════════════════════════════════════════
# 主入口：句子 → 概念键簇
# ═══════════════════════════════════════════════

def decompose_sentence(
    text: str,
    network: KeyNetwork | None = None,
) -> list[Key]:
    """
    将一句话拆解为一个概念键簇。

    返回一组互连的 Key（原语键 + 词键），可直接加入网络。
    如果 network 提供，键会自动创建到网络中。
    否则返回独立的 Key 对象列表（调用方负责加入网络）。

    流程：
    1. 分词
    2. 找匹配的 CD 模式
    3. 槽位填充
    4. 创建概念键 + 词键 + 连线
    """
    if not text or len(text) < 2:
        return []

    tokens = list(jieba.lcut(text))
    tokens = [t.strip() for t in tokens if t.strip()]
    if not tokens:
        return []

    # 1. 找匹配的 CD 模式
    pattern = find_pattern_for_sentence(tokens)
    result_keys: list[Key] = []

    if pattern is not None:
        # 2. 有模式匹配 → 创建结构化概念键簇
        result_keys = _decompose_with_pattern(text, tokens, pattern, network)
    else:
        # 3. 无模式匹配 → 创建自由词汇键簇（用 WORD 原语存储）
        result_keys = _decompose_freeform(text, tokens, network)

    return result_keys


def decompose_document(
    text: str,
    network: KeyNetwork | None = None,
) -> list[list[Key]]:
    """
    将整篇文档拆解为多个概念键簇。
    返回 [[cluster_1_keys], [cluster_2_keys], ...]
    """
    sentences = _split_sentences(text)
    clusters = []
    for sent in sentences:
        keys = decompose_sentence(sent, network)
        if keys:
            clusters.append(keys)
    return clusters


# ═══════════════════════════════════════════════
# 结构化拆解（有模式匹配）
# ═══════════════════════════════════════════════

def _decompose_with_pattern(
    text: str,
    tokens: list[str],
    pattern: CDPattern,
    network: KeyNetwork | None,
) -> list[Key]:
    """
    用 CD 模式拆解一句话。
    创建：1 个原语键 + N 个词键 + 槽位连线。
    """
    keys: list[Key] = []

    # ── B6 修复：比较句方向检测（用分词 token 匹配，避免子串误触发） ──
    SMALLER_MARKERS = {"不如", "比不上", "小", "少", "低", "矮", "短", "细", "慢", "轻", "弱"}
    effective_primitive = pattern.primitive
    if pattern.primitive == Primitive.BIGGER:
        if any(t in SMALLER_MARKERS for t in tokens):
            effective_primitive = Primitive.SMALLER

    # 1. 槽位填充
    fillers = _extract_slot_fillers(text, tokens, pattern)

    # 2. 创建词键（每个槽位填充值 → WORD 键）
    word_keys: dict[str, Key] = {}  # slot_name → 对应的词键
    for slot_name, filler_text in fillers.items():
        if filler_text:
            wk = _create_key(f"词:{filler_text}", Primitive.WORD, network)
            wk.slots = {"text": 0}  # WORD 的 text 槽存自身 ID（占位）
            word_keys[slot_name] = wk
            keys.append(wk)

    # 3. 创建原语键
    slot_fills: dict[str, int] = {}
    for slot_name, target_slot in pattern.slot_map.items():
        wk = word_keys.get(slot_name)
        if wk is not None:
            slot_fills[target_slot] = wk.id

    # 原语键的 content = 原文（保留用于输出参考）
    prim_key = _create_key(text, effective_primitive, network)
    prim_key.slots = slot_fills
    keys.append(prim_key)

    # 4. 创建电线（如果 network 存在）
    if network is not None:
        # 词键 ↔ 原语键
        for wk in word_keys.values():
            network.set_wire(prim_key.id, wk.id, thickness=0.8)
            # 获取默认关系类型
            w = network.get_wire(prim_key.id, wk.id)
            if w:
                w.relation_type = "part-of"
                w.slippage_distance = SLIPPAGE_DISTANCE.get("part-of", 0)

        # 词键之间（如果有多个）
        wk_list = list(word_keys.values())
        for i in range(len(wk_list)):
            for j in range(i + 1, len(wk_list)):
                network.set_wire(wk_list[i].id, wk_list[j].id, thickness=0.4)
                w = network.get_wire(wk_list[i].id, wk_list[j].id)
                if w:
                    w.relation_type = "similar-to"
                    w.slippage_distance = SLIPPAGE_DISTANCE.get("similar-to", 0)

    return keys


# ═══════════════════════════════════════════════
# 自由拆解（无模式匹配时）
# ═══════════════════════════════════════════════

def _decompose_freeform(
    text: str,
    tokens: list[str],
    network: KeyNetwork | None,
) -> list[Key]:
    """
    无法匹配 CD 模式时：将句子整体作为一个 WORD 键存储。
    后续可以通过传导和推理逐渐建立与其他概念键的关联。

    这不是"放弃理解"——这是"先记住，慢慢消化"。
    """
    key = _create_key(text, Primitive.WORD, network)
    key.slots = {"text": 0}

    # 为每个有意义的 token 创建词键并连线
    keys = [key]
    if network is not None:
        for token in tokens:
            if len(token) >= 2:
                wk = _create_key(f"词:{token}", Primitive.WORD, network)
                wk.slots = {"text": 0}
                network.set_wire(key.id, wk.id, thickness=0.5)
                w = network.get_wire(key.id, wk.id)
                if w:
                    w.relation_type = "part-of"
                    w.slippage_distance = SLIPPAGE_DISTANCE.get("part-of", 0)
                keys.append(wk)

    return keys


# ═══════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════

def _create_key(
    content: str,
    primitive: Primitive | str,
    network: KeyNetwork | None,
) -> Key:
    """创建键（如果有网络则加入网络，否则独立创建）。"""
    prim_str = primitive.value if isinstance(primitive, Primitive) else primitive

    if network is not None:
        key = network.create_key(content=content, source="external")
        key.primitive = prim_str
        return key
    else:
        import random
        kid = random.randint(100000, 999999)
        key = Key(id=kid, content=content)
        key.primitive = prim_str
        key.source = "external"
        return key


def _wire_keys(
    a: Key,
    b: Key,
    network: KeyNetwork | None,
    thickness: float = 0.5,
):
    """在两个键之间建立电线（如果 network 存在）。"""
    if network is not None:
        network.set_wire(a.id, b.id, thickness=thickness)
