"""
键盘模型 AI — 文档投喂模块

把投喂的 txt 文档切成经历碎片，去重后创建键，并批量消化。
投喂来源：对话中 [投喂] 标记 或 data/ 目录下的 txt 文件。

碎片不拆成实体/属性/关系三元组——保留经历的原样。
结构（理解）由后续对话中电线的自然生长形成。
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.network import KeyNetwork
    from src.core.key import Key
    from src.core.garden import Garden, Pot

from src.core.digestion import digest_batch_new_keys
from src.core.wire import SLIPPAGE_DISTANCE
from .input import find_similar_key


# ── 切分参数 ──────────────────────────────────

MIN_FRAGMENT_LEN = 5     # 短于此值 → 合并到相邻碎片
MAX_FRAGMENT_LEN = 200   # 长于此值 → 在逗号处再切


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

def ingest_document(
    network: KeyNetwork,
    text: str,
    source: str = "external",
) -> list[Key]:
    """
    投喂一份文档：
    1. 切成经历碎片
    2. 去重（已知道的事不新建键，改为激活旧键）
    3. 创建新键
    4. 批量消化（新键互连 + 与上下文弱连）
    5. 返回新创建的键列表
    """
    fragments = split_into_fragments(text)
    if not fragments:
        return []

    # 质量过滤（Step 6c）：去噪音 + 批次内去重
    fragments = [f for f in fragments if not _is_noise(f)]
    fragments = _dedup_within_batch(fragments)

    if not fragments:
        return []

    new_keys: list[Key] = []
    activated_existing: list[Key] = []

    for frag in fragments:
        # 去重
        existing = find_similar_key(network, frag, threshold=0.8)
        if existing is not None:
            # 已知道 → 激活旧键
            existing.activate(1)
            existing.activation_count += 1
            existing.activation_history.append(time.time())
            activated_existing.append(existing)
        else:
            # 新经历 → 创建键
            key = network.create_key(content=frag, source=source)
            new_keys.append(key)

    # 批量消化：新键互连 + 与上下文（已激活的旧键）弱连
    if new_keys:
        context = activated_existing + new_keys
        digest_batch_new_keys(network, new_keys, context)

    return new_keys


# ═══════════════════════════════════════════════
# 质量过滤（Step 6c）
# ═══════════════════════════════════════════════

def _is_noise(fragment: str) -> bool:
    """
    判断一个片段是否为明显噪音，不应建键。

    过滤项：
    - 纯数字 / 日期 / 时间
    - 重复单字符 5 次以上（如"哈哈哈哈哈"）
    - URL 或文件路径
    - 中文字符占比 < 30%
    - 标题/分隔标记（===、═══、---、***等）
    """
    if not fragment or len(fragment.strip()) < 2:
        return True

    stripped = fragment.strip()

    # 纯数字/日期/时间
    if re.match(r'^[\d\s\-/:.\s年月日时分秒]+$', stripped):
        return True

    # 重复单字符（>4 次）
    if re.match(r'^(.)\1{4,}$', stripped):
        return True

    # URL 或文件路径
    if re.search(r'[A-Za-z]:[\\/]|https?://|www\.', stripped):
        return True

    # 标题/分隔标记
    if re.match(r'^[=═\-－*#]{3,}', stripped):
        return True
    if stripped.startswith("===") or stripped.startswith("═══"):
        return True
    if stripped.startswith("---") or stripped.startswith("***"):
        return True

    # 中文字符占比 < 30%（大量符号/乱码）
    cjk_chars = sum(1 for ch in stripped if '一' <= ch <= '鿿')
    total = len(stripped)
    if total > 0 and cjk_chars / total < 0.3:
        return True

    return False


def _bigram_similarity(a: str, b: str) -> float:
    """
    基于字符 bigram 的 Jaccard 相似度。
    比 jieba 分词快，适合批量内去重。
    """
    if not a or not b:
        return 0.0
    set_a = set(a[i:i+2] for i in range(len(a)-1))
    set_b = set(b[i:i+2] for i in range(len(b)-1))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _dedup_within_batch(fragments: list[str]) -> list[str]:
    """
    同一批次内去除近似重复的片段。
    只与最近 5 个已保留片段比较，避免 O(n²)。
    """
    result: list[str] = []
    for frag in fragments:
        is_dup = False
        # 只检查最近的 5 条，兼顾效率和准确性
        for seen in result[-5:]:
            if _bigram_similarity(frag, seen) > 0.85:
                is_dup = True
                break
        if not is_dup:
            result.append(frag)
    return result


# ═══════════════════════════════════════════════
# 碎片切分
# ═══════════════════════════════════════════════

def split_into_fragments(document: str) -> list[str]:
    """
    将文档按自然停顿切成经历碎片。

    分隔符：句号、换行、段落空行。
    一个碎片 ≈ 1-3 句话（一个完整的"小印象"）。
    """
    # 1. 按句号、换行切分
    raw = re.split(r'[。\n]{1,2}', document)
    raw = [c.strip() for c in raw if c.strip()]

    # 2. 过滤空格和纯符号行
    raw = [c for c in raw if any('一' <= ch <= '鿿' for ch in c)]

    fragments: list[str] = []
    for chunk in raw:
        if len(chunk) < MIN_FRAGMENT_LEN:
            # 太短 → 合并到上一个碎片
            if fragments:
                fragments[-1] += "。" + chunk
            else:
                fragments.append(chunk)
        elif len(chunk) > MAX_FRAGMENT_LEN:
            # 太长 → 在逗号、分号处再次切分
            sub = re.split(r'[，,;；]', chunk)
            for s in sub:
                s = s.strip()
                if len(s) >= MIN_FRAGMENT_LEN:
                    fragments.append(s)
                elif fragments:
                    fragments[-1] += "，" + s
        else:
            fragments.append(chunk)

    return fragments


# ═══════════════════════════════════════════════
# 批量投喂（从文件）
# ═══════════════════════════════════════════════

def ingest_file(
    network: KeyNetwork,
    filepath: str,
    source: str = "external",
) -> int:
    """
    投喂一个 txt 文件。
    返回新创建的键数量。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    new_keys = ingest_document(network, text, source=source)
    return len(new_keys)


# ═══════════════════════════════════════════════
# 投喂队列
# ═══════════════════════════════════════════════

class IngestQueue:
    """
    投喂队列：管理 data/ 目录下待投喂的 txt 文件。
    启动时扫描，对话间隙逐个消化。
    """

    def __init__(self, data_dir: str = "data"):
        import os
        self.data_dir = data_dir
        self._pending: list[str] = []
        self._processed: set[str] = set()

    def scan(self):
        """扫描 data/ 目录，发现新文件加入队列。"""
        import os
        if not os.path.isdir(self.data_dir):
            return
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".txt") and fname not in self._processed:
                self._pending.append(os.path.join(self.data_dir, fname))
                self._processed.add(fname)

    def has_pending(self) -> bool:
        return len(self._pending) > 0

    def next(self) -> str | None:
        """取出下一个待投喂的文件路径。"""
        if self._pending:
            return self._pending.pop(0)
        return None


# ═══════════════════════════════════════════════
# 花园模式投喂（v0.3 新增）
# ═══════════════════════════════════════════════

def ingest_document_to_garden(
    garden: "Garden",
    text: str,
    source: str = "external",
) -> tuple[list["Key"], str, bool]:
    """
    投喂文档到花园（花盆模型）。

    1. 切碎片 → 质量过滤
    2. 计算文档主题签名 → 找匹配的盆（必要时创建新盆）
    3. 在目标盆内去重 + 创建键
    4. 盆内批量消化
    5. 自动创建跨盆桥接

    返回：(new_keys, pot_id, is_new_pot)
    """
    fragments = split_into_fragments(text)
    if not fragments:
        return [], "", False

    # 质量过滤
    fragments = [f for f in fragments if not _is_noise(f)]
    fragments = _dedup_within_batch(fragments)
    if not fragments:
        return [], "", False

    # 路由到盆
    target_pot, is_new_pot = garden.route_ingest(fragments, source)

    # 在目标盆内创建键
    network = target_pot.network
    new_keys: list["Key"] = []
    activated_existing: list["Key"] = []

    for frag in fragments:
        existing = find_similar_key(network, frag, threshold=0.8)
        if existing is not None:
            existing.activate(1)
            existing.activation_count += 1
            existing.activation_history.append(time.time())
            activated_existing.append(existing)
        else:
            key = network.create_key(content=frag, source=source)
            new_keys.append(key)

    # 盆内批量消化
    if new_keys:
        context = activated_existing + new_keys
        digest_batch_new_keys(network, new_keys, context)

    # 更新盆的主题签名
    target_pot.maybe_update_signature()

    # 自动创建跨盆桥接
    if new_keys:
        garden.auto_create_bridges_for_new_keys(new_keys, target_pot.id)

    return new_keys, target_pot.id, is_new_pot


def ingest_file_to_garden(
    garden: "Garden",
    filepath: str,
    source: str = "external",
) -> int:
    """
    投喂一个 txt 文件到花园。
    返回新创建的键数量。
    """
    import os
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    new_keys, _, _ = ingest_document_to_garden(garden, text, source=source)
    return len(new_keys)


# ═══════════════════════════════════════════════
# v1.0: CD 概念拆解投喂路径
# ═══════════════════════════════════════════════

def ingest_document_to_garden_v1(
    garden: "Garden",
    text: str,
    source: str = "external",
) -> tuple[list["Key"], str, bool, list[list["Key"]]]:
    """
    v1.0 投喂：CD 概念拆解 + 花园路由。

    与旧版 ingest_document_to_garden 的核心区别：
    旧版：文本 → 切句 → 每句存为一个键（原文原样）
    v1.0：文本 → 切句 → 每句 CD 拆解为概念键簇（原语+槽位）

    流程：
    1. 文本 → 切句
    2. 每句 → CD 拆解 → 概念键簇
    3. 用簇中键的 content 做主题路由（与旧版兼容）
    4. 在目标盆内去重 + 批量创建概念键
    5. 盆内消化 + 跨盆桥接

    返回：(new_keys, pot_id, is_new_pot, concept_clusters)
    """
    from .cd_engine import decompose_document

    # 1. CD 拆解整篇文档 → 概念键簇列表
    concept_clusters = decompose_document(text, network=None)
    if not concept_clusters:
        return [], "", False, []

    # 2. 收集所有概念键的 content 用于路由
    all_contents = []
    for cluster in concept_clusters:
        for k in cluster:
            if k.content and len(k.content) >= 2:
                all_contents.append(k.content)

    if not all_contents:
        # 没有可路由的内容 → 用原文片段
        from .cd_engine import _split_sentences
        all_contents = _split_sentences(text)

    # 3. 路由到盆（用 content 做主题匹配，与旧版兼容）
    target_pot, is_new_pot = garden.route_ingest(all_contents, source)

    # 4. 在目标盆内批量创建概念键
    network = target_pot.network
    all_new_keys: list["Key"] = []

    for cluster in concept_clusters:
        # 分类簇中的键
        prim_keys_in_cluster = [k for k in cluster if k.primitive and k.primitive != "WORD"]
        word_keys_in_cluster = [k for k in cluster if k.primitive == "WORD"]

        # ── 先创建/查找词键，建立 old_id → new_id 映射 ──
        word_id_map: dict[int, int] = {}  # old_word_id → new_word_id

        for wk in word_keys_in_cluster:
            existing_w = find_similar_key(network, wk.content, threshold=0.9)
            if existing_w is not None:
                word_id_map[wk.id] = existing_w.id
            else:
                new_wk = network.create_key(content=wk.content, source=source)
                new_wk.primitive = "WORD"
                new_wk.slots = {"text": new_wk.id}  # 指向自身
                all_new_keys.append(new_wk)
                word_id_map[wk.id] = new_wk.id

        # ── 再处理原语键（此时 word_id_map 已就绪，去重时可以合并槽位） ──
        prim_key_map: list[tuple["Key", "Key"]] = []  # [(new_key, old_pk), ...]

        for pk in prim_keys_in_cluster:
            existing = find_similar_key(network, pk.content, threshold=0.85)
            if existing is not None:
                # 已存在 → 激活 + 增强 evidence + 合并新槽位（B4 修复）
                existing.activate(1)
                existing.activation_count += 1
                existing.activation_history.append(time.time())
                existing.evidence_freq = min(1.0, existing.evidence_freq + 0.05)
                existing.evidence_conf = min(1.0, existing.evidence_conf + 0.02)
                # 合并原旧键没有的槽位
                for slot_name, old_wid in pk.slots.items():
                    if slot_name not in existing.slots and old_wid in word_id_map:
                        existing.slots[slot_name] = word_id_map[old_wid]
                continue

            # 创建新原语键
            new_key = network.create_key(content=pk.content, source=source)
            new_key.primitive = pk.primitive
            new_key.slots = {}  # 槽位待重映射
            all_new_keys.append(new_key)
            prim_key_map.append((new_key, pk))

        # ── 重映射原语键的槽位 + 创建带关系类型的电线（B1 + B2 修复） ──
        for new_key, old_pk in prim_key_map:
            # 槽位 ID 重映射：old_word_id → new_word_id
            remapped: dict[str, int] = {}
            for slot_name, old_wid in old_pk.slots.items():
                if old_wid in word_id_map:
                    remapped[slot_name] = word_id_map[old_wid]
            new_key.slots = remapped

            # 原语键 ↔ 词键：part-of 关系
            for word_id in remapped.values():
                network.set_wire(new_key.id, word_id, thickness=0.8)
                w = network.get_wire(new_key.id, word_id)
                if w:
                    w.relation_type = "part-of"
                    w.slippage_distance = SLIPPAGE_DISTANCE.get("part-of", 0)

            # 词键之间：similar-to 关系
            wk_ids = list(remapped.values())
            for i in range(len(wk_ids)):
                for j in range(i + 1, len(wk_ids)):
                    network.set_wire(wk_ids[i], wk_ids[j], thickness=0.4)
                    w = network.get_wire(wk_ids[i], wk_ids[j])
                    if w:
                        w.relation_type = "similar-to"
                        w.slippage_distance = SLIPPAGE_DISTANCE.get("similar-to", 0)

    if not all_new_keys:
        return [], target_pot.id, is_new_pot, concept_clusters

    # 5. 盆内批量消化（在概念键之间建立电线）
    from src.core.digestion import digest_batch_new_keys
    digest_batch_new_keys(network, all_new_keys, all_new_keys)

    # 6. 更新盆的签名
    target_pot.maybe_update_signature()

    # 7. 跨盆桥接
    if all_new_keys:
        garden.auto_create_bridges_for_new_keys(all_new_keys, target_pot.id)

    return all_new_keys, target_pot.id, is_new_pot, concept_clusters


def ingest_file_to_garden_v1(
    garden: "Garden",
    filepath: str,
    source: str = "external",
) -> int:
    """
    v1.0 投喂一个 txt 文件到花园（CD 拆解版）。
    返回新创建的键数量。
    """
    import os
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    new_keys, _, _, _ = ingest_document_to_garden_v1(garden, text, source=source)
    return len(new_keys)
