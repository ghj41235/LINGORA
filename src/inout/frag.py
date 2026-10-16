"""
键盘模型 AI — .frag 碎片知识入网  v1.0

.frag 是师父手写的结构化知识三元组（`主体 关系 客体`），
直接入网，绕过 cd_engine 的自然语言拆解。解决「种子不够」：
cd_engine 的 17 种生活句式匹配不了世界知识句式（X能Y / X是Y的一种 / X连接Y），
整句落 freeform 变死 WORD 键。.frag 把这些句式直接结构化投喂。

格式（三列最简，空白分隔，`#` 注释）：
    @概念
    手机 能 打电话
    猫 是一种 动物
    猫 不是 植物

一个三元组展开为（完整方案 A）：
    2 个词键（WORD）+ 1 个原语键（带命名槽位）+ 线（带 relation_type + 极性）。

极性三态 -1/0/+1 落在「主体↔客体」的词键间线上，由关系词决定 relation_type：
    猫 是 动物   → 词猫↔词动物 = part-of（+1）
    猫 不是 植物 → 词猫↔词植物 = opposite-of（-1）
原语键↔词键的线固定 part-of（+1）。

纯符号：不调 LLM，不碰禁区。只把结构化的碎片建成键和线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.primitives import Primitive
from src.core.wire import SLIPPAGE_DISTANCE

if TYPE_CHECKING:
    from src.core.key import Key
    from src.core.network import KeyNetwork
    from src.core.garden import Garden


# ── 分区头 ──────────────────────────────────────

SECTIONS = ("@概念", "@经历", "@因果")


# ── 关系词映射表：关系词 → (原语, (主体槽位, 客体槽位), 词键间 relation_type) ──
# 词键间 relation_type 决定极性（opposite-of → -1，其余 → +1），见 wire.RELATION_POLARITY。

RELATION_WORD_MAP: dict[str, tuple[Primitive, tuple[str, str], str]] = {
    # 判断 / 归属
    "是":   (Primitive.IS, ("subject", "category"), "part-of"),
    "就是": (Primitive.IS, ("subject", "category"), "part-of"),
    "叫做": (Primitive.IS, ("subject", "category"), "part-of"),
    "是一种": (Primitive.IS, ("subject", "category"), "part-of"),
    "属于": (Primitive.IS, ("subject", "category"), "part-of"),
    "不是": (Primitive.NOT, ("subject", "category"), "opposite-of"),
    # 拥有
    "有":   (Primitive.HAVE, ("owner", "owned"), "part-of"),
    "拥有": (Primitive.HAVE, ("owner", "owned"), "part-of"),
    "具有": (Primitive.HAVE, ("owner", "owned"), "part-of"),
    "没有": (Primitive.NOT_HAVE, ("owner", "lacking"), "opposite-of"),
    "缺少": (Primitive.NOT_HAVE, ("owner", "lacking"), "opposite-of"),
    # 能力
    "能":   (Primitive.CAN, ("agent", "action"), "part-of"),
    "会":   (Primitive.CAN, ("agent", "action"), "part-of"),
    "可以": (Primitive.CAN, ("agent", "action"), "part-of"),
    # 连接 / 相似
    "连接": (Primitive.CONNECT, ("a", "b"), "similar-to"),
    "连着": (Primitive.CONNECT, ("a", "b"), "similar-to"),
    "相连": (Primitive.CONNECT, ("a", "b"), "similar-to"),
    "像":   (Primitive.SAME, ("a", "b"), "similar-to"),
    "类似": (Primitive.SAME, ("a", "b"), "similar-to"),
    # 因果
    "导致": (Primitive.CAUSE, ("reason", "effect"), "cause-of"),
    "引起": (Primitive.CAUSE, ("reason", "effect"), "cause-of"),
    # 情感
    "喜欢": (Primitive.LIKE, ("liker", "liked"), "part-of"),
    "爱":   (Primitive.LIKE, ("liker", "liked"), "part-of"),
    "讨厌": (Primitive.DISLIKE, ("disliker", "disliked"), "opposite-of"),
    "怕":   (Primitive.DISLIKE, ("disliker", "disliked"), "opposite-of"),
}


@dataclass
class FragTriple:
    """一条 .frag 三元组（含所属分区）。"""
    section: str      # "@概念" / "@经历" / "@因果"
    subject: str      # 主体
    relation: str     # 关系词
    object: str       # 客体


# ═══════════════════════════════════════════════
# 解析
# ═══════════════════════════════════════════════

def parse_frag(text: str) -> list[FragTriple]:
    """
    解析 .frag 文本为三元组列表。

    - `#` 开头或行内 `#` 之后是注释。
    - `@概念` / `@经历` / `@因果` 是分区头，标记后续三元组。
    - 每行三列：`主体 关系 客体`，空白分隔。
    """
    triples: list[FragTriple] = []
    section = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        # 空行 / 纯注释
        if not line or line.startswith("#"):
            continue
        # 行内注释
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue

        # 分区头
        if line in SECTIONS:
            section = line
            continue

        # 三列
        parts = line.split()
        if len(parts) < 3:
            continue  # 列数不足，跳过（容错：不炸整份文件）
        subject, relation, object_ = parts[0], parts[1], " ".join(parts[2:])

        if relation not in RELATION_WORD_MAP:
            continue  # 未知关系词，跳过（避免自由词键污染）
        triples.append(FragTriple(section, subject, relation, object_))

    return triples


# ═══════════════════════════════════════════════
# 三元组 → 概念键簇
# ═══════════════════════════════════════════════

def frag_triple_to_cluster(
    subject: str,
    relation: str,
    object_: str,
    network: "KeyNetwork | None",
) -> list["Key"]:
    """
    一个三元组 → 概念键簇（2 词键 + 1 原语键 + 线）。

    复用 cd_engine 的建键结构：词键（WORD）+ 原语键（带命名槽位）+ 线。
    与 cd_engine 的区别：词键间线的 relation_type 由关系词决定（而非固定 similar-to），
    这是极性 -1（对立）的落点。
    """
    spec = RELATION_WORD_MAP.get(relation)
    if spec is None:
        return []

    primitive, (subj_slot, obj_slot), word_rel_type = spec

    # 词键
    subj_key = _create_key(f"词:{subject}", Primitive.WORD, network)
    subj_key.slots = {"text": 0}
    obj_key = _create_key(f"词:{object_}", Primitive.WORD, network)
    obj_key.slots = {"text": 0}

    # 原语键：content = 三元组原文（保留用于输出参考）
    prim_key = _create_key(f"{subject}{relation}{object_}", primitive, network)
    prim_key.slots = {subj_slot: subj_key.id, obj_slot: obj_key.id}

    keys = [subj_key, obj_key, prim_key]

    if network is not None:
        # 原语键 ↔ 词键：part-of（固定，+1）
        for wk in (subj_key, obj_key):
            network.set_wire(prim_key.id, wk.id, thickness=0.8)
            w = network.get_wire(prim_key.id, wk.id)
            if w:
                w.relation_type = "part-of"
                w.slippage_distance = SLIPPAGE_DISTANCE.get("part-of", 0)

        # 词键 ↔ 词键：关系词决定的 relation_type（极性落点）
        network.set_wire(subj_key.id, obj_key.id, thickness=0.4)
        w = network.get_wire(subj_key.id, obj_key.id)
        if w:
            w.relation_type = word_rel_type
            w.slippage_distance = SLIPPAGE_DISTANCE.get(word_rel_type, 0)

    return keys


# ═══════════════════════════════════════════════
# 入网（复用 ingest 的去重 / 桥接 / SELF 桥）
# ═══════════════════════════════════════════════

def ingest_frag_file_to_garden(
    garden: "Garden",
    filepath: str,
    source: str = "external",
) -> tuple[list["Key"], str, bool]:
    """
    投喂一个 .frag 文件到花园。

    1. 读文件 → 解析三元组
    2. 用三元组 content 做主题路由（与 ingest v1 兼容）
    3. 跨三元组去重词键（共享词如「猫」只建一次）
    4. 每个三元组建原语键 + 词键间线（relation_type 由关系词定）
    5. 跨盆桥接 + SELF 桥

    返回 (new_keys, pot_id, is_new_pot)。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    triples = parse_frag(text)
    if not triples:
        return [], "", False

    # 路由（用三元组 content 做主题匹配）
    contents = [f"{t.subject}{t.relation}{t.object}" for t in triples]
    target_pot, is_new_pot = garden.route_ingest(contents, source)
    network = target_pot.network

    from .input import find_similar_key

    all_new_keys: list["Key"] = []

    # 词键跨三元组去重缓存：词文本 → 词键 ID
    word_cache: dict[str, int] = {}

    def get_or_create_word(word: str) -> int:
        if word in word_cache:
            return word_cache[word]
        existing = find_similar_key(network, f"词:{word}", threshold=0.9)
        if existing is not None:
            word_cache[word] = existing.id
            return existing.id
        wk = network.create_key(content=f"词:{word}", source=source)
        wk.primitive = "WORD"
        wk.slots = {"text": wk.id}
        all_new_keys.append(wk)
        word_cache[word] = wk.id
        return wk.id

    # 每个三元组：词键（去重）→ 原语键（去重）→ 线
    for t in triples:
        spec = RELATION_WORD_MAP.get(t.relation)
        if spec is None:
            continue
        primitive, (subj_slot, obj_slot), word_rel_type = spec

        subj_id = get_or_create_word(t.subject)
        obj_id = get_or_create_word(t.object)

        # 原语键去重
        content = f"{t.subject}{t.relation}{t.object}"
        existing = find_similar_key(network, content, threshold=0.85)
        if existing is not None:
            existing.activate(1)
            existing.activation_count += 1
            existing.evidence_freq = min(1.0, existing.evidence_freq + 0.05)
            existing.evidence_conf = min(1.0, existing.evidence_conf + 0.02)
            # 合并缺失槽位
            for slot_name, wid in ((subj_slot, subj_id), (obj_slot, obj_id)):
                if slot_name not in existing.slots:
                    existing.slots[slot_name] = wid
            continue

        prim_key = network.create_key(content=content, source=source)
        prim_key.primitive = primitive.value
        prim_key.slots = {subj_slot: subj_id, obj_slot: obj_id}
        all_new_keys.append(prim_key)

        # 原语键 ↔ 词键：part-of
        for wid in (subj_id, obj_id):
            network.set_wire(prim_key.id, wid, thickness=0.8)
            w = network.get_wire(prim_key.id, wid)
            if w:
                w.relation_type = "part-of"
                w.slippage_distance = SLIPPAGE_DISTANCE.get("part-of", 0)

        # 词键 ↔ 词键：关系词决定的 relation_type（极性落点）
        network.set_wire(subj_id, obj_id, thickness=0.4)
        w = network.get_wire(subj_id, obj_id)
        if w:
            w.relation_type = word_rel_type
            w.slippage_distance = SLIPPAGE_DISTANCE.get(word_rel_type, 0)

    if not all_new_keys:
        return [], target_pot.id, is_new_pot

    # 跨盆桥接 + SELF 桥（复用 ingest 逻辑）
    target_pot.maybe_update_signature()
    garden.auto_create_bridges_for_new_keys(all_new_keys, target_pot.id)

    from .ingest import wire_self_reference
    wire_self_reference(garden, all_new_keys, target_pot.id)

    # 战场二：为盆里已有的原语补建疑问键 + same-as 线（疑问词→目标原语概念键）
    from src.core.grammar import ensure_question_keys_for_pot
    ensure_question_keys_for_pot(garden, target_pot)

    return all_new_keys, target_pot.id, is_new_pot


# ═══════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════

def _create_key(
    content: str,
    primitive: Primitive | str,
    network: "KeyNetwork | None",
) -> "Key":
    """创建键（有网络则入网，否则独立创建）。与 cd_engine._create_key 同构。"""
    prim_str = primitive.value if isinstance(primitive, Primitive) else primitive

    if network is not None:
        key = network.create_key(content=content, source="external")
        key.primitive = prim_str
        return key
    else:
        import random
        from src.core.key import Key
        kid = random.randint(100000, 999999)
        key = Key(id=kid, content=content)
        key.primitive = prim_str
        key.source = "external"
        return key


# ═══════════════════════════════════════════════
# 备课查重（脑外，纯符号——不用向量、不碰禁区）
# ═══════════════════════════════════════════════

def _term_similarity(a: str, b: str) -> float:
    """
    词项相似度（纯 char bigram Jaccard，不用向量）。

    为什么不做「a in b」包含关系：中文上下位词（"动物"⊂"哺乳动物"）
    字符全包含，会误判成近重复——但它们只是不同粒度的分类知识，该共存。
    纯 bigram Jaccard 对短词近似「精确相等」，宁漏勿错。
    """
    a, b = a.strip(), b.strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) <= 1 or len(b) <= 1:
        return 0.0
    set_a = set(a[i:i + 2] for i in range(len(a) - 1))
    set_b = set(b[i:i + 2] for i in range(len(b) - 1))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _relation_primitive(relation: str) -> Primitive | None:
    """关系词 → 原语（同原语 = 同义关系，用于近重复判定）。"""
    spec = RELATION_WORD_MAP.get(relation)
    return spec[0] if spec else None


@dataclass
class DupReport:
    """备课查重报告。"""
    kept: list["FragTriple"] = None                # 保留的新三元组（非重复）
    exact_dups: list[tuple["FragTriple", "FragTriple"]] = None  # (新, 已有) 精确重复
    near_dups: list[tuple["FragTriple", "FragTriple", float]] = None  # (新, 已有, 相似度)
    redundant: list[tuple[str, str, int]] = None   # (主体, 客体, 出现次数≥3) 冗余提示

    def __post_init__(self):
        self.kept = self.kept or []
        self.exact_dups = self.exact_dups or []
        self.near_dups = self.near_dups or []
        self.redundant = self.redundant or []

    @property
    def is_clean(self) -> bool:
        return not self.exact_dups and not self.near_dups and not self.redundant


def load_triples_from_files(paths: list[str]) -> list["FragTriple"]:
    """
    读多个 .frag 文件，返回全部三元组（含来源文件路径，存在 FragTriple 不追踪——
    备课查重只需内容，不需要知道重复来自哪个文件）。
    """
    triples: list["FragTriple"] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        triples.extend(parse_frag(text))
    return triples


def deduplicate_triples(
    new_triples: list["FragTriple"],
    existing_triples: list["FragTriple"],
    near_threshold: float = 0.85,
) -> "DupReport":
    """
    备课查重：去掉与已有 .frag 重复/近重复的三元组。

    判定（离散重叠，不用向量）：
    - 精确重复：主体/关系词/客体 三项全同。
    - 近重复：主体、客体相似度都 ≥ near_threshold，且关系词同原语（同义）。
    - 冗余上限：同一 (主体, 客体) 出现 ≥3 次 → 提示合并（重点概念允许 2~3 次强化）。

    返回 DupReport；kept 是可直接入网的新三元组。
    """
    report = DupReport()
    seen: set[tuple[str, str, str]] = set()
    for t in existing_triples:
        seen.add((t.subject, t.relation, t.object))

    # (主体, 客体) 出现次数，用于冗余上限
    so_count: dict[tuple[str, str], int] = {}
    for t in existing_triples:
        so_count[(t.subject, t.object)] = so_count.get((t.subject, t.object), 0) + 1

    for t in new_triples:
        # ── 精确重复 ──
        if (t.subject, t.relation, t.object) in seen:
            report.exact_dups.append((t, t))
            continue

        # ── 近重复 ──
        near_hit: "FragTriple | None" = None
        near_score = 0.0
        for e in existing_triples:
            if _relation_primitive(e.relation) != _relation_primitive(t.relation):
                continue  # 关系不同义，不判近重复
            subj_sim = _term_similarity(t.subject, e.subject)
            obj_sim = _term_similarity(t.object, e.object)
            if subj_sim >= near_threshold and obj_sim >= near_threshold:
                score = min(subj_sim, obj_sim)
                if score > near_score:
                    near_score = score
                    near_hit = e
        if near_hit is not None:
            report.near_dups.append((t, near_hit, near_score))
            continue

        # ── 通过：保留 + 计数 ──
        report.kept.append(t)
        so_count[(t.subject, t.object)] = so_count.get((t.subject, t.object), 0) + 1

    # ── 冗余上限（≥3 次）──
    for (subj, obj), count in so_count.items():
        if count >= 3:
            report.redundant.append((subj, obj, count))

    return report
