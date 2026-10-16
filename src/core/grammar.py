"""
键盘模型 AI — 语法键网络（语言输出重构） v1.0

将硬编码模板替换为"格局→输出"的键+线映射。

核心思想（参考 case.py 的 CBR 模式）：
1. 每次传导产生一个"激活格局"——哪些概念原语亮了、哪些关系类型被使用了
2. 这个格局作为"触发签名"，匹配已有的语法输出模式
3. 匹配到的语法模式提供句子结构模板，槽位填充后输出
4. 每次输出后，Hebbian 强化成功使用的语法键

架构：
- GrammarPattern dataclass → 存储在 garden.grammar_library 中（类似 case_library）
- OUTPUT 原语键 → 存储在专用语法盆中
- 概念键 ↔ 语法键之间的电线 → "这个格局产生这个句子"

一切皆键+线：语言输出由传导驱动，不靠硬编码模板。
禁区：见 src/core/__init__.py 完整清单（名称+概念+代码，永久锁定）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .garden import Garden, GardenConductionResult
    from .key import Key


# ═══════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════

# 语法模式匹配的最小 Jaccard 相似度
GRAMMAR_MIN_MATCH_SIMILARITY = 0.25

# 语法模式匹配的最小签名元素数
GRAMMAR_MIN_SIGNATURE_SIZE = 2

# 语法模式库最大容量
GRAMMAR_MAX_PATTERNS = 300

# 语法模式休眠阈值（天）
GRAMMAR_DORMANCY_DAYS = 21

# Hebbian 强化幅度
GRAMMAR_HEBBIAN_DELTA = 0.05

# 新语法模式的初始可靠性
GRAMMAR_INITIAL_RELIABILITY = 0.4

# 种子语法键（从现有模板转化）的最小使用次数
GRAMMAR_SEED_FREQUENCY = 3


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class GrammarPattern:
    """
    一个语法输出模式：记录"什么样的激活格局产生什么样的句子"。

    与 Case 的区别：
    - Case 记录完整的激活快照（哪些键亮了多少）→ 用于跳过传导
    - GrammarPattern 记录激活签名（哪些原语类型参与了）→ 用于选择句子模板
    - Case 是"这个情况我见过"，GrammarPattern 是"这种感觉我表达过"
    """

    id: int
    # 激活签名：参与的概念原语名称 + 关系类型的排序元组
    # 例如：("HAVE", "IS", "LIKE", "cause-of", "part-of")
    activation_signature: tuple[str, ...] = ()

    # 句子结构模板列表（含 {slot_N} 占位符）
    # 例如：["我觉得{slot_0}。", "应该是{slot_0}吧。"]
    sentence_structures: list[str] = field(default_factory=list)

    # 连接到概念键的电线签名——记录哪些概念键 ID 曾触发过这个模式
    # 用于后续精确匹配和 Hebbian 强化
    linked_concept_ids: list[int] = field(default_factory=list)

    # 语气
    tone: str = "normal"  # "normal" | "brief" | "gentle" | "serious" | "excited"

    # 统计
    frequency: int = 0           # 被使用的总次数
    success_count: int = 0       # 输出后被接受的次数
    fail_count: int = 0          # 输出后效果不佳的次数

    # 时间戳
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    # KeyNetwork 中的 OUTPUT 键 ID 列表（一个模板 = 一个 OUTPUT 键）
    output_key_ids: list[int] = field(default_factory=list)

    # 状态
    is_dormant: bool = False
    is_seed: bool = False        # 种子模式（从旧模板转化），不会被完全删除

    def __post_init__(self):
        if self.last_used == 0.0:
            self.last_used = self.created_at

    @property
    def reliability(self) -> float:
        """模式的可靠性分数 (0.0–1.0)。"""
        total = self.success_count + self.fail_count
        if total == 0:
            return GRAMMAR_INITIAL_RELIABILITY
        return self.success_count / total

    @property
    def idle_days(self) -> float:
        """上次使用以来的天数。"""
        if self.last_used == 0:
            return (time.time() - self.created_at) / 86400
        return (time.time() - self.last_used) / 86400

    def record_success(self):
        """记录一次成功的复用。"""
        self.frequency += 1
        self.success_count += 1
        self.last_used = time.time()

    def record_failure(self):
        """记录一次失败的复用。"""
        self.frequency += 1
        self.fail_count += 1
        self.last_used = time.time()

    def snapshot(self) -> dict:
        """可序列化快照。"""
        return {
            "id": self.id,
            "activation_signature": list(self.activation_signature),
            "sentence_structures": self.sentence_structures,
            "linked_concept_ids": self.linked_concept_ids,
            "tone": self.tone,
            "frequency": self.frequency,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "output_key_ids": self.output_key_ids,
            "is_dormant": self.is_dormant,
            "is_seed": self.is_seed,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "GrammarPattern":
        """从快照恢复语法模式。"""
        sig = tuple(data.get("activation_signature", []))
        return cls(
            id=data["id"],
            activation_signature=sig,
            sentence_structures=data.get("sentence_structures", []),
            linked_concept_ids=data.get("linked_concept_ids", []),
            tone=data.get("tone", "normal"),
            frequency=data.get("frequency", 0),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            created_at=data.get("created_at", time.time()),
            last_used=data.get("last_used", 0.0),
            output_key_ids=data.get("output_key_ids", []),
            is_dormant=data.get("is_dormant", False),
            is_seed=data.get("is_seed", False),
        )

    def __repr__(self) -> str:
        status = "💤" if self.is_dormant else ("🌱" if self.is_seed else "✓")
        return (f"Grammar#{self.id} {status} sig={self.activation_signature[:3]}... "
                f"templates={len(self.sentence_structures)} "
                f"rel={self.reliability:.2f} ×{self.frequency}")


# ═══════════════════════════════════════════════
# 1. 激活签名提取
# ═══════════════════════════════════════════════

def compute_activation_signature(
    conduction_result: "GardenConductionResult",
    garden: "Garden",
    question_primitive: str | None = None,
) -> tuple[str, ...]:
    """
    从传导结果中提取"激活签名"——一个描述当前认知格局的排序元组。

    签名包含两类元素：
    1. 概念原语名称：亮键的 primitive 字段（如 "IS", "HAVE", "LIKE"）
    2. 关系类型：传导路径中出现的 relation_type（如 "cause-of", "part-of"）

    签名用于匹配已有的语法输出模式——相似的格局应产生相似的句子结构。

    疑问键（content 前缀「问:」）是路由信号，不是答案内容，不进签名。
    疑问词主导（question_primitive）时，签名聚焦目标原语——「猫能干什么」
    的认知格局是 CAN，不是 IS+CAN+HAVE+DISLIKE 的大杂烩。

    算法：
    1. 收集所有亮键（brightness ≥ 1）的 primitive 类型
    2. 收集传导路径中的 relation_type
    3. 去重、排序，返回不可变元组
    """
    from .primitives import Primitive

    # 内部/机制原语不属于"语义格局"，不该进签名。否则 OUTPUT/WORD/SITUATION
    # 等机制键会污染签名，导致语法匹配永远命中同一模式（"已读乱回"根因之一）。
    # 用裸成员判断——Primitive(str, Enum) 的 str() 返回 "Primitive.OUTPUT" 而非
    # "OUTPUT"，str() 化后反而漏掉；裸成员对枚举和纯字符串两种存储都成立。
    _INTERNAL_PRIMITIVES = {
        Primitive.WORD, Primitive.PATTERN, Primitive.RULE,
        Primitive.CASE, Primitive.OUTPUT, Primitive.SITUATION,
        # M3 机制键（seedplan.py 用裸字符串 primitive）：SELF/MASTER/QUEST
        # 不进签名，否则「你是谁」「手机是什么」共享 QUEST 元素，降低语法匹配区分度。
        "SELF", "MASTER", "QUEST",
    }

    def _collect_primitive(key) -> None:
        """把概念键的原语收进签名（过滤机制键/疑问键/非目标原语）。"""
        if not key.primitive or key.primitive in _INTERNAL_PRIMITIVES:
            return
        if key.content.startswith("问:"):
            return  # 疑问键是路由信号，不是答案内容
        if question_primitive is not None and key.primitive != question_primitive:
            return  # 疑问词聚焦：只看目标原语
        primitives_seen.add(key.primitive)

    primitives_seen: set[str] = set()
    relations_seen: set[str] = set()

    # 1. 从 round1（盆内传导）收集
    for pot_id, cr in conduction_result.round1.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue

        # 收集亮键的原语类型
        for kid, brightness in cr.final_pattern.items():
            if brightness < 1:
                continue
            key = pot.network.get_key(kid)
            if key is None:
                continue
            _collect_primitive(key)

        # 收集路径中的关系类型
        for dst_id, (src_id, src_c, dst_c, wire_thickness, push) in cr.activation_paths.items():
            wire = pot.network.get_wire(src_id, dst_id)
            if wire is not None and wire.relation_type:
                relations_seen.add(wire.relation_type)

    # 2. 从 round3（跨盆传导）收集
    for pot_id, cr in conduction_result.round3.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue

        for kid, brightness in cr.final_pattern.items():
            if brightness < 1:
                continue
            key = pot.network.get_key(kid)
            if key is None:
                continue
            _collect_primitive(key)

        for dst_id, (src_id, src_c, dst_c, wire_thickness, push) in cr.activation_paths.items():
            wire = pot.network.get_wire(src_id, dst_id)
            if wire is not None and wire.relation_type:
                relations_seen.add(wire.relation_type)

    # 3. 合并：原语在前，关系在后，统一排序
    # 用前缀区分，确保排序稳定
    all_elements = sorted(primitives_seen) + sorted(f"rel:{r}" for r in relations_seen)
    return tuple(all_elements)


def signature_to_key(signature: tuple[str, ...]) -> str:
    """将激活签名序列化为可存储的字符串键。"""
    return "|".join(signature)


def key_to_signature(key: str) -> tuple[str, ...]:
    """从字符串键恢复激活签名。"""
    if not key:
        return ()
    return tuple(key.split("|"))


# ═══════════════════════════════════════════════
# 2. 语法盆初始化
# ═══════════════════════════════════════════════

def init_grammar_pot(garden: "Garden") -> str:
    """
    初始化语法盆——存储 OUTPUT 原语键的专用内部盆。

    如果语法盆已存在，直接返回其 ID。
    否则创建新盆并播种初始语法模式（从旧模板转化）。

    返回语法盆的 ID。
    """
    # 检查是否已有语法盆（通过 garden 上可能存在的属性）
    grammar_pot_id = getattr(garden, 'grammar_pot_id', None)
    if grammar_pot_id and grammar_pot_id in garden.pots:
        # 向前兼容：已有 dict 数据但缺少 KeyNetwork OUTPUT 键的，补建
        _backfill_output_keys(garden)
        return grammar_pot_id

    # 创建新盆
    pot = garden.create_pot(
        source_type="grammar",
        topic_label="语法键网络",
        is_internal=False,  # 不覆盖 internal_pot_id
    )
    garden.grammar_pot_id = pot.id

    # 初始化 grammar_library（如果还没有）
    if not hasattr(garden, 'grammar_library') or garden.grammar_library is None:
        garden.grammar_library = {}

    # 播种初始语法模式（从旧硬编码模板转化）
    _seed_initial_patterns(garden)

    # 构建语法盆内部结构：原语标记键 + OUTPUT 键电线
    _ensure_primitive_marker_keys(garden)
    _wire_markers_to_outputs(garden)

    return pot.id


def _seed_initial_patterns(garden: "Garden"):
    """
    将现有硬编码模板转化为种子语法模式。

    这些种子模式标记为 is_seed=True，初始可靠性较低，
    会在实际使用中被 Hebbian 学习调整优化。
    不会被完全删除（is_seed 保护）。
    """
    if not hasattr(garden, 'grammar_library'):
        garden.grammar_library = {}

    # 如果已有种子，不重复播种
    existing_seeds = [p for p in garden.grammar_library.values() if p.is_seed]
    if existing_seeds:
        return

    seeds: list[GrammarPattern] = []
    next_id = max(garden.grammar_library.keys(), default=0) + 1

    # ── 陈述型 ──
    seeds.append(GrammarPattern(
        id=next_id,
        activation_signature=("IS", "HAVE", "SAME"),
        sentence_structures=[
            "嗯，{slot_0}。",
            "{slot_0}嘛。",
            "{slot_0}。",
            "喵，{slot_0}。",
            "我记得——{slot_0}。",
        ],
        tone="normal",
        is_seed=True,
    ))
    next_id += 1

    # ── 回答型 ──
    seeds.append(GrammarPattern(
        id=next_id,
        activation_signature=("IS", "NOT", "CAUSE"),
        sentence_structures=[
            "我觉得是{slot_0}。",
            "应该是{slot_0}吧。",
            "{slot_0}。",
            "嗯……{slot_0}。",
        ],
        tone="normal",
        is_seed=True,
    ))
    next_id += 1

    # ── 困惑型 ──
    seeds.append(GrammarPattern(
        id=next_id,
        activation_signature=(),  # 空签名 = 默认匹配（低优先级）
        sentence_structures=[
            "师父，这个我不太懂……",
            "喵？我没太听明白。",
            "唔……这个好难。",
            "嘿嘿，我不懂这个。",
            "师父，你说的是什么呢？",
        ],
        tone="gentle",
        is_seed=True,
    ))
    next_id += 1

    # ── 兴奋型 ──
    seeds.append(GrammarPattern(
        id=next_id,
        activation_signature=("LIKE", "HAVE", "SEE"),
        sentence_structures=[
            "师父！{slot_0}！",
            "喵！{slot_0}！",
            "啊！{slot_0}！",
            "师父师父！{slot_0}！",
        ],
        tone="excited",
        is_seed=True,
    ))
    next_id += 1

    # ── 主动搭话型 ──
    seeds.append(GrammarPattern(
        id=next_id,
        activation_signature=("rel:solitude",),  # 独处模式触发
        sentence_structures=[
            "师父，今天还没跟我说过话呢。",
            "师父，你在忙什么呀？",
            "喵……师父好久没理我了。",
        ],
        tone="gentle",
        is_seed=True,
    ))
    next_id += 1

    # ── 因果解释型 ──
    seeds.append(GrammarPattern(
        id=next_id,
        activation_signature=("CAUSE", "rel:cause-of"),
        sentence_structures=[
            "{slot_0}，所以{slot_1}。",
            "因为{slot_0}，所以{slot_1}。",
            "{slot_0}导致了{slot_1}。",
        ],
        tone="normal",
        is_seed=True,
    ))
    next_id += 1

    # ── 拥有/归属型 ──
    seeds.append(GrammarPattern(
        id=next_id,
        activation_signature=("HAVE", "NOT_HAVE"),
        sentence_structures=[
            "{slot_0}有{slot_1}。",
            "{slot_0}没有{slot_1}。",
            "{slot_0}带着{slot_1}呢。",
        ],
        tone="normal",
        is_seed=True,
    ))

    for pattern in seeds:
        garden.grammar_library[pattern.id] = pattern

    # ── 在语法盆 KeyNetwork 中创建 OUTPUT 键 ──
    grammar_pot = garden.pots.get(garden.grammar_pot_id)
    if grammar_pot is not None:
        from .primitives import Primitive
        for pattern in seeds:
            for template in pattern.sentence_structures:
                key = grammar_pot.network.create_key(
                    content=template,
                    source="internal",
                )
                key.primitive = Primitive.OUTPUT
                pattern.output_key_ids.append(key.id)


# ═══════════════════════════════════════════════
# 3. 语法盆内部结构 —— 原语标记键 + OUTPUT 键电线
# ═══════════════════════════════════════════════

def _ensure_primitive_marker_keys(garden: "Garden"):
    """
    在语法盆 KeyNetwork 中创建原语标记键。

    每个参与语法匹配的 Primitive 类型一个键，作为传导的种子入口。
    标记键的 content 以 "语法锚:" 为前缀，primitive 字段存原语值。

    已存在的标记键不会重复创建（向前兼容旧快照）。
    """
    from .primitives import Primitive

    grammar_pot = garden.pots.get(garden.grammar_pot_id)
    if grammar_pot is None:
        return

    MARKER_PRIMITIVES = [
        # 感知
        Primitive.SEE, Primitive.HEAR, Primitive.SMELL, Primitive.TASTE, Primitive.TOUCH,
        # 存在
        Primitive.IS, Primitive.NOT, Primitive.HAVE, Primitive.NOT_HAVE,
        # 行动
        Primitive.MOVE, Primitive.GIVE, Primitive.EAT, Primitive.SAY, Primitive.MAKE, Primitive.USE,
        Primitive.CAN,
        # 关系
        Primitive.BIGGER, Primitive.SMALLER, Primitive.INSIDE, Primitive.OUTSIDE,
        Primitive.BEFORE, Primitive.AFTER, Primitive.CAUSE, Primitive.LIKE, Primitive.DISLIKE,
        Primitive.SAME, Primitive.CONNECT,
        # 数量
        Primitive.MANY, Primitive.FEW,
    ]

    # 收集已有标记键
    existing_markers: dict[str, int] = {}
    for key in grammar_pot.network.all_keys():
        if key.primitive and key.primitive != "OUTPUT" and key.content.startswith("语法锚:"):
            existing_markers[key.primitive] = key.id

    for prim in MARKER_PRIMITIVES:
        prim_str = prim.value if isinstance(prim, Primitive) else prim
        if prim_str in existing_markers:
            continue
        key = grammar_pot.network.create_key(
            content=f"语法锚:{prim_str}",
            source="internal",
        )
        key.primitive = prim_str
        existing_markers[prim_str] = key.id

    # 标记键之间建弱线
    _wire_markers_together(grammar_pot, existing_markers)


def _wire_markers_together(grammar_pot, markers: dict[str, int]):
    """
    在原语标记键之间建弱线。

    同类别原语之间线稍粗（0.3），不同类别之间线很弱（0.15）。
    这创建了"结构共振"——传导时相关原语相互支持。
    """
    from .primitives import PRIMITIVE_CATEGORIES

    # 构建 primitive → category 的查找
    prim_category: dict[str, str] = {}
    for cat_name, prim_list in PRIMITIVE_CATEGORIES.items():
        for p in prim_list:
            prim_category[p.value] = cat_name

    marker_ids = list(markers.values())
    for i in range(len(marker_ids)):
        for j in range(i + 1, len(marker_ids)):
            a_id, b_id = marker_ids[i], marker_ids[j]
            # 避免重复建线
            existing = grammar_pot.network.get_wire(a_id, b_id)
            if existing is not None:
                continue
            a_prim = next((p for p, mid in markers.items() if mid == a_id), "")
            b_prim = next((p for p, mid in markers.items() if mid == b_id), "")
            cat_a = prim_category.get(a_prim, "")
            cat_b = prim_category.get(b_prim, "")
            thickness = 0.3 if cat_a == cat_b else 0.15
            grammar_pot.network.set_wire(a_id, b_id, thickness=thickness)


# ═══════════════════════════════════════════════
# 3b. 疑问键 —— 一词多原语键的传导式路由
# ═══════════════════════════════════════════════

# 疑问词 → 目标原语。这是「建键声明」，不是运行时查表：
# 每个疑问词建成一个键（疑问键），键的 primitive 字段声明它指向哪个原语。
# 识别时靠内容匹配键（像找概念键一样），路由时靠 primitive 共振——
# 「问:是什么」这个键亮了，直接引起同原语 IS 的概念键亮，环环相扣。
QUESTION_WORD_TO_PRIMITIVE: dict[str, str] = {
    "是什么": "IS",
    "是什么东西": "IS",
    "属于什么": "IS",
    "能干什么": "CAN",
    "会做什么": "CAN",
    "能做什么": "CAN",
    "怕什么": "DISLIKE",
    "讨厌什么": "DISLIKE",
    "有什么": "HAVE",
    "喜欢什么": "LIKE",
    "爱吃什么": "LIKE",
    "导致什么": "CAUSE",
    "引起什么": "CAUSE",
}


# 疑问词 → 关系词（内容匹配用）。同一原语内的精细路由：疑问键只连
# 「内容含这些关系词」的概念键。「属于什么」→ 只连「猫属于猫科动物」，
# 「是什么」→ 只连「猫是一种哺乳动物」等。建线时定死，运行时纯传导——
# 疑问键亮 → 直接引起目标概念键亮，比词键多一次推力，自然更亮（环环相扣）。
QUESTION_WORD_TO_RELATIONS: dict[str, tuple[str, ...]] = {
    "是什么": ("是", "叫做"),
    "是什么东西": ("是", "叫做"),
    "属于什么": ("属于",),
    "能干什么": ("能", "会", "可以"),
    "会做什么": ("会", "能", "可以"),
    "能做什么": ("能", "会", "可以"),
    "怕什么": ("怕",),
    "讨厌什么": ("讨厌",),
    "有什么": ("有", "拥有", "具有"),
    "喜欢什么": ("喜欢",),
    "爱吃什么": ("爱", "喜欢"),
    "导致什么": ("导致",),
    "引起什么": ("引起",),
}


def ensure_question_keys_for_pot(garden: "Garden", pot):
    """
    在概念盆为每个已存在的原语建疑问键，并连线到同 primitive 的概念键。

    疑问键 content = 「问:<疑问词>」，primitive = 目标原语。
    疑问键 ↔ 概念键之间建 same-as 强线（0.8）——传导时疑问键亮了，
    直接点亮目标原语概念键（「这个亮了直接引起另一个」），不是查表选键。
    """
    from src.core.wire import SLIPPAGE_DISTANCE

    if pot is None:
        return

    # 盆里所有概念原语键（非 WORD/OUTPUT、非疑问键自身），按原语分组
    concept_by_prim: dict[str, list] = {}
    for key in pot.network.all_keys():
        if not key.primitive or key.primitive in ("WORD", "OUTPUT"):
            continue
        if key.content.startswith("问:"):
            continue
        concept_by_prim.setdefault(key.primitive, []).append(key)

    for word, prim in QUESTION_WORD_TO_PRIMITIVE.items():
        concept_keys = concept_by_prim.get(prim, [])
        if not concept_keys:
            continue  # 该盆还没有这个原语的概念键 → 不建空疑问键
        # 同一原语内的精细路由：疑问词自带关系词，只连内容含该关系词的概念键。
        relations = QUESTION_WORD_TO_RELATIONS.get(word, ())
        if relations:
            concept_keys = [
                k for k in concept_keys
                if any(r in k.content for r in relations)
            ]
            if not concept_keys:
                continue
        content = f"问:{word}"
        qkey = next(
            (k for k in pot.network.all_keys() if k.content == content),
            None,
        )
        if qkey is None:
            qkey = pot.network.create_key(content=content, source="internal")
            qkey.primitive = prim
        for ck in concept_keys:
            if pot.network.get_wire(qkey.id, ck.id) is None:
                pot.network.set_wire(qkey.id, ck.id, thickness=0.8)
                w = pot.network.get_wire(qkey.id, ck.id)
                if w:
                    w.relation_type = "same-as"
                    w.slippage_distance = SLIPPAGE_DISTANCE.get("same-as", 0)


def detect_question_key(garden: "Garden", text: str):
    """
    识别输入里的疑问词，返回 (目标原语, 疑问键 id, 盆 id)，无命中返回 None。

    传导式识别：遍历概念盆的疑问键（content 前缀「问:」），
    取 content 在 text 里的最长命中（「能干什么」优先于「什么」）。
    不查表——命中的是键，返回的是键的 primitive 字段 + 其盆。
    """
    best_word = ""
    best: tuple[str, int, str] | None = None
    for pot in garden.pots.values():
        if pot.id == getattr(garden, "grammar_pot_id", ""):
            continue
        if pot.source_type == "internal":
            continue
        for key in pot.network.all_keys():
            if not key.content.startswith("问:"):
                continue
            word = key.content[2:]
            if word and word in text and len(word) > len(best_word):
                best_word = word
                best = (key.primitive, key.id, pot.id)
    return best


def detect_question_primitive(garden: "Garden", text: str) -> str | None:
    """薄封装：只返回目标原语（兼容旧调用点）。"""
    hit = detect_question_key(garden, text)
    return hit[0] if hit else None


def _wire_markers_to_outputs(garden: "Garden"):
    """
    将原语标记键连接到包含该原语的所有 OUTPUT 键。

    对每个 GrammarPattern，读取其 activation_signature，
    找到签名中每个 Primitive 对应的标记键 → 连线到该 pattern 的所有 OUTPUT 键。

    线粗细按签名长度调节：签名越短（越聚焦），线越粗。
    """
    grammar_pot = garden.pots.get(garden.grammar_pot_id)
    if grammar_pot is None:
        return

    # 建标记键查找表
    marker_by_primitive: dict[str, int] = {}
    for key in grammar_pot.network.all_keys():
        if key.primitive and key.primitive != "OUTPUT" and key.content.startswith("语法锚:"):
            marker_by_primitive[key.primitive] = key.id

    if not marker_by_primitive:
        return

    for pattern in garden.grammar_library.values():
        if pattern.is_dormant:
            continue
        if not pattern.activation_signature:
            continue  # 空签名模式（困惑型）不需要连线

        sig_len = len([e for e in pattern.activation_signature if not e.startswith("rel:")])
        if sig_len == 0:
            continue
        thickness = 0.6 if sig_len <= 3 else (0.5 if sig_len <= 5 else 0.4)

        for output_kid in pattern.output_key_ids:
            output_key = grammar_pot.network.get_key(output_kid)
            if output_key is None:
                continue
            for sig_elem in pattern.activation_signature:
                if sig_elem.startswith("rel:"):
                    continue  # 关系类型标记不参与标记键连线
                marker_id = marker_by_primitive.get(sig_elem)
                if marker_id is None:
                    continue
                # 避免重复建线
                existing = grammar_pot.network.get_wire(marker_id, output_kid)
                if existing is None:
                    grammar_pot.network.set_wire(marker_id, output_kid, thickness=thickness)


def _wire_one_output_to_markers(
    garden: "Garden",
    output_key_id: int,
    pattern: "GrammarPattern",
):
    """
    将一个新创建的 OUTPUT 键连接到原语标记键。

    在 _ensure_output_keys 创建新 OUTPUT 键后调用，
    确保动态生成的语法模式也能参与传导匹配。
    """
    grammar_pot = garden.pots.get(garden.grammar_pot_id)
    if grammar_pot is None:
        return

    marker_by_primitive: dict[str, int] = {}
    for key in grammar_pot.network.all_keys():
        if key.primitive and key.primitive != "OUTPUT" and key.content.startswith("语法锚:"):
            marker_by_primitive[key.primitive] = key.id

    if not marker_by_primitive:
        return

    sig_prims = [e for e in pattern.activation_signature if not e.startswith("rel:")]
    sig_len = len(sig_prims)
    if sig_len == 0:
        return
    thickness = 0.6 if sig_len <= 3 else (0.5 if sig_len <= 5 else 0.4)

    for sig_elem in sig_prims:
        marker_id = marker_by_primitive.get(sig_elem)
        if marker_id is not None:
            existing = grammar_pot.network.get_wire(marker_id, output_key_id)
            if existing is None:
                grammar_pot.network.set_wire(marker_id, output_key_id, thickness=thickness)


# ═══════════════════════════════════════════════
# 4. 传导驱动语法匹配（Step 2 核心）—— 替代 Jaccard 查表
# ═══════════════════════════════════════════════

def conduct_grammar_match(
    garden: "Garden",
    activation_signature: tuple[str, ...],
    tone: str = "normal",
    question_primitive: str | None = None,
) -> "GrammarPattern | None":
    """
    通过语法盆内部传导选出最匹配的 OUTPUT 键。

    替代原来的 match_output_pattern() Jaccard 字典遍历。
    签名不是用来"查表"，而是作为种子播进语法盆 → 传导扩散 →
    结构决定哪个 OUTPUT 键最亮。

    流程：
    1. 找到签名中每个 Primitive 对应的原语标记键 → 设为种子
    2. 疑问词主导（question_primitive）：目标原语标记键亮度 3，
       其余原语降为辅助亮度 1——「是什么」亮，IS 标记键就比别的更亮
    3. Tone 引导：同 tone pattern 的签名原语获得微弱加成
    4. 运行标准传导（conduct）
    5. 收集亮度 ≥ 1 的 OUTPUT 键
    6. 按亮度排序 → 找所属 GrammarPattern → 返回最亮的

    返回 None 表示未命中（调用方降级到困惑型或路径构建）。
    """
    from .conduction import conduct, ConductionRequest

    grammar_pot = garden.pots.get(garden.grammar_pot_id)
    if grammar_pot is None:
        return None

    if not hasattr(garden, 'grammar_library') or not garden.grammar_library:
        return None

    # ── 空签名 → 降级 ──
    if not activation_signature:
        return _find_fallback_pattern(garden)

    # ── 1. 建标记键查找表 ──
    marker_by_primitive: dict[str, int] = {}
    for key in grammar_pot.network.all_keys():
        if key.primitive and key.primitive != "OUTPUT" and key.content.startswith("语法锚:"):
            marker_by_primitive[key.primitive] = key.id

    if not marker_by_primitive:
        return _find_fallback_pattern(garden)

    # ── 2. 签名 → 种子 ──
    seeds: list[tuple[int, int]] = []
    for sig_elem in activation_signature:
        if sig_elem.startswith("rel:"):
            continue  # 关系类型标记暂不参与种子（MVP 简化）
        marker_id = marker_by_primitive.get(sig_elem)
        if marker_id is not None:
            if question_primitive is not None:
                # 疑问词主导：目标原语亮度 3，其余原语降为辅助 1
                seeds.append((marker_id, 3 if sig_elem == question_primitive else 1))
            else:
                seeds.append((marker_id, 2))

    if not seeds:
        return _find_fallback_pattern(garden)

    # ── 3. Tone 引导 ──
    # 同 tone 的 pattern 所用的原语标记键获得 +1 额外亮度
    if tone != "normal":
        for pattern in garden.grammar_library.values():
            if pattern.is_dormant or not pattern.output_key_ids:
                continue
            if pattern.tone == tone:
                for sig_elem in pattern.activation_signature:
                    if sig_elem.startswith("rel:"):
                        continue
                    marker_id = marker_by_primitive.get(sig_elem)
                    if marker_id is not None:
                        # 已有种子 → 加亮；否则新增
                        found = False
                        for idx, (mid, b) in enumerate(seeds):
                            if mid == marker_id:
                                seeds[idx] = (mid, min(3, b + 1))
                                found = True
                                break
                        if not found:
                            seeds.append((marker_id, 1))

    # ── 4. 运行传导 ──
    request = ConductionRequest(
        seed_keys=seeds,
        max_hops=1,  # 匹配期走最短路径（标记→OUTPUT一跳）。学习期可升为3+峰值记录
        timeout_ms=5000,
        source="grammar_match",
        temperature=0.3,  # 低温度 → 精准扩散，不随机跳跃
    )
    result = conduct(grammar_pot.network, request)

    # ── 5. 收集 OUTPUT 键（不设绝对门槛，取最亮）──
    # 所有 OUTPUT 键参与竞争，亮度最高的胜出。
    # 只有全部 OUTPUT 键亮度为 0 时才降级到困惑型。
    output_brightness: list[tuple[int, int]] = []
    for key in grammar_pot.network.all_keys():
        if key.primitive != "OUTPUT":
            continue
        b = result.final_pattern.get(key.id, 0)
        if b > 0:
            output_brightness.append((key.id, b))

    if not output_brightness:
        return _find_fallback_pattern(garden)

    # 按亮度降序 — 最亮的 OUTPUT 键胜出
    output_brightness.sort(key=lambda x: x[1], reverse=True)

    # ── 6. OUTPUT 键 → GrammarPattern（最亮的优先）──
    for out_kid, _brightness in output_brightness:
        pattern = _find_pattern_by_output_key(garden, out_kid)
        if pattern is not None:
            return pattern

    return _find_fallback_pattern(garden)


def _find_pattern_by_output_key(
    garden: "Garden",
    output_key_id: int,
) -> "GrammarPattern | None":
    """根据 OUTPUT 键 ID 反查所属 GrammarPattern。"""
    if not hasattr(garden, 'grammar_library'):
        return None
    for pattern in garden.grammar_library.values():
        if not pattern.is_dormant and output_key_id in pattern.output_key_ids:
            return pattern
    return None


def _find_fallback_pattern(garden: "Garden") -> "GrammarPattern | None":
    """
    找降级模式。
    优先返回空签名的种子模式（困惑型），其次返回任意活跃模式。
    """
    if not hasattr(garden, 'grammar_library'):
        return None
    for pattern in garden.grammar_library.values():
        if pattern.is_dormant:
            continue
        if not pattern.activation_signature:
            return pattern
    # 没有空签名模式 → 返回第一个活跃模式
    for pattern in garden.grammar_library.values():
        if not pattern.is_dormant:
            return pattern
    return None


# ═══════════════════════════════════════════════
# 5. 旧版语法匹配（已废弃，保留供参考）
# ═══════════════════════════════════════════════

def match_output_pattern(
    garden: "Garden",
    activation_signature: tuple[str, ...],
    tone: str = "normal",
) -> GrammarPattern | None:
    """
    在语法模式库中寻找与当前激活签名最匹配的输出模式。

    算法（类似 match_case）：
    1. 过滤：只考虑非休眠模式，优先同 tone
    2. 评分：Jaccard 相似度 × reliability × tone 匹配加成
    3. 空签名（困惑型）只在无其他匹配时使用
    4. 返回最佳匹配（如果分数 ≥ GRAMMAR_MIN_MATCH_SIMILARITY）

    tone 匹配逻辑：
    - 同 tone：分数 × 1.3（鼓励风格一致）
    - 不同 tone 但非 excited vs serious 对立：分数 × 1.0
    - 对立 tone：分数 × 0.5
    """
    if not hasattr(garden, 'grammar_library') or not garden.grammar_library:
        return None

    sig_set = set(activation_signature)
    best_pattern: GrammarPattern | None = None
    best_score = 0.0
    fallback_empty: GrammarPattern | None = None  # 空签名模式作为最后降级

    for pattern in garden.grammar_library.values():
        if pattern.is_dormant:
            continue

        # 空签名模式 → 保留作为降级，不参与正常评分
        if not pattern.activation_signature:
            fallback_empty = pattern
            continue

        pattern_set = set(pattern.activation_signature)

        # Jaccard 相似度
        intersection = len(sig_set & pattern_set)
        union = len(sig_set | pattern_set)
        if union == 0:
            continue
        jaccard = intersection / union

        # 综合评分：Jaccard × reliability
        score = jaccard * pattern.reliability

        # tone 匹配加成
        if pattern.tone == tone:
            score *= 1.3
        elif _are_opposite_tones(pattern.tone, tone):
            score *= 0.5

        if score > best_score:
            best_score = score
            best_pattern = pattern

    if best_score >= GRAMMAR_MIN_MATCH_SIMILARITY and best_pattern is not None:
        return best_pattern

    # 降级：使用空签名模式（困惑型）
    if fallback_empty is not None and len(activation_signature) < GRAMMAR_MIN_SIGNATURE_SIZE:
        return fallback_empty

    return None


def _are_opposite_tones(tone_a: str, tone_b: str) -> bool:
    """判断两种语气是否对立（不应互换使用）。"""
    opposites = {
        ("excited", "serious"),
        ("serious", "excited"),
        ("brief", "excited"),
        ("excited", "brief"),
    }
    return (tone_a, tone_b) in opposites


# ═══════════════════════════════════════════════
# 6. 语法模式存储（Step 2 核心）
# ═══════════════════════════════════════════════

def store_output_pattern(
    garden: "Garden",
    activation_signature: tuple[str, ...],
    sentences: list[str],
    tone: str = "normal",
    linked_concept_ids: list[int] | None = None,
    concept_key_to_pot: dict[int, str] | None = None,
) -> GrammarPattern | None:
    """
    存储一个新的语法输出模式（或更新已有模式）。

    去重检查：
    - 如果已有 Jaccard ≥ 0.8 的模式 → 合并 sentence_structures，不创建新条目
    - 否则创建新的 GrammarPattern

    concept_key_to_pot: {概念键ID: 所在盆ID}，用于创建跨盆桥接线。
    如果提供，会在语法盆中为新增模板创建 OUTPUT 键，并桥接概念键。

    返回存储/更新后的 GrammarPattern。
    """
    if not activation_signature or len(activation_signature) < GRAMMAR_MIN_SIGNATURE_SIZE:
        return None

    if not sentences:
        return None

    if not hasattr(garden, 'grammar_library'):
        garden.grammar_library = {}

    sig_set = set(activation_signature)

    # ── 去重检查 ──
    for existing in garden.grammar_library.values():
        if existing.is_dormant:
            continue
        existing_set = set(existing.activation_signature)
        intersection = len(sig_set & existing_set)
        union = len(sig_set | existing_set)
        if union > 0 and intersection / union >= 0.8:
            # 太相似 → 合并
            new_templates = [s for s in sentences if s not in existing.sentence_structures]
            _merge_grammar_pattern(existing, sentences, linked_concept_ids or [])
            _ensure_output_keys(garden, existing, new_templates, concept_key_to_pot)
            return existing

    # ── 容量检查 ──
    if len(garden.grammar_library) >= GRAMMAR_MAX_PATTERNS:
        _evict_oldest_grammar(garden)

    # ── 创建新模式 ──
    pattern_id = _next_grammar_id(garden)
    pattern = GrammarPattern(
        id=pattern_id,
        activation_signature=activation_signature,
        sentence_structures=list(sentences),
        linked_concept_ids=list(linked_concept_ids or []),
        tone=tone,
    )
    garden.grammar_library[pattern_id] = pattern
    _ensure_output_keys(garden, pattern, list(sentences), concept_key_to_pot)
    return pattern


def _merge_grammar_pattern(
    existing: GrammarPattern,
    new_sentences: list[str],
    new_concept_ids: list[int],
):
    """
    将新句子结构合并到已有模式中。

    - 新句子追加到 sentence_structures（去重，最多保留 10 个）
    - concept IDs 合并
    - 更新 last_used 时间戳
    """
    for s in new_sentences:
        if s not in existing.sentence_structures:
            existing.sentence_structures.append(s)

    # 限制模板数量，保留最新的
    if len(existing.sentence_structures) > 10:
        # 保留前 5 个（最老的）+ 后 5 个（最新的）
        existing.sentence_structures = (
            existing.sentence_structures[:5] + existing.sentence_structures[-5:]
        )

    # 合并 concept IDs
    existing_ids = set(existing.linked_concept_ids)
    for cid in new_concept_ids:
        if cid not in existing_ids:
            existing.linked_concept_ids.append(cid)
            existing_ids.add(cid)

    existing.last_used = time.time()


def _ensure_output_keys(
    garden: "Garden",
    pattern: GrammarPattern,
    templates: list[str],
    concept_key_to_pot: dict[int, str] | None = None,
):
    """
    确保语法盆 KeyNetwork 中存在对应的 OUTPUT 键，并（可选）桥接概念键。

    - 为 templates 中尚未对应 OUTPUT 键的模板创建新键
    - 如果提供了 concept_key_to_pot，为所有概念键→OUTPUT 键创建跨盆桥接

    被 store_output_pattern 和 init_grammar_pot 共用。
    """
    from .primitives import Primitive

    grammar_pot = garden.pots.get(garden.grammar_pot_id)
    if grammar_pot is None:
        return

    # ── 为模板创建 OUTPUT 键（跳过已存在的）──
    existing_contents = {
        grammar_pot.network.get_key(kid).content
        for kid in pattern.output_key_ids
        if grammar_pot.network.get_key(kid) is not None
    }

    new_output_ids: list[int] = []
    for template in templates:
        if template in existing_contents:
            continue
        key = grammar_pot.network.create_key(
            content=template,
            source="internal",
        )
        key.primitive = Primitive.OUTPUT
        pattern.output_key_ids.append(key.id)
        existing_contents.add(template)
        new_output_ids.append(key.id)

    # ── 新 OUTPUT 键 → 原语标记键连线 ──
    for output_kid in new_output_ids:
        _wire_one_output_to_markers(garden, output_kid, pattern)

    # ── 连接概念键 → OUTPUT 键（跨盆桥接）──
    if concept_key_to_pot:
        for concept_kid, concept_pot_id in concept_key_to_pot.items():
            # 跳过语法盆内部的键（避免自指桥接）
            if concept_pot_id == garden.grammar_pot_id:
                continue
            for output_kid in pattern.output_key_ids:
                garden.create_bridge(
                    src_key_id=concept_kid,
                    src_pot_id=concept_pot_id,
                    dst_key_id=output_kid,
                    dst_pot_id=garden.grammar_pot_id,
                    thickness=0.25,
                )


def _backfill_output_keys(garden: "Garden"):
    """
    向前兼容：已有 grammar_library dict 数据但缺少 KeyNetwork OUTPUT 键的，补建。

    这处理 Phase 7 实现时只写 dict 不创建键的历史数据。
    只有当某 pattern 的 output_key_ids 为空时才补建——已补建过的跳过。
    """
    if not hasattr(garden, 'grammar_library') or not garden.grammar_library:
        return

    grammar_pot = garden.pots.get(garden.grammar_pot_id)
    if grammar_pot is None:
        return

    for pattern in garden.grammar_library.values():
        if not pattern.output_key_ids:
            for template in pattern.sentence_structures:
                key = grammar_pot.network.create_key(
                    content=template,
                    source="internal",
                )
                from .primitives import Primitive
                key.primitive = Primitive.OUTPUT
                pattern.output_key_ids.append(key.id)
                # 新键连上标记键
                _wire_one_output_to_markers(garden, key.id, pattern)

    # 确保标记键存在（旧快照可能没有）
    _ensure_primitive_marker_keys(garden)
    _wire_markers_to_outputs(garden)


def _next_grammar_id(garden: "Garden") -> int:
    """分配下一个语法模式 ID。"""
    if not garden.grammar_library:
        return 1
    return max(garden.grammar_library.keys()) + 1


def _evict_oldest_grammar(garden: "Garden"):
    """驱逐最老的语法模式（优先休眠，跳过种子）。"""
    if not garden.grammar_library:
        return

    # 优先驱逐休眠的非种子模式
    dormant = [
        (gid, gp) for gid, gp in garden.grammar_library.items()
        if gp.is_dormant and not gp.is_seed
    ]
    if dormant:
        oldest = min(dormant, key=lambda x: x[1].last_used)
        del garden.grammar_library[oldest[0]]
        return

    # 驱逐最久未用的非种子模式
    non_seeds = [
        (gid, gp) for gid, gp in garden.grammar_library.items()
        if not gp.is_seed
    ]
    if non_seeds:
        oldest = min(non_seeds, key=lambda x: x[1].last_used)
        del garden.grammar_library[oldest[0]]


# ═══════════════════════════════════════════════
# 7. Hebbian 强化（Step 2 核心）
# ═══════════════════════════════════════════════

def reinforce_output_pattern(
    pattern: GrammarPattern,
    success: bool,
):
    """
    Hebbian 强化/弱化一个语法输出模式。

    - 成功：success_count++，reliability 上升
    - 失败：fail_count++，连续失败可能导致休眠
    """
    if success:
        pattern.record_success()
    else:
        pattern.record_failure()
        # 可靠性过低且非种子 → 休眠
        if pattern.reliability < 0.15 and not pattern.is_seed:
            pattern.is_dormant = True


def _reinforce_grammar_bridges(
    garden: "Garden",
    pattern: GrammarPattern,
    concept_key_to_pot: dict[int, str],
    delta: float = 0.02,
):
    """
    语法模式匹配成功后，加粗概念键 → OUTPUT 键之间的桥接线。

    如果桥接尚不存在（如种子模式首次被使用），则创建之。
    这是 KeyNetwork 层面的 Hebbian 强化，对应 dict 层面的 record_success()。
    """
    for concept_kid, concept_pot_id in concept_key_to_pot.items():
        # 跳过语法盆内部的键
        if concept_pot_id == garden.grammar_pot_id:
            continue
        for output_kid in pattern.output_key_ids:
            bridge = garden.create_bridge(
                src_key_id=concept_kid,
                src_pot_id=concept_pot_id,
                dst_key_id=output_kid,
                dst_pot_id=garden.grammar_pot_id,
                thickness=0.25,
            )
            if bridge is None:
                # 桥接已存在 → 加粗
                existing = garden.bridge_table.get(concept_kid, [])
                for b in existing:
                    if b.dst_key_id == output_kid and b.dst_pot_id == garden.grammar_pot_id:
                        b.strengthen(delta)
                        break


# ═══════════════════════════════════════════════
# 8. 语法模式维护
# ═══════════════════════════════════════════════

def decay_grammar_patterns(garden: "Garden"):
    """
    语法模式库维护（在 garden.maintenance 中调用）。

    1. 长期未用的非种子模式 → 休眠
    2. 休眠超过 60 天的非种子模式 → 删除
    3. 可靠性 < 0.1 且 frequency ≥ 10 的非种子模式 → 删除
    """
    if not hasattr(garden, 'grammar_library') or not garden.grammar_library:
        return

    now = time.time()
    to_delete: list[int] = []

    for gid, pattern in list(garden.grammar_library.items()):
        if pattern.is_seed:
            continue

        idle_days = (now - pattern.last_used) / 86400 if pattern.last_used > 0 else pattern.idle_days

        # 超过 GRAMMAR_DORMANCY_DAYS 天未用 → 休眠
        if idle_days >= GRAMMAR_DORMANCY_DAYS and not pattern.is_dormant:
            pattern.is_dormant = True

        # 休眠超过 60 天 → 删除
        if pattern.is_dormant and idle_days >= 60:
            to_delete.append(gid)
            continue

        # 可靠性过低 → 删除
        if pattern.reliability < 0.1 and pattern.frequency >= 10:
            to_delete.append(gid)
            continue

    for gid in to_delete:
        del garden.grammar_library[gid]


def get_grammar_stats(garden: "Garden") -> dict:
    """获取语法模式库统计信息。"""
    if not hasattr(garden, 'grammar_library') or not garden.grammar_library:
        return {
            "total": 0,
            "active": 0,
            "dormant": 0,
            "seed": 0,
            "avg_reliability": 0.0,
        }

    patterns = list(garden.grammar_library.values())
    active = [p for p in patterns if not p.is_dormant]
    return {
        "total": len(patterns),
        "active": len(active),
        "dormant": sum(1 for p in patterns if p.is_dormant),
        "seed": sum(1 for p in patterns if p.is_seed),
        "avg_reliability": (
            sum(p.reliability for p in active) / len(active) if active else 0.0
        ),
    }
