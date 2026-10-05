"""
键盘模型 AI — 概念原语体系

罗小黑理解世界的原子意义单位。每个原语定义了一种基本的概念结构，
所有投喂文本都会被拆解为以原语为核心的概念键簇。

这不是独立的"消化模块"——原语键本身存在于键网络中，
拆解过程由语言盆中的模式键驱动。

参考：Schank (1972) Conceptual Dependency Theory
      但做了大幅简化，去掉不必要的抽象层级。
"""

from __future__ import annotations

from enum import Enum


# ═══════════════════════════════════════════════
# 原语枚举
# ═══════════════════════════════════════════════

class Primitive(str, Enum):
    """
    概念原语类型。

    分为五大类：感知、存在、行动、关系、数量。
    每个原语有固定的槽位签名，槽位值是对其他概念键的 ID 引用。
    没有槽位的原语（如 MANY）表示内在属性，不需要外部引用。
    """

    # ── 感知原语（罗小黑理解世界的五个入口） ──
    SEE = "SEE"        # 用眼睛接收视觉信息
    HEAR = "HEAR"      # 用耳朵接收声音信息
    SMELL = "SMELL"    # 用鼻子接收气味信息
    TASTE = "TASTE"    # 用舌头接收味道信息
    TOUCH = "TOUCH"    # 用身体接触感知表面/温度/质地

    # ── 存在原语（判断事物的存在与归属） ──
    IS = "IS"          # 一个东西属于某个类别
    NOT = "NOT"        # 否定一个说法
    HAVE = "HAVE"      # 一个东西存在或属于谁
    NOT_HAVE = "NOT_HAVE"  # 一个东西不存在或缺少

    # ── 行动原语（发生在世界中的动作） ──
    MOVE = "MOVE"      # 自己或物体改变物理位置
    GIVE = "GIVE"      # 一个东西从A转移到B
    EAT = "EAT"        # 把东西放进嘴里吞下去
    SAY = "SAY"        # 用嘴发出有意义的语言
    MAKE = "MAKE"      # 用手或工具创造/改变一个东西
    USE = "USE"        # 用某个工具来完成一个动作
    CAN = "CAN"        # 有能力做某个动作（世界知识：X能Y）

    # ── 关系原语（两个事物之间的关系） ──
    BIGGER = "BIGGER"    # A比B在某个维度上更大
    SMALLER = "SMALLER"  # A比B在某个维度上更小
    INSIDE = "INSIDE"    # A在B的内部空间里
    OUTSIDE = "OUTSIDE"  # A在B的外部
    BEFORE = "BEFORE"    # A在时间上先于B
    AFTER = "AFTER"      # A在时间上后于B
    CAUSE = "CAUSE"      # A导致B发生
    LIKE = "LIKE"        # A对B有好感
    DISLIKE = "DISLIKE"  # A对B有厌恶
    SAME = "SAME"        # A和B没有差别
    CONNECT = "CONNECT"  # A和B相连/连接

    # ── 数量原语（数量的表达） ──
    MANY = "MANY"      # 数量多
    FEW = "FEW"        # 数量少

    # ── 特殊原语 ──
    WORD = "WORD"      # 词汇键——存储具体的中文词汇，用于语言输出
    PATTERN = "PATTERN"  # 模式键——存储句式拆解模板，用于 CD 拆解
    RULE = "RULE"      # 规则键——存储 IF...THEN... 的产生式规则
    CASE = "CASE"      # 案例键——存储对话案例快照的锚点
    OUTPUT = "OUTPUT"  # 输出模式键——存储"格局→句子"的映射，语法键网络核心
    SITUATION = "SITUATION"  # 情境键——存储社交情境感知状态


# ═══════════════════════════════════════════════
# 槽位签名：每个原语有哪些槽位
# ═══════════════════════════════════════════════

PRIMITIVE_SLOTS: dict[Primitive, tuple[str, ...]] = {
    # 感知原语
    Primitive.SEE:    ("agent", "target", "result"),
    Primitive.HEAR:   ("agent", "source", "result"),
    Primitive.SMELL:  ("agent", "source", "result"),
    Primitive.TASTE:  ("agent", "target", "quality"),
    Primitive.TOUCH:  ("agent", "target", "quality"),

    # 存在原语
    Primitive.IS:       ("subject", "category"),
    Primitive.NOT:      ("subject", "category"),
    Primitive.HAVE:     ("owner", "owned"),
    Primitive.NOT_HAVE: ("owner", "lacking"),

    # 行动原语
    Primitive.MOVE: ("actor", "origin", "destination"),
    Primitive.GIVE: ("giver", "receiver", "object"),
    Primitive.EAT:  ("eater", "food"),
    Primitive.SAY:  ("speaker", "content"),
    Primitive.MAKE: ("maker", "product", "tool"),
    Primitive.USE:  ("user", "tool", "purpose"),
    Primitive.CAN:  ("agent", "action"),

    # 关系原语
    Primitive.BIGGER:   ("a", "b", "dimension"),
    Primitive.SMALLER:  ("a", "b", "dimension"),
    Primitive.INSIDE:   ("container", "contained"),
    Primitive.OUTSIDE:  ("outside", "inside_object"),
    Primitive.BEFORE:   ("earlier", "later"),
    Primitive.AFTER:    ("later", "earlier"),
    Primitive.CAUSE:    ("reason", "effect"),
    Primitive.LIKE:     ("liker", "liked"),
    Primitive.DISLIKE:  ("disliker", "disliked"),
    Primitive.SAME:     ("a", "b"),
    Primitive.CONNECT:  ("a", "b"),

    # 数量原语（无槽位——内在属性）
    Primitive.MANY: (),
    Primitive.FEW:  (),

    # 特殊原语
    Primitive.WORD:      ("text",),
    Primitive.PATTERN:   ("pattern", "primitive", "slot_map"),
    Primitive.RULE:      ("condition", "action"),
    Primitive.CASE:      ("trigger", "snapshot"),
    Primitive.OUTPUT:    ("trigger_signature", "template"),
    Primitive.SITUATION: (),  # 情境键：标记键，无数据槽位
}


# ═══════════════════════════════════════════════
# 原语分类
# ═══════════════════════════════════════════════

PRIMITIVE_CATEGORIES: dict[str, list[Primitive]] = {
    "感知": [Primitive.SEE, Primitive.HEAR, Primitive.SMELL,
             Primitive.TASTE, Primitive.TOUCH],
    "存在": [Primitive.IS, Primitive.NOT, Primitive.HAVE, Primitive.NOT_HAVE],
    "行动": [Primitive.MOVE, Primitive.GIVE, Primitive.EAT, Primitive.SAY,
             Primitive.MAKE, Primitive.USE, Primitive.CAN],
    "关系": [Primitive.BIGGER, Primitive.SMALLER, Primitive.INSIDE,
             Primitive.OUTSIDE, Primitive.BEFORE, Primitive.AFTER,
             Primitive.CAUSE, Primitive.LIKE, Primitive.DISLIKE, Primitive.SAME,
             Primitive.CONNECT],
    "数量": [Primitive.MANY, Primitive.FEW],
    "特殊": [Primitive.WORD, Primitive.PATTERN, Primitive.RULE, Primitive.CASE,
             Primitive.OUTPUT, Primitive.SITUATION],
}


# ═══════════════════════════════════════════════
# 默认关系类型：同类原语之间的默认电线类型
# ═══════════════════════════════════════════════

DEFAULT_RELATIONS: dict[tuple[Primitive, Primitive], str] = {
    # 同类型感知之间 → 相似
    (Primitive.SEE, Primitive.HEAR): "similar-to",
    (Primitive.SEE, Primitive.SMELL): "similar-to",
    (Primitive.HEAR, Primitive.SMELL): "similar-to",

    # 对立关系
    (Primitive.IS, Primitive.NOT): "opposite-of",
    (Primitive.HAVE, Primitive.NOT_HAVE): "opposite-of",
    (Primitive.BIGGER, Primitive.SMALLER): "opposite-of",
    (Primitive.INSIDE, Primitive.OUTSIDE): "opposite-of",
    (Primitive.BEFORE, Primitive.AFTER): "opposite-of",
    (Primitive.LIKE, Primitive.DISLIKE): "opposite-of",
    (Primitive.MANY, Primitive.FEW): "opposite-of",

    # 因果链（行动 → 关系）
    (Primitive.GIVE, Primitive.HAVE): "cause-of",
    (Primitive.EAT, Primitive.INSIDE): "cause-of",
    (Primitive.MOVE, Primitive.INSIDE): "cause-of",
    (Primitive.MAKE, Primitive.HAVE): "cause-of",
}

# 保证对称
_symmetric = dict(DEFAULT_RELATIONS)
for (a, b), rel in _symmetric.items():
    DEFAULT_RELATIONS.setdefault((b, a), rel)


# ═══════════════════════════════════════════════
# CD 拆解模式：中文句式 → 原语 + 槽位映射
# ═══════════════════════════════════════════════

class CDPattern:
    """
    一个 CD 拆解模式。

    定义了"什么样的中文句式"对应"什么样的概念原语+槽位填充方式"。
    模式键（PATTERN）存储在语言盆中，拆解时由模式键驱动。
    """
    __slots__ = ("name", "triggers", "primitive", "slot_map", "description")

    def __init__(
        self,
        name: str,
        triggers: list[str],
        primitive: Primitive,
        slot_map: dict[str, str],
        description: str = "",
    ):
        """
        name: 模式名（如 "X把Y放在Z"）
        triggers: 触发这个模式的关键词或结构标记
        primitive: 对应的概念原语
        slot_map: 中文成分 → 槽位名的映射
        description: 人类可读的描述
        """
        self.name = name
        self.triggers = triggers
        self.primitive = primitive
        self.slot_map = slot_map
        self.description = description


# ── 预定义的 CD 拆解模式 ──

CD_PATTERNS: list[CDPattern] = [
    # ── 判断句 ──
    CDPattern(
        name="X是Y",
        triggers=["是", "就是", "是一种", "叫做"],
        primitive=Primitive.IS,
        slot_map={"X": "subject", "Y": "category"},
        description="判断句：X属于Y这个类别",
    ),
    CDPattern(
        name="X不是Y",
        triggers=["不是", "并不是", "不是一种"],
        primitive=Primitive.NOT,
        slot_map={"X": "subject", "Y": "category"},
        description="否定判断：X不属于Y",
    ),

    # ── 存在/拥有句 ──
    CDPattern(
        name="X有Y",
        triggers=["有", "拥有", "长了", "带着"],
        primitive=Primitive.HAVE,
        slot_map={"X": "owner", "Y": "owned"},
        description="拥有：X拥有Y",
    ),
    CDPattern(
        name="X没有Y",
        triggers=["没有", "没", "缺少"],
        primitive=Primitive.NOT_HAVE,
        slot_map={"X": "owner", "Y": "lacking"},
        description="缺少：X没有Y",
    ),

    # ── 把字句 ──
    CDPattern(
        name="X把Y放在Z",
        triggers=["把", "放在", "放到", "放进"],
        primitive=Primitive.GIVE,
        slot_map={"X": "giver", "Y": "object", "Z": "receiver"},
        description="把字句：X把Y转移到了Z",
    ),

    # ── 用字句（工具） ──
    CDPattern(
        name="X用Y做Z",
        triggers=["用", "拿", "使用"],
        primitive=Primitive.USE,
        slot_map={"X": "user", "Y": "tool", "Z": "purpose"},
        description="工具句：X用Y来完成Z",
    ),

    # ── 比较句 ──
    CDPattern(
        name="X比Y大/小",
        triggers=["比", "更", "不如", "像……一样"],
        primitive=Primitive.BIGGER,
        slot_map={"X": "a", "Y": "b", "Z": "dimension"},
        description="比较句：X比Y在某维度上更大或更小（Z=维度词如大小/高低/快慢）",
    ),

    # ── 因果句 ──
    CDPattern(
        name="X导致Y",
        triggers=["因为", "所以", "导致", "结果", "于是"],
        primitive=Primitive.CAUSE,
        slot_map={"X": "reason", "Y": "effect"},
        description="因果句：X导致了Y",
    ),

    # ── 感知句 ──
    CDPattern(
        name="X看Y",
        triggers=["看", "看见", "看到", "盯着"],
        primitive=Primitive.SEE,
        slot_map={"X": "agent", "Y": "target", "Z": "result"},
        description="视觉感知：X看见了Y（Z=看见的结果）",
    ),
    CDPattern(
        name="X听到Y",
        triggers=["听", "听到", "听见"],
        primitive=Primitive.HEAR,
        slot_map={"X": "agent", "Y": "source", "Z": "result"},
        description="听觉感知：X听到了Y发出的声音（Z=听到的结果）",
    ),
    CDPattern(
        name="X闻到Y",
        triggers=["闻", "闻到"],
        primitive=Primitive.SMELL,
        slot_map={"X": "agent", "Y": "source", "Z": "result"},
        description="嗅觉感知：X闻到了Y的气味（Z=闻到的结果）",
    ),

    # ── 移动句 ──
    CDPattern(
        name="X去到Y",
        triggers=["去", "到", "跑向", "走到", "来到", "过去"],
        primitive=Primitive.MOVE,
        slot_map={"X": "actor", "Y": "destination", "Z": "origin"},
        description="移动：X移动到了Y的位置（Z=出发地，可选）",
    ),

    # ── 吃喝句 ──
    CDPattern(
        name="X吃Y",
        triggers=["吃", "喝", "尝", "吞"],
        primitive=Primitive.EAT,
        slot_map={"X": "eater", "Y": "food"},
        description="摄入：X把Y放进嘴里",
    ),

    # ── 制作句 ──
    CDPattern(
        name="X做Y",
        triggers=["做", "制作", "创造", "造", "煮", "烧"],
        primitive=Primitive.MAKE,
        slot_map={"X": "maker", "Y": "product", "Z": "tool"},
        description="制作：X创造了Y（Z=使用的工具，可选）",
    ),

    # ── 说字句 ──
    CDPattern(
        name="X说Y",
        triggers=["说", "告诉", "讲", "问"],
        primitive=Primitive.SAY,
        slot_map={"X": "speaker", "Y": "content"},
        description="说话：X发出了语言Y",
    ),

    # ── 喜欢/不喜欢 ──
    CDPattern(
        name="X喜欢Y",
        triggers=["喜欢", "爱", "喜爱", "中意", "偏好"],
        primitive=Primitive.LIKE,
        slot_map={"X": "liker", "Y": "liked"},
        description="喜好：X对Y有好感",
    ),
    CDPattern(
        name="X不喜欢Y",
        triggers=["不喜欢", "讨厌", "怕", "害怕"],
        primitive=Primitive.DISLIKE,
        slot_map={"X": "disliker", "Y": "disliked"},
        description="厌恶：X对Y有负面感受",
    ),
]


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def find_pattern_for_sentence(tokens: list[str]) -> CDPattern | None:
    """
    根据分词结果找到最匹配的 CD 拆解模式。
    匹配策略：统计每个 pattern 的 triggers 在 tokens 中出现的数量。
    返回触发词命中最多的 pattern。
    """
    best_pattern: CDPattern | None = None
    best_score = 0

    for pattern in CD_PATTERNS:
        score = sum(1 for t in tokens if t in pattern.triggers)
        if score > best_score:
            best_score = score
            best_pattern = pattern

    return best_pattern if best_score > 0 else None
