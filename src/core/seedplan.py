"""
键盘模型 AI — 指代消解种子规划（M3）

把输入里的指代词 / 疑问词转成机制键种子，让「你是谁」这类
问题能被锁定到 SELF / MASTER / QUEST 键上，再由传导够到答案。

纯符号：这里不产生任何新知识，只让某些键亮、调某些线的导电率。
删掉本文件任何一条规则，网络不少一根线。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.garden import Garden


# ── 指代 / 疑问映射 ──────────────────────────

REFERENCE_SEED = {
    "你": "SELF", "小黑": "SELF", "罗小黑": "SELF",
    "师父": "MASTER", "我": "MASTER",
}

QUEST_WORD_MAP = {
    "谁": "WHO", "什么": "WHAT", "为啥": "WHY", "为什么": "WHY", "怎么": "HOW",
}

# 只匹配明确的自我介绍句式或名字身份，不放裸「小黑」前缀——
# 否则「小黑吃饭」「小黑喜欢鱼」这类经历键也会连上 SELF 桥，问「你是谁」答偏。
SELF_CONTENT_RULE = lambda c: c.startswith(("我是", "我叫", "我的名字")) or "罗小黑" in c


# 机制键规格：name → (content, primitive)
_MECHANISM_SPECS = {
    "SELF":   ("自我:罗小黑", "SELF"),
    "MASTER": ("师父:无限", "MASTER"),
    "WHO":    ("问:谁", "QUEST"),
    "WHAT":   ("问:什么", "QUEST"),
    "WHY":    ("问:为什么", "QUEST"),
    "HOW":    ("问:怎么", "QUEST"),
}


def ensure_mechanism_keys(garden: "Garden") -> dict[str, int]:
    """幂等地在 internal 盆常驻机制键。返回 {name: key_id}。"""
    pot = garden.pots.get(garden.internal_pot_id)
    if pot is None:
        return {}
    net = pot.network
    existing = {k.content: k for k in net.all_keys()}
    result: dict[str, int] = {}
    for name, (content, prim) in _MECHANISM_SPECS.items():
        k = existing.get(content)
        if k is None:
            k = net.create_key(content=content, source="internal")
            k.primitive = prim
            existing[content] = k
        result[name] = k.id
    return result


def _spotlight_for(qtype: str) -> dict[str, float]:
    """问什么 → part-of / same-as 探照灯；问谁 → part-of。"""
    if qtype == "WHAT":
        return {"part-of": 1.6, "same-as": 1.3}
    if qtype == "WHO":
        return {"part-of": 1.6}
    return {}


def plan_seeds(
    garden: "Garden",
    tokens: list[str],
) -> tuple[list[tuple[int, int, str]], dict[str, float]]:
    """
    把疑问词 / 指代词转成机制键种子。

    返回 (mechanism_seeds, spotlight)：
    - mechanism_seeds: [(key_id, brightness, pot_id)]
    - spotlight: relation_boost 探照灯（从触发的 QUEST 键类型读出）
    """
    mech = ensure_mechanism_keys(garden)
    if not mech:
        return [], {}

    mechanism_seeds: list[tuple[int, int, str]] = []
    spotlight: dict[str, float] = {}

    # 疑问词 → QUEST 键 @1 + 探照灯
    for token in tokens:
        qtype = QUEST_WORD_MAP.get(token)
        if qtype and qtype in mech:
            mechanism_seeds.append((mech[qtype], 1, garden.internal_pot_id))
            spotlight = _spotlight_for(qtype)

    # 指代词 → SELF / MASTER
    # 「你是谁」里的 SELF 亮度给到 2（有 谁/叫/名字/姓名 时），否则 1
    self_brightness = 2 if (set(tokens) & {"谁", "叫", "名字", "姓名"}) else 1
    for token in tokens:
        target = REFERENCE_SEED.get(token)
        if target is None:
            continue
        if target == "SELF" and "SELF" in mech:
            mechanism_seeds.append((mech["SELF"], self_brightness, garden.internal_pot_id))
        elif target == "MASTER" and "MASTER" in mech:
            mechanism_seeds.append((mech["MASTER"], 1, garden.internal_pot_id))

    return mechanism_seeds, spotlight
