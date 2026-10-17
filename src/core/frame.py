"""
键盘模型 AI — 言外之力框架（Frame）v1.0（MVP-1）

资格层：在亮度赢家层之前，先识别输入的"言外之力"（illocutionary force）
和"话题焦点"（topic），决定哪些语法路径有资格参与回答。

融合原则（师父定）：
- 亮度不变——它仍然是"谁赢"的机制（赢家层）
- 新增的只是"谁有资格比亮度"（资格层）
- 加一张安全网：话题没有可说的内容 → "我还不太懂{topic}"

言外之力（act）来自符号产生式（IF 关键词 THEN act），不是概率、不是向量。

act 枚举：
- query_who     问人："谁"
- query_what    问物："什么 / 啥"
- query_yesno   问是非："吗 / 是不是 / 有没有"
- query_why     问原因："为什么 / 怎么"
- command       令："请 / 帮我 / 去 / 不要"
- greeting      暄："你好 / 再见"
- ostension     光秃单概念（师父只给一个实义词）
- assert        述：默认陈述

设计边界：见 src/core/__init__.py 完整清单（名称+概念+代码）。

English:
Illocutionary-force frame: before the brightness winner layer, symbolic
production rules identify the act and topic to decide which grammar paths
are eligible to answer, plus a safety net for unanswered topics.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── 言外之力枚举 ──

ACT_QUERY_WHO = "query_who"
ACT_QUERY_WHAT = "query_what"
ACT_QUERY_YESNO = "query_yesno"
ACT_QUERY_WHY = "query_why"
ACT_COMMAND = "command"
ACT_GREETING = "greeting"
ACT_OSTENSION = "ostension"
ACT_ASSERT = "assert"


@dataclass
class Frame:
    """一个输入的言外之力 + 话题焦点。"""

    act: str
    topic: str | None = None            # 话题焦点（被谈论的实义词）
    is_self_query: bool = False         # "你是谁 / 你叫什么" → 问的是小黑自己
    is_definition_query: bool = False   # "X是什么" → 在求 X 的定义
    pred: str = ""                      # MVP-1: 谓词原语（IS/EAT/CAUSE/LIKE/…），空=未绑定
    focus: str = ""                     # MVP-1: 焦点角色（identity/patient/cause/…），空=未绑定
    is_farewell: bool = False           # "再见/拜拜/晚安" → 告别，区别于问候


# ── 功能词 / 应答词（抽取话题与判定光秃概念时过滤） ──

_FUNCTION_WORDS = {
    "你", "我", "他", "她", "它", "这", "那", "的", "了", "吗", "呢", "吧",
    "是", "有", "在", "个", "们", "和", "跟", "就", "都", "也", "很", "还",
    "什么", "啥", "谁", "为什么", "怎么", "为啥", "请问", "啊", "呀", "哦", "嗯",
    "一个", "一种", "东西", "时候", "这样", "那样",
}

# 应答/敷衍词：单字输入里这些不是"求解释"，不该触发光秃单概念。
#（师父说"好"是确认，不是让小黑解释"好"是什么意思。）
_ACK_WORDS = {
    "好", "行", "对", "可以", "好的", "没事", "知道了", "辛苦", "收到",
}

# 注：patient/cause 门控（「吃什么/为什么」→ EAT/CAUSE 反查）属 MVP-2/MVP-3，
# 需 CBR CASE + NARS 继承链，MVP-1 不实现（见 grok 落地顺序 449-451）。


def _strip_function(tokens: list[str]) -> list[str]:
    """去掉功能词和应答词，留下实义词（按原顺序）。"""
    return [t for t in tokens if t not in _FUNCTION_WORDS and t not in _ACK_WORDS]


# ═══════════════════════════════════════════════
# 言外之力识别
# ═══════════════════════════════════════════════

def recognize_frame(text: str, tokens: list[str]) -> Frame:
    """
    识别输入的言外之力 + 话题焦点。

    纯符号产生式：按关键词命中决定 act，按内容抽取 topic。
    顺序重要——先判断最明确的问句类型，再落到默认陈述。
    """
    text = (text or "").strip()
    tokens = [t for t in tokens if t.strip()]

    # ── 暄 ──
    # 高频短问候（精确匹配，避免"早"误命中"早点"）
    _greeting_exact = {"早", "早啊", "早呀", "师父好", "小黑好", "师父早", "小黑早",
                       "在吗", "在么", "哈喽", "哈啰"}
    if text in _greeting_exact or text.lower() in ("hello", "hi"):
        return Frame(act=ACT_GREETING)
    if any(g in text for g in ("你好", "您好", "早上好", "中午好", "下午好",
                               "晚上好", "早安", "晚安", "再见", "拜拜", "嗨")):
        is_farewell = any(g in text for g in ("再见", "拜拜", "晚安"))
        return Frame(act=ACT_GREETING, is_farewell=is_farewell)

    # ── 问小黑自己的名字（"你叫什么"无"谁"字，单独拦） ──
    if "你叫什么" in text or "你叫啥" in text:
        return Frame(act=ACT_QUERY_WHO, topic="self", is_self_query=True,
                     pred="IS", focus="identity")

    # ── 问谁 ──
    if "谁" in text:
        # "你" + "谁" → 问小黑自己（你是谁 / 你叫什么）
        if "你" in text and ("叫" in text or "是" in text):
            return Frame(act=ACT_QUERY_WHO, topic="self", is_self_query=True,
                         pred="IS", focus="identity")
        # 否则问别人 → 抽 topic（去掉功能词后剩下的词）
        content = _strip_function(tokens)
        return Frame(act=ACT_QUERY_WHO, topic=content[0] if content else None,
                     pred="IS", focus="identity")

    # ── 问原因 ──
    # 必须放在"问物"之前："为什么"包含子串"什么"，先判断原因疑问
    if "为什么" in text or "怎么" in text or "为啥" in text:
        return Frame(act=ACT_QUERY_WHY)

    # ── 问物 ──
    if "什么" in text or "啥" in text:
        # topic = 紧邻"什么"前的内容词（吃什么→吃；手机是什么→手机）
        topic = None
        is_definition = False
        for i, t in enumerate(tokens):
            if t in ("什么", "啥"):
                # 往前找最近的实义词
                for j in range(i - 1, -1, -1):
                    if tokens[j] == "是":
                        is_definition = True  # "X是什么" → 求定义
                        continue
                    if tokens[j] not in _FUNCTION_WORDS and tokens[j] not in _ACK_WORDS:
                        topic = tokens[j]
                        break
                break
        if topic is None:
            content = _strip_function(tokens)
            topic = content[0] if content else None
        # MVP-1: 只有「X是什么」求定义时绑定 IS/identity 走门控；
        # 其余问物（吃什么/爱吃什么/看什么）交还亮度赢家层，不拦截。
        pred = focus = ""
        if is_definition:
            pred, focus = "IS", "identity"
        return Frame(act=ACT_QUERY_WHAT, topic=topic,
                     is_definition_query=is_definition, pred=pred, focus=focus)

    # ── 问是非 ──
    if text.endswith("吗") or text.endswith("？") or text.endswith("?") or \
       any(w in text for w in ("是不是", "有没有", "行不行", "对不对")):
        return Frame(act=ACT_QUERY_YESNO)

    # ── 令 ──
    if any(text.startswith(c) for c in ("请", "去", "不要", "帮我", "给我", "过来")):
        return Frame(act=ACT_COMMAND)

    # ── 光秃单概念 ──
    content = _strip_function(tokens)
    if len(content) == 1:
        return Frame(act=ACT_OSTENSION, topic=content[0], pred="IS", focus="identity")

    # ── 默认陈述 ──
    return Frame(act=ACT_ASSERT, topic=content[0] if content else None)


# ═══════════════════════════════════════════════
# 门控检索（焦点钉死后的答案来源）
# ═══════════════════════════════════════════════

def retrieve_identity(garden, topic: str, is_self_query: bool = False) -> str | None:
    """
    门控检索：反查 identity 定义键，返回定义句原文（IS 键的 content）。

    只认「IS 键的 subject 槽位 == 主题词键」这一条资格线——不看传导亮度，
    所以「手机屏幕也是亮的」这种前缀误匹配键再亮也没资格（它的 subject 不是「手机」）。

    is_self_query（「你是谁」）时，主题词是「我/小黑/罗小黑」，并且要选
    含名字的完整自我介绍（「我是一只黑色的小猫妖，叫罗小黑」），
    排除「我是猫妖」这类片段式自称。

    返回 None = 话题没有 identity 定义 → 调用方走安全网「我还不太懂{topic}」。
    纯 slots 反查 + 字符串判断，不碰概率/向量（设计边界见 __init__.py）。
    """
    if not topic and not is_self_query:
        return None

    candidates: list = []
    for pot in garden.pots.values():
        if pot.source_type == "internal":
            continue
        target_words = {"我", "小黑", "罗小黑"} if is_self_query else {topic}
        for k in pot.network.all_keys():
            if k.is_dormant or k.primitive != "IS":
                continue
            sub = pot.network.get_key(k.slots.get("subject", 0)) if k.slots.get("subject") else None
            if sub is None or sub.primitive != "WORD":
                continue
            sub_name = (sub.content or "").removeprefix("词:").removeprefix("词：")
            if sub_name in target_words:
                candidates.append(k)

    if not candidates:
        return None

    if is_self_query:
        # 自我介绍：以「我是/我叫」开头且含名字（罗小黑）最优先
        for k in candidates:
            c = k.content or ""
            if c.startswith(("我是", "我叫")) and "罗小黑" in c:
                return c
        # 退而求其次：以「我是/我叫」开头
        for k in candidates:
            c = k.content or ""
            if c.startswith(("我是", "我叫")):
                return c
        return None

    # 定义句：content 以主题词开头（排除「师父拿着手机…」这类主语不是 topic 的误拆句）
    for k in candidates:
        c = k.content or ""
        if c.startswith(topic):
            return c
    return candidates[0].content if candidates[0].content else None
