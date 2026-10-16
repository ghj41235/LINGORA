"""
键盘模型 AI — 语言模块

将亮键格局翻译为罗小黑风格的中文输出。
三个素材来源：记忆网络（亮键）+ 知识库（抽屉查询）+ 性格参数（风格）。

v1.0: 语法键网络集成——优先从语法盆中匹配输出模式，
      硬编码模板降级为种子模式和后备方案。
一切皆键+线：语言输出由传导驱动，不靠硬编码模板。
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.conduction import ConductionResult
    from src.core.key import Key
    from src.core.garden import Garden


# ═══════════════════════════════════════════════
# 罗小黑性格参数
# ═══════════════════════════════════════════════

# 称呼
CALL_MASTER = "师父"

# 语气词（随机选用）
TONE_LIGHT = ["喵", "嗯", "哦", "啊", "呀"]

# 句末语气
SENTENCE_END = ["。", "~", "！", "……"]

# ═══════════════════════════════════════════════
# 模板库
# ═══════════════════════════════════════════════

def _pick_tone() -> str:
    """随机选一个语气词。"""
    return random.choice(TONE_LIGHT)


# ═══════════════════════════════════════════════
# 时间感知
# ═══════════════════════════════════════════════

def _time_ago(created_at: float) -> str:
    """把时间戳翻译为模糊时间描述。"""
    if created_at == 0:
        return ""
    delta = time.time() - created_at
    if delta < 300:         # 5 分钟内
        return "刚刚"
    elif delta < 7200:      # 2 小时内
        return "刚才"
    elif delta < 86400:     # 24 小时内
        return "今天"
    elif delta < 172800:    # 48 小时内
        return "昨天"
    elif delta < 259200:    # 72 小时内
        return "前天"
    elif delta < 604800:    # 7 天内
        return "前几天"
    elif delta < 2592000:   # 30 天内
        return "前一阵"
    else:
        return "很久以前"


def _time_phrase(created_at: float) -> str:
    """生成罗小黑风格的时间描述短句。"""
    ago = _time_ago(created_at)
    phrases = {
        "昨天": "昨天才学会的",
        "前天": "前天知道的",
        "前几天": "前几天想到的呢",
        "前一阵": "前一阵子的事了",
        "很久以前": "很久以前就知道的",
    }
    return phrases.get(ago, "")




# ── 困惑型 ──────────────────────────────────

TEMPLATE_CONFUSED = [
    "师父，这个我不太懂……",
    "喵？我没太听明白。",
    "唔……这个好难。",
    "嘿嘿，我不懂这个。",
    "师父，你说的是什么呢？",
]


# ── 犹豫型（由 hesitate.py 触发） ─────────────

TEMPLATE_HESITATE = [
    "师父，我在想……{detail}",
    "唔，这个有点拿不准。{detail}",
    "让我再想想……{detail}",
]

# ── 投喂确认 ────────────────────────────────

TEMPLATE_INGEST_DONE = [
    "师父，看完了。{count}个新东西。",
    "喵，学到了{count}个。",
    "哦——知道了{count}件事。",
]


# ── 主动搭话（后期） ─────────────────────────

TEMPLATE_PROACTIVE = [
    "师父，今天还没跟我说过话呢。",
    "师父，你在忙什么呀？",
    "喵……师父好久没理我了。",
]


# ── 寒暄 / 告别（MVP-1：跳过花园，直接回应） ─

TEMPLATE_GREETING = [
    "师父好呀。",
    "喵，师父！",
    "师父来啦！",
    "师父好~",
]

TEMPLATE_FAREWELL = [
    "师父再见。",
    "喵，师父再见~",
    "师父拜拜。",
    "嗯，师父再见。",
]


# ═══════════════════════════════════════════════
# 电线粗细 → 语气副词映射
# ═══════════════════════════════════════════════

def _wire_adverb(wire_thickness: float) -> str:
    """
    电线粗细翻译为罗小黑风格的连接词。
    粗线 → 确定语气，细线 → 不确定语气。
    返回的词用作句中连接，必须自然通顺。
    """
    if wire_thickness >= 0.7:
        return "就是"
    elif wire_thickness >= 0.5:
        return "会"
    elif wire_thickness >= 0.3:
        return "，"
    else:
        return "，好像"



def _is_header_like(content: str) -> bool:
    """判断内容是否像标题/标记符号，不应出现在输出中。"""
    if not content:
        return True
    s = content.strip()
    if s.startswith("=") or s.startswith("【") or s.startswith("##"):
        return True
    if s.startswith("//") or s.startswith("--") or s.startswith("══"):
        return True
    if s.startswith("词:") or s.startswith("词："):
        return True  # WORD 原语键，不应直接输出
    if s.startswith("问:") or s.startswith("问："):
        return True  # 疑问键（问题标记），不应直接输出——它是路由信号不是答案
    if len(s) <= 2 and not any('一' <= ch <= '鿿' for ch in s):
        return True  # 纯符号或过短
    return False


# ═══════════════════════════════════════════════
# 语法键网络 — 模板填充（v1.0 新增）
# ═══════════════════════════════════════════════

def _fill_grammar_template(
    template: str,
    slot_values: dict[int, str],
) -> str:
    """
    将语法模板中的 {slot_N} 占位符替换为实际内容。

    例如：
        template = "我觉得{slot_0}，因为{slot_1}。"
        slot_values = {0: "小鱼干很好吃", 1: "它是脆的"}
        → "我觉得小鱼干很好吃，因为它是脆的。"

    未填充的占位符保留为空字符串。
    """
    result = template
    for idx, value in slot_values.items():
        placeholder = f"{{slot_{idx}}}"
        result = result.replace(placeholder, value)
    # 清理未填充的占位符
    import re
    result = re.sub(r'\{slot_\d+\}', '', result)
    # 清理可能产生的多余标点
    result = re.sub(r'，\s*。', '。', result)
    result = re.sub(r'，\s*，', '，', result)
    result = re.sub(r'。\s*。', '。', result)
    return result.strip()


def _extract_slot_values_from_keys(
    bright_keys: list[Key],
    conduction_result: ConductionResult,
    key_lookup: dict[int, str] | None = None,
    question_primitive: str | None = None,
) -> dict[int, str]:
    """
    从亮键和传导路径中提取槽位填充值。

    策略：
    - slot_0: 最重要的根键（种子键或被推最多的源键）
    - slot_1: 第一个子键或关联键
    - slot_2+: 后续子键

    每个值截断到 35 字以内（适合中文输出）。
    """
    slots: dict[int, str] = {}
    if not bright_keys:
        return slots

    # 建查找表
    lookup: dict[int, str] = dict(key_lookup) if key_lookup else {}
    for k in bright_keys:
        if k.id not in lookup:
            lookup[k.id] = k.content

    # 识别根键和子键（复用传导路径信息）
    # 兼容两种传导结果：盆内 ConductionResult（.activation_paths, key=int）
    # 和跨盆 GardenConductionResult（.all_activation_paths, key=(pot_id, dst_id)）
    if hasattr(conduction_result, 'all_activation_paths'):
        raw = conduction_result.all_activation_paths
        paths: dict[int, tuple] = {}
        for (_pot_id, dst_id), info in raw.items():
            if dst_id not in paths:
                paths[dst_id] = info
    else:
        paths = conduction_result.activation_paths
    pushed_ids: set[int] = set(paths.keys())
    source_ids: set[int] = set()
    children_by_parent: dict[int, list[tuple[int, str]]] = {}

    for dst_id, (src_id, src_content, dst_content, _wire, _push) in paths.items():
        source_ids.add(src_id)
        if src_id not in children_by_parent:
            children_by_parent[src_id] = []
        if dst_id not in lookup:
            lookup[dst_id] = dst_content
        if src_id not in lookup:
            lookup[src_id] = src_content
        children_by_parent[src_id].append((dst_id, dst_content))

    # 过滤不可输出的键
    def _ok(cid: int) -> bool:
        content = lookup.get(cid, "")
        return bool(content) and not _is_header_like(content)

    # slot_0: 根键（当过源但没被推的）
    root_ids = source_ids - pushed_ids
    for k in bright_keys:
        if k.id not in pushed_ids and k.id not in source_ids:
            root_ids.add(k.id)

    root_list = [rid for rid in root_ids if _ok(rid)]
    root_list.sort(key=lambda rid: next(
        (k.brightness for k in bright_keys if k.id == rid), 0
    ), reverse=True)

    # 疑问词主导：slot_0 填目标原语的概念键内容（传导共振，非查表选键）。
    # 优先级高于根键——「猫能干什么」要填 CAN 键「猫能捕鼠」，而非最亮的 IS 根键。
    if question_primitive:
        valid_prim = [k for k in bright_keys if _ok(k.id) and k.primitive == question_primitive]
        if valid_prim:
            brightest_prim = max(valid_prim, key=lambda k: k.brightness)
            slots[0] = brightest_prim.content[:35]

    if 0 not in slots:
        if root_list:
            slots[0] = lookup[root_list[0]][:35]
        elif bright_keys:
            # 无明确根键 → 取最亮的可输出键（过滤 "词:" 等头/标记碎片，防泄漏）
            valid = [k for k in bright_keys if _ok(k.id)]
            if valid:
                # 疑问词主导：同原语键优先（「是什么」→ IS 键比 CAN/HAVE 键更该填 slot_0）
                if question_primitive:
                    matched = [k for k in valid if k.primitive == question_primitive]
                    pool = matched or valid
                else:
                    pool = valid
                brightest = max(pool, key=lambda k: k.brightness)
                slots[0] = brightest.content[:35]

    # slot_1+: 从根键的子键中提取
    if root_list:
        root_id = root_list[0]
        children = children_by_parent.get(root_id, [])
        children = [(cid, ct) for cid, ct in children if _ok(cid)]
        for i, (cid, ct) in enumerate(children[:3]):
            slots[i + 1] = ct[:30]

    return slots


def _extract_concept_key_pot_map(
    conduction_result,
    garden: Garden,
) -> dict[int, str]:
    """
    从 GardenConductionResult 中提取非 OUTPUT 原语的亮键及其所在盆 ID。

    用于语法桥接强化：找出哪些概念键参与了当前传导，
    以便加粗它们到 OUTPUT 键的桥接线。
    返回 {key_id: pot_id}。
    """
    if not hasattr(conduction_result, 'round1'):
        return {}

    concept_map: dict[int, str] = {}

    for round_name in ('round1', 'round3'):
        round_data = getattr(conduction_result, round_name, {})
        for pot_id, cr in round_data.items():
            for kid, brightness in cr.final_pattern.items():
                if brightness < 1:
                    continue
                key = garden.pots[pot_id].network.get_key(kid) if pot_id in garden.pots else None
                if key is None:
                    continue
                # 排除 OUTPUT 键自身
                if key.primitive == "OUTPUT":
                    continue
                concept_map[kid] = pot_id

    return concept_map


def _try_grammar_generation(
    garden: Garden,
    conduction_result: ConductionResult,
    bright_keys: list[Key],
    tone: str = "normal",
    question_primitive: str | None = None,
) -> list[str] | None:
    """
    尝试从语法键网络生成句子。

    流程：
    1. 计算激活签名
    2. 匹配语法模式（考虑 tone）
    3. 如果匹配成功 → 槽位填充 → 返回句子列表
    4. 如果无匹配 → 返回 None（调用方降级到 _build_from_paths）

    返回 None 表示"语法键未命中，请使用降级方案"。
    """
    from src.core.grammar import (
        compute_activation_signature,
        conduct_grammar_match,
    )

    # 1. 计算激活签名
    # conduction_result 可能是普通的 ConductionResult 或 GardenConductionResult
    # compute_activation_signature 需要 GardenConductionResult 和 garden
    # 在 generate_from_garden 中调用时，result 就是 GardenConductionResult
    if not hasattr(conduction_result, 'round1'):
        return None  # 不是 GardenConductionResult，无法提取签名

    signature = compute_activation_signature(conduction_result, garden, question_primitive)

    # 2. 传导驱动语法匹配（替代旧版 Jaccard 字典遍历）
    pattern = conduct_grammar_match(garden, signature, tone, question_primitive)
    if pattern is None or not pattern.sentence_structures:
        return None

    # 困惑兜底（空签名种子模式）不覆盖传导路径的真实内容——
    # 有可输出赢家时，语法未命中应降级到 _build_from_paths 直接说概念内容，
    # 困惑留给 generate_from_garden 的赢家门/城市门，不在这里抢输出。
    if not pattern.activation_signature:
        return None

    # 3. 槽位填充（无槽位模板如困惑表达、主动搭话 → 跳过提取）
    if any('{slot_' in t for t in pattern.sentence_structures):
        slot_values = _extract_slot_values_from_keys(
            bright_keys, conduction_result, question_primitive=question_primitive
        )
    else:
        slot_values = {}

    sentences: list[str] = [
        filled for t in pattern.sentence_structures
        if (filled := _fill_grammar_template(t, slot_values))
    ]

    if sentences:
        # 记录成功使用（Hebbian 强化在 main.py 中调用）
        pattern.record_success()
        # KeyNetwork 层面：加粗概念键 → OUTPUT 键的桥接线
        from src.core.grammar import _reinforce_grammar_bridges
        concept_map = _extract_concept_key_pot_map(conduction_result, garden)
        if concept_map:
            _reinforce_grammar_bridges(garden, pattern, concept_map)
        return sentences

    return None


# ═══════════════════════════════════════════════
# 语言生成器（路径驱动版）
# ═══════════════════════════════════════════════

class LanguageModule:
    """
    罗小黑风格的语言模块——传导路径驱动版。

    不再随机选模板。改为读取传导路径（activation_paths），
    理解"哪个键推亮了哪个键"，然后按因果链构建句子。

    用法：
        lang = LanguageModule()
        response = lang.generate(network, knowledge_base, result)
    """


    # ── 路径驱动内容构建 ──────────────────────

    def _build_from_paths(
        self,
        result: ConductionResult,
        bright_keys: list[Key],
        max_sentences: int | None = None,
        garden: Garden | None = None,
        tone: str = "normal",
    ) -> list[str]:
        """
        从传导路径构建句子列表。

        v1.0: 优先尝试语法键匹配，未命中时降级为路径驱动构建。

        步骤（新版）：
        0. 如果 garden 可用 → 尝试语法键匹配 → 命中则直接返回
        1. 建查找表（key_id → content）
        2. 识别根键（推了别人但没被推的 = 种子键）
        3. 按根键分组子键
        4. 每个根键 → 一句（含其子键）
        5. 处理"孙键"链（子键推了孙键 → 因果链）

        max_sentences: 自然上限（None = 不限制）。
        句子按重要性排序（根键 > 孙链 > 残留），截断时保留前 N 句。
        """
        # ── v1.0: 语法键优先 ──
        if garden is not None:
            grammar_sentences = _try_grammar_generation(
                garden, result, bright_keys, tone
            )
            if grammar_sentences:
                if max_sentences is not None and len(grammar_sentences) > max_sentences:
                    grammar_sentences = grammar_sentences[:max_sentences]
                return grammar_sentences

        # ── 降级：路径驱动构建（原有逻辑）──
        paths = result.activation_paths
        sentences: list[str] = []

        # —— 无路径数据 → 降级为平铺 ——
        if not paths:
            return self._fallback_flat(bright_keys)

        # —— 建查找表 ——
        key_lookup: dict[int, str] = {}
        for k in bright_keys:
            key_lookup[k.id] = k.content
        # 补充：路径中提到的源键和目的键可能不在 bright_keys 中（已衰减但推过/被推过）
        for src_id, src_content, dst_content, _, _ in paths.values():
            if src_id not in key_lookup:
                key_lookup[src_id] = src_content
            # dst_content 已经存在路径中，但为了统一查找也加入

        # —— 识别根键（种子键） ——
        pushed_ids = set(paths.keys())           # 被推亮的键
        source_ids = set()                       # 所有当过源的键
        children_by_parent: dict[int, list[tuple[int, str, float]]] = {}  # {src_id: [(dst_id, dst_content, wire)]}

        for dst_id, (src_id, src_content, dst_content, wire, push) in paths.items():
            source_ids.add(src_id)
            if src_id not in children_by_parent:
                children_by_parent[src_id] = []
            # 直接使用路径中自带的 dst_content（不依赖键是否衰减）
            if dst_id not in key_lookup:
                key_lookup[dst_id] = dst_content
            children_by_parent[src_id].append((dst_id, dst_content, wire))

        # 根键 = 当过源但没被推的 + 在 bright_keys 中但不在路径中的亮键
        root_ids = source_ids - pushed_ids
        for k in bright_keys:
            if k.id not in pushed_ids and k.id not in source_ids:
                root_ids.add(k.id)

        # 如果还是没有根键（极少情况），取最亮的两个当根
        if not root_ids:
            root_ids = {k.id for k in bright_keys[:2]}

        # —— 过滤：去掉内部键和标题式内容 ——
        def _ok_for_output(content: str) -> bool:
            if not content:
                return False
            if _is_header_like(content):
                return False
            # 内部键（如体感时间键）不应该出现在输出中
            return True

        # —— 按根键分组构建句子 ——
        used_dst: set[int] = set()  # 已经被某句用过的子键
        # 不再硬截断——句子数由根键数自然决定（通常 2-5 个），
        # 多盆模式下由 generate_from_garden 按参与权重调节

        for root_id in root_ids:

            root_content = key_lookup.get(root_id, "")
            if not _ok_for_output(root_content):
                continue

            children = children_by_parent.get(root_id, [])
            # 过滤掉不适合输出的子键
            children = [(did, ct, w) for did, ct, w in children if _ok_for_output(ct)]
            # 按电线粗细排序（粗的在前）
            children.sort(key=lambda x: x[2], reverse=True)

            if not children:
                # 无子键的根 → 简短陈述
                sentences.append(f"{root_content[:35]}")
            elif len(children) == 1:
                # 单子键 → 自然连接
                dst_id, child_text, wire = children[0]
                used_dst.add(dst_id)
                child_short = child_text[:30]
                if wire >= 0.7:
                    sentences.append(f"{root_content[:25]}就是{child_short}")
                elif wire >= 0.5:
                    sentences.append(f"{root_content[:25]}会{child_short}")
                elif wire >= 0.3:
                    sentences.append(f"{root_content[:25]}，{child_short}")
                else:
                    sentences.append(f"{root_content[:25]}，好像{child_short}")
            else:
                # 多子键 → "X就是Y，还会Z"
                parts = []
                for i, (dst_id, child_text, wire) in enumerate(children[:3]):
                    used_dst.add(dst_id)
                    child_short = child_text[:30]
                    if i == 0:
                        if wire >= 0.7:
                            parts.append(f"就是{child_short}")
                        elif wire >= 0.5:
                            parts.append(f"会{child_short}")
                        elif wire >= 0.3:
                            parts.append(f"，{child_short}")
                        else:
                            parts.append(f"，好像{child_short}")
                    else:
                        if wire >= 0.7:
                            parts.append(f"，还会{child_short}")
                        elif wire >= 0.5:
                            parts.append(f"，也{child_short}")
                        elif wire >= 0.3:
                            parts.append(f"，也{child_short}")
                        else:
                            parts.append(f"，好像也{child_short}")

                sentences.append(f"{root_content[:25]}{''.join(parts)}")

        # —— 处理孙键链（子键又推了孙键） ——
        remaining = [
            (dst_id, src_id, wire)
            for dst_id, (src_id, _, _, wire, _) in paths.items()
            if dst_id not in used_dst and src_id in pushed_ids
        ]
        for dst_id, src_id, wire in remaining:
            dst_content = key_lookup.get(dst_id, "")
            src_content = key_lookup.get(src_id, "")
            if dst_content and src_content:
                adv = _wire_adverb(wire)
                sentences.append(f"{src_content[:20]}{adv}{dst_content[:25]}")
                used_dst.add(dst_id)

        # —— 未被使用的亮键（没有路径关系的） ——
        for k in bright_keys:
            if k.id not in used_dst and k.id not in root_ids:
                if k.id not in pushed_ids:
                    continue  # 根键已经处理过
                if k.source == "internal":
                    continue  # 内部键（时间键等）不输出
                if _is_header_like(k.content):
                    continue
                sentences.append(f"{k.content[:35]}")
                used_dst.add(k.id)

        # 自然截断：句子已按重要性排序（根键 > 孙链 > 残留）
        if max_sentences is not None and len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]

        return sentences

    # ── 降级：无路径时平铺亮键 ────────────────

    def _fallback_flat(self, bright_keys: list[Key]) -> list[str]:
        """无传导路径时的降级方案——平铺展示亮键内容。"""
        parts = []
        for k in bright_keys[:3]:
            if k.source == "internal":
                continue
            if _is_header_like(k.content):
                continue
            text = k.content[:30]
            if len(k.content) > 30:
                text += "…"
            if text not in parts:
                parts.append(text)
        if parts:
            return ["，".join(parts)]
        return ["…"]

    # ── 时间感知 ────────────────────────────

    def _maybe_add_time_context(
        self,
        content: str,
        result: ConductionResult,
    ) -> str:
        """
        如果传导结果中有非今天的记忆，偶尔提一下时间。
        40% 概率触发，选一个"时间最有意思"的键。
        """
        if random.random() > 0.4:
            return content

        # 找时间最值得说的键（非今天、非刚刚）
        candidates = []
        for k in result.top_keys[:5]:
            if k.created_at == 0 or k.source == "internal":
                continue
            ago = _time_ago(k.created_at)
            if ago in ("昨天", "前天", "前几天", "前一阵", "很久以前"):
                candidates.append((k, ago))

        if not candidates:
            return content

        k, ago = random.choice(candidates)
        phrase = _time_phrase(k.created_at)
        if not phrase:
            return content

        # 追加时间短句
        content = content.rstrip("。") + f"。{phrase}。"
        return content

    # ── 花园模式生成（v0.3 多盆参与版）───────────

    def generate_from_garden(
        self,
        garden,
        knowledge_base,
        result,
        seed_counts: dict | None = None,
        output_regulation=None,  # Phase 6: OutputRegulation | None
        turn_seed_ids: set[int] | None = None,  # M5: 本轮种子键（赢家抽取时排除）
        orphan_keys: list | None = None,         # M5: 本轮孤种键（沉默时说"还不懂"）
        question_primitive: str | None = None,   # 战场二：疑问词目标原语（传导式路由）
        question_relations: tuple[str, ...] = (),  # 战场二：疑问词关系词（同原语内精细路由）
    ) -> str:
        """
        从花园跨盆传导结果生成回复——多盆加权参与版。

        概念（师父的话）：
        - "主要城市"：被输入匹配到种子键的盆，按种子密度加权参与
        - "小村庄"：通过桥接线被触发的盆，贡献一两句关联
        - 不单选主导盆——多个城市并存，各按其份

        说话习惯：
        - 小黑自然说短句（2-4 句），话题丰富时自然长（5-6 句）
        - 不去硬截断——句子数由参与盆数和种子密度自然决定
        - Phase 6: output_regulation 调控句子数和语气
        """
        seed_counts = seed_counts or {}

        # ── M5: 赢家抽取门 ──
        # 传导没带出任何「非种子、可输出」的键 = 没有结构可传导。
        # 此时直接说"我还不太懂"（孤种键）或困惑，绝不落入语法盆乱选。
        winners = self._extract_winners(result, turn_seed_ids, garden, question_primitive)
        if not winners:
            return self._add_personality(self._speak_silence(orphan_keys))

        # ── 犹豫检测（任一盆犹豫 → 全局犹豫）──
        for cr in result.round1.values():
            if cr.is_hesitating:
                detail = cr.hesitation_detail or "让我再想想……"
                return random.choice(TEMPLATE_HESITATE).format(detail=detail)

        # ── 战场二：疑问词主导的传导式问答 ──
        # 疑问键亮了 → 直接引起目标原语概念键亮（环环相扣），答案 = 那些键的内容。
        # 这是传导式路由的落点：不做查表，只是把传导结果读出来。
        if question_primitive is not None:
            answer = self._answer_question(result, garden, question_primitive, question_relations)
            if answer:
                return self._add_personality(answer, tone="normal")

        # ── 多盆参与计划 ──
        # 城市：有种子键的盆（输入直接相关的）
        # 村庄：只有桥接触发、没有种子的盆（关联联想的）
        cities: list[tuple] = []     # [(pot, cr, seed_count)]
        villages: list[tuple] = []   # [(pot, cr)]

        for pot_id, cr in result.round1.items():
            pot = garden.pots.get(pot_id)
            if pot is None:
                continue
            seeds = seed_counts.get(pot_id, 0)
            if seeds > 0:
                cities.append((pot, cr, seeds))
            elif cr.top_keys and any(k.brightness >= 1 for k in cr.top_keys):
                # 没有种子但亮了键——微弱参与
                cities.append((pot, cr, 0))

        # 村庄：只被桥接触发的盆（round3）
        for pot_id, cr in result.round3.items():
            if pot_id in result.round1:
                continue  # 已经是城市了
            pot = garden.pots.get(pot_id)
            if pot is None:
                continue
            bright = [k for k in cr.top_keys if k.brightness >= 1]
            if bright:
                villages.append((pot, cr))

        if not cities and not villages:
            return random.choice(TEMPLATE_CONFUSED)

        # ── Phase 6: 确定语气 ──
        tone = "normal"
        if output_regulation is not None:
            tone = getattr(output_regulation, 'tone', 'normal')

        # ── v1.0: 语法键网络优先 ──
        # 收集所有亮键（跨所有参与盆）用于语法匹配和槽位填充
        all_bright_keys: list = []
        for pot, cr, seeds in cities:
            all_bright_keys.extend(k for k in cr.top_keys[:8] if k.brightness >= 1)
        for pot, cr in villages[:2]:
            all_bright_keys.extend(k for k in cr.top_keys[:4] if k.brightness >= 1)

        grammar_sentences = _try_grammar_generation(
            garden, result, all_bright_keys, tone, question_primitive
        )
        if grammar_sentences:
            # ── 语法键命中 → 直接使用（跳过逐盆构建）──
            all_sentences: list[str] = list(grammar_sentences)
        else:
            # ── 降级：按参与权重逐盆生成句子 ──
            all_sentences = []
            # 城市：按种子数排序，按比例分配句子配额
            cities.sort(key=lambda x: x[2], reverse=True)
            total_seeds = sum(s for _, _, s in cities) or 1

            for pot, cr, seeds in cities:
                if pot.source_type == "internal":
                    continue  # 机制键盆（时间键/SELF/QUEST）不产出内容句子
                share = seeds / total_seeds
                # 配额：种子占比 × 5 → 1-3 句，种子为 0 的给 1 句
                if seeds == 0:
                    quota = 1
                else:
                    quota = max(1, min(3, round(share * 5)))

                bright_keys = [k for k in cr.top_keys[:8] if k.brightness >= 1]
                all_bright_keys.extend(bright_keys)

                sentences = self._build_from_paths(
                    cr, bright_keys, max_sentences=quota
                )
                all_sentences.extend(sentences)

            # 村庄：每个 0-1 句
            for pot, cr in villages[:2]:
                if pot.source_type == "internal":
                    continue  # 机制键盆不产出内容句子
                bright_keys = [k for k in cr.top_keys[:4] if k.brightness >= 1]
                all_bright_keys.extend(bright_keys)
                sentences = self._build_from_paths(
                    cr, bright_keys, max_sentences=1
                )
                all_sentences.extend(sentences)

        # ── 知识库补充（查所有参与盆的亮键）──
        if knowledge_base is not None and all_bright_keys:
            kb_hits = knowledge_base.query(all_bright_keys, max_results=1)
            if kb_hits:
                kb_text = kb_hits[0].content
                full_text = "".join(all_sentences)
                if kb_text not in full_text:
                    all_sentences.append(f"哦对了师父，{kb_text}")

        if not all_sentences:
            return random.choice(TEMPLATE_CONFUSED)

        # ── Phase 6: 应用输出调控 ──
        if output_regulation is not None:
            # 句子数上限裁剪
            if hasattr(output_regulation, 'apply_cap'):
                all_sentences = output_regulation.apply_cap(all_sentences)
            # tone 已在前面从 output_regulation 读取

        # ── 组装 + 时间感知 + 性格 ──
        content = "。".join(s for s in all_sentences if s) + "。"
        # 用种子最多的城市盆的结果做时间感知
        best_cr = cities[0][1] if cities else (
            villages[0][1] if villages else None
        )
        if best_cr is not None:
            content = self._maybe_add_time_context(content, best_cr)
        return self._add_personality(content, tone=tone)

    # ── M5: 赢家抽取 + 沉默 ──────────────────────────

    def _extract_winners(self, result, turn_seed_ids, garden, question_primitive=None) -> list:
        """
        赢家 = 传导结束后仍亮着、且不是本轮种子的可输出键。

        排除：本轮种子（turn_seed_ids）、internal 源、header 式内容、休眠键。
        遍历 round1 + round3 的 final_pattern（brightness >= 1）。
        排序：疑问词目标原语优先 → 亮度 → 离内容种子的路径跳数 → NARS 期望值。
        """
        turn_seed_ids = turn_seed_ids or set()
        candidates: list[tuple] = []  # [(key, pot_id, cr, brightness)]
        seen: set[int] = set()

        for round_name in ('round1', 'round3'):
            round_data = getattr(result, round_name, {})
            for pot_id, cr in round_data.items():
                pot = garden.pots.get(pot_id)
                if pot is None:
                    continue
                for kid, brightness in cr.final_pattern.items():
                    if brightness < 1 or kid in turn_seed_ids or kid in seen:
                        continue
                    key = pot.network.get_key(kid)
                    if key is None or key.is_dormant:
                        continue
                    if key.source == "internal":
                        continue
                    if _is_header_like(key.content):
                        continue
                    seen.add(kid)
                    candidates.append((key, pot_id, cr, brightness))

        if not candidates:
            return []

        from src.core.evidence import truth_expectation

        def _depth(cr, kid: int) -> int:
            # 沿 activation_paths 反向走回种子，数跳数；走不到种子给一个大值。
            paths = cr.activation_paths
            depth = 0
            cur = kid
            guard = 0
            while cur in paths and guard < 32:
                src_id = paths[cur][0]
                if src_id in turn_seed_ids:
                    return depth + 1
                cur = src_id
                depth += 1
                guard += 1
            return 999

        def _sort_key(item) -> tuple:
            key, _pot_id, cr, brightness = item
            exp = truth_expectation(key.evidence_freq, key.evidence_conf)
            # 疑问词主导：与目标原语同 primitive 的键排最前（传导式共振，非查表选键）
            prim_match = 0 if (question_primitive and key.primitive == question_primitive) else 1
            return (prim_match, -brightness, _depth(cr, key.id), -exp)

        candidates.sort(key=_sort_key)
        return [k for k, _p, _c, _b in candidates]

    def _answer_question(self, result, garden, question_primitive, question_relations=()) -> str | None:
        """
        战场二：疑问词主导的传导式问答。

        疑问键（持久种子）亮了，直接引起目标原语概念键亮（环环相扣）。
        答案 = 那些被点亮的目标原语概念键的内容，按亮度排序取最亮的几个自然成句。

        不做查表——只是把传导结果「读」出来；路由已在传导阶段完成。
        关系词（question_relations）区分同一原语内的精细路由（「属于什么」vs「是什么」）。
        返回 None 表示没有可答的目标原语键（降级到常规生成）。
        """
        collected: list[tuple] = []  # (key, brightness)
        seen: set[int] = set()
        for round_name in ('round1', 'round3'):
            for pot_id, cr in getattr(result, round_name, {}).items():
                pot = garden.pots.get(pot_id)
                if pot is None:
                    continue
                for kid, b in cr.final_pattern.items():
                    if b < 1 or kid in seen:
                        continue
                    key = pot.network.get_key(kid)
                    if key is None or key.is_dormant:
                        continue
                    if key.source == "internal":
                        continue  # 疑问键/机制键不是答案内容
                    if key.primitive != question_primitive:
                        continue
                    if _is_header_like(key.content):
                        continue
                    seen.add(kid)
                    collected.append((key, b))

        if not collected:
            return None

        # 关系词精细路由：优先取内容含关系词的概念键（如「属于」），
        # 空则回退到全部同原语键（宽容：不因关系词缺配而答不出）。
        if question_relations:
            rel_hits = [
                (k, b) for k, b in collected
                if any(r in k.content for r in question_relations)
            ]
            if rel_hits:
                collected = rel_hits

        # 最亮的在前，同亮度按 id 稳定排序
        collected.sort(key=lambda item: (-item[1], item[0].id))
        return "。".join(k.content[:35] for k, _ in collected[:3]) + "。"

    def _speak_silence(self, orphan_keys) -> str:
        """
        赢家为空 = 传导没带来新东西 = 没有结构可传导。
        取最亮的孤种键名字：topic = content.removeprefix("词:")。
        有 topic → "我还不太懂{topic}。"；无 → 困惑模板。
        """
        if not orphan_keys:
            return random.choice(TEMPLATE_CONFUSED)
        best = None
        for k in orphan_keys:
            if best is None or k.brightness > best.brightness:
                best = k
        if best is None:
            return random.choice(TEMPLATE_CONFUSED)
        topic = (best.content or "").removeprefix("词:").removeprefix("词：")
        if not topic:
            return random.choice(TEMPLATE_CONFUSED)
        return f"我还不太懂{topic}。"

    # ── 罗小黑性格修饰 ────────────────────────

    def _add_personality(self, text: str, tone: str = "normal") -> str:
        """添加罗小黑的语气特点。tone 参数影响风格。"""

        # 根据 tone 调整语气风格
        if tone == "brief":
            # 简洁模式：不加"师父"前缀和语气词
            return text
        if tone == "gentle":
            # 温和模式：更高概率加"师父"
            if random.random() < 0.5 and not text.startswith(CALL_MASTER):
                text = f"{CALL_MASTER}，{text}"
            return text
        if tone == "serious":
            # 认真模式：不加语气词，不加感叹
            return text

        # ── normal / excited（原有逻辑）──

        # 偶尔在前面叫"师父"
        if random.random() < 0.35 and not text.startswith(CALL_MASTER):
            text = f"{CALL_MASTER}，{text}"

        # 偶尔在前面加语气词（如果不以师父开头）
        if random.random() < 0.25 and not text.startswith(CALL_MASTER) and not text.startswith("喵"):
            text = _pick_tone() + "，" + text

        # 如果内容里有兴奋词，偶尔在末尾加感叹
        excitement_words = ["厉害", "好", "太", "真", "喜欢", "开心", "棒", "强"]
        excitement_chance = 0.6 if tone == "excited" else 0.4
        if any(w in text for w in excitement_words):
            if random.random() < excitement_chance and not text.endswith("！"):
                text = text.rstrip("。") + "！"

        return text


# ═══════════════════════════════════════════════
# 快捷函数
# ═══════════════════════════════════════════════

# 全局单例
_lang_module: LanguageModule | None = None


def get_language_module() -> LanguageModule:
    """获取语言模块单例。"""
    global _lang_module
    if _lang_module is None:
        _lang_module = LanguageModule()
    return _lang_module



def startup_message(
    sleep_hours: float = 0.0,
    fatigue_before: float = 0.0,
    fatigue_after: float = 0.0,
) -> str:
    """
    启动时的问候。根据时间和睡眠情况生成自然的表达。

    不是"假装刚醒"——小黑真的感受到了状态变化：
    睡了 N 小时 → 脑子清爽了 → 疲劳恢复了。
    如果没睡（刚关了一会儿），就不说"醒了"，说"回来了"。
    """
    now = datetime.now()
    hour = now.hour

    # 确定时间段
    if 5 <= hour < 10:
        period = "morning"
    elif 10 <= hour < 14:
        period = "noon"
    elif 14 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 21:
        period = "evening"
    else:
        period = "night"

    # ── 真的睡了一觉（sleep_hours > 1 小时） ──
    if sleep_hours > 1.0:
        fatigue_recovered = fatigue_before - fatigue_after

        morning_lines = [
            "师父早。小黑醒了。",
            "喵……早上好师父。",
            f"睡了一觉，脑子清楚多了。师父早。",
            "嗯……天亮了啊。师父早。",
        ]
        noon_lines = [
            "师父中午好。小黑醒了。",
            "喵，师父。睡饱了。",
            f"睡了{sleep_hours:.0f}个小时，现在精神很好。师父好。",
        ]
        afternoon_lines = [
            "师父下午好。",
            f"喵……睡了{sleep_hours:.0f}个小时。师父下午好。",
            "醒了。师父！",
        ]
        evening_lines = [
            "师父……天都黑了啊。",
            f"喵。这一觉睡得真好。师父晚上好。",
            "师父！我刚醒。晚上好。",
        ]
        night_lines = [
            "师父……这么晚了还没睡啊。",
            "喵。师父晚上好。",
            f"师父还不睡吗？小黑刚醒。",
        ]

        lines_by_period = {
            "morning": morning_lines,
            "noon": noon_lines,
            "afternoon": afternoon_lines,
            "evening": evening_lines,
            "night": night_lines,
        }

        # 疲劳大幅恢复 → 可能提一句
        if fatigue_recovered > 0.3 and random.random() < 0.4:
            lines = lines_by_period.get(period, morning_lines)
            return random.choice(lines)
        else:
            lines = [l for l in lines_by_period.get(period, morning_lines)
                     if "睡" not in l and "精神" not in l]
            return random.choice(lines) if lines else random.choice(morning_lines)

    # ── 没怎么睡（sleep_hours <= 1）——刚关了一会儿，不是真的睡 ──
    else:
        returning_lines = [
            "师父！小黑来了~",
            "喵，师父。",
            "师父好。",
        ]
        return random.choice(returning_lines)


def shutdown_message() -> str:
    """
    关闭时的告别。根据时间自然表达。
    """
    now = datetime.now()
    hour = now.hour

    if 21 <= hour or hour < 5:
        # 深夜 → 该睡觉了
        night_lines = [
            "小黑睡了。师父晚安。",
            "喵……困了。师父也早点睡。",
            "师父晚安。小黑去床上了。",
            "睡了……师父拜拜。",
        ]
        return random.choice(night_lines)
    elif 17 <= hour < 21:
        # 傍晚 → 可能吃东西或休息
        evening_lines = [
            "小黑去休息一下。师父拜拜。",
            "师父拜拜~",
            "喵，小黑去躺一会儿。",
        ]
        return random.choice(evening_lines)
    else:
        # 白天 → 不一定是睡觉
        day_lines = [
            "师父拜拜~",
            "喵。师父再见。",
            "小黑在房间里。师父拜拜。",
        ]
        return random.choice(day_lines)


def tired_expression(level: str = "tired") -> str:
    """
    小黑感到疲劳时的自然表达。
    level: "tired" | "very_tired"
    不是"假装累了"——疲劳值真的高了才会触发。
    """
    if level == "very_tired":
        return random.choice([
            "师父……小黑有点累了，脑子转不动了……",
            "喵……今天想了好多东西，小黑想去床上躺一会儿。",
            "师父，小黑困了……",
        ])
    else:
        return random.choice([
            "师父，小黑觉得有点累……",
            "喵……脑子有点不太清楚了。",
        ])


def ingest_done_message(count: int) -> str:
    """投喂完成后的确认。"""
    return random.choice(TEMPLATE_INGEST_DONE).format(count=count)
