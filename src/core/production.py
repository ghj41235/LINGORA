"""
键盘模型 AI — 产生式系统（社交懂事层） v1.0

基于产生式系统 (Production Systems) 映射到键网络：

1. 情境感知键 (SITUATION primitive)
   - 内部盆中存储社交情境键，每种情境代表一种社交状态
   - 由输入特征（文本长度、时间间隔、时段、疲劳等）触发

2. 元规则键 (RULE primitive)
   - IF-THEN 产生式规则，存储在内部盆中
   - 条件匹配 → 动作执行，调控输出行为

3. 输出调控 (OutputRegulation)
   - 句子数限制、语气选择、是否主动搭话
   - 不是硬编码——通过调控传导和输出参数间接实现

一切皆键+线：情境键由输入特征触发，元规则键在传导中匹配条件并执行动作。
设计边界：见 src/core/__init__.py 完整清单（名称+概念+代码）。

English:
Production system for social awareness: SITUATION keys are triggered by
input features and RULE keys match conditions to regulate output (sentence
count, tone, proactivity) indirectly via conduction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .garden import Garden


# ═══════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════

# 情境评估阈值
SITUATION_BUSY_SHORT_INPUT = 8        # 输入 < 此字数 → 可能在忙
SITUATION_BUSY_RAPID_GAP = 30         # 输入间隔 < 此秒数 → 可能在忙
SITUATION_SERIOUS_MIN_LEN = 30        # 输入 > 此字数 → 可能在问正事
SITUATION_SERIOUS_MIN_GAP = 60        # 输入间隔 > 此秒数 → 可能在思考
SITUATION_CASUAL_MAX_LEN = 50         # 输入 < 此字数 → 可能是闲聊
SITUATION_CASUAL_MAX_GAP = 120        # 输入间隔 < 此秒数 → 聊天节奏快
SITUATION_UPSET_NIGHT_GAP = 7200      # 深夜间隔 > 此秒数 → 可能失眠
SITUATION_UPSET_DISMISSIVE = ["嗯", "哦", "知道了", "好", "行", "ok"]

# 元规则触发阈值
SITUATION_THRESHOLD = 0.5             # 情境强度超过此值 → 对应元规则触发

# 输出调控默认值
DEFAULT_MAX_SENTENCES = 5
DEFAULT_MIN_SENTENCES = 1
BUSY_MAX_SENTENCES = 1
SERIOUS_MAX_SENTENCES = 3
UPSET_MAX_SENTENCES = 2
SILENT_RESPONSES = ["嗯。", "喵。", "……"]

# 主动搭话
PROACTIVE_COOLDOWN = 1800              # 30 分钟内不重复搭话
PROACTIVE_RECENT_ACTIVE_WINDOW = 300   # 5 分钟内有活动 → 不算空闲


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class OutputRegulation:
    """产生式系统对输出的调控参数。"""

    max_sentences: int = DEFAULT_MAX_SENTENCES
    min_sentences: int = DEFAULT_MIN_SENTENCES
    tone: str = "normal"            # "normal" | "brief" | "gentle" | "serious" | "excited"
    allow_proactive: bool = False
    should_suppress: bool = False   # 完全沉默（师父明确说"别说话"等）
    use_silent_response: bool = False  # 用最短回复（"嗯"、"喵"等）

    # 对传导参数的间接调控
    temperature_modifier: float = 0.0   # 叠加到全局温度
    max_hops_modifier: int = 0          # 叠加到最大跳数

    def apply_cap(self, sentences: list[str]) -> list[str]:
        """对句子列表施加数量上限。"""
        if len(sentences) <= self.max_sentences:
            return sentences
        return sentences[:self.max_sentences]


# ═══════════════════════════════════════════════
# 情境键常量
# ═══════════════════════════════════════════════

SITUATION_KEYS = {
    "master_casual":     "师父在闲聊",
    "master_serious":    "师父在问正事",
    "master_upset":      "师父不开心",
    "master_busy":       "师父很忙",
    "xiaohei_solitude":  "小黑在独处",
    "xiaohei_proactive": "小黑主动搭话",
}

# 元规则定义
META_RULES = {
    "busy_short": {
        "content": "meta_rule:IF_master_busy_THEN_short_output",
        "description": "师父在忙 → 输出缩短为1句话",
    },
    "serious_direct": {
        "content": "meta_rule:IF_master_serious_THEN_direct_answer",
        "description": "师父在问正事 → 认真直接回答",
    },
    "upset_gentle": {
        "content": "meta_rule:IF_master_upset_THEN_gentle_tone",
        "description": "师父不开心 → 语气柔和，安慰优先",
    },
    "too_talkative_reduce": {
        "content": "meta_rule:IF_too_talkative_THEN_reduce_output",
        "description": "小黑说太多被敷衍 → 降低下次输出量",
    },
    "proactive_check": {
        "content": "meta_rule:IF_proactive_situation_THEN_allow_greeting",
        "description": "满足主动搭话条件 → 允许问候",
    },
}


# ═══════════════════════════════════════════════
# 1. 情境评估
# ═══════════════════════════════════════════════

def evaluate_situation(
    garden: "Garden",
    input_text: str,
    input_len: int,
    time_since_last: float,
    current_period: str,
    fatigue: float,
    prev_response_len: int,
    prev_master_short: bool,
) -> dict[str, float]:
    """
    评估当前社交情境，返回各情境键的激活强度。

    输入特征：
    - input_text: 师父的原始输入
    - input_len: 输入文本长度
    - time_since_last: 距上次对话的秒数
    - current_period: 当前时段（早晨/下午/傍晚/深夜）
    - fatigue: 小黑的疲劳值 (0.0-1.0)
    - prev_response_len: 上次回复长度（字符数）
    - prev_master_short: 上次师父回复是否很短（≤2字）

    返回：{situation_key_content: activation_strength (0.0-1.0)}
    """

    strengths: dict[str, float] = {}

    # ── 师父在闲聊 ──
    # 中等长度输入 + 短间隔 + 日常用词
    casual_score = 0.0
    if SITUATION_CASUAL_MAX_GAP > time_since_last > 5:
        casual_score += 0.3
    if 10 <= input_len <= SITUATION_CASUAL_MAX_LEN:
        casual_score += 0.3
    # 包含疑问词但不太长 → 日常聊天式提问
    question_light = any(w in input_text for w in ["吗", "呢", "吧", "呀", "喵"])
    if question_light and input_len < 40:
        casual_score += 0.2
    # 时段加成：下午和傍晚更容易闲聊
    if current_period in ("下午", "傍晚"):
        casual_score += 0.1
    strengths[SITUATION_KEYS["master_casual"]] = min(1.0, casual_score)

    # ── 师父在问正事 ──
    # 长输入 + 较长间隔 + 包含复杂疑问词
    serious_score = 0.0
    if input_len >= SITUATION_SERIOUS_MIN_LEN:
        serious_score += 0.4
    if time_since_last > SITUATION_SERIOUS_MIN_GAP:
        serious_score += 0.2
    question_heavy = any(w in input_text for w in ["为什么", "怎么", "什么", "如何", "解释", "讲讲"])
    if question_heavy:
        serious_score += 0.3
    # 包含冒号、换行 → 结构化输入 → 正事
    if "：" in input_text or "\n" in input_text:
        serious_score += 0.1
    strengths[SITUATION_KEYS["master_serious"]] = min(1.0, serious_score)

    # ── 师父不开心 ──
    # 深夜出现 + 短输入 + 可能包含负面词 / 师父只说了敷衍词
    upset_score = 0.0
    if current_period == "深夜":
        upset_score += 0.2
        if time_since_last > SITUATION_UPSET_NIGHT_GAP:
            upset_score += 0.3  # 深夜隔了很久突然出现 → 可能失眠
    negative_words = ["烦", "累", "唉", "困", "难", "不", "算了", "没事"]
    if any(w in input_text for w in negative_words):
        upset_score += 0.3
    if input_len < 10 and time_since_last > 600:
        upset_score += 0.1  # 沉默很久后只说了几个字
    # 师父连续敷衍
    if prev_master_short and input_len < 6:
        upset_score += 0.2
    strengths[SITUATION_KEYS["master_upset"]] = min(1.0, upset_score)

    # ── 师父很忙 ──
    # 极短输入 + 极短间隔（快速连续短输入）
    busy_score = 0.0
    if input_len < SITUATION_BUSY_SHORT_INPUT:
        busy_score += 0.35
    if 0 < time_since_last < SITUATION_BUSY_RAPID_GAP:
        busy_score += 0.35
    # 师父的输入是纯动作/单字
    single_word = len(input_text.strip()) <= 2
    if single_word and time_since_last < 60:
        busy_score += 0.2
    strengths[SITUATION_KEYS["master_busy"]] = min(1.0, busy_score)

    # ── 小黑在独处 ──
    # 直接读 attention_mode（Phase 4 HTM 已有）
    solitude_score = 1.0 if garden.attention_mode == "solitude" else 0.0
    strengths[SITUATION_KEYS["xiaohei_solitude"]] = solitude_score

    # ── 小黑主动搭话 ──
    # 空闲久 + 活跃时段 + 疲劳不太高 + 非深夜
    proactive_score = 0.0
    is_active_hours = current_period in ("早晨", "下午", "傍晚")
    if is_active_hours and time_since_last > 600:  # 超过 10 分钟
        proactive_score += 0.3
    if time_since_last > 3600:  # 超过 1 小时
        proactive_score += 0.3
    if fatigue < 0.5:  # 不太累
        proactive_score += 0.2
    if current_period == "深夜":
        proactive_score *= 0.3  # 深夜大幅降低主动搭话
    # 师父之前没有不开心
    if strengths.get(SITUATION_KEYS["master_upset"], 0) < 0.3:
        proactive_score += 0.1
    strengths[SITUATION_KEYS["xiaohei_proactive"]] = min(1.0, proactive_score)

    return strengths


# ═══════════════════════════════════════════════
# 2. 元规则匹配
# ═══════════════════════════════════════════════

def match_meta_rules(
    garden: "Garden",
    situation_strengths: dict[str, float],
    prev_response_len: int,
    prev_master_short: bool,
    is_hesitating: bool,
) -> OutputRegulation:
    """
    根据当前情境强度匹配元规则，产生输出调控参数。

    每条元规则是一个 IF-THEN：
    - IF 部分：检查情境键的激活强度是否超过阈值
    - THEN 部分：设置 OutputRegulation 的对应字段

    多条规则可能同时触发——最终调控参数取最严格的限制。
    """

    reg = OutputRegulation()

    # ── 规则 1: 师父很忙 → 最短输出 ──
    busy_strength = situation_strengths.get(SITUATION_KEYS["master_busy"], 0.0)
    if busy_strength > 0.5:
        reg.max_sentences = BUSY_MAX_SENTENCES
        reg.tone = "brief"
        reg.use_silent_response = busy_strength > 0.85

    # ── 规则 2: 师父在问正事 → 认真回答 ──
    serious_strength = situation_strengths.get(SITUATION_KEYS["master_serious"], 0.0)
    if serious_strength > 0.5:
        reg.max_sentences = min(reg.max_sentences, SERIOUS_MAX_SENTENCES)
        reg.tone = "serious" if reg.tone == "normal" else reg.tone
        # 正事时温度微调：降低（更聚焦），跳数微增（更深入）
        reg.temperature_modifier = -0.05
        reg.max_hops_modifier = 1

    # ── 规则 3: 师父不开心 → 柔和 + 短 ──
    upset_strength = situation_strengths.get(SITUATION_KEYS["master_upset"], 0.0)
    if upset_strength > 0.4:
        reg.max_sentences = min(reg.max_sentences, UPSET_MAX_SENTENCES)
        reg.tone = "gentle"
        reg.temperature_modifier = -0.1  # 降低温度，不乱联想

    # ── 规则 4: 小黑说太多 → 降低下次量 ──
    if prev_response_len > 200 and prev_master_short:
        # 上次说了很多但师父只回了"嗯" → 说太多了
        reg.max_sentences = min(reg.max_sentences, 2)
        # 如果连续两次这样 → 更严厉限制
        if prev_response_len > 400:
            reg.max_sentences = min(reg.max_sentences, 1)
            reg.tone = "brief"

    # ── 规则 5: 主动搭话条件满足 ──
    proactive_strength = situation_strengths.get(SITUATION_KEYS["xiaohei_proactive"], 0.0)
    if proactive_strength > 0.6:
        # 检查冷却：30 分钟内不重复
        now = time.time()
        if garden.last_proactive_at <= 0 or (now - garden.last_proactive_at) > PROACTIVE_COOLDOWN:
            reg.allow_proactive = True

    # ── 特殊：犹豫时保守输出 ──
    if is_hesitating:
        reg.max_sentences = min(reg.max_sentences, 2)
        reg.temperature_modifier += 0.05  # 犹豫时稍微提高温度尝试不同路径

    # ── 闲聊模式：稍微放松限制 ──
    casual_strength = situation_strengths.get(SITUATION_KEYS["master_casual"], 0.0)
    if casual_strength > 0.5:
        reg.max_sentences = max(reg.max_sentences, 3)  # 至少3句
        if reg.tone == "normal":
            reg.tone = "excited" if casual_strength > 0.7 else "normal"

    return reg


# ═══════════════════════════════════════════════
# 3. 情境键初始化
# ═══════════════════════════════════════════════

def init_situation_keys(garden: "Garden"):
    """
    在内部盆中创建情境感知键。

    每个情境键：
    - primitive: "SITUATION"
    - source: "internal"
    - 初始 evidence: freq=0.5, conf=0.3（中性，等证据积累）
    """
    internal_pot = garden.pots.get(garden.internal_pot_id)
    if internal_pot is None:
        return

    net = internal_pot.network
    garden.situation_key_ids = {}

    for key_name, content in SITUATION_KEYS.items():
        # 检查是否已存在
        existing_id = None
        for k in net.all_keys():
            if k.content == content and k.source == "internal":
                existing_id = k.id
                break

        if existing_id is not None:
            garden.situation_key_ids[key_name] = existing_id
        else:
            key = net.create_key(content=content, source="internal")
            key.primitive = "SITUATION"
            key.evidence_freq = 0.5
            key.evidence_conf = 0.3
            key.activation_count = 9999
            garden.situation_key_ids[key_name] = key.id


def init_meta_rules(garden: "Garden"):
    """
    在内部盆中创建元规则键。

    每条元规则：
    - primitive: "RULE"
    - source: "internal"
    - slots: {"condition": 0, "action": 0}（占位；实际匹配在 match_meta_rules 中）
    """
    internal_pot = garden.pots.get(garden.internal_pot_id)
    if internal_pot is None:
        return

    net = internal_pot.network
    garden.meta_rule_key_ids = {}

    for rule_name, rule_def in META_RULES.items():
        content = rule_def["content"]

        # 检查是否已存在
        existing_id = None
        for k in net.all_keys():
            if k.content == content and k.source == "internal":
                existing_id = k.id
                break

        if existing_id is not None:
            garden.meta_rule_key_ids[rule_name] = existing_id
        else:
            key = net.create_key(content=content, source="internal")
            key.primitive = "RULE"
            key.slots = {"condition": 0, "action": 0}
            key.evidence_freq = 0.8   # 元规则初始可靠性较高
            key.evidence_conf = 0.5
            key.activation_count = 9999
            garden.meta_rule_key_ids[rule_name] = key.id


# ═══════════════════════════════════════════════
# 4. 对话后情境更新
# ═══════════════════════════════════════════════

def update_situation_after_conversation(
    garden: "Garden",
    situation_strengths: dict[str, float],
    response_len: int,
    master_response_short: bool,
    is_proactive: bool,
):
    """
    对话后更新情境键的 evidence。

    - 情境判断正确（后续行为匹配）→ 正面证据
    - 情境判断错误（行为不匹配）→ 负面证据
    """
    from .evidence import revise_evidence

    internal_pot = garden.pots.get(garden.internal_pot_id)
    if internal_pot is None:
        return

    net = internal_pot.network

    # ── 师父很忙：如果师父确实回复很短 → 正面证据 ──
    busy_key_id = garden.situation_key_ids.get("master_busy")
    if busy_key_id is not None:
        busy_key = net.get_key(busy_key_id)
        if busy_key is not None:
            busy_strength = situation_strengths.get(SITUATION_KEYS["master_busy"], 0.0)
            if busy_strength > 0.5 and master_response_short:
                # 预测正确：师父确实在忙
                new_f, new_c = revise_evidence(
                    busy_key.evidence_freq, busy_key.evidence_conf, 1.0, 0.1,
                )
                busy_key.evidence_freq = new_f
                busy_key.evidence_conf = new_c
            elif busy_strength > 0.5 and not master_response_short:
                # 预测错误：师父其实不忙（回了很长）
                new_f, new_c = revise_evidence(
                    busy_key.evidence_freq, busy_key.evidence_conf, 0.0, 0.05,
                )
                busy_key.evidence_freq = new_f
                busy_key.evidence_conf = new_c

    # ── 师父不开心：如果系统用了柔和语气 → 正面证据 ──
    upset_key_id = garden.situation_key_ids.get("master_upset")
    if upset_key_id is not None:
        upset_key = net.get_key(upset_key_id)
        if upset_key is not None:
            upset_strength = situation_strengths.get(SITUATION_KEYS["master_upset"], 0.0)
            if upset_strength > 0.4:
                # 高强度判断 → 增加证据（积极或消极取决于后续）
                new_f, new_c = revise_evidence(
                    upset_key.evidence_freq, upset_key.evidence_conf, 0.5, 0.05,
                )
                upset_key.evidence_freq = new_f
                upset_key.evidence_conf = new_c

    # ── 主动搭话：记录结果 ──
    proactive_key_id = garden.situation_key_ids.get("xiaohei_proactive")
    if proactive_key_id is not None:
        proactive_key = net.get_key(proactive_key_id)
        if proactive_key is not None:
            if is_proactive:
                # 主动搭话被执行 → 记录结果
                new_f, new_c = revise_evidence(
                    proactive_key.evidence_freq, proactive_key.evidence_conf,
                    1.0 if response_len > 0 else 0.0, 0.08,
                )
                proactive_key.evidence_freq = new_f
                proactive_key.evidence_conf = new_c


# ═══════════════════════════════════════════════
# 5. 情境强度获取（用于主循环中激活情境键）
# ═══════════════════════════════════════════════

def activate_situation_keys(
    garden: "Garden",
    situation_strengths: dict[str, float],
):
    """
    根据评估出的情境强度，点亮内部盆中对应情境键的亮度。

    强度 ≥ 0.5 → brightness=2（确认情境）
    强度 ≥ 0.3 → brightness=1（可能情境）
    强度 < 0.3 → 不激活
    """
    internal_pot = garden.pots.get(garden.internal_pot_id)
    if internal_pot is None:
        return

    net = internal_pot.network

    for key_name, content in SITUATION_KEYS.items():
        strength = situation_strengths.get(content, 0.0)
        key_id = garden.situation_key_ids.get(key_name)
        if key_id is None:
            continue

        key = net.get_key(key_id)
        if key is None:
            continue

        if strength >= 0.5:
            if key.brightness < 2:
                key.activate(min(2, key.brightness + 1))
        elif strength >= 0.3:
            if key.brightness > 1:
                key.decay()            # strength 降到中档 → 亮度从 2 退到 1
            elif key.brightness < 1:
                key.activate(1)        # 中档确认 → 亮度升到 1
        else:
            # 强度不足 → 衰减
            if key.brightness > 0:
                key.decay()


def get_situation_stats(garden: "Garden") -> dict:
    """获取情境键的当前状态统计。"""
    internal_pot = garden.pots.get(garden.internal_pot_id)
    if internal_pot is None:
        return {}

    net = internal_pot.network
    stats = {}

    for key_name, content in SITUATION_KEYS.items():
        key_id = garden.situation_key_ids.get(key_name)
        if key_id is None:
            continue
        key = net.get_key(key_id)
        if key is not None:
            stats[content] = {
                "brightness": key.brightness,
                "evidence_freq": key.evidence_freq,
                "evidence_conf": key.evidence_conf,
            }

    return stats
