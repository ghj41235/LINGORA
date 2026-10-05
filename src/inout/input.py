"""
键盘模型 AI — 输入解析模块

把用户的自然语言输入转化为种子键列表。
流程：分词 → 分层匹配 → 种子键 + 初始亮度。

师父的输入享有 +1 亮度特权（罗小黑信任师父）。
匹配不到时创建短期记忆键（7 天内再触发才升级）。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jieba

if TYPE_CHECKING:
    from src.core.network import KeyNetwork
    from src.core.key import Key


# ── 同义词表 ──────────────────────────────────

SYNONYM_MAP: dict[str, list[str]] = {
    "开心":   ["高兴", "快乐", "兴奋", "嗨", "欢喜", "愉悦"],
    "难过":   ["伤心", "难受", "不开心", "低落", "郁闷", "悲伤"],
    "师父":   ["主人", "老师", "无限", "师傅"],
    "法术":   ["技能", "能力", "魔法", "招数", "本领", "妖术"],
    "害怕":   ["担心", "恐惧", "怕", "畏惧", "慌张"],
    "厉害":   ["强大", "强", "了不起", "牛逼", "凶猛"],
    "喜欢":   ["喜爱", "中意", "爱好", "偏爱", "爱"],
    "朋友":   ["伙伴", "同伴", "好友", "哥们"],
    "家":     ["住处", "屋子", "房子", "窝"],
    "吃":     ["进食", "用餐", "吃饭", "食用"],
    "睡":     ["睡觉", "入睡", "休息", "眠"],
}

# 反向索引：同义词 → 标准词
_synonym_to_canonical: dict[str, str] = {}
for _canon, _syns in SYNONYM_MAP.items():
    for _s in _syns:
        _synonym_to_canonical[_s] = _canon


# ── 功能词 / 应答词（M1：概念提及不做这些词的子串匹配）──

_FUNCTION_WORDS = {
    "你", "我", "他", "她", "它", "这", "那", "的", "了", "吗", "呢", "吧",
    "是", "有", "在", "个", "们", "和", "跟", "就", "都", "也", "很", "还",
    "什么", "啥", "谁", "为什么", "怎么", "为啥", "请问", "啊", "呀", "哦", "嗯",
    "一个", "一种", "东西", "时候", "这样", "那样",
}
_ACK_WORDS = {"好", "行", "对", "可以", "好的", "没事", "知道了", "辛苦", "收到"}

# ── .frag 关系词（修正 C：关系词是三元组的连接器，不是独立概念）──
# 单字关系词（能/会/像/导致…）在概念提及层会 substring 误命中原语键，
# 把「手机能打电话」这类 CAN 键抢成种子（turn_seed_ids），挤掉真正的赢家。
# 情感关系词（喜欢/爱/讨厌/怕）保留为实义——它们是可独立提问的焦点。
_RELATION_WORDS = {
    "能", "会", "就是", "叫做", "是一种", "属于", "不是",
    "拥有", "具有", "没有", "缺少",
    "连接", "连着", "相连", "像", "类似",
    "导致", "引起",
}
_STOPWORDS = _FUNCTION_WORDS | _ACK_WORDS | _RELATION_WORDS


def is_content_token(token: str) -> bool:
    """token 是否实义词（非功能词 / 应答词）。"""
    return token not in _STOPWORDS


def _synonym_words(token: str) -> set[str]:
    """token 及其同义词的所有词形（含标准词）。"""
    words = {token}
    if token in SYNONYM_MAP:
        words.update(SYNONYM_MAP[token])
    canon = _synonym_to_canonical.get(token)
    if canon:
        words.add(canon)
        words.update(SYNONYM_MAP.get(canon, []))
    return words


def _token_mentions(token: str, content: str) -> bool:
    """token 是否「提及」概念键内容。功能词 → False（修正 B：按功能词不按长度）。"""
    if not is_content_token(token):
        return False
    return token in content


def materialize_orphan(network: KeyNetwork, token: str) -> Key | None:
    """
    把没见过（零证据）的词落成孤种键（M2）。
    精确匹配，不用 find_similar_key（避免 jieba 切「词:英语」的坑）。
    """
    for key in network.all_keys():
        if not key.is_dormant and key.content == f"词:{token}":
            key.is_orphan = True
            key.evidence_conf = 0.0
            return key
    k = network.create_key(content=f"词:{token}", source="master")
    k.primitive = "WORD"
    k.evidence_conf = 0.0
    k.is_orphan = True
    return k


# ── 短期记忆 ──────────────────────────────────

# 短期记忆池：{content_hash: (fragment_text, created_at)}
# 7 天内再次触发 → 升级为正式键
_short_term_pool: dict[int, tuple[str, float]] = {}
SHORT_TERM_TTL = 7 * 86400  # 7 天


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

def parse_user_input(
    network: KeyNetwork,
    text: str,
    is_master: bool = False,
    global_total: int | None = None,
    global_matching: dict[str, int] | None = None,
) -> list[tuple[int, int]]:
    """
    解析用户输入，返回种子键列表 [(key_id, brightness), ...]。

    - 分词
    - 计算每个 token 的特异度（IDF 类权重：稀有词权重高，高频词权重低）
    - 分层匹配（精确 → 同义 → 关联 → 短期记忆）
    - 师父输入所有匹配到的键亮度 +1

    global_total: 全花园的键总数。
    global_matching: 每个 token 在全花园中命中的键数（跨所有盆）。
                    不传则退化为盆内统计（小盆会高估常见词的权重）。
    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    # ── 预计算 token 特异度 ──
    # 关键：分子（总键数）和分母（匹配键数）都必须用全花园的全局统计。
    # 否则"是"在食物盆（15/30 匹配 → 看起来不常见 → 高权重）和概念盆
    #（200/259 匹配 → 看起来泛滥 → 低权重）权重不一致，小盆永远赢。
    all_keys = [k for k in network.all_keys() if not k.is_dormant]
    local_total = len(all_keys)
    effective_total = global_total if global_total and global_total > local_total else local_total

    token_specificity: dict[str, float] = {}
    for token in tokens:
        if effective_total == 0:
            token_specificity[token] = 1.0
            continue
        # 优先使用全局匹配计数（跨所有盆的命中数）
        if global_matching and token in global_matching:
            matching = global_matching[token]
        else:
            matching = sum(1 for k in all_keys if token in k.content)
        # specificity = total / (total + matching * 9)
        # 匹配 0 键 → 1.0，匹配 total/10 键 → 0.53，匹配 total/2 键 → 0.11
        token_specificity[token] = effective_total / (effective_total + matching * 9)

    # ── 分层匹配（M1：词键优先，命中词键则跳过概念提及）──
    seeds: dict[int, int] = {}  # key_id → max brightness

    for token in tokens:
        specificity = token_specificity.get(token, 0.5)
        # base: 0（到处都有）→ 2（独一无二）
        base_brightness = round(2 * specificity)
        # 师父信任：词键亮度 +1（上限 2）。提及种子不享受此特权。
        word_brightness = min(base_brightness + (1 if is_master else 0), 2)
        word_forms = {f"词:{w}" for w in _synonym_words(token)}

        has_word_key = False
        for key in all_keys:
            if key.primitive != "WORD":
                continue
            if key.content == f"词:{token}":
                # 第一层：词键精确命中。孤种键亮度恒 1（弱键）。
                b = 1 if key.is_orphan else word_brightness
                seeds[key.id] = max(seeds.get(key.id, 0), b)
                has_word_key = True
            elif key.content in word_forms:
                # 第二层：词键同义词命中（低 1 级，至少 1）。
                b = 1 if key.is_orphan else max(1, word_brightness - 1)
                seeds[key.id] = max(seeds.get(key.id, 0), b)
                has_word_key = True

        matched_any = has_word_key

        # 第三层：概念提及（修正 A：有词键就整层跳过；修正 B：按功能词不按长度）。
        # 提及种子亮度恒 1，永不升级——只能靠传导被进一步推亮。
        if not has_word_key:
            for key in all_keys:
                if key.primitive in ("WORD", ""):
                    continue
                if _token_mentions(token, key.content):
                    if key.id not in seeds:
                        seeds[key.id] = 1
                    matched_any = True

        # 第四层：无任何命中 → 短期记忆（实义词才记）
        if not matched_any and is_content_token(token):
            _add_short_term(token)

    # ── 种子数控制：太多 → 取最相关的 ──
    MAX_SEEDS = 8
    if len(seeds) > MAX_SEEDS:
        scored = []
        for kid, brightness in seeds.items():
            key = network.get_key(kid)
            if key is None:
                continue
            score = (brightness * 100) + min(key.activation_count, 50)
            scored.append((kid, brightness, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        seeds = {kid: b for kid, b, _ in scored[:MAX_SEEDS]}

    # ── 兜底：零种子时用字符 bigram 模糊匹配 ──
    # 只对含实义词的输入兜底；纯功能词/指代/疑问（"你是谁"这类）交给 M3 机制键，
    # 否则"是/你"等共享字会把"手机是…"这类定义键误拉进种子（污染指代消解）。
    if not seeds and len(text) >= 2 and any(is_content_token(t) for t in tokens):
        best_matches = []
        for key in all_keys:
            sim = _text_overlap_score(text, key.content)
            if sim > 0.15:
                best_matches.append((key.id, sim))
        best_matches.sort(key=lambda x: x[1], reverse=True)
        for kid, sim in best_matches[:3]:
            seeds[kid] = 1

    # 检查短期记忆升级
    _upgrade_short_term(network, text, seeds)

    return [(kid, b) for kid, b in seeds.items()]


# ═══════════════════════════════════════════════
# 分词
# ═══════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """
    分层分词：中文走 jieba，符号/数字/英文直接保留。

    处理顺序：
    1. 正则切分文本为「中文段」和「符号段」
    2. 中文段 → jieba 分词
    3. 符号段 → 按空白和运算符边界切分，逐段保留
    4. 合并，去空，返回
    """
    import re

    tokens: list[str] = []

    # 将文本按「中文」和「非中文」交替切分
    # 一-鿿 = CJK 统一汉字区
    segments = re.split(r'([一-鿿　-〿＀-￯]+)', text)

    for seg in segments:
        if not seg:
            continue

        # 中文段 → jieba
        if re.search(r'[一-鿿]', seg):
            words = jieba.lcut(seg)
            for w in words:
                w = w.strip()
                if w and len(w) >= 1:
                    tokens.append(w)

        # 非中文段 → 符号/数字/英文
        else:
            # 按空白和常见运算符边界切分
            symbol_tokens = re.findall(
                r'[A-Za-zΑ-ωА-я]+'   # 英文 / 希腊 / 俄文
                r'|\d+(?:\.\d+)?'                          # 数字（含小数）
                r'|[+\-*/×÷=<>()\[\]{}^%&|!~@#$\\:;,.?]+' # 运算符 / 标点
                r'|[^\s]',                                  # 其他非空白单字符
                seg
            )
            for st in symbol_tokens:
                st = st.strip()
                # 过滤纯空白和纯标点噪音（但保留数学运算符）
                if not st:
                    continue
                if st in (' ', '\t', '\n', '\r'):
                    continue
                # 单个逗号句号等中文已处理过的 → 跳过
                if st in ('，', '。', '！', '？', '、', '：', '；', '"', '"', ''', '''):
                    continue
                tokens.append(st)

    # 去重但保持顺序
    seen = set()
    result = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ═══════════════════════════════════════════════
# 短期记忆
# ═══════════════════════════════════════════════

def _add_short_term(token: str):
    """将匹配不到的词加入短期记忆池。"""
    h = hash(token)
    if h not in _short_term_pool:
        _short_term_pool[h] = (token, time.time())


def _upgrade_short_term(
    network: KeyNetwork,
    text: str,
    seeds: dict[int, int],
):
    """
    检查输入文本是否与短期记忆中的内容相似。
    相似的 → 升级为正式键 + 弱连线到当前种子。
    """
    now = time.time()
    expired = []

    for h, (fragment, created) in _short_term_pool.items():
        # 过期检查
        if now - created > SHORT_TERM_TTL:
            expired.append(h)
            continue

        # 相似度检查：当前输入文本与短期记忆片段
        if _jaccard_similarity(text, fragment) > 0.6:
            # 升级为正式键
            new_key = network.create_key(
                content=fragment,
                source="external",
                brightness=1,
            )
            # 弱连线到当前种子
            for kid in seeds:
                network.set_wire(new_key.id, kid, thickness=0.05)
            expired.append(h)

    for h in expired:
        _short_term_pool.pop(h, None)


# ── 短期记忆清理（后台调用） ──────────────────

def cleanup_short_term():
    """清理过期的短期记忆。"""
    now = time.time()
    expired = [
        h for h, (_, created) in _short_term_pool.items()
        if now - created > SHORT_TERM_TTL
    ]
    for h in expired:
        _short_term_pool.pop(h, None)


# ═══════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════

def _jaccard_similarity(a: str, b: str) -> float:
    """两段文本的 Jaccard 相似度（基于 jieba 分词）。"""
    set_a = set(jieba.lcut(a))
    set_b = set(jieba.lcut(b))
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _text_overlap_score(a: str, b: str) -> float:
    """
    max(Jaccard, containment) — 升级版相似度。
    containment 能捕获前缀/包含关系，避免长文本被漏检。
    """
    set_a = set(jieba.lcut(a))
    set_b = set(jieba.lcut(b))
    if not set_a or not set_b:
        return 0.0
    jaccard = len(set_a & set_b) / len(set_a | set_b)
    containment = len(set_a & set_b) / min(len(set_a), len(set_b))
    return max(jaccard, containment)


def find_similar_key(
    network: KeyNetwork,
    text: str,
    threshold: float = 0.8,
) -> Key | None:
    """
    在现有键中找一个与给定文本最相似的键。
    用于文档投喂去重。使用升级版相似度（Jaccard + containment）。
    """
    best_key = None
    best_score = 0.0

    for key in network.all_keys():
        if key.is_dormant:
            continue
        score = _text_overlap_score(text, key.content)
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= threshold and best_key is not None:
        return best_key
    return None


def get_short_term_count() -> int:
    """短期记忆池中当前条目数。"""
    return len(_short_term_pool)
