# 外包任务 · Claude 网页版 —— 修复"已读乱回"的 4 个根因

## 项目背景与铁律
luoxiaohei 是一个**纯符号**认知 AI（"键 + 电线 + 网络 + 传导"），代码是 Python、中文注释。

**铁律（绝对禁止，最高优先级）**：禁止一切神经网络相关技术——LLM、Transformer、神经网络、Embedding/向量嵌入、余弦相似度、Softmax、反向传播/梯度下降、矩阵乘法、GPU、PyTorch/TensorFlow、tokenization、概率采样，任何形式任何规模都不行。只允许符号化/图论/逻辑运算（传导、Hebbian、电线衰减、NARS 证据量、HTM、CBR、语法键网络、产生式系统）。**你写的代码里绝不能出现这些被禁的词/概念/模式。**（完整宪法清单见 `src/core/__init__.py` 顶部。）

---

## 症状（已复现，非推测）

输入不同的话题，小黑要么回"嗯，你好。"，要么回一句跟问题毫不相干的日常事实，偶尔还漏出"词:"碎片。真实例子：

| 输入 | 回复（错） |
|---|---|
| 你认识无限吗？ | 嗯，师父说夏天穿短袖就够了…（毫不相干） |
| 你爱吃什么？ | 嗯，天黑了屋子里什么都看不见…（毫不相干） |
| 你知道手机吗？ | 嗯，你好。 |
| 你是谁 | 嗯，你好。 |
| 我不好 | 师父，嗯，词:小鱼干是我最喜欢。（漏出"词:"碎片） |

已用诊断脚本 `reproduce_bug.py`（在项目根目录，可直接跑 `python reproduce_bug.py`）逐句复现，确认是下面 4 个根因叠加，**不是随机、不是理解错**。

---

## 四个根因 + 修复

### 根因 1（P0）：语法盆的 OUTPUT 键被当成概念种子匹配，污染传导与签名

**证据**：复现时 `[pot-14]` 里出现了 `师父！{slot_0}！`、`我觉得是{slot_0}。`、`我记得——{slot_0}。` 这类**带 `{slot_N}` 占位符的 OUTPUT 模板键**被当成种子点亮；激活签名里也出现了 `<Primitive.OUTPUT: 'OUTPUT'>`。

**位置**：
- `src/core/grammar.py:293` — `init_grammar_pot` 用 `is_internal=False` 创建语法盆，导致它被 `get_active_pots()` 当作普通概念盆返回。
- `src/runtime/main.py:479` — `for pot in self.garden.get_active_pots():` 对每个盆（含语法盆）跑 `parse_user_input`。

**修法**：在 main.py 的种子匹配循环里跳过语法盆：

```python
for pot in self.garden.get_active_pots():
    if pot.id == self.garden.grammar_pot_id:
        continue  # 语法盆是内部 OUTPUT 结构，不参与概念种子匹配
    seeds = parse_user_input(pot.network, text, is_master=is_master,
                             global_total=global_total, global_matching=global_matching)
    ...
```

### 根因 2（P0）：激活签名把内部原语（OUTPUT/WORD/SITUATION/RULE/PATTERN/CASE）也算了进去

**证据**：几乎每句输入的签名都是同一个 30+ 元素的巨型元组，含 `WORD`、`SITUATION`、`RULE`、`<Primitive.OUTPUT>`、`rel:part-of`、`rel:similar-to` 等内部标记。

**位置**：`src/core/grammar.py:221-222` 和 `242-243`：

```python
if key.primitive:
    primitives_seen.add(key.primitive)
```

**修法**：定义内部原语集合并跳过（两处都要改）。注意 `Primitive` 是 `class Primitive(str, Enum)`，`key.primitive` 可能是 str 也可能是枚举，两者 `==` 互通；稳妥起见用 `str(...)` 归一化再判：

```python
from .primitives import Primitive

_INTERNAL_PRIMITIVES = {
    Primitive.OUTPUT, Primitive.WORD, Primitive.SITUATION,
    Primitive.RULE, Primitive.PATTERN, Primitive.CASE,
}

# 在收集处改成：
if key.primitive and str(key.primitive) not in _INTERNAL_PRIMITIVES:
    primitives_seen.add(key.primitive)
```

### 根因 3（P1）：种子爆炸——每盆独立返回 ≤8 个，14 盆合计 50~102 个，全亮度 2，无主次

**证据**：复现时 `你认识无限吗？` 匹配到 102 个种子、`你会英语吗` 85 个、`我不好` 95 个，全部亮度 2。原因是 `parse_user_input` 里 `MAX_SEEDS=8` 是**每盆**上限（`src/inout/input.py:150-161`），而 main.py 对 14 个盆各自独立调用、累加。同义词表（`无限→师父`、`爱→喜欢`）放大了匹配面。

**位置**：`src/inout/input.py:150-161`（每盆 cap）+ `src/runtime/main.py:479-485`（每盆累加）。

**修法**（给两个方案，任选其一，推荐 A 起步）：

- **方案 A（最小改动）**：main.py 收集完所有盆的种子后，做一次**全局 top-N 截断**——按亮度降序、再按 `activation_count` 降序，全局只保留 **12~16 个**种子再进传导。
- **方案 B（更根本）**：把种子匹配从"每盆独立"改为"全局统计"——跨盆统一算特异度、统一 top-N，而不是每盆各算各的。

**关键验收点**：让输入真正相关的键（如"手机""无限""吃"）排到最前，而不是 100 个"师父""你"糊满。修完跑 `reproduce_bug.py`，每个输入的种子数应降到 20 个以内，且前几名是话题相关的键。

### 根因 4（P1）：slot_0 兜底分支漏过 `_is_header_like`，泄漏"词:"碎片

**证据**：`我不好` → slot_0 填成了 `词:小鱼干是我最喜欢`。`_is_header_like`（`src/inout/language.py:156`）本来会过滤 `词:` 前缀，但这条兜底分支没走它。

**位置**：`src/inout/language.py:263-266`：

```python
elif bright_keys:
    brightest = max(bright_keys, key=lambda k: k.brightness)
    slots[0] = brightest.content[:35]   # ← 没过滤！
```

**修法**：先筛掉不可输出的键，再取最亮；若全被筛掉，slot_0 留空（交给困惑型降级）：

```python
elif bright_keys:
    ok = [k for k in bright_keys if not _is_header_like(k.content)]
    if ok:
        brightest = max(ok, key=lambda k: k.brightness)
        slots[0] = brightest.content[:35]
```

---

## 约束（重要，别越界）
- **最小改动**：只动上面 4 处相关的行，别重构、别顺手改别的模块。
- **别改传导/语法匹配的算法语义**：`conduct()`、`conduct_grammar_match()`、`compute_activation_signature()` 的**匹配逻辑**不动，只做"排除内部原语"和"排除语法盆"这两个过滤。
- **别动 `_seed_initial_patterns()` 的种子签名**（陈述型/回答型/困惑型等模板是设计好的，别改）。
- **别动 `MAX_SEEDS=8` 这个常量本身**（除非你选方案 B 全局化，否则每盆逻辑不动）。
- 铁律全程生效。

## 你要交付
1. 改好的 `src/runtime/main.py`、`src/core/grammar.py`、`src/inout/input.py`、`src/inout/language.py`（以及你方案 B 涉及的其它文件）。
2. 一段「改了哪几处、每处为什么」的说明（每个根因 1-2 句）。
3. 修复后的复现结果：`python reproduce_bug.py` 的输出，展示 8 句输入的种子数都降下来、回复不再"嗯，你好。"卡死、不再漏"词:"碎片。

## 运行方式
在项目根目录（`D:\树洞\luoxiaohei`）运行：
```
python reproduce_bug.py
```
脚本会自动新建一个全新临时实例（不碰真实存档，跑完自删），喂入 8 句问题输入，打印每句的「种子 → 激活签名 → 胜出模式 → 槽位 → 回复」。修复前先跑一遍看现状，改完再跑一遍对照。

也可以直接跑真实交互验证：
```
python -m src.runtime.main --new
```
然后依次输入：你认识无限吗 / 你爱吃什么 / 你知道手机吗 / 你是谁 / 我不好，观察回复是否还乱回。

## 预期结果（修好后）
- 每句种子数从 50~102 降到 ≤20，且排前面的是话题相关键。
- "你认识无限吗" 之类不再回"夏天穿短袖""天黑了灯亮"这种无关事实。
- 不再出现"嗯，你好。"反复卡死。
- 不再漏"词:"碎片。
