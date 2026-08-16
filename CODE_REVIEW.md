# 罗小黑 代码全面审查报告

**审查日期**：2026-08-03  
**审查范围**：`src/` 下全部 30 个 Python 文件  
**审查方法**：代码阅读 + 实际运行验证 + 属性检查 + 逻辑推演  
**状态**：✅ 5 个 Bug 已修复 / ✅ 12 项死代码已清理 / ✅ 概念偏差 #2 已修复 / ⏳ 概念偏差 #3 待 B2

---

## 零、审查总览

| 类别 | 数量 | 状态 |
|------|------|------|
| 🔴 崩溃级 Bug | 4 | ✅ 已修复 |
| 🟡 逻辑死代码 Bug | 1 | ✅ 已修复 |
| 🟡 兼容性问题 | 3 | ✅ 已核查，无需修改 |
| 🟢 冗余/死代码 | 9 项 | ✅ 已删除 |
| 🟢 二次扫描死代码 | 3 项 | ✅ 已删除 |
| 🟡 概念偏差 | 3 项 | ✅ #2 已修复 / #3 待 B2 / #1 已消除 |
| 禁区违规 | **0** | ✅ 完全合规 |

### 核心判断（更新）

> 语法键网络的代码框架完整。5 个 Bug 修复后，语法键优先路径已可正常工作。概念偏差 #2 已修复：语法盆中现在有 27 个种子 OUTPUT 键，运行时 store_output_pattern 会创建新 OUTPUT 键和概念键↔OUTPUT 键的桥接线。语法匹配仍是 Jaccard dict 查表（概念偏差 #3），计划在 B2 阶段切换为传导驱动。

---

## 一、禁区合规审查 ✅

对全部 30 个文件进行了三层面扫描（名称、概念、代码模式），结果：

| 检查项 | 结果 |
|--------|------|
| LLM / GPT / Claude / Transformer / BERT 等名称 | 仅在 `__init__.py` 禁区清单中出现（文档性质） |
| Neural Network / CNN / RNN / LSTM / GRU | 零命中 |
| Embedding / word2vec / GloVe / 稠密向量 | 零命中 |
| Softmax / Sigmoid / ReLU / 激活函数 | 零命中 |
| Backpropagation / Gradient Descent / Adam / SGD | 零命中 |
| Matrix Multiplication 作为核心运算 | 零命中 |
| GPU / CUDA / 并行矩阵 | 零命中 |
| PyTorch / TensorFlow / Keras / JAX / HuggingFace / ONNX | 零命中（零 import） |
| Diffusion / GAN / Reinforcement Learning | 零命中 |
| Fine-tuning / Pre-training / Tokenization (BPE) | 零命中 |
| .pt / .bin / .safetensors 模型权重格式 | 零命中 |

**判定为合法的边界项：**

| 术语 | 位置 | 合法理由 |
|------|------|----------|
| `cosine_similarity` | `pot.py:418`, `garden.py:468` | bigram 浅层向量文档路由，非 embedding 比较 |
| `attention_mode` | `temporal.py`, `garden.py` | HTM 认知注意力聚焦 (dialogue/solitude)，非 transformer self-attention |
| `token` | `input.py`, `cd_engine.py` | jieba 中文分词结果，非 BPE/WordPiece tokenizer |
| `activation` | `case.py`, 多处 | 键亮度激活（symbolic activation），非神经网络激活函数 |
| `reinforce` | `grammar.py:637` | 符号化 Hebbian 强化计数器，非 RL policy gradient |
| `checkpoint` | `format.py` | JSON 应用状态持久化，非模型权重保存 |

**结论：零违规。禁区清单在所有三层面（名称/概念/代码）均得到严格遵守。**

---

## 二、逻辑 Bug 🐛

### 🔴 Bug #1（崩溃级）：`grammar.py:695` — 双重 `garden.garden` 拼写错误

**文件**：`src/core/grammar.py`  
**行号**：695  
**代码**：
```python
# 当前（错误）：
for gid in to_delete:
    del garden.garden.grammar_library[gid]

# 应为：
for gid in to_delete:
    del garden.grammar_library[gid]
```

**触发条件**：语法模式满足以下任一删除条件时触发：
- 非种子模式休眠超过 60 天
- 非种子模式 reliability < 0.1 且 frequency ≥ 10

**实际运行验证**：
```
AttributeError: 'Garden' object has no attribute 'garden'
```

**影响范围**：`decay_grammar_patterns()` → 被 `garden.maintenance()` 调用 → 崩溃后阻断所有后续维护（盆拆分/合并/休眠检查 + 桥接衰减 + 证据衰减 + CBR 维护等）

**严重程度**：🔴 崩溃级。不触发删除时静默（种子模式有保护不会触发删除），一旦有非种子模式满足删除条件即崩溃。

---

### 🔴 Bug #2（崩溃级）：`storage/format.py:92, 128` — 导入路径缺少 `src.` 前缀

**文件**：`src/storage/format.py`  
**行号**：92, 128  
**代码**：
```python
# 当前（错误）：
from knowledge.desk import Desk

# 应为：
from src.knowledge.desk import Desk
```

**触发条件**：系统启动 → `CheckpointManager.load()` → `load_checkpoint()`  
**影响范围**：系统完全无法启动（包括全新创建和从检查点加载）

**实际运行验证**：
```
ModuleNotFoundError: No module named 'knowledge'
```

**严重程度**：🔴 崩溃级。100% 触发，系统无法启动。

---

### 🔴 Bug #3（崩溃级）：`grammar.py:220, 240` — 传导路径迭代解包类型错误

**文件**：`src/core/grammar.py`  
**行号**：220, 240  
**代码**：
```python
# 当前（错误）：
for (path_pot_id, dst_id), (src_id, src_c, dst_c, wire_thickness, push) in cr.activation_paths.items():
```

**根因分析**：

`ConductionResult.activation_paths` 的实际结构是：
```python
dict[int, tuple[int, str, str, float, float]]
# key = dst_id (int)
# value = (src_id, src_content, dst_content, wire_thickness, push)
```

但 for 循环的 key 解包写成了 `(path_pot_id, dst_id)` —— 这是 `GardenConductionResult.all_activation_paths` 的结构（key 是 `(pot_id, dst_id)` 元组）。

`cr` 来自 `conduction_result.round1.items()`，其中每个 `cr` 是 `ConductionResult`（盆级），不是 `GardenConductionResult`。

**触发条件**：传导产生了实际的激活路径（`activation_paths` 非空 dict）  
**影响范围**：`compute_activation_signature()` 崩溃 → 语法键匹配失败 → 降级到路径驱动

**实际运行验证**：
```
TypeError: cannot unpack non-iterable int object
```

**严重程度**：🔴 崩溃级。但仅在 activation_paths 非空时触发——空路径静默不触发，掩盖了 bug。

---

### 🔴 Bug #4（崩溃级）：`language.py:270` — `GardenConductionResult` 属性不匹配

**文件**：`src/inout/language.py`  
**行号**：270（在 `_extract_slot_values_from_keys` 函数内）  
**调用链**：`generate_from_garden():742` → `_try_grammar_generation():358` → `_extract_slot_values_from_keys():270`

**代码**：
```python
# _extract_slot_values_from_keys 第 270 行：
paths = conduction_result.activation_paths  # ❌ GardenConductionResult 无此属性
```

**根因分析**：

| 类 | 路径属性名 | Key 类型 |
|----|-----------|----------|
| `ConductionResult` | `.activation_paths` | `int` (dst_id) |
| `GardenConductionResult` | `.all_activation_paths` | `(pot_id, dst_id)` 元组 |

`_try_grammar_generation` 接收的是 `GardenConductionResult`（从 `generate_from_garden` 传入），但传给 `_extract_slot_values_from_keys` 时后者期望的是 `ConductionResult`（有 `.activation_paths`）。

**触发条件**：激活签名长度 ≥ 2（触发槽位填充）  
**影响范围**：`AttributeError` → 有意义的签名无法完成槽位填充

**实际运行验证**：
```
AttributeError: 'GardenConductionResult' object has no attribute 'activation_paths'
```

**严重程度**：🔴 崩溃级。但仅在被 Bug #3 没拦住（空路径时 Bug #3 不触发）且签名 ≥ 2 时暴露。Bug #3 和 Bug #4 形成了"互相掩盖"的格局。

---

### 🟡 Bug #5（逻辑死代码）：空签名"困惑型"种子模式永远不可达

**文件**：`src/inout/language.py` + `src/core/grammar.py`  
**涉及行号**：`language.py:349`, `grammar.py:491`

**代码流程**：
```python
# _try_grammar_generation() 第 349 行：
if not signature or len(signature) < 2:
    return None  # ← 在此提前返回，永远不会到达 match_output_pattern

# match_output_pattern() 第 491 行（永远不会被调用）：
if fallback_empty is not None and len(activation_signature) < GRAMMAR_MIN_SIGNATURE_SIZE:
    return fallback_empty  # ← 困惑型种子模式 (activation_signature=())
```

**影响**：种子模式 #3（困惑型，空签名，5 个模板如"师父，这个我不太懂……"）永远无法通过语法键匹配路径被使用。困惑表达仍可通过旧硬编码 `TEMPLATE_CONFUSED` 降级路径输出，但不是由语法键网络驱动的。

**严重程度**：🟡 功能静默失效。不影响用户可见行为（有降级路径），但语法键网络的一个设计意图（"困惑格局→困惑句式"的传导驱动）未实现。

---

## 三、兼容性问题 🔗

### 兼容问题 #1：`set_wire()` 不接受 `relation_type` 参数

**文件**：`src/core/network.py:90-98`  
**问题**：`KeyNetwork.set_wire(a, b, thickness)` 在创建新 Wire 时不传递 `relation_type`：
```python
# network.py:98
self.wires[pair] = Wire(key_from=pair[0], key_to=pair[1], thickness=thickness)
#                                                      ↑ 没有 relation_type=xxx
```

而 `Wire.__init__` 接受 `relation_type` 参数。当前所有代码通过"先 `set_wire` 再 `w.relation_type = "..."`"模式工作，所以不会崩溃，但：
- 这是一种 API 不一致：创建时不能设关系类型，必须分两步
- `set_wire` 在更新已有 wire 时只升级 thickness，不更新 relation_type

**严重程度**：🟡 设计不一致。不影响运行但增加了出错可能。

---

### 兼容问题 #2：`GardenConductionResult` vs `ConductionResult` 属性名不一致

两个类的路径存储使用了不同的属性名和 key 结构：

| | `ConductionResult` | `GardenConductionResult` |
|----|----|----|
| 路径属性 | `.activation_paths` | `.all_activation_paths` |
| Key 类型 | `int` (dst_id) | `tuple[str, int]` (pot_id, dst_id) |
| Value 类型 | `tuple[int,str,str,float,float]` | `tuple[int,str,str,float,float]` |

`grammar.py` 中的 `compute_activation_signature` 和 `language.py` 中的 `_extract_slot_values_from_keys` 在不同层级间传递数据时混用了这些结构。当前通过 `hasattr` 检查分流，但属性名不匹配导致 Bug #3 和 #4。

**严重程度**：🟡 导致 Bug #3 和 #4 的根因。

---

### 兼容问题 #3：`_try_grammar_generation` 被两条路径以不同类型调用

| 调用位置 | 传入类型 | 结果 |
|----------|----------|------|
| `generate_from_garden():742` | `GardenConductionResult` | 主路径，触发 Bug #3+#4 |
| `_build_from_paths():462` | `ConductionResult` | 总是返回 None（`hasattr(conduction_result, 'round1')` 为 False） |

第二条调用路径（盆级）中语法键匹配被静默跳过。可能是设计意图（盆内单独传导不走语法键），但无注释说明。

**严重程度**：🟡 需要明确设计意图。

---

## 四、冗余/死代码 🗑️

### 完全未使用的函数

| 死代码 | 文件:行号 | 原因 |
|--------|-----------|------|
| `_connection_word()` | `language.py:186` | 定义后从未被调用（引用计数 = 1，仅定义） |
| `save_checkpoint()` | `storage/format.py:35` | 旧 v0.2 格式保存函数，接收 `KeyNetwork`，Garden 模式下已弃用 |
| `generate_response()` | `language.py:863` | 快捷函数，从未被 import 或调用 |
| `LanguageModule.generate()` | `language.py:389` | 仅被废弃的 `generate_response()` 调用，实际运行走 `generate_from_garden()` |

### 仅定义未使用的旧模板列表

这些模板已被语法种子模式替代，但旧列表仍保留在代码中：

| 模板变量 | 文件:行号 | 模板数 |
|----------|-----------|--------|
| `TEMPLATE_STATEMENT` | `language.py:91` | 5 个模板 |
| `TEMPLATE_ANSWER` | `language.py:101` | 4 个模板 |
| `TEMPLATE_EXCITED` | `language.py:120` | 4 个模板 |
| `TEMPLATE_STARTUP` | `language.py:145` | 3 个模板 |
| `TEMPLATE_SHUTDOWN` | `language.py:151` | 3 个模板 |

> `TEMPLATE_STARTUP` 和 `TEMPLATE_SHUTDOWN` 在 `main.py` 中引用数为 **0**——启动/关闭消息完全走 `startup_message()` 和 `shutdown_message()` 函数的内联模板。

### 仍在使用的模板（保留正确）

| 模板变量 | 原因 |
|----------|------|
| `TEMPLATE_CONFUSED` | 5 处引用（`generate()`, `generate_from_garden()` 的降级路径） |
| `TEMPLATE_HESITATE` | 3 处引用（`generate()`, `generate_from_garden()` 的犹豫表达） |
| `TEMPLATE_INGEST_DONE` | 2 处引用（`ingest_done_message()`） |
| `TEMPLATE_PROACTIVE` | 2 处引用（`main.py` 主动搭话） |
| `_wire_adverb()` | 1 处引用（`_build_from_paths()` 孙键链） |

---

## 五、概念对齐审查 📐

### ✅ 正确对齐的设计要素

| 计划要求 | 实现状态 |
|----------|----------|
| `OUTPUT` 原语 + 槽位 `("trigger_signature", "template")` | ✅ `primitives.py` 已添加 |
| `SITUATION` 原语 + 空槽位 `()` | ✅ `primitives.py` 已添加 |
| `GrammarPattern` dataclass 含全部字段 | ✅ `grammar.py` 已实现 |
| 7 个种子语法模式对应旧模板分类 | ✅ 已播种（陈述/回答/困惑/兴奋/主动搭话/因果/拥有） |
| `Garden` 含 `grammar_library` + `grammar_pot_id` | ✅ `garden.py` 已添加 |
| `init_grammar_pot()` 在启动时调用 | ✅ `main.py:_init_garden()` |
| `_store_grammar_pattern()` 在对话后调用 | ✅ `main.py:_handle_conversation()` |
| 语法优先 → 路径降级的生成流程 | ✅ `generate_from_garden()` 结构正确 |
| `decay_grammar_patterns()` 在维护中调用 | ✅ `garden.maintenance()` |
| 快照/恢复 包含完整语法库 | ✅ `snapshot()` / `from_snapshot()` |
| Hebbian 强化/弱化机制 | ✅ `reinforce_output_pattern()` |
| Jaccard 相似度匹配 | ✅ `match_output_pattern()` |
| 种子模式 `is_seed=True` 保护 | ✅ 不被完全删除 |
| 旧模板向后兼容 | ✅ 作为种子模式保留 |

### ⚠️ 概念偏差

**偏差 1：语法键优先实际上不工作**

计划描述："优先从语法盆中匹配输出模式，硬编码模板降级为种子模式和后备方案。"

实际状态：由于 Bug #3 + #4，语法匹配路径在传导有实际路径或签名 ≥ 2 时必定崩溃。降级路径（`_build_from_paths` + 硬编码模板）成了事实上的主路径。

**偏差 2：语法盆中未存储 OUTPUT 键** ✅ 已修复（2026-08-03 B1）

修复内容：
- `_seed_initial_patterns()`: 7 个种子模式 → 27 个 OUTPUT 键存入语法盆 KeyNetwork
- `store_output_pattern()`: 新语法模式 → 自动在语法盆建 OUTPUT 键 + 概念键↔OUTPUT 跨盆桥接
- `_reinforce_grammar_bridges()`: 匹配成功时加粗桥接线（KeyNetwork 层面 Hebbian）
- `_backfill_output_keys()`: 向前兼容，补建历史数据中缺失的 OUTPUT 键
- 快照/恢复：OUTPUT 键通过 KeyNetwork 序列化自动持久化，output_key_ids 通过 GrammarPattern snapshot 持久化

**偏差 3：语法键网络的"传导驱动"未实现**

计划描述："一切皆键+线：语言输出由传导驱动，不靠硬编码模板。"

实际情况：
- 语法匹配使用 Jaccard 集合相似度（`match_output_pattern` 中的纯 Python 计算），不通过 KeyNetwork 传导
- 概念键（亮键）→ 语法模式的选择不经过电线
- 不存在"概念原语键 → 语法盆中的 OUTPUT 键"的跨盆桥接/传导

简言之：语法选择不是传导驱动的，是 dict 查表驱动的。

---

## 六、运行行为审查 🏃

### 当前运行流程（实际路径标注 bug）

```
师父输入
  │
  ├─ parse_user_input()            ← 分词 + 种子键匹配
  │
  ├─ garden_conduct()              ← 三轮跨盆传导
  │
  ├─ 预测/证据/推理/犹豫检测        ← v1.0 各子系统
  │
  ├─ generate_from_garden()        ← 语言生成
  │   │
  │   ├─ _try_grammar_generation() ← 语法键优先
  │   │   ├─ compute_activation_signature()   [Bug #3 潜伏]
  │   │   ├─ match_output_pattern()           [Jaccard 匹配，非传导]
  │   │   └─ _extract_slot_values_from_keys() [Bug #4 潜伏]
  │   │
  │   └─ _build_from_paths()       ← 降级路径（事实上的主路径）
  │       ├─ 识别根键/子键/孙键
  │       └─ 按电线粗细拼接句子
  │
  ├─ _store_grammar_pattern()      ← 存储格局→输出映射 [Bug #3 潜伏]
  │
  └─ 输出 + 疲劳表达
```

### 实际运行测试日志

**启动测试**（`LuoXiaoheiRuntime(filepath='data/test_review.luoxiaohei')`）：
```
ModuleNotFoundError: No module named 'knowledge'
```
→ 因 Bug #2 无法启动

**独立模块测试**（绕过 Bug #2，直接测试 Garden + Grammar）：
```
Garden initialized: 3 pots
Grammar pot created: pot-3
Grammar stats: total=7 active=7 seed=7
Snapshot restore: total=7 (match)
maintenance() OK - no crash (Bug #1 未触发，因无满足删除条件的模式)
```

**语法匹配测试**（模拟"小鱼干很好吃"输入）：
```
Input: "小鱼干很好吃"
Seeds found: 1 (content="师父很忙" prim="SITUATION")  ← 匹配质量差
Activation signature: ('SITUATION',)  ← 只有 1 个元素
Grammar matched: #3 (困惑型, 空签名) ← 降级匹配
Response: "…。" ← 空槽位导致无语义输出
```

**Bug 触发测试**：
- Bug #1：✅ 已确认 `AttributeError: 'Garden' object has no attribute 'garden'`
- Bug #2：✅ 已确认 `ModuleNotFoundError: No module named 'knowledge'`
- Bug #3：✅ 已确认 `TypeError: cannot unpack non-iterable int object`
- Bug #4：✅ 已确认 `AttributeError: 'GardenConductionResult' object has no attribute 'activation_paths'`

---

## 七、受影响文件总览

| 文件 | 涉及问题 |
|------|----------|
| `src/core/grammar.py` | Bug #1 (L695), Bug #3 (L220, L240), 偏差 #2 (缺少键+线创建) |
| `src/storage/format.py` | Bug #2 (L92, L128), 冗余 (`save_checkpoint`) |
| `src/inout/language.py` | Bug #4 (L270), Bug #5 (L349), 冗余 (旧模板+死函数), 偏差 #3 (非传导匹配) |
| `src/core/network.py` | 兼容问题 #1 (`set_wire` API 不一致) |
| `src/core/garden.py` | 兼容问题 #2 (属性名不一致), 偏差 #2 (语法盆为空盆) |
| `src/core/primitives.py` | 无问题 |
| `src/runtime/main.py` | 偏差 #2 (语法盆未存储键) |
| 其余 23 个文件 | 无问题 |

---

## 八、附录：审查方法清单

| 方法 | 说明 |
|------|------|
| 全文件阅读 | 逐行阅读所有 30 个 Python 源文件 |
| 导入测试 | `import` 全部 17 个核心模块，确认无 ImportError |
| 独立运行测试 | 对 `grammar.py`, `language.py`, `garden.py` 进行独立单元测试 |
| 端到端模拟 | 模拟完整对话流程：花园初始化 → 输入解析 → 传导 → 语法匹配 → 语言生成 |
| Bug 触发验证 | 对每个疑似 Bug 编写最小复现代码并实际运行确认 |
| 属性对比 | 对比 `ConductionResult` vs `GardenConductionResult` 的属性名和类型 |
| 引用计数 | 对每个函数/变量统计跨文件引用次数，识别死代码 |
| 概念对照 | 逐条对比计划文件 (`goofy-swinging-teapot.md`) 中的设计要求与实际实现 |
| 正则搜索 | 对禁区术语、导入路径模式、属性访问模式进行全项目正则搜索 |

---

## 九、修复记录（2026-08-03）

### Bug 修复（5 项）

| # | 文件 | 改动 | 验证 |
|---|------|------|------|
| 1 | `grammar.py:695` | `garden.garden` → `garden` | ✅ |
| 2 | `format.py:92,128` | 加 `src.` 前缀 | ✅ |
| 3 | `grammar.py:220,240` | 修正解包 `(pot_id, dst_id)` → `dst_id` | ✅ |
| 4 | `language.py:270` | 消费端扁平化兼容 `all_activation_paths` | ✅ |
| 5 | `language.py:349-368` | 删 guard + 有槽位才提取 + 统一清洗 | ✅ |

### 死代码删除（12 项）

**第一轮（9 项）**：
`_connection_word()`、`save_checkpoint()`、`generate_response()`、`LanguageModule.generate()`、
`TEMPLATE_STATEMENT`、`TEMPLATE_ANSWER`、`TEMPLATE_EXCITED`、`TEMPLATE_STARTUP`、`TEMPLATE_SHUTDOWN`

**第二轮（3 项）**：
`KeyNetwork`、`KnowledgeBase`、`GardenConductionResult` 的 TYPE_CHECKING 导入

### 保留的基础设施字段（5 项）

`cycles`、`cycle_patterns`、`total_pots_activated`、`has_bridge_spillover`、`get_language_module()`
→ 未接入的仪表盘/API，保留待消费端接入

### 兼容性问题核查（3 项）

| # | 判定 |
|---|------|
| C1 `set_wire` 不接受 `relation_type` | 低优先优化，非 Bug |
| C2 属性名不一致 | ✅ Bug #3 #4 修复后已解决 |
| C3 双路径调用 | 一条路径是预留 hook，保留 |

### 概念偏差修复（策略 B — B1 阶段，2026-08-03）

| # | 偏差 | 处置 |
|---|------|------|
| D1 语法键优先不工作 | ✅ Bug #3 #4 #5 修复后已消除 |
| D2 语法盆为空盆 | ✅ B1 修复：种子→27 OUTPUT 键，运行时建键+桥接 |
| D3 匹配非传导驱动 | ⏳ 待 B2：生长验证通过后切传导驱动 |

**B1 实现要点：**
- `GrammarPattern.output_key_ids`：新模式字段，记录对应 KeyNetwork 中的 OUTPUT 键 ID
- `_seed_initial_patterns()`：每条模板 → `grammar_pot.network.create_key(primitive="OUTPUT")`
- `store_output_pattern()`：新增参数 `concept_key_to_pot`，创建 OUTPUT 键 + `garden.create_bridge()`
- `_reinforce_grammar_bridges()`：匹配成功 → `BridgeEntry.strengthen()` 加粗桥接线
- `_backfill_output_keys()`：`init_grammar_pot()` 中调用，向前兼容历史数据
- `language.py:_extract_concept_key_pot_map()`：从传导结果提取非 OUTPUT 亮键→盆映射
- `main.py:_extract_concept_key_pot_map()`：同上，供 `_store_grammar_pattern` 使用

**验证结果（7/7 通过）：**
1. ✅ 种子创建：27 OUTPUT 键，每个 pattern 的 output_key_ids 与模板数一致
2. ✅ 向前兼容：重复 init 不重复创建
3. ✅ 快照恢复：OUTPUT 键 + output_key_ids 完整恢复
4. ✅ 运行时存储：`store_output_pattern` 创建键 + 跨盆桥接
5. ✅ 桥接加粗：`_reinforce_grammar_bridges` 正确加粗 (0.250→0.270)
6. ✅ 语法统计：total=8 active=8 seed=7
7. ✅ 无回归：所有现有逻辑（Jaccard 匹配、降级路径）保持不变
