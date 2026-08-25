# 外包任务 · Claude 网页版 —— 修复 evidence.py 的 3 个隐患

## 项目背景与铁律
luoxiaohei 是一个**纯符号**认知 AI，代码是 Python、中文注释。

**铁律（绝对禁止，最高优先级）**：禁止一切神经网络相关技术——LLM、Transformer、神经网络、Embedding/向量嵌入、余弦相似度、Softmax、反向传播/梯度下降、矩阵乘法、GPU、PyTorch/TensorFlow、tokenization、概率采样，任何形式任何规模都不行。只允许符号化/图论/逻辑运算。**你写的代码里绝不能出现这些被禁的词/概念/模式。**（完整宪法清单见 `src/core/__init__.py` 顶部。）

---

## 任务：只修 3 个隐患（1/2/3），4/5/6 别动

上一轮你给 `src/core/evidence.py` 写了 30 个单元测试，并报了 6 个隐患。现在**只修前 3 个**，剩下 3 个（顺序依赖 / docstring 的 0.5 措辞 / decay 抬升处女键）**这次不动**，留给后面。

### 隐患 1：`deltas` 字典漏记（最优先）
`update_evidence_after_conduction` 里，**种子键**（步骤 1）和**亮度分级**（步骤 2-3）两段在改证据后都会写 `deltas[key_id]`；但下面三段**改了证据却漏了写 deltas**：
- 3b 共激活（`event_freq=0.8`）
- 4 桥接触发（`event_freq=0.7`）
- 5 犹豫/方向切换（`event_freq=0.0`）

**修法**：这三段也按步骤 1/2-3 的写法累加 delta：
```python
delta = abs(key.evidence_freq - new_f) + abs(key.evidence_conf - new_c)
deltas[key_id] = deltas.get(key_id, 0.0) + delta
```
保证「本轮每个被更新过的键，deltas 里都有它本次的总变化量」。

### 隐患 2：`frequency` 没有 clamp
`revise_evidence` 里 `new_conf` 每次都过 `min(0.99, max(EVIDENCE_CONF_MIN, ...))`，但 `new_freq` 没任何夹紧——负的 `event_conf` 或脏的 `old_freq` 会让 frequency 越出 [0,1] 且不自愈。

**修法**：给 `new_freq` 夹到 `[0.0, 1.0]`。

### 隐患 3：短路分支绕开下限
`if old_conf + event_conf == 0.0: return old_freq, old_conf` 这个短路分支直接 return，绕开了 `EVIDENCE_CONF_MIN`。例如 `revise_evidence(0.77, 0.0, 0.5, 0.0)` 会返回 conf=0.0（低于 0.01）。

**修法**：短路分支也要保证返回的 confidence ∈ `[EVIDENCE_CONF_MIN, 0.99]`（frequency ∈ `[0,1]`）。和隐患 2 统一夹紧逻辑，别写成两套。

## 约束（重要，别越界）
- **最小改动**：只动 `revise_evidence` 和 `update_evidence_after_conduction` 里跟上面 3 点相关的行，别重构、别顺手改别的。
- **别改公式语义**：frequency 的加权平均公式、confidence 的"增量趋近 1.0"模型都不动，只加夹紧和补记账。
- **别加输入校验/assert**：不要给纯函数加参数断言——只夹输出，不校验输入。
- **别动 4/5/6**、**别动 EVIDENCE_* 常量**。
- 铁律全程生效。

## 你要交付
1. 改好的 `src/core/evidence.py`。
2. 改好的 `src/core/test_evidence.py`：
   - 隐患 1/2/3 对应的 `test_hazard_*` 用例，**改成断言"修复后"的正确行为**（变成回归测试——以后 bug 复现就会红）。
   - 隐患 4/5/6 的 `test_hazard_*` 用例**保持原样不动**（它们继续断言"还没修"的现状）。
   - 最终 **30 个测试仍然全绿**。
3. 一段「改了哪三处、每处为什么」的说明（每个隐患 1-2 句）。

## 运行方式
解压后 `src/core/` 下有 `evidence.py` / `key.py` / `__init__.py` / `test_evidence.py`，四者同目录。运行：
```
python3 -m unittest test_evidence -v
```
跑完 30 个全绿即可交付。
