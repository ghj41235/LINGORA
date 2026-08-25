# 外包任务 · Claude 网页版 —— 为 NARS 证据量写单元测试

## 项目背景与铁律
luoxiaohei 是一个**纯符号**认知 AI，核心隐喻「键 + 电线 + 网络 + 传导」。代码是 Python、中文注释。

**铁律（绝对禁止，最高优先级）**：本项目禁止一切神经网络相关技术——LLM、Transformer、神经网络、Embedding/向量嵌入、余弦相似度、Softmax、反向传播/梯度下降、矩阵乘法、GPU、PyTorch/TensorFlow、tokenization、概率采样，任何形式、任何规模都不行。只允许符号化/图论/逻辑运算（传导、Hebbian、电线衰减、NARS 证据量、HTM、CBR、产生式）。**你写的代码里绝不能出现这些被禁的词/概念/模式。**

**范围**：这是小任务测试，只做下面这一小块，别重构别的地方、别扩展开，交付能跑的代码。

---

## 任务：给 evidence.py 两个纯函数写单元测试 + 找边界 bug

背景：NARS 证据量是本项目「信念 = 概率、拧螺丝式调教」的核心机制。下面是 `src/core/evidence.py` 的**完整源码**（已贴给你，无需读文件）。注意 `Key` 对象上有 `evidence_freq: float` 和 `evidence_conf: float` 两个字段，被下面这些函数读写：

```python
"""
键盘模型 AI — NARS 证据量模块  v1.0

truth = (frequency, confidence)
- frequency ∈ [0.0, 1.0]: 正面证据占比
- confidence ∈ [0.0, 1.0]: 总证据量
期望值 expectation = confidence × (frequency - 0.5) + 0.5
"""

from __future__ import annotations
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .garden import Garden
    from .key import Key

# ── NARS 参数 ──
EVIDENCE_SEED_CONF = 0.15            # 种子键（师父提到的）→ 强正面
EVIDENCE_BRIGHT2_CONF = 0.10         # 亮度 = 2 的激活键
EVIDENCE_BRIGHT1_CONF = 0.05         # 亮度 = 1 的激活键
EVIDENCE_BRIDGE_CONF = 0.12          # 桥接触发的跨域键
EVIDENCE_COACT_CONF = 0.06           # 共激活关联键
EVIDENCE_HESITATION_NEG = 0.08       # 犹豫中的参与键（负面）
EVIDENCE_DIRECTION_NEG = 0.05        # 方向切换的参与键（负面）

EVIDENCE_DECAY_INTERVAL = 86400      # 1 天（秒）
EVIDENCE_CONF_DECAY_RATE = 0.001     # 每天 confidence 衰减量
EVIDENCE_CONF_MIN = 0.01             # confidence 最低值


def truth_expectation(frequency: float, confidence: float) -> float:
    """期望值 = confidence * (frequency - 0.5) + 0.5"""
    return confidence * (frequency - 0.5) + 0.5


def revise_evidence(
    old_freq: float, old_conf: float,
    event_freq: float, event_conf: float,
) -> tuple[float, float]:
    """NARS 证据修正。
    event_freq: 1.0=正面 / 0.0=负面 / 0.5=中性(不改 frequency)
    new_freq = (old_freq*old_conf + event_freq*event_conf) / (old_conf + event_conf)
    new_conf = old_conf + event_conf*(1.0 - old_conf)，再夹到 [0.01, 0.99]
    """
    if old_conf + event_conf == 0.0:
        return old_freq, old_conf
    new_freq = (old_freq * old_conf + event_freq * event_conf) / (old_conf + event_conf)
    new_conf = old_conf + event_conf * (1.0 - old_conf)
    new_conf = min(0.99, max(EVIDENCE_CONF_MIN, new_conf))
    return new_freq, new_conf


def update_evidence_after_conduction(garden, seeds_by_pot, result):
    """传导后批量更新证据。按优先级：种子键→强正面，亮度≥2→正面，
    亮度=1→弱正面，共激活→0.8，桥接触发→0.7，犹豫/方向切换→0.0(负面)。"""
    deltas = {}

    # 1. 种子键：师父直接提到的 → 强正面
    for pot_id, seed_list in seeds_by_pot.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue
        for key_id, _brightness in seed_list:
            key = pot.network.get_key(key_id)
            if key is not None:
                new_f, new_c = revise_evidence(
                    key.evidence_freq, key.evidence_conf,
                    1.0, EVIDENCE_SEED_CONF,
                )
                key.evidence_freq, key.evidence_conf = new_f, new_c

    # 2-3. 传导激活键：按亮度分级
    for pot_id, cr in result.round1.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue
        for key_id, brightness in cr.final_pattern.items():
            key = pot.network.get_key(key_id)
            if key is None:
                continue
            conf = EVIDENCE_BRIGHT2_CONF if brightness >= 2 else EVIDENCE_BRIGHT1_CONF
            new_f, new_c = revise_evidence(
                key.evidence_freq, key.evidence_conf, 1.0, conf,
            )
            key.evidence_freq, key.evidence_conf = new_f, new_c

        # 3b. 共激活：同盆同时亮的键 → 关联证据(event_freq=0.8)
        active_ids = list(cr.final_pattern.keys())
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                ki, kj = active_ids[i], active_ids[j]
                wire = pot.network.get_wire(ki, kj)
                if wire is not None:
                    for kid in (ki, kj):
                        k = pot.network.get_key(kid)
                        if k is not None:
                            new_f, new_c = revise_evidence(
                                k.evidence_freq, k.evidence_conf,
                                0.8, EVIDENCE_COACT_CONF,
                            )
                            k.evidence_freq, k.evidence_conf = new_f, new_c

    # 4. 桥接触发键 → 跨域正面(event_freq=0.7)
    for pot_id, cr in result.round3.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue
        for key_id in cr.final_pattern:
            key = pot.network.get_key(key_id)
            if key is not None:
                new_f, new_c = revise_evidence(
                    key.evidence_freq, key.evidence_conf,
                    0.7, EVIDENCE_BRIDGE_CONF,
                )
                key.evidence_freq, key.evidence_conf = new_f, new_c

    # 5. 犹豫/方向切换 → 负面证据(event_freq=0.0)
    for pot_id, cr in result.round1.items():
        pot = garden.pots.get(pot_id)
        if pot is None:
            continue
        if cr.is_hesitating:
            for key_id in cr.final_pattern:
                key = pot.network.get_key(key_id)
                if key is not None:
                    new_f, new_c = revise_evidence(
                        key.evidence_freq, key.evidence_conf,
                        0.0, EVIDENCE_HESITATION_NEG,
                    )
                    key.evidence_freq, key.evidence_conf = new_f, new_c
        if cr.direction_changed:
            for key_id in cr.final_pattern:
                key = pot.network.get_key(key_id)
                if key is not None:
                    new_f, new_c = revise_evidence(
                        key.evidence_freq, key.evidence_conf,
                        0.0, EVIDENCE_DIRECTION_NEG,
                    )
                    key.evidence_freq, key.evidence_conf = new_f, new_c

    return deltas


def decay_all_evidence(garden):
    """长期未激活的键 confidence 缓慢下降，永不低于 EVIDENCE_CONF_MIN(0.01)。"""
    now = time.time()
    for pot in garden.pots.values():
        if pot.is_dormant:
            continue
        for key in pot.network.all_keys():
            last_active = max(key.last_activated_at, key.created_at)
            idle = now - last_active
            if idle > EVIDENCE_DECAY_INTERVAL:
                days = idle / 86400.0
                key.evidence_conf = max(
                    EVIDENCE_CONF_MIN,
                    key.evidence_conf - EVIDENCE_CONF_DECAY_RATE * days,
                )


def evidence_label(freq: float, conf: float) -> str:
    """返回证据的人类可读标签。"""
    exp = truth_expectation(freq, conf)
    if conf < 0.1:
        return "未知"
    if exp > 0.75:
        return "确信"
    if exp > 0.6:
        return "比较确定"
    if exp > 0.4:
        return "不确定"
    if exp > 0.25:
        return "怀疑"
    return "否定"
```

### 你要交付的
写一个 `test_evidence.py`（用 Python 标准库 `unittest`，放 `src/core/` 下），覆盖：

1. `truth_expectation` 文档声明的 5 个例子：
   (0.9,0.8)→0.82、(0.9,0.1)→0.54、(0.1,0.8)→0.18、(0.1,0.1)→0.46、(0.5,任意)→0.50
2. `revise_evidence`：
   - 正面(1.0) 使 frequency 上升；负面(0.0) 使 frequency 下降；中性(0.5) 不改变 frequency；
   - confidence 单调上升且封顶 0.99、兜底 0.01；
   - 零权重边界 old_conf + event_conf == 0 → 原样返回；
   - 结果 frequency 始终 ∈ [0,1]、confidence ∈ [0.01, 0.99]。

### 重点：找隐患
除了写测试，请指出你发现的**边界 bug / 公式隐患**。至少检查：
- frequency 有没有被显式夹在 [0,1]？（没有的话，什么输入能让它越界？）
- 置信度用"增量趋近 1.0"是简化模型（原版 NARS 是更复杂的加权公式）——这个简化在什么情况下失真？
- 有没有除零 / 浮点 / 参数未校验的问题？

### 交付物
`test_evidence.py`（能跑、全绿）+ 一段「我发现的隐患」清单。
