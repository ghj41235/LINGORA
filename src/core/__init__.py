"""
键盘模型 AI — 核心引擎

键 + 电线 + 网络 + 传导 + 消化 + 犹豫检测。

English:
Core engine of the keyboard model — keys, wires, network, conduction,
digestion and hesitation detection, plus the design boundaries listed below.

═══════════════════════════════════════════════════════════════
设计边界（Design Boundaries）

本项目不使用以下技术——不仅是名称，包括其底层概念、数学结构
和代码模式，无论以任何形式、任何规模、任何包装出现：

不采用的技术（名称）            对应的概念/代码形态
──────────────────────────────────────────────────────────
LLM / 大语言模型                自回归 token 生成
GPT / Claude / LLaMA / 等       next-token prediction
小模型 / Small LM / SLM         任何规模的参数化语言模型
Transformer                     self-attention / QKV 注意力
Neural Network / 神经网络       层、神经元、激活函数
CNN / RNN / LSTM / GRU          卷积核、循环单元、门控
Embedding / 向量嵌入            稠密向量表示、余弦相似度
word2vec / GloVe / BERT         分布式语义向量
Softmax / Sigmoid / ReLU        任何激活函数作为计算单元
Backpropagation / 反向传播      链式法则求导、梯度下降
Gradient Descent / 梯度下降     损失函数、优化器、学习率
Adam / SGD / 优化器             参数更新规则
Matrix Multiplication           大规模矩阵乘法作为核心运算
GPU / CUDA / 并行矩阵           硬件加速、批量矩阵运算
PyTorch / TensorFlow / JAX      任何深度学习框架
Keras / HuggingFace / ONNX      模型加载、模型格式
Diffusion / Stable Diffusion    扩散模型、去噪过程
GAN / 生成对抗网络              生成器-判别器博弈
Reinforcement Learning          神经网络策略梯度
Fine-tuning / Pre-training      预训练+微调范式
Tokenization (BPE/WordPiece)    子词分词器、vocabulary 词表
.pt / .bin / .safetensors       模型权重文件格式
──────────────────────────────────────────────────────────

这条边界覆盖三个维度：
- 名称：代码中不出现这些词（注释中的否定声明除外）
- 概念：不以任何其他名词实现上述数学结构
- 代码：不引入上述计算模式，包括被其他名称包装的等效实现

本项目定义的"学习"机制是：
- 传导（Conduction）：推力沿电线扩散，逐跳衰减
- Hebbian 增强：一起亮的键之间电线加粗
- 电线衰减：不同时亮的键之间电线缓慢变细
- NARS 证据量：频率 × 置信度 → 期望值
- HTM 时序状态机：预测状态 0→1→2
- CBR 案例推理：签名匹配 + 快照预热（跳过传导）
- 语法键网络：激活格局 → OUTPUT 键 → 句子模板
- 产生式系统：SITUATION 情境感知 + META_RULE 元规则
以上全部是符号化/图论/逻辑运算，不含微分、矩阵、概率采样。

v1.0 模块结构：
────────────────────────────────────────────────────
primitives.py    — 概念原语体系（29 原语 + CD 拆解模式）
key.py           — 键（记忆原子）：亮度、证据、预测状态
wire.py          — 电线：关系类型、滑移距离、时序上下文
network.py       — 键网络：键+线的 CRUD + 快照
conduction.py    — 传导引擎：推力扩散 + 重播种
pot.py           — 花盆：主题空间 + 拆分/合并
garden.py        — 花园：多盆 + 桥接 + 维护
digestion.py     — 消化：新键创建时的 Hebbian 调整
evidence.py      — NARS 证据量：频率×置信度→期望值
temporal.py      — HTM 时序：预测状态机 + 注意力聚焦
temperature.py   — Copycat 温度：4 因子调控传导
inference.py     — NARS 推理：演绎/归纳/类比
case.py          — CBR 案例：快照存储 + 快速路径
grammar.py       — 语法键网络：格局→输出的键+线映射
hesitate.py      — 犹豫检测：振幅振荡→不确定→坦率
production.py    — 产生式系统：情境评估 + 元规则 + 输出调控
═══════════════════════════════════════════════════════════════
"""