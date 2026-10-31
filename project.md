下面我为你细化这套 **“scRAG-Flow” (Retrieval-Augmented Data-to-Data Flow Matching)** 的完整方案架构。所有选用的模块都满足：**成熟、小巧、防过拟合、生物学解释性强**。

注意使用环境：conda activate gf310

---

### 🏛️ 全局方案架构细化（scRAG-Flow）

#### 🟢 阶段一：跨模态对比自编码器 (Stage 1: Contrastive VAE)

**目标**：在不丢失底层生物学信息的前提下，把 RNA 和 Protein 对齐到同一个特征空间。
**小巧先进的模块选择**：

1. **编码器 (Encoders)**：弃用 Transformer，采用 **Gated Residual Network (GRN)**。
* *为什么选它*：GRN 是表格数据/单细胞数据处理的神器（常用于顶级表格模型如 TabNet 等）。它比普通 MLP 多了门控机制（GLU）和残差连接，能在极少的参数下捕获基因间的非线性关系，且在小数据上极难过拟合。
* *结构*：$RNA \rightarrow \text{GRN\_Block} \times 2 \rightarrow F_{rna}$（维度如 128 维）。


2. **对齐头与损失 (Alignment)**：
* 采用经典的 **SimCLR 范式**：加一层非线性 Projection Head，将 $F_{rna}$ 和 $F_{pro}$ 映射到对齐空间。
* **Loss**: `InfoNCE Loss`（带可学习的 Temperature 参数）。


3. **解码器与重构损失 (Decoders & Recon Loss)**：
* *成熟但极具针对性的设计*：不要用单纯的 MSE。单细胞数据有大量的“零失落”（Dropout）。
* 对 RNA 和 Protein 的解码，全部使用 **ZINB Loss (Zero-Inflated Negative Binomial)**。ZINB 让模型去预测均值、方差和 dropout 概率，这在小数据上能极大地提供正则化（Regularization）作用，防止模型死记硬背。



#### 🟡 阶段二：建立记忆检索库 (Stage 2: FAISS Memory Bank)

**目标**：将训练集的先验知识固化，供下游提取。
**方法**：

1. **构建索引**：小数据（1万细胞以内），直接使用 `FAISS Exact Search`（如 `IndexFlatIP` 计算余弦相似度）。不需要近似搜索，速度在毫秒级，且绝对不会爆显存。
2. **保存特征**：跑一遍训练集，将所有细胞的 $F_{rna}^{train}$ 存入 FAISS 库，对应的真实的 $Pro^{train}$ 存为一个 PyTorch Tensor 字典。

#### 🟠 阶段三：基于检索的增强流匹配 (Stage 3: RAG Data-to-Data Flow)

**目标**：输入测试集 RNA，检索最近的训练集蛋白，计算残差，生成目标蛋白。
**小巧先进的模块选择**：

1. **检索增强机制 (KNN Smoothing Retrieval)**：
* 测试集 $RNA \rightarrow F_{rna}^{test}$。
* 去 FAISS 库找 Top-K（建议 K=3 或 5）个最相似的 $F_{rna}^{train}$。
* **巧思**：不要只取 Top-1，取 Top-K 对应的真实 $Pro^{train}$，利用余弦相似度进行**加权平均**，得到 $Pro_{retrieved}$。这极大地平滑了单一细胞带来的噪声，提供了一个极其稳健的“初始蛋白表达估计”。


2. **Data-to-Data 流匹配构建 (Optimal Transport Rectified Flow)**：
* 传统流匹配从高斯噪声 $N(0,1)$ 开始。
* 我们使用 **I2I (Image-to-Image) Rectified Flow** 公式：

$$x_t = (1-t) \cdot Pro_{retrieved} + t \cdot Pro_{target}$$



*(其中 $t \in [0,1]$，起点是检索到的蛋白，终点是真实蛋白)*


3. **流场预测网络 ($v_\theta$ 网络)**：
* **核心模块**：**AdaLN-Zero MLP (Adaptive Layer Normalization with Zero Init)**。
* *为什么选它*：这是 DiT（Diffusion Transformers）能稳定训练的核心秘密，但我们把它用在 MLP 上。它把时间 $t$ 和条件 $F_{rna}$ 作为输入，去动态缩放（Scale）和平移（Shift）网络层的特征。Zero Init 保证了网络在初始阶段就是个恒等映射，训练极度平稳。
* *任务*：预测流动的方向（即 $Pro_{target} - Pro_{retrieved}$ 的残差）。



---

### 📂 代码结构规划 (Code Directory)

我们将整个项目解耦为极其清晰的模块，这不仅方便我们一步步写代码，也方便未来论文开源时审稿人阅读。

```text
scRAG_Flow/
│
├── data/                       # 存放数据 (.h5ad)
│
├── core_modules/               # 【核心算子库】(保持小巧、可复用)
│   ├── networks.py             # 存放 GRN (Gated Residual Net) 和 AdaLN-MLP
│   ├── losses.py               # 存放 InfoNCE Loss 和 ZINB Loss
│   └── retrieval.py            # FAISS 的轻量封装类
│
├── models/                     # 【两阶段模型定义】
│   ├── clip_vae.py             # Stage 1: 双塔对比自编码器
│   └── rag_flow.py             # Stage 3: 基于 AdaLN 的条件流匹配网络
│
├── pipeline/                   # 【执行流程】
│   ├── 01_train_clip.py        # 训练 Stage 1
│   ├── 02_build_bank.py        # 建立 FAISS 库并保存 Tensor
│   ├── 03_train_flow.py        # 训练 Stage 3
│   └── 04_inference.py         # 端到端测试与指标计算
│
├── utils.py                    # 数据加载 (h5ad)、PCC/CMD 评估模块
└── requirements.txt            # torch, scanpy, faiss-cpu, torchdiffeq 等

---

### 🌟 方案评估与亮点总结（用来写论文 Introduction）

1. **为什么适合小数据？**
全程没有使用 Transformer 或庞大的 Attention 矩阵。**GRN** 和 **AdaLN-MLP** 都是极度参数高效的结构，加上 **ZINB 重构正则化**，模型在几千个细胞上就能充分收敛。
2. **最大的创新点在哪里？**
提出了 **"Data-to-Data 流生成"** 的概念。以往的生成模型都是“无中生有”（从纯噪声生成数据），这需要极大的数据量去拟合复杂的流形。我们利用 **RAG (检索)** 直接把生成起点拉到了终点附近，流模型只需要学习“最后的一公里微调（Residual Shift）”。这在单细胞跨模态预测中是首次提出。
3. **极佳的可解释性**：
在推理时，我们可以清晰地输出：模型检索到了训练集里的哪几个细胞？初始的猜测状态是什么？流匹配网络最后加上了什么样的残差修正？过程完全白盒化。

这是一个非常专业且务实的要求。在开始写代码之前，把每一个阶段的数据流向（Input/Output）、优化目标（Loss）和工程暗坑（注意点）彻底理清，是确保代码一次跑通的关键。

下面我为你详细梳理这四个阶段的**标准操作规范（SOP）**：

---

### 🟢 阶段一：跨模态对比自编码器 (Stage 1: Contrastive VAE)

本阶段的目标是训练一个极其稳健的特征提取器，将 RNA 和 Protein 映射到同一个对齐的隐空间，同时不丢失它们自身的生物学分布特征。

* **输入 (Input)**：
* `RNA_batch`: 形状 `[Batch_Size, Gene_Dim]` (如 [256, 4000])。
* `Pro_batch`: 形状 `[Batch_Size, Pro_Dim]` (如 [256, 134])。


* **输出 (Output)**：
* 对齐特征：`F_rna` 和 `F_pro`，形状均为 `[Batch_Size, Hidden_Dim]` (如 [256, 128])。
* RNA 重构参数：`mu_rna, theta_rna, pi_rna`。
* Pro 重构参数：`mu_pro, theta_pro, pi_pro`。


* **损失函数 (Loss)**：
* $Loss_{total} = \alpha \cdot Loss_{InfoNCE} + \beta \cdot Loss_{ZINB}^{RNA} + \gamma \cdot Loss_{ZINB}^{Pro}$
* **InfoNCE**：计算一整个 Batch 内 `F_rna` 和 `F_pro` 的相似度矩阵，拉近同细胞的对角线元素，推开非对角线元素。
* **ZINB Loss**：利用输出的 (均值, 离散度, Dropout率) 评估与真实输入 `RNA_batch` 和 `Pro_batch` 的负对数似然。


* **⚠️ 需要注意的点 (Gotchas)**：
1. **L2 归一化**：在送入 InfoNCE 计算前，`F_rna` 和 `F_pro` 必须经过 `F.normalize(dim=-1)` 进行 L2 归一化，否则对比学习极易崩溃或出现 NaN。
2. **Temperature 参数**：InfoNCE 中的温度系数 $\tau$ 建议设为可学习参数（`nn.Parameter`），初始化为 0.07。
3. **小数据防过拟合**：RNA Encoder 建议使用 GRN (Gated Residual Network) 并加入适量的 Dropout (0.2~0.3)，因为小数据上直接全连接容易死记硬背。



---

### 🟡 阶段二：建立记忆检索库 (Stage 2: Build Memory Bank)

本阶段不进行任何模型训练，只是单纯地将训练集的先验知识固化到内存/硬盘中。

* **输入 (Input)**：
* 整个训练集的真实 `RNA_train` 和 `Pro_train`。
* 阶段一训练好的冻结的 `RNA_Encoder`。


* **输出 (Output)**：
* **FAISS 索引对象 (`.index` 文件)**：存储所有训练集细胞的 `F_rna_train` 特征。
* **真实蛋白字典 (`.pt` 文件)**：保存所有训练集细胞的真实 `Pro_train` 张量，按照索引与 FAISS 对应。


* **损失函数 (Loss)**：
* 无。纯推断（Inference）与保存。


* **⚠️ 需要注意的点 (Gotchas)**：
1. **模型模式**：必须极其严格地设置 `model.eval()` 和 `with torch.no_grad():`，禁止在此阶段更新梯度，节省显存。
2. **FAISS 距离度量**：因为我们在阶段一使用了余弦相似度（L2 归一化后的内积），构建 FAISS 时必须选用 `faiss.IndexFlatIP`（内积距离），**切忌使用 L2 距离 (`IndexFlatL2`)**。
3. **特征入库前必须归一化**：入库前 `F_rna_train` 必须再一次保证做过 `L2 Normalize` 并且转为 `numpy.float32` 格式，FAISS 才能正确工作。



---

### 🟠 阶段三：基于检索的增强流匹配 (Stage 3: RAG-Flow Training)

这是整个架构最核心的创新点，利用 AdaLN-Zero 预测“从邻居蛋白流向目标蛋白”的残差路径。

* **输入 (Input)**：
* `RNA_batch` 和对应的真实目标蛋白 `Pro_target` (此时它是流向的终点 $x_1$)。


* **中间过程 (Retrieval & ODE Setup)**：
1. 用冻结的 RNA Encoder 提取 `F_rna`。
2. 去 FAISS 中查询 Top-K 最相似的特征，拿到对应的真实蛋白进行平均，作为初始起点 `Pro_retrieved` ($x_0$)。
3. 随机采样时间 $t \sim \text{Uniform}(0, 1)$。
4. 构造中间状态：$x_t = (1-t) \cdot x_0 + t \cdot x_1$。


* **输出 (Output)**：
* 神经网络 $v_\theta(x_t, t, F_{rna})$ 预测出的当前时刻的流场速度。


* **损失函数 (Loss)**：
* **MSE Loss**（最优传输条件流匹配公式）：$Loss = || v_\theta(x_t, t, F_{rna}) - (x_1 - x_0) ||_2^2$
* *解释：真实的运动速度是终点减去起点，我们让神经网络去逼近这个真实速度。*


* **⚠️ 需要注意的点 (Gotchas) —— 【极其重要】**：
1. **排除自身 (Exclude Self in Retrieval)**：因为在训练集上训练，FAISS 检索时 Top-1 必然是它自己！如果取了自己，$x_0$ 会完全等于 $x_1$，真实速度为 0，模型什么都学不到（退化为恒等映射）。**必须在检索时取 Top-(K+1)，然后丢弃第 1 个（距离为 1.0 的那个）**，用后面的邻居来求平均。
2. **梯度阻断 (Stop Gradient)**：送入流模型的条件变量 `F_rna` 必须使用 `.detach()`，不要让流匹配的 Loss 反向传播回 Stage 1 的 Encoder 中，这会破坏好不容易对齐的隐空间。
3. **零初始化 (Zero Init)**：AdaLN 最后一层的缩放参数必须初始化为 0。保证模型在最开始什么都不干（输出速度为0），慢慢学习残差。



---

### 🔴 阶段四：端到端推理阶段 (Stage 4: Inference)

来了一个全新的测试集细胞，端到端预测其连续的蛋白质表达量。

* **输入 (Input)**：
* 新的 `RNA_test` 向量。


* **计算流向 (Forward Pass)**：
1. 提取 `F_rna_test`。
2. 去 FAISS 库检索 Top-K，加权平均得到初始估计起点 `Pro_start` ($x_0$)。（注意：**测试集检索不需要排除自身**，因为库里根本没有它）。
3. 设置流模型的初始状态 $X = Pro_{start}$。
4. 利用 ODE Solver（常微分方程求解器），基于网络 $v_\theta$，将时间 $t$ 从 0 积分到 1。


* **输出 (Output)**：
* 积分到 $t=1$ 时的状态，即为最终预测结果 `Pro_pred`。


* **损失函数 (Loss)**：
* 无。直接用 `Pro_pred` 计算 PCC（皮尔逊相关系数）、RMSE 等指标。


* **⚠️ 需要注意的点 (Gotchas)**：
1. **求解器选择 (ODE Solver)**：使用 `torchdiffeq` 库。如果追求极致速度，设置 `method='euler'`，步数设置 10~20 步即可；如果追求准确度刷指标，设置 `method='dopri5'`（自适应步长，但较慢）。
2. **加权检索平滑**：在测试时，不要简单对 Top-K 的蛋白求算术平均，建议**利用余弦相似度进行 Softmax 加权平均**。距离越近的细胞，其参考权重越大。这在 OOD（分布外）样本上极其有效。

注意baseline:/data3/liuyinbo/fastpaper/cell_gene_model/scChord_base
注意使用环境：conda activate gf310

```
