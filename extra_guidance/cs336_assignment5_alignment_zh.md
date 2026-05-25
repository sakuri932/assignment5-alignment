# CS336 Assignment 5：语言模型对齐技术详解

> **核心主题**：后训练（Post-training）技术栈，包括监督微调（SFT）、GRPO 强化学习算法及其变体、离策略扩展、以及直接偏好优化（DPO）。本文基于 Stanford CS336 Spring 2026 第五次作业材料编写。

---

## 目录

1. [背景：后训练与对齐](#1-背景后训练与对齐)
2. [监督微调（SFT）](#2-监督微调sft)
3. [GRPO：组相对策略优化](#3-grpo组相对策略优化)
4. [GRPO 变体算法](#4-grpo-变体算法)
5. [离策略强化学习](#5-离策略强化学习)
6. [直接偏好优化（DPO）](#6-直接偏好优化dpo)
7. [代码实现总结](#7-代码实现总结)

---

## 1. 背景：后训练与对齐

### 1.1 为什么需要后训练？

大型语言模型的训练分为两个阶段：

**预训练（Pre-training）**：在海量文本上做下一个 token 的预测，目标是让模型学会语言的统计规律。但这种训练方式产生的模型只会"接续文本"——给定任意前缀，它会补全成自然语言，却不一定能有用地回答问题或遵循指令。

**后训练（Post-training）**：对预训练模型做进一步优化，使其行为符合人类期望。后训练包含三个递进的阶段：

1. **SFT（Supervised Fine-Tuning）**：用高质量"问题—答案"对微调模型，教会它正确的回答格式和风格。
2. **RLHF/GRPO（Reinforcement Learning from Human/Automated Feedback）**：用强化学习让模型在给定任务上持续改进，超越 SFT 数据的质量上限。
3. **DPO（Direct Preference Optimization）**：用人类或模型的偏好数据直接优化，无需显式奖励模型。

本作业覆盖以上所有阶段。

### 1.2 问题设置与符号约定

| 符号 | 含义 |
|------|------|
| $x$ | prompt（提示词） |
| $y$ | response（模型响应） |
| $\pi_\theta$ | 待训练的策略（policy），即语言模型 |
| $\pi_\text{ref}$ | 参考策略（reference policy），通常是 SFT 模型，训练时冻结 |
| $R(x, y)$ | 奖励函数，对完整 response 给分 |
| $G$ | group size，每个 prompt 采样的 rollout 数量 |
| $B$ | batch size |
| $N = B \times G$ | 一次更新中的总 rollout 数量 |
| $\beta$ | KL 惩罚系数，控制策略偏离参考模型的程度 |

强化学习的优化目标是：

$$
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \left[ R(x, y) \right] - \beta \cdot D_{\mathrm{KL}}(\pi_\theta \| \pi_\text{ref})
$$

其中 KL 散度项防止训练策略过度偏离参考策略（即"奖励黑客"问题）。

### 1.3 奖励函数的设计（以数学推理任务为例）

本作业使用格式奖励 + 准确性奖励的组合：

- **格式奖励**：模型输出必须包含 `<think>...</think>` 推理块和 `<answer>...</answer>` 答案块，格式正确才给分。
- **准确性奖励**：提取 `<answer>` 内的数值，与标准答案比较，答对得满分。

这种设计来自 DeepSeek-R1-Zero：不提供任何思维链示例，让模型通过 RL 自主学习推理格式。

---

## 2. 监督微调（SFT）

SFT 是后训练的第一步，相当于给模型"看范例"——展示高质量的问答对，让模型学会模仿。

### 2.1 Alpaca 指令格式

本作业使用 Alpaca 模板将指令和响应格式化：

```
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
```

这个模板的作用是给模型明确的角色定位（"按任务要求写响应"），并通过固定格式让模型学会区分 prompt 部分和 response 部分。

**关键细节**：训练时只计算 response 部分的 loss，prompt 部分不参与反向传播。这体现在 `response_mask`：mask=1 的位置才计入损失。

### 2.2 打包数据集（PackedSFTDataset）

**为什么需要"打包"？**

朴素的做法是把每条对话单独 padding 到最大长度，这会导致大量 padding token 浪费算力。PackedSFTDataset 的思路是把所有文档拼接成一条超长序列，再滑动窗口切分——每个训练样本都是实际内容，无浪费。

**实现流程**：

1. 将每条文档用 Alpaca 模板格式化后分词，末尾追加 EOS token。
2. 将所有文档的 token 序列拼接成一条长序列 $T$（长度为 $|T|$）。
3. 以步长 `seq_length` 做滑动窗口切分：

$$
\text{window}_i = T[i \cdot L : i \cdot L + L + 1]
$$

其中 $L = \text{seq\_length}$。每个窗口：
- `input_ids` = `window[:-1]`（长度 $L$）
- `labels` = `window[1:]`（长度 $L$，向左偏移一位）

**注意**：窗口之间不重叠（步长等于序列长度），总共产生

$$
\left\lfloor \frac{|T| - 1}{L} \right\rfloor
$$

个样本。相邻窗口共享一个边界 token（`window_i` 的最后一个 token 是 `window_{i+1}` 的第一个 `input_ids` token）。

### 2.3 自回归训练的 Token 掩码

在分词完成后，需要构建 `response_mask`，指示哪些位置的 label 是真正的 response token（需计入 loss）：

```python
# prompt_len - 1：第一个 response token 在 labels 中的位置
# seq_len - 2：最后一个真实（非 padding）response token 的位置
response_mask[j] = 1  iff  (prompt_len - 1) <= j < (seq_len - 1)
```

**重要实现细节**：必须先对完整序列（`prompt_ids + output_ids`）做 padding，再截取 `input_ids` 和 `labels`。若先截取再 padding，则边界 pad 对不上 full_ids 末尾的 pad，导致 `response_mask` 位置错误。

### 2.4 SFT 损失函数

SFT 的损失是负对数似然（negative log-likelihood）：

$$
\mathcal{L}_{\mathrm{SFT}} = -\frac{1}{C \cdot G \cdot B} \sum_{b=1}^{B} \sum_{t=1}^{L} \mathbf{1}[\text{mask}_{b,t}=1] \cdot \log \pi_\theta(y_{b,t} \mid y_{b,<t}, x_b)
$$

其中 $C$ 是额外的归一化常数，$G$ 是梯度累积步数，$B$ 是微批次大小（microbatch size）。

### 2.5 梯度累积

**问题背景**：80GB 显存的 GPU 只能支持 batch size=2 的训练（序列长度 512），但实践中我们希望有效 batch size 达到 32 或更大。

**梯度累积的核心思想**：不是每个批次后都更新权重，而是在 $G$ 个批次后才更新一次。每个小批次（microbatch）计算 loss 并调用 `.backward()` 累积梯度，到第 $G$ 步再调用 `optimizer.step()`。

**梯度等价性**：设大批次 $\{x_1, \ldots, x_{GB}\}$ 被拆成 $G$ 个微批次 $\{x_1,\ldots,x_B\}, \{x_{B+1},\ldots,x_{2B}\}, \ldots$。对每个微批次计算 $\mathcal{L}_i / G$ 并反向传播，累积梯度的结果等价于在整个大批次上计算梯度——这是因为梯度对期望是线性的。

```python
gradient_accumulation_steps = 4
for idx, (inputs, labels) in enumerate(data_loader):
    logits = model(inputs)
    loss = loss_fn(logits, labels) / gradient_accumulation_steps
    loss.backward()  # 梯度累积，不清零

    if (idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()       # 每 G 步更新一次
        optimizer.zero_grad()  # 清零梯度
```

**注意**：必须将 loss 除以 `gradient_accumulation_steps`，才能使累积梯度与真正大批次的梯度等价（平均而非求和）。

---

## 3. GRPO：组相对策略优化

GRPO（Group Relative Policy Optimization）是本作业的核心算法，它将强化学习用于大型语言模型的后训练。

### 3.1 语言模型作为强化学习策略

将语言模型理解为一个**自回归策略（autoregressive policy）**：

- **状态（state）**：当前已生成的前缀 $(x, y_{<t})$
- **动作（action）**：下一个 token $y_t$（从词表中采样）
- **轨迹（trajectory）**：完整的 response $y = (y_1, y_2, \ldots, y_T)$
- **奖励（reward）**：对完整 response 的评分 $R(x, y)$（稀疏奖励，只在序列末尾给出）

策略 $\pi_\theta$ 生成完整 response 的概率为：

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t})
$$

### 3.2 REINFORCE 策略梯度

**目标**：最大化期望奖励

$$
J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ R(x, y) \right]
$$

**策略梯度定理**（也称对数导数技巧，log-derivative trick）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ R(x, y) \cdot \nabla_\theta \log \pi_\theta(y \mid x) \right]
$$

**推导**（关键步骤）：

$$
\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta \sum_y \pi_\theta(y|x) R(x,y) \\
&= \sum_y R(x,y) \nabla_\theta \pi_\theta(y|x) \\
&= \sum_y R(x,y) \cdot \pi_\theta(y|x) \cdot \frac{\nabla_\theta \pi_\theta(y|x)}{\pi_\theta(y|x)} \\
&= \mathbb{E}_{y \sim \pi_\theta} \left[ R(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x) \right]
\end{align}
$$

第三步利用了恒等式 $\nabla_\theta \log f = \frac{\nabla_\theta f}{f}$（即对数导数技巧）。

**实用意义**：$\nabla_\theta \log \pi_\theta(y|x)$ 就是将 response $y$ 视为 ground truth 时的 SFT 梯度——因此 REINFORCE 等价于用奖励作为权重的加权 SFT。

**蒙特卡洛估计**：用采样估计期望：对每个 prompt $x$，采样多个 response $y^{(i)} \sim \pi_\theta$，用

$$
\hat{\nabla}_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} R(x, y^{(i)}) \cdot \nabla_\theta \log \pi_\theta(y^{(i)} \mid x)
$$

来近似真实梯度。

### 3.3 基线减法与方差缩减

**高方差问题**：REINFORCE 的梯度估计方差很大（因为 response 之间的奖励差异巨大），导致训练不稳定、收敛慢。

**关键洞察**：对任意不依赖于 $y$ 的基线函数 $b(x)$，有：

$$
\mathbb{E}_{y \sim \pi_\theta} \left[ b(x) \cdot \nabla_\theta \log \pi_\theta(y \mid x) \right] = 0
$$

**证明**：

$$
\mathbb{E}_y \left[ b(x) \cdot \nabla_\theta \log \pi_\theta(y|x) \right] = b(x) \cdot \mathbb{E}_y \left[ \nabla_\theta \log \pi_\theta(y|x) \right] = b(x) \cdot \nabla_\theta \underbrace{\sum_y \pi_\theta(y|x)}_{=1} = 0
$$

因此，减去任意基线 $b(x)$ 不改变梯度的**期望**，但可以显著降低**方差**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ \left(R(x, y) - b(x)\right) \cdot \nabla_\theta \log \pi_\theta(y \mid x) \right]
$$

定义**优势（advantage）**：$A(x, y) = R(x, y) - b(x)$

### 3.4 GRPO 核心：组内相对优势

**GRPO 的思路**：对每个 prompt $x$，采样 $G$ 个 response，用**组内均值**作为基线：

$$
b_i = \frac{1}{G} \sum_{j=1}^{G} R(x, y^{(i,j)})
$$

每个 response 的优势为：

$$
A^{(i,j)} = \frac{R(x, y^{(i,j)}) - \mu_i}{\sigma_i + \varepsilon}
$$

其中：
- $\mu_i = \frac{1}{G}\sum_j R(x, y^{(i,j)})$ 是组内平均奖励
- $\sigma_i = \mathrm{std}(R(x, y^{(i,j)}))$ 是组内标准差（PyTorch 默认无偏估计，分母 $G-1$）
- $\varepsilon$ 是防除零的小常数（如 $10^{-6}$）

**为什么用组内归一化？**

- 减去均值：高于平均水平的 response 得到正优势（被鼓励），低于平均的得到负优势（被抑制）。
- 除以标准差：使优势的量级在不同 prompt 之间一致，避免某些 prompt 奖励方差大而主导训练。

### 3.5 GRPO 训练损失

对应的损失函数（最大化目标 → 最小化负目标）：

$$
\mathcal{L}_{\mathrm{GRPO}} = -\frac{1}{N} \sum_{i=1}^{B} \sum_{j=1}^{G} A^{(i,j)} \cdot \frac{1}{|y^{(i,j)}|} \sum_{t=1}^{|y^{(i,j)}|} \log \pi_\theta\left(y^{(i,j)}_t \;\Big|\; x^{(i)}, y^{(i,j)}_{<t}\right)
$$

其中外层平均先对序列内 token 取平均（sequence normalization），再对所有 rollout 取平均。

### 3.6 完整 GRPO 算法

**Algorithm 1（GRPO 训练）**：

```
输入：训练数据集 D，初始策略 π_θ，超参数 G, B, lr 等

对每个训练步：
  1. 从 D 中采样 B 个 prompt：x^(1), ..., x^(B)
  2. 对每个 prompt x^(i)，用当前策略采样 G 个 response：
       y^(i,1), ..., y^(i,G) ~ π_θ(·|x^(i))
  3. 计算每个 rollout 的奖励：r^(i,j) = R(x^(i), y^(i,j))
  4. 组内归一化，得到优势 A^(i,j)
  5. 计算策略梯度损失，反向传播，更新 θ
```

实践中步骤 1+2 合并为 rollout 阶段，步骤 3+4+5 为更新阶段，更新阶段可用梯度累积拆成多个 microbatch。

### 3.7 代码实现要点

**`tokenize_prompt_and_output`**：将 `(prompt, output)` 对分词，返回 `input_ids (B,L)`, `labels (B,L)`, `response_mask (B,L)`。核心是先对完整序列 padding，再 slice——避免 mask 位置错位。

**`get_response_log_probs`**：模型前向传播，用 `log_softmax` + `gather` 提取每个 label token 的 log 概率，返回 `(B,L)` 的 per-token log 概率张量。

**`compute_rollout_rewards`**：批量调用 reward_fn，返回 `(N,)` 的奖励张量和统计元数据。

**`compute_group_normalized_rewards`**：将 `(N,)` 奖励 reshape 为 `(n_groups, G)`，组内做均值/标准差归一化，flatten 回 `(N,)`。

---

## 4. GRPO 变体算法

原始 GRPO 并非唯一选择，不同的归一化策略对应不同的算法变体。

### 4.1 四种变体的对照

| 算法 | baseline | advantage_normalizer | loss_normalization |
|------|----------|----------------------|--------------------|
| GRPO | mean | std | sequence |
| Dr.GRPO | mean | none | constant |
| MaxRL | mean | mean | sequence |
| RFT | none | none | sequence |

### 4.2 Dr.GRPO：序列长度归一化

**问题**：在 GRPO 的 sequence normalization 下，长 response 和短 response 的梯度贡献相同（每条序列贡献 1 单位）。但实际上，长 response 更难生成，其梯度信号应有更大权重。

**Dr.GRPO 的改进**（来自 DeepSeek 团队）：

1. **不除以标准差**（`advantage_normalizer="none"`）：保留奖励信号的绝对量级，短 response 组的奖励差异小，自然贡献更小的梯度。
2. **constant loss normalization**：将 loss 除以固定常数 $C$（通常 = batch中总 token 数 $B \times G \times L$），而非按序列数平均。

这使得长 response（包含更多 token）对总损失的贡献更大，纠正了 sequence normalization 下的隐式 token 数偏差。

### 4.3 RFT：拒绝采样微调

**思路**：最简单的强化学习变体——只保留奖励为正（答对）的 response，用 SFT 训练这些样本。

对应 `baseline="none"`（不减均值），`advantage_normalizer="none"`（奖励直接作为权重）。等价于用过滤后的成功样本做 SFT，但训练效率更高（可以批量采样然后过滤）。

**局限性**：没有基线减法，对"容易的" prompt（每次都答对，$A=1$）和"难的" prompt（很少答对）一视同仁，梯度信号的质量不均匀。

### 4.4 MaxRL：均值归一化

**思路**：不用标准差归一化，而是除以 $|\mu_i|$（组均值的绝对值）。

这等价于用 prompt 难度（平均得分的倒数）对梯度重新加权：难的 prompt（得分低，$|\mu_i|$ 小）获得更大的梯度权重，引导模型多学习难题。

**对比 GRPO（std 归一化）**：std 归一化关注组内奖励的**相对分散程度**；MaxRL 关注组内奖励的**绝对水平**。

---

## 5. 离策略强化学习

到目前为止讨论的 GRPO 都是**在策略（on-policy）**的：每次更新前都要重新采样 rollout。这效率很低（采样是昂贵的）。**离策略（off-policy）RL** 允许用旧策略 $\pi_0$ 采样的数据来更新当前策略 $\pi_\theta$。

### 5.1 重要性重加权

**核心公式**：利用重要性采样（importance sampling），可以用 $\pi_0$ 的采样来估计 $\pi_\theta$ 下的期望：

$$
\mathbb{E}_{y \sim \pi_\theta}[A \cdot \nabla_\theta \log \pi_\theta] = \mathbb{E}_{y \sim \pi_0}\left[ \frac{\pi_\theta(y|x)}{\pi_0(y|x)} \cdot A \cdot \nabla_\theta \log \pi_\theta \right]
$$

其中重要性权重 $w = \frac{\pi_\theta(y|x)}{\pi_0(y|x)}$ 修正了分布偏移。

**Token 级分解**：由于 $\pi_\theta(y|x) = \prod_t \pi_\theta(y_t|y_{<t},x)$，有：

$$
w = \frac{\pi_\theta(y|x)}{\pi_0(y|x)} = \prod_{t=1}^{T} \frac{\pi_\theta(y_t|y_{<t},x)}{\pi_0(y_t|y_{<t},x)} = \prod_{t} \exp(\log\pi_\theta(y_t|\cdot) - \log\pi_0(y_t|\cdot))
$$

对应 per-token 重要性权重：$w_t = \exp(\log\pi_\theta(y_t|\cdot) - \log\pi_0(y_t|\cdot))$

**`noclip` 方法**：直接用 $-A \cdot w_t$ 作为 per-token loss，梯度无偏但方差极大（$w_t$ 可能很大或很小）。

### 5.2 Token 级截断（PPO/GRPO 风格）

**问题**：无截断时，若 $\pi_\theta$ 偏离 $\pi_0$ 较远，$w_t$ 可能极大，导致梯度爆炸和训练不稳定。

**PPO 的解决方案**：对每个 token 的重要性权重截断到 $[1-\varepsilon, 1+\varepsilon]$：

$$
\mathcal{L}_{\mathrm{token}}^{(t)} = -\min\left(w_t \cdot A, \; \mathrm{clip}(w_t, 1-\varepsilon, 1+\varepsilon) \cdot A\right)
$$

**`min` 的作用**：

- 当 $A > 0$（鼓励该 response）：若 $w_t > 1+\varepsilon$，裁剪后的值更小（更保守），避免过度更新。
- 当 $A < 0$（抑制该 response）：若 $w_t < 1-\varepsilon$，裁剪后的值更大（绝对值更小），避免过度惩罚。

取 `min` 相当于总选"更保守"的更新方向，保证单步更新幅度可控。

### 5.3 序列级截断（GSPO）

**问题**：token 级截断处理的是 per-token 的分布比，但 token 的差异会累乘，导致长序列整体分布漂移更严重。

**GSPO（Group Sequence Policy Optimization）的思路**：用序列级别的**几何均值重要性权重**代替 token 级权重：

$$
s = \exp\left(\frac{1}{L}\sum_{t=1}^{L} \log\frac{\pi_\theta(y_t|\cdot)}{\pi_0(y_t|\cdot)}\right) = \exp\left(\frac{1}{L}\sum_t (\log\pi_\theta - \log\pi_0)\right)
$$

然后对序列级权重 $s$ 做截断：

$$
\mathcal{L}_{\mathrm{seq}} = -\min\left(s \cdot A, \; \mathrm{clip}(s, 1-\varepsilon, 1+\varepsilon) \cdot A\right)
$$

**梯度的 $1/L$ 因子**：$s$ 的表达式中含 $\frac{1}{L}\sum_t \log\pi_\theta$，对 $\theta$ 求梯度时链式法则自然产生 $\frac{1}{L}$ 因子——等价于对所有 token 梯度取平均（序列长度归一化）。

**实现技巧**：将序列级 loss 值展开到 `(B, L)` 形状（每条序列内所有 token 共享同一 loss 值），然后正常反向传播。梯度会通过每个 token 的 $\log\pi_\theta$ 自动引入 $1/L$ 权重，无需手动处理。

### 5.4 四种重要性重加权方式对比

| 方法 | 权重 | 截断 | 特点 |
|------|------|------|------|
| `none` | 无（在策略） | 无 | 标准 GRPO，每步需重新采样 |
| `noclip` | token 级 $w_t$ | 无 | 无偏但高方差，不稳定 |
| `grpo` | token 级 $w_t$ | $[1-\varepsilon, 1+\varepsilon]$ | PPO 风格，稳定，略有偏差 |
| `gspo` | 序列级 $s$（几何均值）| $[1-\varepsilon, 1+\varepsilon]$ | 低方差，含隐式序列长度归一化 |

---

## 6. 直接偏好优化（DPO）

### 6.1 RLHF 的挑战

传统 RLHF（Reinforcement Learning from Human Feedback）流程复杂：

1. 收集人类对 response 的排序偏好数据
2. 训练奖励模型 $r_\theta(x, y)$ 拟合偏好
3. 用 PPO 等 RL 算法优化 LM 使其最大化 $r_\theta$

这个流程有三个主要缺点：
- 奖励模型训练不稳定，容易过拟合（reward hacking）
- RL 训练本身极难调参
- 整体管线复杂，难以复现

### 6.2 DPO 的核心推导

**从最优策略反推**：DPO 的出发点是观察到，对于带 KL 约束的 RL 目标，最优策略 $\pi_r$ 与奖励函数 $r$ 存在解析关系：

$$
r(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)
$$

其中 $Z(x) = \sum_y \pi_\text{ref}(y|x) e^{r(x,y)/\beta}$ 是配分函数（只与 $x$ 有关，与 $y$ 无关）。

**RLHF 的偏好损失**：Bradley-Terry 偏好模型下，人类更偏好 $y_w$ 而非 $y_l$ 的概率为：

$$
p^*(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

对应损失：

$$
\ell_\theta^r(x, y_w, y_l) = -\log\sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

**将奖励替换为策略**：把最优策略公式代入上式，$Z(x)$ 对应在差中相消：

$$
r(x, y_w) - r(x, y_l) = \beta\log\frac{\pi_r(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_r(y_l|x)}{\pi_\text{ref}(y_l|x)}
$$

因此，DPO 的单条偏好样本损失为：

$$
\ell_{\mathrm{DPO}}(\pi_\theta, \pi_\text{ref}, x, y_w, y_l) = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)
$$

**直观理解**：DPO 鼓励 $\pi_\theta$ 相对于 $\pi_\text{ref}$ 更加偏向 $y_w$（偏好响应），同时相对压低 $y_l$（非偏好响应）。$\beta$ 控制偏离参考策略的力度。

### 6.3 与 RLHF 的关键区别

- **无需显式奖励模型**：DPO 直接用 $\pi_\theta$ 隐式表达奖励，简化了训练管线。
- **无需在线采样**：DPO 只需计算条件 log 概率，不需要从模型采样，因此不是"强化学习"的常规意义。
- **偏好数据来源灵活**：偏好数据可以来自人类标注，也可以来自其他 LM（如 GPT-4）对同一 prompt 的多个 response 打分，无需真正的人类反馈。

### 6.4 实现细节

**Log 概率计算**：需要分别计算 $\log\pi_\theta(y_w|x)$ 和 $\log\pi_\theta(y_l|x)$，即对应 response token 序列（含 EOS）的条件 log 概率之和。

**使用 Alpaca 模板**：与 SFT 保持一致，将 prompt 格式化为 Alpaca 前缀（`"Below is an instruction..."` 到 `"### Response:\n"`），response 为实际内容，EOS 追加在 response 末尾。

**实现公式**（代码逻辑）：

对于模型 `lm` 和响应 `y`，计算：

$$
\ell_{\mathrm{resp}}(\mathrm{lm}, x, y) = \sum_{t \in \text{response}} \log \pi_{\mathrm{lm}}(y_t \mid \text{Alpaca}(x), y_{<t})
$$

然后 DPO 损失：

$$
\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\!\left(\beta\bigl[(\ell_{\mathrm{resp}}(\pi_\theta, x, y_w) - \ell_{\mathrm{resp}}(\pi_\text{ref}, x, y_w)) - (\ell_{\mathrm{resp}}(\pi_\theta, x, y_l) - \ell_{\mathrm{resp}}(\pi_\text{ref}, x, y_l))\bigr]\right)
$$

**重要实现 Bug 警示**：

1. **必须使用 Alpaca 模板**：DPO 的 log 概率必须在完整 Alpaca 上下文下计算，与 SFT 保持一致。若直接用裸 prompt，上下文不匹配，log 概率值完全不同。
2. **EOS 必须追加为 token ID**：不能将 EOS 作为字符串（如 `"<eos>"`）编码，因为 BPE tokenizer 会把 `<eos>` 拆成多个 token（如 `["<", "eos", ">"]`）。必须直接将 `tokenizer.eos_token_id` 追加到 token ID 列表。
3. **梯度隔离**：参考模型 `lm_ref` 的 log 概率计算必须在 `torch.no_grad()` 上下文中，否则会构建不必要的计算图。

**`response_mask` 的边界**：设 Alpaca 前缀 token 数为 $P$，response（含 EOS）token 数为 $R$，完整序列长度 $n = P + R$。`input_ids = full_ids[:-1]`，`labels = full_ids[1:]`，则 response mask 覆盖 `labels` 中索引 $[P-1, n-2]$ 的位置：

- 位置 $P-1$：`input_ids` 中是最后一个前缀 token，`labels` 中是第一个 response token
- 位置 $n-2$：`input_ids` 中是最后一个 response token（EOS 之前），`labels` 中是 EOS token

### 6.5 DPO 训练配置

相比 SFT，DPO 训练有独特挑战：

- 每个样本需要**两个模型**各跑两次前向（chosen/rejected × policy/reference），显存占用是 SFT 的 4 倍。
- 因此不能用 AdamW（需要额外显存存优化器状态），改用 **RMSprop**。
- 推荐配置：batch size=64，$\beta=0.1$，学习率 $1\times10^{-6}$，梯度累积。
- 监控指标：**隐式奖励分类准确率**（当 chosen 的 log 概率高于 rejected 时，视为正确分类），用于衡量偏好对齐程度。

---

## 7. 代码实现总结

### 7.1 文件结构

```
cs336_alignment/
├── grpo.py          # GRPO 核心：分词、log 概率、奖励归一化、PG loss、训练步
├── sft.py           # SFT：masked_normalize、微批次训练步
├── dpo.py           # DPO：per-instance loss 计算
└── utils.py         # 工具：PackedSFTDataset、response 解析器
```

### 7.2 函数调用关系

```
grpo_train_step
├── compute_rollout_rewards         # 批量调用 reward_fn
├── compute_group_normalized_rewards # 组内优势归一化
├── tokenize_prompt_and_output      # 分词
└── [microbatch loop]
    ├── get_response_log_probs      # 前向传播
    ├── compute_policy_gradient_loss # PG loss（含重要性重加权）
    └── aggregate_loss_across_microbatch # 聚合为标量

compute_per_instance_dpo_loss
└── _response_log_prob_sum × 4    # lm_ref×2 + lm×2（chosen/rejected）
    └── get_response_log_probs
```

### 7.3 各算法变体参数速查

```python
# GRPO（标准）
grpo_train_step(...,
    baseline="mean", advantage_normalizer="std",
    importance_reweighting_method="none",
    loss_normalization="sequence")

# Dr.GRPO
grpo_train_step(...,
    baseline="mean", advantage_normalizer="none",
    importance_reweighting_method="none",
    loss_normalization="constant", normalization_constant=B*G*L)

# RFT
grpo_train_step(...,
    baseline="none", advantage_normalizer="none",
    importance_reweighting_method="none",
    loss_normalization="sequence")

# MaxRL
grpo_train_step(...,
    baseline="mean", advantage_normalizer="mean",
    importance_reweighting_method="none",
    loss_normalization="sequence")

# Off-policy GRPO（PPO 风格）
grpo_train_step(...,
    importance_reweighting_method="grpo",
    old_log_probs=old_lp,  # 旧策略 log 概率
    cliprange=0.2)

# GSPO（序列级截断）
grpo_train_step(...,
    importance_reweighting_method="gspo",
    old_log_probs=old_lp,
    cliprange=0.2)
```

### 7.4 关键超参数及其物理意义

| 超参数 | 典型值 | 物理意义 |
|--------|--------|----------|
| `group_size` G | 8~16 | 每个 prompt 采样多少个 response，决定基线估计的质量 |
| `advantage_eps` ε | 1e-6 | 标准差归一化的防除零常数 |
| `cliprange` ε | 0.2 | PPO 截断范围，控制单步策略更新的最大幅度 |
| `beta` β | 0.1 | DPO 的 KL 惩罚强度，越大越保守（越接近参考策略） |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值，防止梯度爆炸 |
| `gradient_accumulation_steps` | 4~16 | 有效 batch size 的放大倍数 |

---

## 附录：关键公式速查

**REINFORCE 梯度**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\left[(R(x,y) - b(x)) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]
$$

**GRPO 优势**：

$$
A^{(i,j)} = \frac{R(x^{(i)}, y^{(i,j)}) - \mu_i}{\sigma_i + \varepsilon}, \quad \mu_i = \frac{1}{G}\sum_j R^{(i,j)}, \quad \sigma_i = \mathrm{std}(R^{(i,j)})
$$

**PPO token 级截断**：

$$
\mathcal{L}_t = -\min\!\left(w_t A,\; \mathrm{clip}(w_t, 1-\varepsilon, 1+\varepsilon) A\right), \quad w_t = \exp(\log\pi_\theta(y_t|\cdot) - \log\pi_0(y_t|\cdot))
$$

**GSPO 序列级截断**：

$$
s = \exp\!\left(\frac{1}{L}\sum_t (\log\pi_\theta - \log\pi_0)\right), \quad \mathcal{L} = -\min(sA,\; \mathrm{clip}(s, 1-\varepsilon, 1+\varepsilon)A)
$$

**DPO 损失**：

$$
\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\!\left(\beta\!\left[\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right]\right)
$$
