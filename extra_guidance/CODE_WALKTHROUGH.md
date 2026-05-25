# CS336 Assignment 5 代码逐段详解

> 本文档面向学习者，以"功能是什么 + 代码如何实现"为主线，对四个源文件逐段讲解。  
> 全程中文；代码片段、技术术语保留英文。

---

## 目录

1. [grpo.py — GRPO 核心算法](#1-grpopy--grpo-核心算法)
2. [sft.py — 监督微调工具](#2-sftpy--监督微调工具)
3. [dpo.py — 直接偏好优化](#3-dpopy--直接偏好优化)
4. [utils.py — 数据集与解析工具](#4-utilspy--数据集与解析工具)

---

## 1. grpo.py — GRPO 核心算法

### 1.1 `tokenize_prompt_and_output`

**功能**：将一批 `(prompt, output)` 字符串对转化为模型训练所需的张量三元组：`input_ids`、`labels`、`response_mask`。

**为什么要单独实现这个函数？**

Hugging Face 的 tokenizer 默认对 batch 左对齐或右对齐，但不直接给出"哪些位置属于 response"的 mask。自回归 LM 训练时，只需对 response 部分计算 loss，prompt 部分的 loss 应被忽略，因此必须手动构建 `response_mask`。

**实现关键：先 padding 完整序列，再截取**

```python
# ✅ 正确：先 padding，再截 input_ids/labels
padded = full_ids + [pad_id] * pad_len   # 先 padding 到最大长度
input_ids = padded[:-1]                  # 去掉最后一个
labels    = padded[1:]                   # 去掉第一个（向左移一位）
```

为什么不能"先截取再 padding"？假设有两条序列：
- 序列 A：`[p1, p2, r1, r2]`（prompt 2 tokens，response 2 tokens）
- 序列 B：`[p1, p2, r1, r2, r3, r4]`（prompt 2 tokens，response 4 tokens）

正确做法是先把两条序列都 padding 到长度 6，再截取。若先截取再 padding，A 的 `labels` 末尾的 pad 会错位，导致 `response_mask` 计算的边界不对。

**`response_mask` 边界推导**：

设 prompt 长度为 $P$，完整序列（含 output）长度为 $S$，则 `input_ids/labels` 长度均为 $\max\_len - 1$。

- `labels[j]`：是 `full_ids[j+1]`
- 第一个 response token 在 `full_ids` 中是索引 $P$，即 `labels` 中索引 $P-1$
- 最后一个真实 token（非 padding）在 `full_ids` 中是索引 $S-1$，即 `labels` 中索引 $S-2$

因此：

```python
response_mask[j] = 1 if (prompt_len - 1 <= j < seq_len - 1) else 0
```

---

### 1.2 `get_response_log_probs`

**功能**：对 causal LM 做一次前向传播，提取每个位置的 per-token 条件 log 概率（以及可选的信息熵）。

**实现核心：`log_softmax` + `gather`**

```python
outputs = model(input_ids=input_ids)
logits = outputs.logits                          # (B, L, V)
log_probs_all = F.log_softmax(logits, dim=-1)    # (B, L, V)

per_token_log_probs = log_probs_all.gather(
    dim=2, index=labels.unsqueeze(2)             # (B, L, 1)
).squeeze(2)                                     # (B, L)
```

`gather` 操作的含义：在词表维度（dim=2）上，根据 `labels` 指定的 token ID，取出对应位置的 log 概率。等价于"对第 $b$ 条序列的第 $t$ 个位置，取 `log_probs_all[b, t, labels[b, t]]`"，即该位置真实下一 token 的 log 概率。

**为什么用 `log_softmax` 而不是 `softmax` 再 `log`？**

数值稳定性：`log_softmax(x)_i = x_i - log(sum(exp(x)))` 通过 log-sum-exp trick 避免 `exp` 溢出，而 `softmax` → `log` 会在概率极小时出现 `log(0)` 的 `-inf`。

**信息熵（可选）**：

```python
probs = torch.exp(log_probs_all)              # (B, L, V)
entropy = -(probs * log_probs_all).sum(dim=-1) # (B, L)
```

熵 $H = -\sum_v p_v \log p_v$，衡量模型在该位置的不确定程度。训练过程中监控熵可以了解模型的"探索程度"——熵过低说明模型已过度自信，可能出现了 collapse。

---

### 1.3 `compute_rollout_rewards`

**功能**：批量调用奖励函数，收集每条 rollout 的奖励，同时统计均值等元数据。

**实现简单直接**：

```python
for response, gt in zip(rollout_responses, repeated_ground_truths):
    result = reward_fn(response, gt)
    rewards.append(result["reward"])
    format_rewards.append(result["format_reward"])
```

奖励函数 `reward_fn` 的接口为 `(response, ground_truth) → {"reward": float, "format_reward": float}`，其中 `reward` 是总奖励（格式奖励 + 准确性奖励），`format_reward` 单独记录格式部分。两者分开统计，方便训练时分析模型在格式遵循和答题正确性上的进展。

---

### 1.4 `compute_group_normalized_rewards`

**功能**：在每个 prompt 组内对奖励做归一化，得到优势估计（advantage）。这是 GRPO 的核心计算。

**输入格式**：`raw_rewards` 形状为 `(N,)`，其中 $N = n\_prompts \times G$，前 $G$ 个属于第 0 组，以此类推。

**先 reshape 再计算**：

```python
rewards_grouped = raw_rewards.reshape(n_groups, group_size)  # (n_groups, G)
group_mean = rewards_grouped.float().mean(dim=1, keepdim=True)  # (n_groups, 1)
```

`keepdim=True` 使得 `group_mean` 形状为 `(n_groups, 1)`，可以直接广播减去 `rewards_grouped`（形状 `(n_groups, G)`），实现组内每个元素减去本组均值。

**四种归一化变体**：

```python
# GRPO：baseline=mean, normalizer=std
advantages = (rewards - group_mean) / (group_std + eps)

# Dr.GRPO：baseline=mean, normalizer=none
advantages = rewards - group_mean

# MaxRL：baseline=mean, normalizer=mean
advantages = (rewards - group_mean) / (group_mean.abs() + eps)

# RFT：baseline=none, normalizer=none
advantages = rewards  # 直接用原始奖励
```

**注意 `std` 的无偏估计**：PyTorch 的 `.std()` 默认 `unbiased=True`（分母 $G-1$），这在 $G$ 较小时（如 $G=8$）比有偏估计（分母 $G$）更准确地反映真实方差，但当组内所有奖励相同时 `std=0`，需要 `eps` 防止除零。

---

### 1.5 `compute_policy_gradient_loss`

**功能**：根据优势和当前策略的 log 概率，计算 per-token 策略梯度 loss。支持 on-policy 和三种 off-policy 变体。

**输入维度处理**：

```python
A = raw_rewards_or_advantages
if A.dim() == 1:
    A = A.unsqueeze(1)  # (B,) → (B, 1)，用于广播到 (B, L)
```

**四种方法实现对比**：

| 方法 | per-token loss | 说明 |
|------|---------------|------|
| `none` | `-A * log_π` | 标准 REINFORCE，A 广播到每个 token |
| `noclip` | `-A * exp(log_π - log_π₀)` | 无截断重加权，梯度无偏但方差大 |
| `grpo` | `-min(w*A, clip(w)*A)` | PPO 风格 token 级截断 |
| `gspo` | `-min(s*A, clip(s)*A)` 展开到每个 token | 序列级几何均值截断 |

**GSPO 实现的关键细节**：

```python
# 只对 response token 求几何均值（排除 prompt 和 padding）
mask_f = response_mask.float()
L = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
mean_log_ratio = (log_ratio * mask_f).sum(dim=1, keepdim=True) / L

s = torch.exp(mean_log_ratio)                       # (B, 1)
per_seq_obj = torch.min(s * A, clipped_s * A)       # (B, 1)
per_token_loss = -per_seq_obj.expand_as(policy_log_probs)  # (B, L)
```

`expand_as` 将序列级标量扩展到每个 token 位置（不复制数据）。反向传播时，梯度通过 `mean_log_ratio` 回传到每个 `log_ratio[b, t]`，链式法则自动引入 $1/L$ 因子——这正是序列长度归一化的效果。

**截断监控**：

```python
is_clipped = (w != clipped_w)
metadata["clip_fraction"] = is_clipped.float().mean()
```

`clip_fraction` 记录被截断的 token（或序列）比例，是训练稳定性的重要指标。若比例过高（>50%），说明当前策略与旧策略偏差过大，应减小学习率或更频繁地重新采样。

---

### 1.6 `aggregate_loss_across_microbatch`

**功能**：将 `(B, L)` 的 per-token loss 聚合为标量，支持两种归一化策略。

**sequence 归一化**（GRPO 默认）：

```python
seq_token_count = mask_f.sum(dim=1).clamp(min=1)  # (B,)，每条序列的 response token 数
seq_loss = masked_loss.sum(dim=1) / seq_token_count  # (B,)，每条序列的平均 loss
loss = seq_loss.mean()                               # scalar，对 batch 取平均
```

每条序列贡献相同权重（不论长短），适合奖励函数对序列整体评分的场景。

**constant 归一化**（Dr.GRPO）：

```python
loss = masked_loss.sum() / normalization_constant
```

所有 response token 的 loss 之和除以固定常数（通常 = $B \times G \times L$），使长序列贡献更大，纠正了 sequence 归一化下的隐式偏差。

---

### 1.7 `grpo_train_step`

**功能**：执行一次完整的 GRPO 更新，包括奖励计算、优势归一化、分词、梯度累积反向传播、梯度裁剪和优化器步骤。

**梯度累积的实现逻辑**：

```python
for step_i in range(gradient_accumulation_steps):
    # 取第 step_i 个 microbatch
    s, e = step_i * microbatch_size, (step_i + 1) * microbatch_size
    # ... 前向传播、loss 计算 ...

    # sequence 规范化：需除以 G，使梯度等价于全 batch 平均
    if loss_normalization == "sequence":
        backward_loss = mb_loss / gradient_accumulation_steps
    else:
        backward_loss = mb_loss  # constant 规范化已正确缩放，无需再除
    backward_loss.backward()
```

**为什么 constant 规范化不需要额外除以 G？**

constant 规范化的分母是 `normalization_constant`（通常 = 全 batch 总 token 数）。每个 microbatch 计算的是"本 microbatch token 的 loss 总和 / 全 batch 总 token"，G 个 microbatch 累积后自然等价于全 batch 的 loss。

sequence 规范化的每个 microbatch 计算的是"本 microbatch 的序列平均 loss"，G 个 microbatch 累积后是"G × 全 batch 序列平均 loss"，必须除以 G 才能消除累积效应。

---

## 2. sft.py — 监督微调工具

### 2.1 `masked_normalize`

**功能**：对 tensor 中 mask=1 的元素做加权求和并归一化。

```python
def masked_normalize(tensor, mask, dim=None, normalize_constant=1.0):
    masked = tensor * mask.float()      # 非 mask 位置变为 0
    if dim is None:
        total = masked.sum()            # 全局求和
    else:
        total = masked.sum(dim=dim)     # 沿指定维度求和
    return total / normalize_constant
```

这是一个通用工具函数，对 mask=0 的位置乘以 0（而非跳过），因此输出形状与输入一致（dim 指定时对应维度被消去），便于后续运算。

### 2.2 `sft_microbatch_train_step`

**功能**：SFT 微批次训练步骤，计算负对数似然损失并反向传播。

```python
def sft_microbatch_train_step(policy_log_probs, response_mask,
                               gradient_accumulation_steps, normalize_constant=1.0):
    microbatch_size = policy_log_probs.shape[0]
    total_scale = normalize_constant * gradient_accumulation_steps * microbatch_size
    loss = -masked_normalize(policy_log_probs, response_mask,
                             dim=None, normalize_constant=total_scale)
    loss.backward()
    return loss, {}
```

**损失公式**：

$$
\mathcal{L} = -\frac{\sum_{b,t} \text{mask}_{b,t} \cdot \log\pi_\theta(y_{b,t}|\cdot)}{C \times G \times B}
$$

其中 $C$ = `normalize_constant`，$G$ = `gradient_accumulation_steps`，$B$ = microbatch size。

**注意**：函数内部调用了 `loss.backward()`，调用方**不应**也不需要再次调用 backward。外层训练循环只需调用 `optimizer.step()` 和 `optimizer.zero_grad()`。

---

## 3. dpo.py — 直接偏好优化

### 3.1 `_response_log_prob_sum`（内部函数）

**功能**：计算单个 response 在给定 prompt 上下文下的条件 log 概率之和（含 EOS）。

**实现步骤**：

```python
# 1. 用 Alpaca 模板构造上下文前缀（只填 instruction，response 留空）
alpaca_prefix = _ALPACA_TEMPLATE.format(instruction=prompt, response="")

# 2. 分别编码前缀和 response
p_ids = tokenizer.encode(alpaca_prefix, add_special_tokens=False)
r_ids = tokenizer.encode(response, add_special_tokens=False)

# 3. 手动追加 EOS（注意：直接追加 token ID，不能用字符串编码）
eos_id = tokenizer.eos_token_id
if eos_id is not None:
    r_ids = r_ids + [eos_id]

# 4. 拼接完整序列
full_ids = p_ids + r_ids   # 长度 n

# 5. 构建 input_ids 和 labels
input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long)  # (1, n-1)
labels    = torch.tensor([full_ids[1:],  ], dtype=torch.long) # (1, n-1)

# 6. 构建 response_mask：只对 labels 中属于 response 的位置计分
mask = torch.zeros(1, n - 1)
mask[0, len(p_ids) - 1 : n - 1] = 1.0

# 7. 前向传播并加权求和
lp_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
return (lp_dict["log_probs"] * mask).sum()   # scalar
```

**两个容易踩坑的细节**：

**坑 1：必须用 Alpaca 模板**

若直接用 prompt 字符串（不套模板），前缀 token 与 SFT 训练时不一致，log 概率值错误。DPO 的 log 概率计算必须与 SFT 训练完全对齐——相同的上下文格式，相同的 tokenizer 设置。

**坑 2：EOS 必须追加 token ID，不能追加字符串**

Llama 的 tokenizer 使用 BPE：
```python
# ❌ 错误：字符串 "<eos>" 会被分成多个 token
r_ids += tokenizer.encode("<eos>", add_special_tokens=False)
# → 可能变成 ["<", "eos", ">"] 三个 token！

# ✅ 正确：直接追加 token ID
r_ids = r_ids + [tokenizer.eos_token_id]
```

EOS token 的字符串表示（`<eos>`、`</s>` 等）在 BPE 词表中是一个完整的特殊 token，但若当作普通字符串编码，BPE 会把尖括号和字母分开处理。

### 3.2 `compute_per_instance_dpo_loss`

**功能**：计算单条偏好数据 $(x, y_w, y_l)$ 的 DPO 损失。

```python
def compute_per_instance_dpo_loss(lm, lm_ref, tokenizer, beta,
                                   prompt, response_chosen, response_rejected):
    # 参考模型：不需要梯度
    with torch.no_grad():
        lp_ref_chosen   = _response_log_prob_sum(lm_ref, prompt, response_chosen,   tokenizer)
        lp_ref_rejected = _response_log_prob_sum(lm_ref, prompt, response_rejected, tokenizer)

    # 训练模型：需要梯度（用于反向传播）
    lp_chosen   = _response_log_prob_sum(lm, prompt, response_chosen,   tokenizer)
    lp_rejected = _response_log_prob_sum(lm, prompt, response_rejected, tokenizer)

    # 对数比率：log π_θ(y|x) - log π_ref(y|x)
    log_ratio_chosen   = lp_chosen   - lp_ref_chosen
    log_ratio_rejected = lp_rejected - lp_ref_rejected

    # DPO 损失
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss
```

**`F.logsigmoid` 的数值稳定性**：

等价于 $\log\sigma(z) = -\log(1 + e^{-z})$，PyTorch 内部用数值稳定实现，避免了 $e^{-z}$ 在 $z$ 很大时溢出或精度丢失。

**梯度流向**：

`lm_ref` 的 log 概率在 `no_grad` 上下文中计算，为常数；只有 `lm` 的 `lp_chosen` 和 `lp_rejected` 带梯度，梯度通过 `logsigmoid` → `log_ratio_chosen - log_ratio_rejected` → `lp_chosen`/`lp_rejected` 回传到 `lm` 的参数。

**跨设备兼容性**：两个模型可能在不同 GPU 上，函数返回的 `loss` 设备与 `lm`（训练模型）一致。

---

## 4. utils.py — 数据集与解析工具

### 4.1 `parse_mmlu_response`

**功能**：从模型输出文本中提取 MMLU 答案字母（A/B/C/D）。

**三级降级策略**：

1. 匹配 `"the answer is X"` 等明确格式（最可靠）
2. 匹配 `"(A)"` 或 `"A."` 等选项格式（中等可靠，要求唯一）
3. 匹配孤立大写字母 `\b[A-D]\b`（兜底，要求所有字母一致）

若三级都无法提取到唯一且有效的字母，返回 `None`（标记为"无法解析"）。这种降级设计能处理不同模型的不同输出风格，最大化解析成功率。

### 4.2 `parse_gsm8k_response`

**功能**：从 GSM8K 推理链中提取最终数值答案。

```python
numbers = re.findall(r"\d[\d,]*", model_output)
if not numbers:
    return None
last_number = numbers[-1].replace(",", "")
return last_number
```

**取最后一个数字的原因**：GSM8K 标准评测协议约定取最后出现的数字。模型在 CoT 推理过程中会产生多个中间计算结果，最终答案通常出现在 `"The answer is ..."` 之后，即文本末尾。

### 4.3 `PackedSFTDataset`

**功能**：将指令微调数据（JSONL 格式）打包为固定长度序列的 PyTorch Dataset。

**核心思路——"流式拼接 + 滑动窗口"**：

```python
# 1. 将所有文档格式化后分词，拼成一条超长序列
all_tokens = []
for doc in docs:
    text = _ALPACA_TEMPLATE.format(instruction=doc["prompt"], response=doc["response"])
    toks = tokenizer.encode(text.rstrip(), add_special_tokens=True)
    if toks[-1] != eos_id:
        toks = toks + [eos_id]
    all_tokens.extend(toks)

# 2. 滑动窗口切分（步长 = seq_length，无重叠）
n_windows = (len(all_tokens) - 1) // seq_length
for i in range(n_windows):
    start = i * seq_length
    window = all_tokens[start : start + seq_length + 1]  # 长度 L+1
    items.append({
        "input_ids": torch.tensor(window[:-1]),  # 长度 L
        "labels":    torch.tensor(window[1:]),   # 长度 L（向左移一位）
    })
```

**`text.rstrip()` 的必要性**：Alpaca 模板末尾有换行符 `\n`，BPE 可能把它与下一个文档的第一个字节合并为一个 token，影响 EOS 追加逻辑（`toks[-1] != eos_id` 判断可能失效）。`rstrip()` 去掉末尾空白，确保每个文档以 EOS 结尾。

**滑动窗口不跨越文档边界吗？**

不保证——这正是"packed dataset"的特点。窗口可能横跨两个文档的边界（一部分是前一篇文章的结尾，另一部分是下一篇的开头）。这是合理的：每个文档末尾有 EOS token，模型可以从 EOS 学会识别文档边界；少量跨文档序列对大规模训练的影响可以忽略不计，但显著提高了数据利用率。

### 4.4 `iterate_batches`

**功能**：将 `PackedSFTDataset` 包装为 `DataLoader`，产生批次字典。

```python
return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

每个 batch 包含 `input_ids: (B, L)` 和 `labels: (B, L)`，可直接传入 `get_response_log_probs` 做前向传播。`drop_last=False`（默认）保留最后一个可能不完整的 batch，确保数据不浪费。
