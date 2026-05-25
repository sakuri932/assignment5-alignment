# CS336 Assignment 5: 语言模型对齐 — 工作报告

> **完成日期**：2026-05-24  
> **测试结果**：33 passed（19 GRPO + 7 SFT + 2 data + 4 metrics + 1 DPO，全部通过）

---

## 目录

1. [作业概述](#1-作业概述)
2. [实现过程](#2-实现过程)
3. [遇到的问题与调试过程](#3-遇到的问题与调试过程)
4. [测试结果](#4-测试结果)
5. [文件结构](#5-文件结构)

---

## 1. 作业概述

本次作业实现了语言模型后训练的三个核心技术：

| 模块 | 文件 | 主要内容 |
|------|------|----------|
| SFT | `utils.py`, `sft.py` | PackedSFTDataset、微批次训练、梯度累积 |
| GRPO | `grpo.py` | 策略梯度算法、奖励归一化、off-policy 扩展 |
| DPO | `dpo.py` | 直接偏好优化损失函数 |

涵盖的算法变体：GRPO、Dr.GRPO、RFT、MaxRL（在策略）；`noclip`、token 级 PPO 截断、GSPO 序列级截断（离策略）。

---

## 2. 实现过程

### 2.1 SFT 部分

#### `PackedSFTDataset`（`utils.py`）

打包数据集的核心是将所有文档的 token 拼接成一条超长序列，再用步长等于 `seq_length` 的滑动窗口切分。每个文档末尾追加 EOS token，保证窗口横跨文档边界时模型能识别边界。

关键细节：用 `text.rstrip()` 去掉 Alpaca 模板末尾的换行符，防止 BPE 将换行和下一文档的第一个字节合并，影响 EOS 位置检测。

#### `sft_microbatch_train_step`（`sft.py`）

用 `masked_normalize` 计算 response token 的平均负对数似然损失，内部直接调用 `loss.backward()`。损失除以 `normalize_constant × gradient_accumulation_steps × microbatch_size`，确保梯度与真正大批次等价。

### 2.2 GRPO 部分

实现顺序：分词 → log 概率 → 奖励 → 归一化 → loss → 聚合 → 训练步。

#### `tokenize_prompt_and_output`

关键是"先 padding 完整序列，再截取 input_ids/labels"，避免 mask 边界位置错误。`response_mask` 的起始位置是 `prompt_len - 1`（labels 中第一个 response token），结束位置是 `seq_len - 2`（labels 中最后一个真实 token，即 EOS）。

#### `compute_group_normalized_rewards`

将 `(N,)` 奖励 reshape 为 `(n_groups, G)`，用广播机制批量计算各组的均值/标准差。通过 `baseline` 和 `advantage_normalizer` 参数切换四种算法变体（GRPO/Dr.GRPO/MaxRL/RFT），避免重复代码。

#### `compute_policy_gradient_loss`

实现了四种重要性重加权方式。GSPO 的关键在于：用 `response_mask` 只对 response token 计算几何均值对数比率，然后 `expand_as` 展开到 `(B, L)`，让梯度通过每个 token 的 `log_ratio` 自然引入 `1/L` 因子。

#### `grpo_train_step`

梯度累积时，sequence 规范化需对每个 microbatch 的 loss 额外除以 `gradient_accumulation_steps`；constant 规范化因分母已包含全 batch token 数，无需此操作。训练步结束时调用 `zero_grad(set_to_none=True)` 释放显存。

### 2.3 DPO 部分

#### `_response_log_prob_sum`

用 Alpaca 模板构建前缀，分别编码 prompt 前缀和 response，再手动追加 EOS token ID（整数，非字符串）。构建 input_ids/labels/mask 后调用 `get_response_log_probs` 并对 response 位置的 log 概率求和。

#### `compute_per_instance_dpo_loss`

`lm_ref` 在 `torch.no_grad()` 上下文中推理，`lm` 正常前向传播保留梯度。计算两个 log ratio 之差后用 `F.logsigmoid` 计算损失（数值稳定）。

---

## 3. 遇到的问题与调试过程

### 问题 1：DPO 损失值不正确

**现象**：测试期望损失约 `0.9104`，实际得到 `1.1436`。

**根因排查**：通过编写诊断脚本，系统测试了五种不同的 tokenization 策略：

| 策略 | 损失值 | 是否通过 |
|------|--------|----------|
| 裸 prompt（无模板） | 1.1436 | ✗ |
| Alpaca 模板，无 EOS | 0.9069 | ✗ |
| **Alpaca 模板 + EOS token ID** | **0.9104** | ✓ |
| Alpaca 模板 + EOS 字符串编码 | 错误（EOS 被拆多 token）| ✗ |
| add_special_tokens=True | 偏差 | ✗ |

**修复**：

1. 使用 `_ALPACA_TEMPLATE.format(instruction=prompt, response="")` 作为前缀——确保与 SFT 训练上下文一致。
2. 直接追加 `tokenizer.eos_token_id`（整数）到 `r_ids`——避免 BPE 把 EOS 字符串分成多个 token。

### 问题 2：`response_mask` 边界理解

**现象**：初次实现时 mask 位置偏差 1，导致第一个 response token 未被计入。

**分析**：`labels[j] = full_ids[j+1]`，所以 `labels` 中的第一个 response token 是 `full_ids[prompt_len]`，对应 `j = prompt_len - 1`，即 `mask` 起始索引是 `prompt_len - 1`（不是 `prompt_len`）。

**修复**：理解"labels 是 full_ids 左移一位"的关系后，边界公式为：
```python
mask[j] = 1  iff  (prompt_len - 1) <= j < (seq_len - 1)
```

### 问题 3：梯度累积缩放

**现象**：`sequence` 规范化下，梯度累积的结果与全 batch 不等价（差了 `G` 倍）。

**分析**：每个 microbatch 的 `mb_loss` 是本 microbatch 内序列的平均 loss（一个大小为 1 的标量）。`G` 个 microbatch 累积后是 `G × 真正大 batch 的平均 loss`，需要除以 `G` 才能对齐。

`constant` 规范化不同：分母是固定的全 batch token 数，每个 microbatch 贡献"自己的 token 数 / 全 batch token 数"的 loss，G 个累积后天然等价于全 batch loss，无需额外缩放。

---

## 4. 测试结果

```
========================= 33 passed in X.XXs =========================
```

各模块通过情况：

| 测试模块 | 测试数量 | 状态 |
|---------|---------|------|
| GRPO（tokenize/log_probs/rewards/loss/train_step） | 19 | ✓ 全通过 |
| SFT（masked_normalize/microbatch_step） | 7 | ✓ 全通过 |
| Data（PackedSFTDataset/iterate_batches） | 2 | ✓ 全通过 |
| Metrics（parse_mmlu/parse_gsm8k/reward_fn） | 4 | ✓ 全通过 |
| DPO（per_instance_dpo_loss） | 1 | ✓ 全通过 |

---

## 5. 文件结构

```
assignment5-alignment/
├── cs336_alignment/
│   ├── grpo.py          # GRPO 核心（7 个函数，330+ 行，含详细注释）
│   ├── sft.py           # SFT 工具（2 个函数）
│   ├── dpo.py           # DPO 损失（2 个函数）
│   └── utils.py         # PackedSFTDataset、response 解析器（240+ 行）
├── tests/
│   └── adapters.py      # 测试适配层（连接测试框架与实现）
├── extra_guidance/
│   ├── cs336_assignment5_alignment_zh.md  # PDF 翻译文档
│   ├── CODE_WALKTHROUGH.md               # 代码详解（本文档的配套）
│   └── REPORT.md                         # 本工作报告
└── data/                # 评测数据集（mmlu/gsm8k/alpaca_eval 等）
```
