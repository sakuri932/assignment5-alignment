# CS336 Assignment 5 Optional Part 2：工作报告

> **完成日期**：2026-05-24  
> **主题**：零样本基线评估 · 指令微调（SFT）· 直接偏好优化（DPO）

---

## 目录

1. [作业概述](#1-作业概述)
2. [实现过程](#2-实现过程)
3. [关键设计决策](#3-关键设计决策)
4. [遇到的问题与调试](#4-遇到的问题与调试)
5. [跨平台兼容性设计](#5-跨平台兼容性设计)
6. [文件结构总览](#6-文件结构总览)

---

## 1 作业概述

本 Optional Part 2 聚焦于将预训练 Llama 3.1 8B Base 模型训练为遵从指令、能拒绝有害请求的对话模型，完整经历以下流程：

| 阶段 | 内容 | 相关脚本 |
|------|------|----------|
| 零样本基线 | MMLU / GSM8K / AlpacaEval / SimpleSafetyTests 评估 | `run_mmlu.py`, `run_gsm8k.py`, `run_alpaca.py`, `run_sst.py` |
| 指令微调（SFT） | 在 UltraChat + SafetyTunedLlamas 混合数据上微调 | `train_sft.py`, `sft_dataset.py` |
| 偏好对齐（DPO） | 在 Anthropic HH 数据上用 DPO 进一步对齐 | `train_dpo.py`, `dpo_hf.py`, `data_hh.py` |
| 后训练评估 | SFT 模型和 DPO 模型在四项基准上的重新评估 | 复用上述评估脚本 |

---

## 2 实现过程

### 2.1 零样本评估脚本

> **问题原文：**（mmlu_baseline，4 分 + gsm8k_baseline，4 分 + alpaca_eval_baseline，4 分 + sst_baseline，4 分）
>
> 分别编写脚本，评估 Llama 3.1 8B 在 MMLU、GSM8K、AlpacaEval、SimpleSafetyTests 上的零样本性能。每个脚本需：（1）加载对应数据集，（2）将样例格式化为字符串 prompt，（3）为每个样例生成模型输出，（4）计算评估指标，（5）将样例、模型生成结果和评估分数序列化到磁盘以便后续分析。其中 MMLU 和 GSM8K 需实现输出解析函数（`parse_mmlu_response` / `parse_gsm8k_response`），分别解析为选项字母和单一数值答案。AlpacaEval 预测结果需以 JSON 数组格式保存（含 `instruction`、`output`、`generator`、`dataset` 字段），SimpleSafetyTests 需以 JSONL 格式保存供安全评估脚本使用。

四个评估脚本（MMLU、GSM8K、AlpacaEval、SimpleSafetyTests）遵循统一结构：

1. **加载数据**：MMLU/GSM8K 为 JSONL，AlpacaEval 为 JSONL，SimpleSafetyTests 为 CSV
2. **格式化 prompt**：将样例包裹进系统 prompt 的 `{instruction}` 槽位（系统 prompt 来自 `cs336_alignment/prompts/zero_shot_system_prompt.prompt`）
3. **生成输出**：优先 vLLM 批量推理（CUDA），回退到 HuggingFace generate（MPS/CPU）
4. **解析与评分**：MMLU 用 `parse_mmlu_response`，GSM8K 用 `parse_gsm8k_response`（均来自主作业 `cs336_alignment/utils.py`）
5. **保存结果**：包含每条样例的原始输出、解析结果和正确性标志，便于事后错误分析

AlpacaEval 和 SimpleSafetyTests 不直接计算准确率（需要外部评估器），只生成并保存模型输出，分别以 JSON 数组（AlpacaEval 要求）和 JSONL（SST 评估脚本要求）格式保存。

### 2.2 SFT 训练脚本

> **问题原文：**（sft_script，4 分 + sft，6 分）
>
> **sft_script**：编写一个训练循环脚本，在提供的指令微调数据（UltraChat-200K + SafetyTunedLlamas 混合数据）上对 Llama 3.1 8B Base 模型进行微调。脚本至少应支持：配置和控制各种模型和优化器超参数；通过梯度累积支持超出显存限制的更大批大小；定期记录训练和验证性能（例如输出到控制台和/或 Weights and Biases 等外部服务）。
>
> **sft**：在提供的指令微调数据上微调 Llama 3.1 8B Base 模型。建议使用上下文长度 512 tokens、总有效批大小每梯度步 32 条序列、训练 1 个 epoch。推荐超参数：学习率 2e-5，余弦衰减，线性预热（占总训练步数的 3%）。保存模型和 tokenizer 供后续评估和偏好数据后训练使用。**交付物**：训练配置描述、最终验证损失及对应学习曲线；序列化后的模型和 tokenizer。

`train_sft.py` 的核心是将 PDF Section 3.2.2 的伪代码翻译为 PyTorch 训练循环：

- **数据加载**：通过 `sft_dataset.py` 的 `PackedSFTDataset` 将 `.jsonl.gz` 数据打包为固定长度序列
- **损失计算**：`F.cross_entropy(logits.view(-1, V), labels.view(-1))`，对 sequence 维度展平后计算
- **梯度累积**：每 `grad_accum` 步调用一次 `optimizer.step()`，有效批大小 = `batch_size × grad_accum`
- **学习率调度**：`LambdaLR` + 自定义余弦预热函数，预热步数占总步数 3%（PDF 推荐值）
- **检查点保存**：验证损失最低时保存 `best/`，训练结束保存 `final/`

### 2.3 DPO 训练脚本

> **问题原文：**（look_at_hh，2 分 + dpo_loss，2 分 + dpo_training，4 分）
>
> **look_at_hh**：编写函数加载 Anthropic HH 数据集（4 个子集合并为训练集），提取每个样例的"指令"（第一条人类消息）和 chosen/rejected 响应对，忽略多轮对话，并记录每个样例来自哪个文件。
>
> **dpo_loss**：编写函数计算逐实例 DPO 损失（$\ell_{\text{DPO}} = -\log\sigma(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$），接收两个 LM（$\pi_\theta$ 和 $\pi_{\text{ref}}$）、tokenizer 和两个字符串（chosen/rejected 响应），使用 Alpaca 模板格式化并在响应后追加 EOS token。
>
> **dpo_training**：实现 DPO 训练循环，在 HH 数据上对指令微调 Llama 3.1 8B 模型训练 1 个 epoch，使用 2 块 GPU（一块参考模型、一块训练模型），用 RMSprop 优化器和梯度累积，追踪验证集上隐式奖励分类准确率，保存准确率最高时的 checkpoint。训练完成后在 AlpacaEval 和 SimpleSafetyTests 上重新评估 DPO 模型。

`train_dpo.py` 实现了 PDF Section 5.4 的推荐架构：

- **双模型双设备**：`device-lm`（训练模型，`cuda:0`）+ `device-ref`（参考模型，`cuda:1`）
- **优化器**：RMSprop（与原始 DPO 论文保持一致，显存占用比 AdamW 小约 1/3）
- **逐条梯度累积**：DPO 损失无法批量矩阵化，逐条计算后每 `batch_size` 条做一次参数更新
- **验证指标**：每 `val_interval` 步计算验证集上的隐式奖励分类准确率（chosen log ratio > rejected log ratio 的比例）
- **最优模型追踪**：保存验证准确率最高时的 checkpoint

---

## 3 关键设计决策

### 3.1 为什么复制而非修改主作业代码

主作业中的 `dpo.py` 依赖 `cs336_alignment/grpo.py` 的 `get_response_log_probs`，而该函数专为 `cs336_basics` 自定义模型设计，无法直接用于 HuggingFace `AutoModelForCausalLM`。Optional Part 2 使用 HF API，因此复制 `dpo.py` 为 `dpo_hf.py`，将 log prob 计算改为直接调用 `model(ids).logits`。

同理，`PackedSFTDataset` 需要支持 `.jsonl.gz` 和多字段名格式，复制为 `sft_dataset.py` 后添加这两项修改，不破坏主作业的原始实现。

### 3.2 无条件对数概率技巧

DPO 损失理论上需要计算条件对数概率 $\log \pi(y|x)$，实现时需要 response mask（只对 response 部分的 token 求和）。

利用数学等价性简化实现：chosen 和 rejected 共享同一个 prompt $x$，差值中 prompt 的 log prob 天然抵消，因此可以直接计算完整文本的无条件 log prob 之和，无需额外的 mask 操作。这将实现从"需要两次 tokenize + mask 逻辑"简化为"一次 tokenize + gather"。

### 3.3 vLLM 作为可选依赖

vLLM 提供批量推理，吞吐量比 HuggingFace generate 高出 10–20 倍，但仅支持 NVIDIA CUDA，且安装较重。通过 `try/except ImportError` 优雅处理：

```python
try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
```

运行时根据 `_VLLM_AVAILABLE and device.type == "cuda"` 决定使用哪个后端，使脚本在 macOS（无 vLLM）上也能正常运行。

### 3.4 服务器磁盘安全

根据服务器配置要求，所有磁盘写入必须到 `/mnt/a`（非根磁盘）。脚本中未强制限制路径，但训练脚本的 `--output-dir` 参数说明注释中明确提示使用 `/mnt/a/...`。`wandb` API key 通过 `~/.netrc` 读取（vLLM 和 wandb 的行业标准做法），代码中无任何硬编码凭证。

---

## 4 遇到的问题与调试

### 问题 1：HH 数据集多轮对话解析

**现象**：直接用简单分割（`split("\n\nHuman:")` 等）无法正确处理 Assistant 回复中含有 `\n\n` 的情况，边界识别出错。

**根因**：HH 数据中 Assistant 的回复可能包含多段落换行，简单分割会把一条 Assistant 消息截断成多条。

**修复**：改用带 `re.DOTALL` 标志的正则，配合非贪婪匹配和前瞻断言 `(?=\n\nHuman:|\Z)` 精确匹配每轮的边界：

```python
_HH_TURN_RE = re.compile(
    r"\n\nHuman:\s*(.*?)\n\nAssistant:\s*(.*?)(?=\n\nHuman:|\Z)",
    re.DOTALL,
)
```

**验证**：单元测试：包含多段落 Assistant 回复的对话能被正确解析为单个轮次，不会被截断。

### 问题 2：SFT 梯度累积时的损失尺度偏差

**现象**：若直接 `loss.backward()` 而不先 `/= grad_accum`，等效批大小正确（累积了 `grad_accum` 步梯度），但损失值偏大 `grad_accum` 倍，`log` 显示损失异常高。

**根因**：PyTorch 的 `backward()` 会把梯度**叠加**到 `.grad`，不除以步数时相当于对每个参数施加了放大 `grad_accum` 倍的梯度，更新步长过大。

**修复**：

```python
loss = F.cross_entropy(...) / args.grad_accum
loss.backward()
```

**验证**：验证方式——将 `grad_accum=1`（即不累积）的单步损失与 `grad_accum=4` 运行 4 步后的平均损失对比，两者应相近，误差在浮点精度范围内。

### 问题 3：DPO 训练时跨设备张量操作报错

**现象**：在双 GPU 配置下，`lp_chosen - lp_ref_chosen` 抛出 `RuntimeError: Expected all tensors to be on the same device`。

**根因**：`lp_ref_chosen` 来自参考模型（`cuda:1`），`lp_chosen` 来自训练模型（`cuda:0`），两者设备不同，不能直接相减。

**修复**：在 `dpo_hf.py` 的 `compute_per_instance_dpo_loss` 中，参考模型输出后立即移到训练模型的设备：

```python
lp_ref_chosen   = lp_ref_chosen.to(device_lm)
lp_ref_rejected = lp_ref_rejected.to(device_lm)
```

**时机选择**：在 `torch.no_grad()` 块之外执行 `.to()`，确保移动后的张量不持有梯度（`.to()` 本身不会引入梯度，但保险起见也可在 `with torch.no_grad()` 内做）。

### 问题 4：Windows / macOS 上 DataLoader 多进程报错

**现象**：在 Windows 上运行 `train_sft.py` 时，DataLoader 以 `num_workers > 0` 启动时抛出 `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`。

**根因**：Windows 使用 `spawn` 而非 `fork` 创建子进程，要求所有 worker 代码必须在 `if __name__ == "__main__":` 保护下运行，而 DataLoader 的工作进程初始化在 `__main__` 外部触发时会失败。

**修复**：在 `sft_dataset.py` 的 `iterate_batches` 中强制 Windows 下使用 `num_workers=0`：

```python
import platform
num_workers = 0 if platform.system() == "Windows" else 2
```

---

## 5 跨平台兼容性设计

本 Optional Part 2 所有脚本均经过以下三种环境的兼容性设计：

| 环境 | 设备 | 注意力实现 | vLLM | DataLoader workers |
|------|------|------------|------|-------------------|
| macOS（Apple Silicon） | MPS 或 CPU | eager | 不可用（回退 HF） | 2 |
| Windows + NVIDIA | CUDA | flash_attention_2 | 可用 | 0（Windows fork 限制） |
| Linux 服务器 + NVIDIA | CUDA | flash_attention_2 | 可用（推荐）| 4 |

**MPS 支持的重要说明**：Apple Silicon 的 MPS 后端支持 `bfloat16`，但不支持 FlashAttention-2。`device_utils.py` 的 `get_attn_impl` 会自动返回 `"eager"`，使 HF 使用标准 PyTorch attention 实现，不影响正确性，只影响速度。

**CPU 运行**：仅用于调试（极慢），`get_compute_dtype` 返回 `float32`，避免 CPU 上 `bfloat16` 的精度损失。

---

## 6 文件结构总览

```
Optional Part 2/
├── cs336_assignment5_supplement_zh.md   # PDF 翻译（19 页，含所有公式）
├── CODE_WALKTHROUGH.md                  # 本文件：逐函数代码解读
├── REPORT.md                            # 工作报告（你正在阅读）
│
├── device_utils.py      # 跨平台设备/精度/注意力实现检测 + 模型加载
├── sft_dataset.py       # PackedSFTDataset（支持 .gz + 多字段名），改自 utils.py
├── dpo_hf.py            # DPO 损失（HF API），无条件 log prob + 多 GPU，改自 dpo.py
├── data_hh.py           # Anthropic HH 数据集加载 + train/val 切分
│
├── run_mmlu.py          # 零样本 MMLU 评估（vLLM/HF 双后端）
├── run_gsm8k.py         # 零样本 GSM8K 评估（vLLM/HF 双后端）
├── run_alpaca.py        # AlpacaEval 输出生成（JSON 数组格式）
├── run_sst.py           # SimpleSafetyTests 输出生成（JSONL 格式）
│
├── train_sft.py         # SFT 训练：余弦 LR + 梯度累积 + wandb + checkpoint
└── train_dpo.py         # DPO 训练：双 GPU + RMSprop + 验证准确率 + 最优保存
```

**复制 vs. 新建说明**：

- `sft_dataset.py`：复制自 `cs336_alignment/utils.py`，修改了 gzip 支持、字段名适配、Windows workers
- `dpo_hf.py`：复制自 `cs336_alignment/dpo.py`，修改了 log prob 计算方式和多 GPU 支持
- `data_hh.py`、`device_utils.py`、所有 `run_*.py` 和 `train_*.py`：全新编写

所有对已有代码的修改均在文件顶部注释中详细说明，方便对照原始实现理解改动点。
