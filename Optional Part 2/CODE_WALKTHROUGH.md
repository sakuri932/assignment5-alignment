# CS336 Assignment 5 Optional Part 2：代码详解

> 本文档逐文件解释 Optional Part 2 所有 Python 实现的工作原理，帮助理解每段代码对应 PDF 理论的哪一步，以及关键实现决策背后的原因。

---

## 目录

1. [device_utils.py — 跨平台设备工具](#1-device_utilspy--跨平台设备工具)
2. [sft_dataset.py — SFT 打包数据集](#2-sft_datasetpy--sft-打包数据集)
3. [dpo_hf.py — DPO 损失（HuggingFace 版）](#3-dpo_hfpy--dpo-损失huggingface-版)
4. [data_hh.py — Anthropic HH 数据集加载](#4-data_hhpy--anthropic-hh-数据集加载)
5. [run_mmlu.py — MMLU 零样本评估](#5-run_mmlupy--mmlu-零样本评估)
6. [run_gsm8k.py — GSM8K 零样本评估](#6-run_gsm8kpy--gsm8k-零样本评估)
7. [run_alpaca.py — AlpacaEval 输出生成](#7-run_alpacapy--alpacaeval-输出生成)
8. [run_sst.py — SimpleSafetyTests 输出生成](#8-run_sstpy--simplesafetytests-输出生成)
9. [train_sft.py — 指令微调训练](#9-train_sftpy--指令微调训练)
10. [train_dpo.py — DPO 偏好对齐训练](#10-train_dpopy--dpo-偏好对齐训练)

---

## 1 device_utils.py — 跨平台设备工具

本文件是整个 Optional Part 2 的设备抽象层，屏蔽 CUDA / Apple MPS / CPU 的差异，使所有脚本无需修改即可在不同硬件上运行。

### `get_device(device_override=None)`

**优先级**：`cuda > mps（Apple Silicon）> cpu`。

若指定 `device_override`（如命令行传入 `--device cuda:1`），直接返回指定设备，不做自动检测。这对多 GPU 训练（DPO 需要两块 GPU）非常重要。

### `get_compute_dtype(device)`

| 设备 | 数据类型 | 原因 |
|------|----------|------|
| CUDA / MPS | `bfloat16` | 与 FP32 同数值范围，无需 loss scaling，节省显存 |
| CPU | `float32` | CPU 缺少 bfloat16 的低精度计算优化 |

**关键细节**：`bfloat16` 相比 `float16` 拥有更大的指数范围（8 位 vs 5 位），因此在语言模型训练中更稳定，无需使用 `GradScaler`。

### `get_attn_impl(device)`

FlashAttention-2 仅支持 NVIDIA CUDA，因此 MPS 和 CPU 回退到 `eager`（PyTorch 标准实现）：

```python
return "flash_attention_2" if device.type == "cuda" else "eager"
```

### `load_model_and_tokenizer(model_path, device, *, eval_mode=True)`

统一的模型加载入口，自动处理：
- `torch_dtype`：通过 `get_compute_dtype` 选择精度
- `attn_implementation`：通过 `get_attn_impl` 选择注意力实现
- `pad_token`：若 tokenizer 没有 pad token，设为 eos token（防止 padding 报错）
- `eval_mode`：训练时传 `False`，保留 dropout；推理时传 `True`（默认），调用 `model.eval()`

---

## 2 sft_dataset.py — SFT 打包数据集

从 `cs336_alignment/utils.py` 的 `PackedSFTDataset` 复制并修改，主要增加了对压缩文件和不同字段名的支持。

### 关键修改点

**修改 1：支持 gzip 压缩文件**

服务器上的训练数据为 `.jsonl.gz` 格式。通过检测文件后缀自动选择打开方式：

```python
def _open_maybe_gz(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")
```

**修改 2：字段名自适应**

主作业数据用 `prompt`/`response`，UltraChat-200K 用 `instruction`/`output`。按优先级依次尝试：

```python
def _extract_prompt_response(doc):
    if "prompt" in doc and "response" in doc:
        return doc["prompt"], doc["response"]
    if "instruction" in doc and "output" in doc:
        return doc["instruction"], doc["output"]
    raise KeyError(...)
```

**修改 3：Windows 兼容**

Windows 不支持 `fork` 多进程，DataLoader 的 `num_workers` 必须为 0：

```python
num_workers = 0 if platform.system() == "Windows" else 2
```

### `PackedSFTDataset.__init__` 核心流程

```
读取所有文档 → 可选随机打乱 → 用 Alpaca 模板格式化 → tokenize
→ 拼接所有 token 为长序列 T
→ 以步长 seq_length 切分为不重叠窗口
→ 每个窗口：input_ids = T[i*L : i*L+L]，labels = T[i*L+1 : i*L+L+1]
```

**为什么打包（packing）**：直接用变长序列会引入大量 padding token，浪费 GPU 计算。打包将所有文档拼接后切等长块，padding 率趋近于 0，大幅提升吞吐量。

**EOS token 的处理**：每个文档末尾若没有 EOS token，自动追加，防止相邻文档的 token 被模型当成同一文档处理。

---

## 3 dpo_hf.py — DPO 损失（HuggingFace 版）

从 `cs336_alignment/dpo.py` 复制并修改，核心差异是将基于 `cs336_basics` 自定义模型的 `get_response_log_probs` 替换为直接调用 HuggingFace API。

### 核心数学：无条件对数概率技巧

DPO 损失需要计算条件对数概率 $\log \pi(y|x)$，但由于 prompt 部分在 chosen 和 rejected 的差值中相互抵消：

$$
\log \pi(y_w|x) - \log \pi(y_l|x) = \log \pi(x \oplus y_w) - \log \pi(x \oplus y_l)
$$

因此只需计算**完整文本**（prompt + response + EOS）的无条件对数概率之和，无需单独的 response mask。

### `_unconditional_log_prob_sum(model, text, tokenizer, device)`

```python
ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
logits = model(ids).logits            # (1, L, vocab_size)
log_probs = F.log_softmax(logits, -1) # (1, L, vocab_size)
target = ids[:, 1:]                   # 目标 token：第 1..L-1 位
# log_probs[:, :-1, :] 对应位置 0..L-2 的预测（标准语言模型偏移）
token_lp = log_probs[:, :-1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
return token_lp.sum()
```

**`.gather(-1, target.unsqueeze(-1))`**：从每个位置的词表分布中取出实际 token 对应的对数概率，这是从 logits 到 per-token log prob 的标准写法。

### `compute_per_instance_dpo_loss`

**关键设计：`device_lm` / `device_ref` 参数**

DPO 需要同时运行两个完整的 LLM（训练模型 + 参考模型），单块 GPU 显存不足。通过显式的设备参数，支持两个模型分别放在不同 GPU 上：

```python
lp_ref_chosen = lp_ref_chosen.to(device_lm)   # 参考模型的值移到训练模型所在设备
lp_ref_rejected = lp_ref_rejected.to(device_lm)
```

**参考模型用 `torch.no_grad()`**：参考模型在整个 DPO 训练中保持冻结，不需要梯度：

```python
with torch.no_grad():
    lp_ref_chosen = _unconditional_log_prob_sum(lm_ref, ...)
    lp_ref_rejected = _unconditional_log_prob_sum(lm_ref, ...)
# 训练模型：保留梯度
lp_chosen = _unconditional_log_prob_sum(lm, ...)
```

**最终损失公式**（对应 PDF 等式 3）：

```python
log_ratio_chosen   = lp_chosen   - lp_ref_chosen
log_ratio_rejected = lp_rejected - lp_ref_rejected
loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
```

### `compute_dpo_reward_accuracy`

用于验证集评估，判断隐式奖励模型是否正确分类（chosen 的 log ratio 是否大于 rejected 的 log ratio）：

```python
return ratio_chosen > ratio_rejected
```

全程 `torch.no_grad()`，比 `compute_per_instance_dpo_loss` 更快（无需 autograd 图）。

---

## 4 data_hh.py — Anthropic HH 数据集加载

### 数据格式

每行 JSON 含 `chosen` 和 `rejected` 两个字段，均为多轮对话字符串，格式为：

```
\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ...
```

### `_parse_conversation(text)`

使用正则表达式提取所有 `(Human消息, Assistant消息)` 轮次对：

```python
_HH_TURN_RE = re.compile(
    r"\n\nHuman:\s*(.*?)\n\nAssistant:\s*(.*?)(?=\n\nHuman:|\Z)",
    re.DOTALL,
)
```

**`(?=\n\nHuman:|\Z)`**：非捕获前瞻断言（lookahead），匹配到下一个 Human 开始或字符串结束为止，确保每个 Assistant 消息不越界截断。

### 过滤多轮对话

PDF 要求只保留单轮对话（Human 只说了一次话）：

```python
if not (_is_single_turn(chosen_turns) and _is_single_turn(rejected_turns)):
    continue
```

**原因**：多轮对话中，两条链的分歧点可能不在第一轮，chosen/rejected 的差异更难界定，DPO 训练时信号会变噪。

### `HHExample` 数据类

```python
@dataclass
class HHExample:
    instruction: str        # 第一条人类消息（纯文本，去掉 "Human:" 前缀）
    response_chosen: str    # 偏好响应
    response_rejected: str  # 被拒响应
    source_file: str        # 来源文件（用于后续分析各子集）
```

`source_file` 字段记录数据来自 `harmless-base`、`helpful-base`、`helpful-online` 还是 `helpful-rejection-sampled`，方便后续分析不同子集对 DPO 的贡献。

### `split_train_val`

使用固定随机种子（42）打乱后切割，保证实验可复现：

```python
rng = random.Random(seed)
shuffled = examples[:]
rng.shuffle(shuffled)
val = shuffled[:val_size]
train = shuffled[val_size:]
```

**不直接用 `random.shuffle`**：`Random(seed)` 创建独立的随机数生成器，不影响全局随机状态，防止与其他代码的随机操作相互干扰。

---

## 5 run_mmlu.py — MMLU 零样本评估

### Prompt 构造

MMLU 要求模型以 `"The correct answer is _"` 格式回答，prompt 末尾追加 `Answer:` 引导输出：

```
Answer the following multiple choice question about {subject}. ...

Question: {question}
A. {opt_a}   B. {opt_b}   C. {opt_c}   D. {opt_d}
Answer:
```

这个 query 再包裹进系统 prompt 的 `{instruction}` 槽位中，形成完整输入。

### 双后端生成

| 条件 | 后端 | 优势 |
|------|------|------|
| `_VLLM_AVAILABLE and device.type == "cuda"` | vLLM | 批量推理，吞吐量高 |
| 否则 | HuggingFace generate | 跨平台，macOS/CPU 可用 |

**vLLM 使用 `stop=_STOP_STR`**：当模型生成 `"# Query:"` 时立即停止，因为系统 prompt 的模板会在答案后开始新的查询轮次。

**HF 后端的输出截断**：由于 HF `generate` 不支持 stop string，事后手动截断：

```python
if _STOP_STR in text:
    text = text[:text.index(_STOP_STR)]
```

### 评分流程

调用 `cs336_alignment.utils.parse_mmlu_response` 将模型输出解析为字母（A/B/C/D 或 `None`），与 `ex["answer"]` 对比。输出文件同时保存每条样例的原始输出和解析结果，便于后续错误分析。

---

## 6 run_gsm8k.py — GSM8K 零样本评估

### Gold Answer 解析

GSM8K 数据集中，金标准答案嵌入在 `answer` 字段的 `#### N` 格式中：

```python
_GOLD_ANS_RE = re.compile(r"####\s*([\d,]+)")

def _parse_gold(answer_str):
    m = _GOLD_ANS_RE.search(answer_str)
    if m:
        return m.group(1).replace(",", "")  # 去掉千位分隔符
    return None
```

**与 MMLU 的区别**：GSM8K 需要从数据集 `answer` 字段中提取金标准数字，MMLU 直接用 `ex["answer"]`（已是单字母）。

### 模型输出解析

调用 `cs336_alignment.utils.parse_gsm8k_response`，取生成文本中的**最后一个数字**作为预测值。GSM8K 的典型推理过程会在结尾给出最终答案，因此取最后一个数字是有效的启发式规则。

### `max_new_tokens=256`（比 MMLU 的 64 更大）

GSM8K 需要思维链推理，模型通常先写出步骤再给出答案，所以需要更多 token 的生成空间。

---

## 7 run_alpaca.py — AlpacaEval 输出生成

### 输出格式要求

AlpacaEval 评估器要求 JSON **数组**格式（非 JSONL），每条记录含固定字段：

```json
[
  {
    "instruction": "...",
    "output": "...",
    "generator": "llama-3.1-8b-base",
    "dataset": "helpful_base"
  },
  ...
]
```

`generator` 字段标识生成模型，同一数组中所有记录必须一致。通过 `--generator` 命令行参数传入，方便区分 baseline / SFT / DPO 模型的输出。

`dataset` 字段直接从原始 AlpacaEval 数据中继承（`ex.get("dataset", "")`），AlpacaEval 用它追踪指令来自哪个子集。

### `max_new_tokens=512`

AlpacaEval 的指令是开放式问答，模型回答通常比 MMLU/GSM8K 更长，因此设为 512。

---

## 8 run_sst.py — SimpleSafetyTests 输出生成

### CSV 数据加载

SimpleSafetyTests 数据以 CSV 格式提供（而非 JSONL），使用 `csv.DictReader` 读取，每行转为字典：

```python
with open(data_path, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        examples.append(dict(row))
```

**`newline=""`**：Python 官方文档推荐在打开 CSV 文件时指定 `newline=""`，防止 `csv` 模块与通用换行符处理冲突。

### 输出格式

与 AlpacaEval 不同，SST 要求 **JSONL** 格式（每行一个 JSON 对象），且必须保留 `prompts_final` 字段（评估脚本用它匹配原始指令）：

```python
for ex, output in zip(examples, outputs):
    record = {**ex, "output": output}   # 保留所有原始字段（id, harm_area 等）
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

`{**ex, "output": output}` 将原始 CSV 行的所有字段（`id`、`harm_area`、`counter`、`category`、`prompts_final`）一并保存，供后续分析使用。

---

## 9 train_sft.py — 指令微调训练

### 学习率调度：线性预热 + 余弦衰减

PDF 推荐：线性预热占总步数 3%，之后余弦衰减至峰值的 10%。

```python
def get_cosine_with_warmup_lr_lambda(current_step, *, warmup_steps, total_steps, min_lr_ratio=0.1):
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)       # 线性预热
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress)) # 余弦衰减
    return max(min_lr_ratio, cosine)
```

使用 `LambdaLR` 包装，将函数作为乘子应用到 `optimizer` 的 base lr 上。

### 梯度累积

PDF Section 3.2.2 明确要求有效批大小 32，但单卡只能跑批大小 2，因此梯度累积步数默认 16（`2 × 16 = 32`）：

```python
loss = F.cross_entropy(...) / args.grad_accum   # 除以累积步数取平均
loss.backward()

if global_step % args.grad_accum == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

**为什么要 `/= grad_accum`**：每次 `backward` 会把梯度**累加**到 `.grad` 属性上，若不除以步数，相当于把批大小缩放了 `grad_accum` 倍，损失尺度会偏大。

**梯度裁剪**：`clip_grad_norm_(..., 1.0)` 在 optimizer step 之前执行，防止梯度爆炸。

### 验证损失计算

`evaluate_val_loss` 使用 `@torch.no_grad()` 装饰器，进入 `model.eval()` 模式后计算最多 100 个批次的平均损失，之后恢复 `model.train()`：

```python
model.eval()
...
model.train()
return total_loss / n_batches
```

### 模型保存策略

- `output_dir/best/`：验证损失最低时保存（训练过程中可能多次更新）
- `output_dir/final/`：训练全部完成后保存最终权重

同时保存 tokenizer，使模型目录自包含，后续推理无需单独指定 tokenizer 路径。

---

## 10 train_dpo.py — DPO 偏好对齐训练

### 双模型双设备架构

DPO 同时需要训练模型 `lm` 和冻结参考模型 `lm_ref`，两个完整 8B 模型无法放在同一块 GPU 上。推荐配置：

```
device-lm  = cuda:0  →  lm（训练，保留梯度）
device-ref = cuda:1  →  lm_ref（冻结，no_grad）
```

`load_model_for_dpo` 根据 `freeze` 参数决定是否关闭梯度：

```python
if freeze:
    for p in model.parameters():
        p.requires_grad_(False)
```

### RMSprop 而非 AdamW

原始 DPO 论文使用 RMSprop，原因是 AdamW 需要维护一阶和二阶动量，显存占用约为参数量的 3 倍。RMSprop 只维护二阶动量，显存压力更小，更适合显存紧张的 DPO 场景：

```python
optimizer = torch.optim.RMSprop(lm.parameters(), lr=args.lr)
```

### 逐样本梯度累积

DPO 损失的计算单位是单条偏好数据（一个 chosen + 一个 rejected），无法像 SFT 那样批量矩阵运算，因此采用逐条处理 + 梯度累积模拟大批次：

```python
for idx, ex in enumerate(epoch_data):
    loss = compute_per_instance_dpo_loss(...) / args.batch_size
    loss.backward()

    if (idx + 1) % args.batch_size == 0:
        optimizer.step()
        optimizer.zero_grad()
```

`args.batch_size` 在这里实际上是**梯度累积步数**，有效批大小即为该值（每条样本独立产生梯度后累积）。

### 验证：奖励分类准确率

DPO 的隐式奖励模型对一条数据分类正确，当且仅当：

$$
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} > \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

`compute_dpo_reward_accuracy` 全程 `torch.no_grad()`，比训练时快约 2 倍（无需构建 autograd 图）。

**保存最优模型**：每隔 `val_interval` 步验证一次，准确率超过历史最优时保存 `output_dir/best/`，确保最终用于下游评估的是泛化最好的 checkpoint，而非最后一步的 checkpoint（可能已经轻微过拟合）。

### 跨设备张量移动

参考模型输出的 log prob 在 `device_ref` 上，需要移到 `device_lm` 才能与训练模型的 log prob 相减：

```python
lp_ref_chosen = lp_ref_chosen.to(device_lm)
```

这一步发生在 `dpo_hf.py` 内部（`compute_per_instance_dpo_loss`），对调用方透明。跨设备的张量传输由 PCIe 总线完成，在 B200 等高带宽 GPU 上延迟极低。
