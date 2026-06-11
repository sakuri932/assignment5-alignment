# CS336 Assignment 5: 语言模型对齐 — 工作报告

> **完成日期**：2026-05-24  
> **测试结果**：33 passed（19 GRPO + 7 SFT + 2 data + 4 metrics + 1 DPO，全部通过）

---

## 目录

1. [作业概述](#1-作业概述)
2. [实现过程](#2-实现过程)
3. [提示工程基线评测（Problem `prompting_baselines`）](#3-提示工程基线评测problem-prompting_baselines)
4. [理论问题解答（书面交付物）](#4-理论问题解答书面交付物)
5. [遇到的问题与调试过程](#5-遇到的问题与调试过程)
6. [测试结果](#6-测试结果)
7. [文件结构](#7-文件结构)
8. [GPU 实验题说明](#8-gpu-实验题说明)

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

> **问题原文：**（data_loading，3 分）
>
> 实现一个 PyTorch `Dataset` 子类，为指令微调生成样例。该 `Dataset` 应具有以下接口：`def __init__(self, tokenizer, dataset_path, seq_length, shuffle)` 构造数据集；`def __len__(self)` 返回序列数量（将所有文档 token 拼接后按 `seq_length` 切分，丢弃不足一块的尾部）；`def __getitem__(self, i)` 返回包含 `input_ids` 和 `labels` 的字典（形状均为 `(seq_length,)`）。实现适配器 `[adapters.get_packed_sft_dataset]`，运行 `uv run pytest -k test_packed_sft_dataset`。

打包数据集的核心是将所有文档的 token 拼接成一条超长序列，再用步长等于 `seq_length` 的滑动窗口切分。每个文档末尾追加 EOS token，保证窗口横跨文档边界时模型能识别边界。

关键细节：用 `text.rstrip()` 去掉 Alpaca 模板末尾的换行符，防止 BPE 将换行和下一文档的第一个字节合并，影响 EOS 位置检测。

#### `sft_microbatch_train_step`（`sft.py`）

> **问题原文：**（sft_script，4 分）
>
> 编写一个训练循环脚本，在提供的指令微调数据上对 Llama 3.1 8B Base 模型进行微调。脚本至少应支持：配置和控制各种模型和优化器超参数；通过梯度累积支持超出显存限制的更大批大小；定期记录训练和验证性能（例如输出到控制台和/或 Weights and Biases 等外部服务）。`sft_microbatch_train_step` 封装了单个微批次的前向传播与 masked 损失计算（用 `masked_normalize` 只对 response token 计算平均负对数似然损失，内部调用 `.backward()`），是训练循环的核心组件。

用 `masked_normalize` 计算 response token 的平均负对数似然损失，内部直接调用 `loss.backward()`。损失除以 `normalize_constant × gradient_accumulation_steps × microbatch_size`，确保梯度与真正大批次等价。

### 2.2 GRPO 部分

实现顺序：分词 → log 概率 → 奖励 → 归一化 → loss → 聚合 → 训练步。

#### `tokenize_prompt_and_output`

> **问题原文：**（tokenize_prompt_and_output，1 分）
>
> 实现方法 `tokenize_prompt_and_output`，将 prompt 和 output 分别 tokenize（不添加特殊 token），拼接后构建与 labels 对齐的 `response_mask`（response token 处为 1，prompt 和 padding 处为 0）。返回字典包含：`input_ids`（形状 `(batch_size, max_len - 1)`，去掉末尾 token）、`labels`（形状 `(batch_size, max_len - 1)`，即 full_ids 左移一位）、`response_mask`（形状同上）。实现适配器 `[adapters.run_tokenize_prompt_and_output]`，运行 `uv run pytest -k test_tokenize_prompt_and_output`。

关键是"先 padding 完整序列，再截取 input_ids/labels"，避免 mask 边界位置错误。`response_mask` 的起始位置是 `prompt_len - 1`（labels 中第一个 response token），结束位置是 `seq_len - 2`（labels 中最后一个真实 token，即 EOS）。

#### `compute_group_normalized_rewards`

> **问题原文：**（compute_group_normalized_rewards_grpo，1 分）
>
> 实现方法 `compute_group_normalized_rewards`，在组内对原始奖励进行归一化，返回归一化后的优势、原始奖励及日志元数据。当前只需支持 `baseline = "mean"`（减去组内均值）和 `advantage_normalizer = "std"`（除以组内标准差），对不支持的输入可抛出 `NotImplementedError`。注意在归一化分母上加 `advantage_eps` 防止除零。实现适配器 `[adapters.run_compute_group_normalized_rewards]`，运行 `uv run pytest -k compute_group_normalized_rewards_grpo`。

将 `(N,)` 奖励 reshape 为 `(n_groups, G)`，用广播机制批量计算各组的均值/标准差。通过 `baseline` 和 `advantage_normalizer` 参数切换四种算法变体（GRPO/Dr.GRPO/MaxRL/RFT），避免重复代码。

#### `compute_policy_gradient_loss`

> **问题原文：**（compute_policy_gradient_loss_on_policy，1 分）
>
> 实现方法 `compute_policy_gradient_loss`，计算每个 token 的策略梯度损失，`raw_rewards_or_advantages` 可以是原始奖励或预先计算好的优势。当前只需支持 `importance_reweighting_method = "none"`（在策略模式），不需要 `old_log_prob` 和 `cliprange` 参数，对不支持的输入可抛出 `NotImplementedError`。返回元组 `(per_token_policy_gradient_loss, metadata)`。实现适配器 `[adapters.run_compute_policy_gradient_loss]`，运行 `uv run pytest -k test_compute_policy_gradient_loss_on_policy`。

实现了四种重要性重加权方式。GSPO 的关键在于：用 `response_mask` 只对 response token 计算几何均值对数比率，然后 `expand_as` 展开到 `(B, L)`，让梯度通过每个 token 的 `log_ratio` 自然引入 `1/L` 因子。

#### `grpo_train_step`

> **问题原文：**（grpo_train_step_standard_on_policy，5 分）
>
> 实现单批次策略梯度更新函数 `grpo_train_step`，输入包括模型、tokenizer、optimizer、梯度累积步数、奖励函数、repeated prompts/rollouts/ground truths 及 group_size 等参数。当前只需支持标准在策略 GRPO（`baseline = "mean"`、`advantage_normalizer = "std"`、`importance_reweighting_method = "none"`、`loss_normalization = "sequence"`）。函数需实现梯度累积：将 rollout batch 切分为若干 microbatch，逐 microbatch 前向+反向传播，最后调用 optimizer.step() 和 optimizer.zero_grad()。optimizer.step() 前按 `max_grad_norm` 裁剪梯度范数。返回 `(loss, metadata)`，metadata 中至少包含 loss、梯度范数、token 熵、训练奖励（total/format）。实现适配器 `[adapters.run_grpo_train_step]`，运行 `uv run pytest -k test_grpo_train_step_standard_on_policy`。

梯度累积时，sequence 规范化需对每个 microbatch 的 loss 额外除以 `gradient_accumulation_steps`；constant 规范化因分母已包含全 batch token 数，无需此操作。训练步结束时调用 `zero_grad(set_to_none=True)` 释放显存。

### 2.3 DPO 部分

#### `_response_log_prob_sum`

> **问题原文：**（dpo_loss，2 分）
>
> 编写一个函数，计算逐实例 DPO 损失。函数接收两个语言模型（$\pi_\theta$ 和 $\pi_{\text{ref}}$）和两个字符串（preferred 响应 $y_w$ 和 rejected 响应 $y_l$），使用 Alpaca 模板格式化 prompt 和响应，并在响应后追加"序列结束"token。可利用以下简化：计算条件对数概率之差等价于计算**无条件对数概率**之差（因为 prompt 部分的对数概率抵消），无需额外的 mask 操作。实现适配器 `[adapters.per_instance_dpo]`，运行 `uv run pytest -k test_per_instance_dpo_loss`。

用 Alpaca 模板构建前缀，分别编码 prompt 前缀和 response，再手动追加 EOS token ID（整数，非字符串）。构建 input_ids/labels/mask 后调用 `get_response_log_probs` 并对 response 位置的 log 概率求和。

#### `compute_per_instance_dpo_loss`

`lm_ref` 在 `torch.no_grad()` 上下文中推理，`lm` 正常前向传播保留梯度。计算两个 log ratio 之差后用 `F.logsigmoid` 计算损失（数值稳定）。

---

## 3. 提示工程基线评测（Problem `prompting_baselines`）

### 3.1 题目原文与实验设置

> **问题原文：**
>
> **Problem（`prompting_baselines`）：在 GSM8K 上评测 OLMo-2-0425-1B（5 分）**
>
> **(a)** 编写脚本评测 OLMo-2-0425-1B 在 GSM8K 上使用零样本 `question_only`、零样本 `r1_zero` 以及少样本 `r1_zero_three_shot` 提示的性能。
>
> 运行脚本并观察输出。对每种提示，统计模型生成中有多少落入以下类别：
> - (1) 格式奖励 1 且准确性奖励 1（格式正确且回答正确）
> - (2) 格式奖励 1 且准确性奖励 0（格式正确但回答错误）
> - (3) 格式奖励 0 且准确性奖励 0（格式错误且回答错误）
>
> 观察至少 10 个类别 (2) 的例子，有多少模型输出实际上答案正确但解析失败？类别 (3) 呢？
>
> **(b)** 观察模型输出，描述模型在每种提示下的行为特征。

**实验设置：**

| 项目 | 取值 |
|------|------|
| 模型 | `allenai/OLMo-2-0425-1B`（本地路径 `/Users/admin/Documents/GitHub/CS336/OLMo-2-0425-1B`） |
| 设备 | Apple M 系列 MPS，bf16 |
| 数据 | GSM8K test，按种子 0 抽样 50 条 |
| 生成超参 | 温度 1.0，top-p 1.0，max_new_tokens = 512 |
| 停止字符串 | `r1_zero` 系列：`</answer>`；`question_only`：无 |
| 评分函数 | `r1_zero` 系列 → `r1_zero_reward_fn`；`question_only` → `question_only_reward_fn` |

脚本：[scripts/eval_prompting_baselines.py](../scripts/eval_prompting_baselines.py)
原始结果：[extra_guidance/prompting_baselines_results.json](prompting_baselines_results.json)（含全部 150 条 response 和分类）

**MPS 内存优化备注**：脚本里通过 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85`、`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5` 把内存池上限压在物理内存 85% 内，并在每条推理后 `torch.mps.empty_cache()`，避免 KV cache 累积导致 swap。同步过程中也修复了 [cs336_alignment/checkpoint.py](../cs336_alignment/checkpoint.py) 的跨平台 bug——原版在 MPS 上强制 `flash_attention_2` 会直接 ImportError，改为 CUDA + 装了 flash_attn 才用它，否则回退到 `sdpa`。

### 3.2 题目(a)：三种提示的类别统计

| 提示 | 类别 (1) 正确 | 类别 (2) 格式对+答错 | 类别 (3) 格式错+答错 | 准确率 | 格式遵循率 | 用时 |
|------|------|------|------|------|----------|-----|
| `question_only` | 0 | 15 | 35 | **0.0%** | 30.0% | 293s |
| `r1_zero` | 0 | 32 | 18 | **0.0%** | 64.0% | 141s |
| `r1_zero_three_shot` | **12** | 37 | 1 | **24.0%** | **98.0%** | 186s |

**关键结论：少样本（few-shot）是真正的拐点。** OLMo-2-0425-1B 是基础模型未经任何指令调优，零样本提示下完全无法在 GSM8K 上正确解题（无论是 `\boxed{}` 还是 `<think>/<answer>` 格式）；但只要在提示里给出 3 个示范，准确率立刻跳到 24%、格式几乎完美。这与 README 中"OLMo-2-0425-1B 官方 benchmark 用 8-shot 提示"的描述一致。

### 3.3 题目(a)：实际答案正确但解析失败的占比

| 提示 | 类别 (2) 总数 | 其中"答对但解析失败"数 |
|------|------|------|
| `question_only` | 15 | **0** |
| `r1_zero` | 32 | **1** |
| `r1_zero_three_shot` | 37 | **1** |

**分析方法：** 对每个类别 (2) 样例，提取 `<answer>...</answer>`（或 `\boxed{...}`）内最后出现的数值，与 ground truth 比对。若数值一致但 reward 为 0，则视为解析失败。

**`r1_zero` 漏判 1 例**：John 香蕉题，GT=22。模型在 `<answer>` 内写了完整推理"On Wednesday, John picks 4 bananas. ... Therefore, John has 4 + 6 + 12 = 22 bananas in total."——最终数字确实是 22，但 `grade()` 把整段文字当作答案表达式比对，"22 bananas in total"未匹配上裸数字 22。

**`r1_zero_three_shot` 漏判 1 例**：Daisy 体重题，GT=7。模型写`<answer> 7 pounds </answer>`——带单位的 "7 pounds" 同样被 grader 判错。

**类别 (3) 没有"答对"案例**：手工抽查 question_only 的 5 个 (3) 类样本和 r1_zero 的 5 个 (3) 类样本，全部确属格式错误**且**推理错误——大多数情况下模型根本没在解题（重新提了新问题、向用户求助、或输出无关补全），少量在生成过程中误用标签（出现两次 `</think>`、把答案放在 `<answer>` 之外等）。

### 3.4 题目(a)：类别 (2) 和类别 (3) 样例展示

**`question_only` 类别 (2)（格式对但答错）：**

> Q: John picks 4 bananas on Wednesday. Then he picks 6 bananas on Thursday. On Friday, he picks triple ... How many bananas does John have? (GT = **22**)
>
> Response: `...The reason why we don't have to write the parentheses ( ) is that the problem states that on Wednesday he picks 4 bananas, ... \boxed{4}`
>
> 模型输出了 `\boxed{4}` —— 解析成功但只识别了 Wednesday 的 4，没做加法。

**`question_only` 类别 (3)（格式错+答错）：**

> Q: Miguel uses 2 pads of paper a week for his drawing. ... how many sheets ... every month? (GT = **240**)
>
> Response: *（空字符串）*
>
> 模型直接什么都没生成。

> Q: Comet Halley orbits the sun every 75 years. Bill's dad saw the Comet when he was 30 years old. ... (GT = **15**)
>
> Response: `Q: 5 years ago you were half of your mother's age. If your mom is 8 times younger than you are now, how old was your mom 5 years ago?  A: 8 * (4 - 5) = 8 * 1 = 64`
>
> 模型直接重新提了一个新问题。

**`r1_zero` 类别 (2)（格式对但答错）：**

> Q: Amber, Micah, and Ahito ran 52 miles in total. Amber ran 8 miles. Micah ran 3.5 times what Amber ran. How many miles did **Ahito** run? (GT = **16**)
>
> Response: `8 * 3.5 = 28. So, Ahito ran 28 miles. </think> <answer> Ahito ran 28 miles. </answer>`
>
> 模型把"Ahito"当成了 Micah，算的是 Micah 跑的距离，没意识到 52 - 8 - 28 = 16 才是 Ahito。

> Q: Comet Halley ... Bill saw the comet a second time when he was three times older than ... (GT = **15**)
>
> Response: `75 / 3 = 25 </think> <answer> 3 * 30 = 90 </answer>`
>
> `<think>` 里写 25，`<answer>` 里写 90，内部前后不一致。

**`r1_zero` 类别 (3)（格式错+答错）：**

> Q: 同 Peyton 题。Response: `On Monday, her yoga takes <answer> 0 </answer>`
>
> 模型在没完成思考时就直接闭合 `<answer>`，缺少 `</think>` 标签，整段被判格式错误。

> Q: Tom 树木题。Response: `... 70% of 50 trees is...` (后续略)。
>
> 模型用了 `</change>` 这种非法标签代替 `</think>`，格式不符。

**`r1_zero_three_shot` 类别 (2)（格式对但答错）：**

> Q: At a certain grade level, three-fourths of students have a desktop computer at home. If 20 students do not have desktop computers, how many students are in this grade? (GT = **80**)
>
> Response: `Three-fourths of students have a desktop computer at home. That means two-fourths do not. If 20 students do not have the computer, then 20 x 2 = 40 do have it. In total, that's 40 + 20 = 60 students. So the answer is 60. </think> <answer> 60 </answer>`
>
> 模型把 1/4 误算成 2/4，思路结构对但分数被算错。

### 3.5 题目(b)：三种提示下的模型行为特征

**`question_only`（30% 格式遵循，0% 正确）**

模型大量"不在解题"：
- 重新提出无关问题（如上述 Comet 例）；
- 向用户求助（"Would you please help me with the following problem?"）；
- 输出无关补全或纯空白响应；
- 即使勉强输出 `\boxed{}`，常常只填入题目里第一个出现的数字。

基础模型缺乏指令跟随能力，仅凭一句"please put your final answer within \boxed{}"无法让它进入解题状态。

**`r1_zero`（64% 格式遵循，0% 正确）**

`<think>/<answer>` 的双标签对生成有强约束力——格式遵循率从 30% 跳到 64%。模型表面上"装出"了 R1-Zero 风格的推理：先输出一段 `<think>`，再给出 `<answer>`。但仔细看常见错误模式：

- **阅读理解错位**：把目标主语认错（Ahito vs Micah）；
- **`<think>` 与 `<answer>` 内部不一致**：思考算出 25，答案写 90；
- **错过题目隐含条件**：忽略"剩下的是 Ahito 跑的"、忽略"4 周/月"的常识；
- **标签误用**：少数样例用 `</change>`、双重 `</think>` 等非法标签。

简言之，模型形似而非神似——能套住外壳，但推理本身仍然乱。这正好对应了 GRPO 训练的起点状态：格式信号已经能学到（reward 频次不低），但需要 RL 把 0% 的准确率拉高。

**`r1_zero_three_shot`（98% 格式遵循，24% 正确）**

加了 3 个示范后，模型从"形似 R1"变成"真在做 R1"：

- 几乎 100% 严格按 `<think> ... </think> <answer> ... </answer>` 输出；
- 对简单算术分解题（Miguel 纸张、John 香蕉、Tom 树木的简化版本）能正确推理；
- 失败集中在多步分解或主语错位（Amber/Micah/Ahito 题对所有提示都难住）；
- 偶尔在 `<answer>` 里写"7 pounds"这种带单位输出，被严格 grader 判错——这是少样本激发的"完整描述"习惯与 r1 grader"只看裸表达式"之间的不匹配。

少样本既约束了格式，又通过示范激活了基础模型在预训练里已经具备的算术推理能力，是远比改提示词更有效的零成本提升手段。

---

## 4. 理论问题解答（书面交付物）

本节回答 PDF 中所有**只需书面推导/讨论**的理论题（不含需要 GPU 训练的实验题，后者见第 8 节）。

### 4.1 Problem `baseline_calcs`：策略梯度估计器的方差（5 分）

> **问题原文：** 设 $\pi_\theta$ 在二元动作空间 $\mathcal{A}=\{0,1\}$ 上定义策略，$\pi_\theta(A=1)=p=\sigma(\theta)$，$\sigma(\theta)=\frac{1}{1+e^{-\theta}}$；奖励 $r(A)=\mathbb{1}\{A=1\}$。
> **(a)** 估计器 $\frac{1}{n}\sum_{i=1}^n r(A_i)\nabla_\theta\log\pi_\theta(A_i)$（$A_i\overset{\text{iid}}{\sim}\pi_\theta$）的方差是多少？
> **(b)** 带基线估计器 $\frac{1}{n}\sum_{i=1}^n (r(A_i)-b)\nabla_\theta\log\pi_\theta(A_i)$ 的方差是多少？
> **(c)** 代入"总体均值"基线 $b=p$ 后方差是多少？与无基线估计器相比，它总是更低、总是更高、还是取决于 $p$？

**预备：得分函数。** 由 $\nabla_\theta\log\sigma(\theta)=1-\sigma(\theta)$、$\nabla_\theta\log(1-\sigma(\theta))=-\sigma(\theta)$：

$$
\nabla_\theta\log\pi_\theta(A=1)=1-p,\qquad \nabla_\theta\log\pi_\theta(A=0)=-p.
$$

**(a)** 记单样本项 $Z=r(A)\nabla_\theta\log\pi_\theta(A)$。则 $A=1$（概率 $p$）时 $Z=1\cdot(1-p)=1-p$；$A=0$（概率 $1-p$）时 $Z=0$。于是

$$
E[Z]=p(1-p),\quad E[Z^2]=p(1-p)^2,\quad \operatorname{Var}(Z)=p(1-p)^2-p^2(1-p)^2=p(1-p)^3.
$$

$n$ 个独立样本求平均：

$$
\boxed{\operatorname{Var}_{(a)}=\frac{p(1-p)^3}{n}.}
$$

**(b)** 记 $Z_b=(r(A)-b)\nabla_\theta\log\pi_\theta(A)$。$A=1$ 时 $Z_b=(1-b)(1-p)$；$A=0$ 时 $Z_b=(0-b)(-p)=bp$。先验证期望不变：

$$
E[Z_b]=p(1-b)(1-p)+(1-p)\,bp=p(1-p)\big[(1-b)+b\big]=p(1-p),
$$

与 (a) 相同，印证基线保持期望。再算二阶矩并化简（关键：括号内恰好配成完全平方）：

$$
\operatorname{Var}(Z_b)=p(1-p)\big[(1-b)^2(1-p)+b^2p\big]-p^2(1-p)^2=p(1-p)\big[(1-p)-b\big]^2.
$$

$$
\boxed{\operatorname{Var}_{(b)}=\frac{p(1-p)\big[(1-p)-b\big]^2}{n}.}
$$

这是关于 $b$ 的抛物线，在 $b^*=1-p$ 处取得**零方差**（最优基线）。

**(c)** 代入 $b=p$：

$$
\boxed{\operatorname{Var}_{(c)}=\frac{p(1-p)(1-2p)^2}{n}.}
$$

与 (a) 之比为 $\dfrac{(1-2p)^2}{(1-p)^2}$。判断 $|1-2p|\le 1-p$：

- $p\le\frac12$：$1-2p\le1-p\iff -p\le0$，恒成立 → 方差更低；
- $\frac12<p\le\frac23$：$2p-1\le1-p\iff p\le\frac23$，成立 → 方差更低（$p=\frac23$ 时相等）；
- $p>\frac23$：$2p-1>1-p$ → **方差反而更高**（如 $p=0.9$：$0.64$ vs $0.01$）。

**结论**：$b=p$ **并非总是**降低方差。它在 $p\le\frac23$ 时降低方差，但 $p>\frac23$ 时反而升高。真正的最优基线是 $b=1-p$（恒为零方差），$b=p$ 只是次优近似。

---

### 4.2 Problem `think_about_length_normalization`：思考长度归一化（1 分）

> **问题原文：** 思考按序列长度归一化（每条序列除以其长度）与按同一常数归一化所有序列之间的区别。各自优缺点？是否有特定设置使某种更好？

- **序列长度归一化**：每条序列对总 loss 的贡献相等，与长度无关；长序列中每个 token 的梯度被 $1/\text{len}$ 稀释。优点是防止长序列主导更新。缺点是引入**长度偏置**——对于错误（负优势）的长 response，逐 token 惩罚被稀释，等于变相鼓励"写得更长以摊薄惩罚"，这正是 Dr. GRPO 指出的长度膨胀问题；且每 token 梯度幅度依赖序列长度，使优化与长度耦合。
- **常数归一化**：所有 token 等权，无逐序列长度偏置，更忠实于 token 级策略梯度。缺点是含较多长序列的批次会产生更大的总梯度，且需要选定归一化常数（如最大长度或固定值），loss 的尺度依赖该选择。
- **何者更好**：若关心去除长度偏置、避免错误答案被鼓励变长，常数归一化更优（Dr. GRPO 的论点）；若各样本长度差异极大且希望每个样本等权（按题目准确率而非 token 数衡量），序列归一化更合适。

---

### 4.3 Problem `think_about_rft`：思考 RFT（2 分）

> **问题原文：** RFT 目标（常数归一化，公式 35）的梯度 $\nabla_\theta J_\theta$，与在策略 Dr. GRPO 估计器（公式 36，组均值 $\mu$）相比。假设二元奖励，比较两者：期望相同吗？哪个方差更低？是否有更偏好其一的情形？

设二元奖励 $r\in\{0,1\}$，故 $\mathbb{1}\{r=1\}=r$。

- **RFT 梯度** $=\frac1Z\sum_x\sum_j r(y^{(j)})\,\nabla_\theta\log\pi_\theta(y^{(j)})$，这正是**无基线的 REINFORCE** 估计器。
- **Dr. GRPO** $=\frac1Z\sum_x\sum_j (r(y^{(j)})-\mu)\,\nabla_\theta\log\pi_\theta(y^{(j)})$，是**带组均值基线**的 REINFORCE。

二者相差一项 $-\frac1Z\sum_x\mu\sum_j\nabla_\theta\log\pi_\theta(y^{(j)})$。

**期望**：相同。组均值基线 $\mu$ 不依赖于被求导的那个 $y$（在 $G\to\infty$ 极限下趋于常数难度 $\eta(x)$），由 $E_y[\nabla_\theta\log\pi_\theta]=0$，减去它不改变期望（见公式 16、21–24 的 $\frac{G-1}{G}$ 缩放）。因此二者都是 $\nabla_\theta J_\theta$ 的无偏估计（二元奖励下）。

**方差**：Dr. GRPO 更低。基线减法正是 §4.1.5 的方差缩减技巧，组均值接近最优基线，能压低梯度估计的方差。

**偏好情形**：RFT 更简单——只是对正确样本做 SFT，无需把错误样本前向传播，更省显存、更快；且它从不对错误 response 施加负梯度，避免了破坏性的负更新，更稳定（代价是浪费了错误样本的信号）。当奖励稀疏、主要想模仿正确解时，RFT 是廉价稳定之选；当希望充分利用对比信号、追求更低方差时选 Dr. GRPO。

---

### 4.4 Problem `derive_difficulty_reweightings`：推导优势归一化诱导的难度重加权（6 分）

> **问题原文：** 代理目标形如 $\nabla_\theta J_{\theta,w}=\nabla_\theta E_{x\sim\rho}[w(x,\text{stopgrad}(\pi_\theta))\,E_{y\sim\pi_\theta}r(y\mid x)]$。令常数归一化 $Z=G$、组大小 $G\to\infty$，分别求 **(a)** Dr. GRPO（公式 41）、**(b)** GRPO 除以 std（公式 42）、**(c)** MaxRL 除以 $\mu$（公式 43）等价优化的重加权函数 $w$。

**统一框架**：记 prompt 难度 $\eta(x)=E_{y\sim\pi_\theta}r(y\mid x)$（二元奖励下即成功率 $\in[0,1]$）。当 $G\to\infty$、$Z=G$：组均值 $\mu\to\eta$，组标准差 $\text{std}\to\sqrt{\eta(1-\eta)}$（Bernoulli 标准差），组内平均 $\frac1G\sum_j\to E_y$。又因 $E_y[(r-\eta)\nabla_\theta\log\pi_\theta]=E_y[r\nabla_\theta\log\pi_\theta]=\nabla_\theta\eta(x)$（基线项期望为零），归一化分母只是 $x$ 的函数（经 stopgrad 视为常数），可提到期望外作为 $w(x)$。

**(a) Dr. GRPO**：

$$
\frac1G\sum_j(r-\mu)\nabla_\theta\log\pi_\theta \;\longrightarrow\; E_y[(r-\eta)\nabla_\theta\log\pi_\theta]=\nabla_\theta\eta(x).
$$

逐 prompt 即标准策略梯度，故 $\boxed{w(x)=1}$（常数，无难度重加权，Dr. GRPO 优化的就是真实 $J_\theta$）。

**(b) GRPO（除以 std）**：

$$
\frac1G\sum_j\frac{r-\mu}{\text{std}}\nabla_\theta\log\pi_\theta \;\longrightarrow\; \frac{1}{\sqrt{\eta(1-\eta)}}\,\nabla_\theta\eta(x).
$$

故 $\boxed{w(x)=\dfrac{1}{\sqrt{\eta(x)(1-\eta(x))}}}$。该权重在 $\eta\to0$ 或 $\eta\to1$（极易或极难，Bernoulli 方差小）时最大，在 $\eta=\tfrac12$（最不确定）时最小——即 std 归一化**上调极端难度、下调中等难度**的 prompt。

**(c) MaxRL（除以 $\mu$）**：

$$
\frac1G\sum_j\frac{r-\mu}{\mu}\nabla_\theta\log\pi_\theta \;\longrightarrow\; \frac{1}{\eta}\,\nabla_\theta\eta(x).
$$

故 $\boxed{w(x)=\dfrac{1}{\eta(x)}}$，即**按难度倒数重加权**：$\eta$ 越小（越难）权重越大，上调困难 prompt、下调简单 prompt，与正文给出的直觉一致。

---

### 4.5 Problem `think_about_advantage_normalization`：思考优势归一化（2 分）

> **问题原文：** 思考按组标准差归一化优势、按组均值归一化、或不做优势归一化之间的区别。三种方法各自优缺点？是否有特定设置使某种更好？

- **不归一化（Dr. GRPO）**：优势 $=r-\mu$。忠实于策略梯度（仅保期望的常数缩放），无难度重加权（见 §4.4(a)，$w(x)=1$）。缺点是奖励方差大的 prompt 会贡献更大梯度，组间不均衡。
- **std 归一化（GRPO）**：优势 $=(r-\mu)/\text{std}$。把各 prompt 的梯度范数拉到大致相等（稳定性技巧）。缺点：(1) 改变期望，不再优化真实 $J_\theta$；(2) 小 $G$ 时 std 估计噪声大；(3) 引入难度重加权 $w(x)=1/\sqrt{\eta(1-\eta)}$（§4.4(b)），实际上**上调极端难度、下调中等难度**的 prompt。
- **均值归一化（MaxRL）**：优势 $=(r-\mu)/\mu$。$w(x)=1/\eta$（§4.4(c)），**上调困难 prompt**，形成类似课程学习的效果。缺点：$\mu\to0$（全错）时会爆炸，需加 `advantage_eps`；同样改变期望，可能不稳。
- **何者更优**：std 归一化是跨异质 prompt 的稳健默认；当希望优先攻克难题时选均值归一化（MaxRL）；当希望避免长度/难度偏置、保持梯度忠实并规避除以噪声 std 的方差膨胀时选不归一化（Dr. GRPO）。

---

### 4.6 Problem `derive_surrogate_objectives`：成对重加权的代理目标（2 分）

> **问题原文：** 求"成对"重要性重加权估计器（公式 55，对每对 $(2t-1,2t)$ 联合重加权）所优化的代理目标，附推导。

仿照 token 级代理（公式 50–51），定义**成对代理策略** $\tilde\pi_t$：除第 $t$ 对 token $(2t-1,2t)$ 从 $\pi_\theta$ 采样外，其余所有时刻均从 $\pi_0$ 采样：

$$
\tilde\pi_t(y\mid x)=\Big(\prod_{s\neq 2t-1,2t}\pi_0(y_s\mid x,y_{<s})\Big)\,\pi_\theta(y_{2t-1}\mid x,y_{<2t-1})\,\pi_\theta(y_{2t}\mid x,y_{<2t}).
$$

则该估计器优化的代理目标为各成对代理策略下期望奖励之和：

$$
\boxed{J_\theta^{\text{pair}}=E_{x}\!\left[\sum_{t=1}^{L/2}E_{y\sim\tilde\pi_t(y\mid x)}\big[r(y\mid x)\big]\right].}
$$

**推导**：对单个 $t$，用从 $\pi_0$ 的重要性重加权改写

$$
E_{y\sim\tilde\pi_t}[r]=E_{y\sim\pi_0}\!\left[\frac{\pi_\theta(y_{2t-1}\mid x,y_{<2t-1})\pi_\theta(y_{2t}\mid x,y_{<2t})}{\pi_0(y_{2t-1}\mid x,y_{<2t-1})\pi_0(y_{2t}\mid x,y_{<2t})}\,r\right],
$$

对 $\theta$ 求梯度，对成对乘积用对数导数技巧 $\nabla_\theta(\cdot)=(\cdot)\nabla_\theta\log(\pi_\theta(y_{2t-1})\pi_\theta(y_{2t}))$，再对 $t$ 求和、令 $y\sim\pi_0$，即得公式 55。直觉上：成对重加权正确处理了每一对内部的前/后缀，但仍以 $\pi_0$ 处理对与对之间的耦合，故偏差介于 token 级与完整序列级之间。

---

### 4.7 Problem `think_about_importance_reweighting`：思考重要性重加权（2 分）

> **问题原文：** 比较三种离策略重要性重加权策略：(a) 不重加权；(b) PPO/GRPO 风格截断 token 级；(c) GSPO 风格几何均值截断序列级。在偏差-方差谱系上各处何处？何时某种更优？

- **(a) 不重加权（朴素）**：**高偏差、低方差**。完全忽略 $\pi_\theta$ 与 $\pi_0$ 的分布偏移，把陈旧样本当作在策略处理；最廉价，但越离策略偏差越大。
- **(b) PPO/GRPO 截断 token 级**：**中等偏差、中等方差**。token 级重加权使权重项不再随长度指数增长（相比序列级乘积大幅降方差），但忽略前/后缀重加权而引入偏差（公式 50–54）；截断进一步压缩大权重以降方差，代价是再添偏差。
- **(c) GSPO 几何均值序列级**：**亦居中、但权衡不同**。$1/L$ 次幂的几何均值避免了序列乘积的指数爆炸，得到方差可控的序列级权重；截断再降方差。它比 token 级"更忠于整条序列"，但几何均值本身是对真实序列比值的有偏近似。

**何时更优**：(a) 近似在策略（每批次只走少数几步）时，偏差小且省去方差与复杂度；(b) response 很长、需要稳定的逐 token 更新时，是经充分检验的主流选择；(c) 当 token 级重加权不稳定时（GSPO 论文的动机即 GRPO 不稳），尤其序列很长或策略偏移较大时，序列级几何均值更稳。

---

## 5. 遇到的问题与调试过程

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

## 6. 测试结果

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

## 7. 文件结构

```
assignment5-alignment/
├── cs336_alignment/
│   ├── grpo.py          # GRPO 核心（7 个函数，330+ 行，含详细注释）
│   ├── sft.py           # SFT 工具（2 个函数）
│   ├── dpo.py           # DPO 损失（2 个函数）
│   ├── utils.py         # PackedSFTDataset、response 解析器（240+ 行）
│   └── checkpoint.py    # HuggingFace 模型加载（跨平台 attn 选择 + 内存优化）
├── scripts/
│   └── eval_prompting_baselines.py   # Problem prompting_baselines 评测脚本
├── tests/
│   └── adapters.py      # 测试适配层（连接测试框架与实现）
├── extra_guidance/
│   ├── cs336_assignment5_alignment_zh.md  # PDF 翻译文档
│   ├── CODE_WALKTHROUGH.md               # 代码详解（本文档的配套）
│   ├── REPORT.md                         # 本工作报告
│   └── prompting_baselines_results.json  # 提示工程评测原始结果（50×3 条）
└── data/                # 评测数据集（mmlu/gsm8k/alpaca_eval 等）
```

---

## 8. GPU 实验题说明

PDF 中以下 Problem 属于**需要 GPU 训练**的实验题，交付物为"评论 + 指标图表"，单题动辄需 8×B200 小时量级算力，在本机（Apple Silicon, 16GB 统一内存）无法运行，故本报告暂不提供训练结果，留待在 kong 服务器（RTX 4090）或云端 B200 上补做：

| Problem | 分值 | 内容 | 依赖 |
|---------|------|------|------|
| `grpo_learning_rate` | — | 标准 GRPO 学习率调优 | GPU 训练 |
| `grpo_prompt_ablation` | — | r1_zero vs question_only 提示消融 | GPU 训练 |
| `grpo_experiments_standard_on_policy` | 10 | 标准在策略 GRPO 完整运行 | GPU 训练 |
| `grpo_experiments_variants_on_policy` | 10 | GRPO/Dr.GRPO/RFT/MaxRL 对比（各 4 seed） | GPU 训练 |
| `grpo_experiments_off_policy` | 10 | naive/noclip/clip/gspo 离策略对比（各 4 seed） | GPU 训练 |
| `try_your_own` | 10 | 自创策略梯度估计器并与基线对比 | GPU 训练 |

这些题目的**算法实现已全部完成并通过单元测试**（见第 6 节），缺的只是大规模训练的运行与作图。第 4 节的理论推导（尤其 `derive_difficulty_reweightings`）已从数学上刻画了 GRPO/Dr.GRPO/MaxRL 各变体的难度重加权行为，可作为这些实验预期结果的理论参照。
