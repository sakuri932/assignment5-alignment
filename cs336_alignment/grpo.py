"""
CS336 Assignment 5 — GRPO (Group Relative Policy Optimization) 核心实现

实现了以下函数：
  tokenize_prompt_and_output   — prompt+output 分词，构建 input_ids/labels/response_mask
  get_response_log_probs       — 模型前向传播，计算每 token 的条件 log 概率（及可选的熵）
  compute_rollout_rewards      — 批量计算 rollout 奖励
  compute_group_normalized_rewards — 组内奖励归一化（支持 GRPO/Dr.GRPO/MaxRL）
  compute_policy_gradient_loss — 策略梯度 loss（on-policy 和 off-policy，含 GSPO）
  aggregate_loss_across_microbatch — 跨 microbatch 聚合（sequence / constant 两种规范化）
  grpo_train_step              — 含梯度累积的完整训练步骤

对应 assignment 章节：Section 4（GRPO 基础）、Section 5（变体）、Section 6（off-policy）
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# 分词工具
# ---------------------------------------------------------------------------

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """
    将 prompt 和 output 字符串分词，构建自回归训练所需的张量。

    Args:
        prompt_strs : 长度为 B 的 prompt 字符串列表
        output_strs : 长度为 B 的 output 字符串列表（与 prompt_strs 一一对应）
        tokenizer   : HuggingFace tokenizer，需含 pad_token_id

    Returns:
        dict，包含以下键（设 L = max_full_len - 1）：
          "input_ids"     : (B, L) long  — prompt+output 拼接后右侧 padding，去掉最后一个 token
          "labels"        : (B, L) long  — 同上但去掉第一个 token（即 input_ids 向左移一位）
          "response_mask" : (B, L) long  — 1 表示该位置的 label 是真实 output token，0 表示 prompt 或 padding

    关键设计：先对完整序列右侧 padding，再 slice 出 input_ids/labels。
    （若先 slice 再 padding，边界 pad 与 full_ids 末尾 pad 对不上，导致 response_mask 错位。）

    流程：
      1. 将 prompt_ids + output_ids 拼成 full_ids（不加特殊 token）
      2. 全批右侧 padding 到 max_full_len
      3. input_ids = padded_full[:, :-1]，labels = padded_full[:, 1:]
      4. response_mask[j] = 1 iff j ∈ [prompt_len-1, seq_len-2]
           prompt_len-1 : labels 中第一个 output token 的位置
           seq_len-2    : labels 中最后一个真实（非 padding）output token 的位置
    """
    all_full_ids: list[list[int]] = []
    all_prompt_lens: list[int] = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        all_full_ids.append(prompt_ids + output_ids)
        all_prompt_lens.append(len(prompt_ids))

    # 步骤 2：先对完整序列 padding 到最大长度
    max_len = max(len(ids) for ids in all_full_ids)
    pad_id = tokenizer.pad_token_id

    padded_input_ids: list[list[int]] = []
    padded_labels: list[list[int]] = []
    padded_response_masks: list[list[int]] = []

    for full_ids, prompt_len in zip(all_full_ids, all_prompt_lens):
        seq_len = len(full_ids)
        pad_len = max_len - seq_len
        padded = full_ids + [pad_id] * pad_len   # (max_len,)

        # 步骤 3：slice — 两者长度均为 max_len - 1
        input_ids = padded[:-1]
        labels = padded[1:]

        # 步骤 4：response_mask[j] = 1 iff j in [prompt_len-1, seq_len-2]
        # prompt_len-1：第一个 output token 在 labels 中的位置
        # seq_len-2：最后一个真实（非 padding）output token 在 labels 中的位置
        response_mask = [
            1 if (prompt_len - 1 <= j < seq_len - 1) else 0
            for j in range(max_len - 1)
        ]

        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        padded_response_masks.append(response_mask)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_response_masks, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 模型 log 概率
# ---------------------------------------------------------------------------

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    """
    对 causal LM 做前向传播，提取每个 label token 的条件 log 概率，
    以及可选的下一 token 分布熵（用于监控探索程度）。

    Args:
        model               : causal LM（HuggingFace 风格，输出 .logits）
        input_ids           : (B, L) long  — 输入 token 序列
        labels              : (B, L) long  — 目标 token 序列（input_ids 向左移一位）
        return_token_entropy: 若为 True，则额外计算每位置的分布熵

    Returns:
        dict，包含：
          "log_probs"    : (B, L) float  — log p(labels[t] | input_ids[:t+1])，逐位置条件 log 概率
          "token_entropy": (B, L) float  — 仅当 return_token_entropy=True 时存在；
                           H_t = -Σ_v p_v log p_v，衡量模型在该位置的不确定程度

    注：调用方负责管理 no_grad 上下文：
      - 训练时（需要反向传播）：不包裹 no_grad
      - 生成旧策略 log probs（reference model 或 old policy）时：包裹 torch.no_grad()
    """
    # 前向传播获得 logits，shape: (batch, seq_len, vocab_size)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    # 数值稳定的 log softmax
    log_probs_all = F.log_softmax(logits, dim=-1)

    # 在每个位置取 labels 指定的 token 的 log 概率
    per_token_log_probs = log_probs_all.gather(
        dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)   # (batch, seq_len)

    result: dict[str, Tensor] = {"log_probs": per_token_log_probs}

    if return_token_entropy:
        # 熵 H = -sum(p * log_p)，用于衡量模型的不确定性
        probs = torch.exp(log_probs_all)
        entropy = -(probs * log_probs_all).sum(dim=-1)  # (batch, seq_len)
        result["token_entropy"] = entropy

    return result


# ---------------------------------------------------------------------------
# Rollout 奖励计算
# ---------------------------------------------------------------------------

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    """
    对每条 rollout 响应调用 reward_fn，收集奖励并统计元数据。

    Args:
        reward_fn              : (response, ground_truth) → {"reward": float, "format_reward": float}
        rollout_responses      : 长度为 N = n_prompts * group_size 的模型生成响应列表
        repeated_ground_truths : 长度为 N 的参考答案列表（与 rollout_responses 一一对应）

    Returns:
        raw_rewards : (N,) float32 tensor — 每条 rollout 的原始奖励值
        metadata    : dict，包含 "mean_reward" 和 "mean_format_reward"（float）
    """
    rewards: list[float] = []
    format_rewards: list[float] = []

    for response, gt in zip(rollout_responses, repeated_ground_truths):
        result = reward_fn(response, gt)
        rewards.append(result["reward"])
        format_rewards.append(result["format_reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)

    metadata = {
        "mean_reward": float(raw_rewards.mean()),
        "mean_format_reward": float(sum(format_rewards) / len(format_rewards)),
    }
    return raw_rewards, metadata


# ---------------------------------------------------------------------------
# 组内奖励归一化（GRPO / Dr.GRPO / MaxRL / RFT）
# ---------------------------------------------------------------------------

def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """
    在每个 prompt 组内对奖励做 baseline 减法和规范化，得到优势估计。

    Args:
        raw_rewards          : (N,) float  — 原始奖励，N = n_prompts * group_size，
                               前 group_size 个属于第 0 个 prompt 组，以此类推
        group_size           : 每个 prompt 采样的 rollout 数量 G
        baseline             : "mean" 减去组均值；"none" 不减（RFT 场景）
        advantage_eps        : 防止除以零的小常数 ε
        advantage_normalizer : "std" 除以组标准差（GRPO）；"mean" 除以 |组均值|（MaxRL）；
                               "none" 不除（Dr.GRPO / RFT）

    Returns:
        advantages  : (N,) float  — 归一化后的优势估计
        raw_rewards : (N,) float  — 原始奖励（直接透传，便于调用方记录日志）
        metadata    : dict，包含 "reward_mean/std/max/min"（float）

    对应算法变体：
      GRPO    : baseline="mean", advantage_normalizer="std"
      Dr.GRPO : baseline="mean", advantage_normalizer="none"
      MaxRL   : baseline="mean", advantage_normalizer="mean"
      RFT     : baseline="none", advantage_normalizer="none"

    数学细节（以 GRPO 为例，组大小 G）：
      μ_i  = (1/G) Σ_j r^{(i,j)}
      σ_i  = std(r^{(i,j)})     （PyTorch 默认 unbiased=True，分母 G-1）
      A^{(i,j)} = (r^{(i,j)} - μ_i) / (σ_i + ε)
    """
    batch_size = raw_rewards.shape[0]
    n_groups = batch_size // group_size

    # 按组 reshape：(n_groups, group_size)
    rewards_grouped = raw_rewards.reshape(n_groups, group_size)
    advantages_grouped = rewards_grouped.clone().float()

    # ── 计算组均值（用于 baseline 减法和 MaxRL 规范化）──────────────────
    group_mean = rewards_grouped.float().mean(dim=1, keepdim=True)  # (n_groups, 1)

    # ── Baseline 减法 ────────────────────────────────────────────────────
    if baseline == "mean":
        advantages_grouped = advantages_grouped - group_mean
    # baseline="none": 不减均值（RFT 场景，只保留正样本的权重）

    # ── 优势规范化 ────────────────────────────────────────────────────────
    if advantage_normalizer == "std":
        # 除以组内标准差（unbiased），防止 std≈0 时数值爆炸
        group_std = rewards_grouped.float().std(dim=1, keepdim=True)  # (n_groups, 1)
        advantages_grouped = advantages_grouped / (group_std + advantage_eps)

    elif advantage_normalizer == "mean":
        # MaxRL：除以 |组均值|，使更难的 prompt 获得更大梯度权重
        advantages_grouped = advantages_grouped / (group_mean.abs() + advantage_eps)

    elif advantage_normalizer == "none":
        pass  # Dr.GRPO / RFT

    advantages = advantages_grouped.reshape(batch_size)

    metadata = {
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std()),
        "reward_max": float(raw_rewards.max()),
        "reward_min": float(raw_rewards.min()),
    }
    return advantages, raw_rewards, metadata


# ---------------------------------------------------------------------------
# 策略梯度 Loss（on-policy & off-policy）
# ---------------------------------------------------------------------------

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    计算每个 token 的策略梯度 loss（取负，以最小化 loss 等价于最大化目标函数）。

    Args:
        raw_rewards_or_advantages  : (B,) 或 (B, 1) float  — 每条序列的优势（或原始奖励）
        policy_log_probs           : (B, L) float  — 当前策略 π_θ 的 per-token log 概率
        importance_reweighting_method : 重要性重加权方式（见下方说明）
        old_log_probs              : (B, L) float  — off-policy 时旧策略 π_0 的 per-token log 概率；
                                     "none" 方法时可为 None
        cliprange                  : PPO 裁剪范围 ε；"grpo"/"gspo" 时必须提供
        response_mask              : (B, L) long/float  — GSPO 中用于只对 response token 求均值；
                                     其他方法可为 None

    Returns:
        per_token_loss : (B, L) float  — 每个 token 位置的策略梯度 loss（未做 mask 过滤）
        metadata       : dict；"grpo"/"gspo" 时含 "clip_fraction"（被裁剪的 token/序列比例）

    四种重要性重加权方式：
      "none"   — on-policy，直接 -A * log_π（GRPO 标准 on-policy）
      "noclip" — off-policy，不裁剪：-A * w，w = π_θ/π_0（高方差，无偏）
      "grpo"   — PPO/GRPO 风格 token-level 裁剪：
                   -min(w*A, clip(w, [1-ε, 1+ε])*A)，w_t = π_θ(y_t)/π_0(y_t)
      "gspo"   — GSPO 序列级别裁剪（geometric mean 重加权，低方差）：
                   -min(s*A, clip(s, [1-ε, 1+ε])*A)，s = exp((1/L)Σ_t log(π_θ/π_0))
                   每条序列的 per-token loss 为同一常数，梯度通过链式法则引入 1/L 规范化
    """
    metadata: dict[str, Tensor] = {}

    # A 的 shape: (batch,) 或 (batch, 1) → 统一变为 (batch, 1) 以便广播
    A = raw_rewards_or_advantages
    if A.dim() == 1:
        A = A.unsqueeze(1)

    # ── On-policy（无重要性重加权）─────────────────────────────────────
    if importance_reweighting_method == "none":
        # 标准 REINFORCE/GRPO：最大化 A * log_π → 最小化 -A * log_π
        per_token_loss = -A * policy_log_probs

    # ── Off-policy：无裁剪 token-level 重加权 ────────────────────────────
    elif importance_reweighting_method == "noclip":
        # 重要性权重 w_t = π_θ(y_t) / π_0(y_t) = exp(log_π - log_π_0)
        # 梯度通过 policy_log_probs 传播
        w = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = -A * w

    # ── Off-policy：PPO/GRPO 风格 token-level 裁剪 ───────────────────────
    elif importance_reweighting_method == "grpo":
        w = torch.exp(policy_log_probs - old_log_probs)           # (batch, seq)
        clipped_w = torch.clamp(w, 1.0 - cliprange, 1.0 + cliprange)

        # PPO 目标：取较小值防止 ratio 过大的更新
        # min(w*A, clip(w)*A)：对正优势裁剪大 ratio，对负优势裁剪小 ratio
        per_token_obj = torch.min(w * A, clipped_w * A)
        per_token_loss = -per_token_obj

        # 记录被裁剪的 token 比例（用于监控训练稳定性）
        is_clipped = (w != clipped_w)
        metadata["clip_fraction"] = is_clipped.float().mean()

    # ── Off-policy：GSPO 序列级别裁剪 ────────────────────────────────────
    elif importance_reweighting_method == "gspo":
        log_ratio = policy_log_probs - old_log_probs   # (batch, seq)

        # 几何均值重要性权重 s = exp( (1/L) * Σ_t log(π_θ/π_0) )
        # 只对 response token 求均值，忽略 padding 和 prompt
        if response_mask is not None:
            mask_f = response_mask.float()
            L = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
            mean_log_ratio = (log_ratio * mask_f).sum(dim=1, keepdim=True) / L
        else:
            mean_log_ratio = log_ratio.mean(dim=1, keepdim=True)

        s = torch.exp(mean_log_ratio)                             # (batch, 1)
        clipped_s = torch.clamp(s, 1.0 - cliprange, 1.0 + cliprange)

        # 序列级目标值（与序列内所有 token 共享同一值）
        per_seq_obj = torch.min(s * A, clipped_s * A)            # (batch, 1)

        # 展开到 (batch, seq_len)：梯度通过每个 token 的 log_ratio 回传（1/L 权重由链式法则自然产生）
        per_token_loss = -per_seq_obj.expand_as(policy_log_probs)

        # 记录被裁剪的序列比例
        is_clipped = (s != clipped_s).squeeze(1)
        metadata["clip_fraction"] = is_clipped.float().mean()

    else:
        raise ValueError(f"未知的重要性重加权方式：{importance_reweighting_method}")

    return per_token_loss, metadata


# ---------------------------------------------------------------------------
# Microbatch loss 聚合
# ---------------------------------------------------------------------------

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    """
    将 per-token loss 在 batch 和 seq 维度上聚合为标量。

    Args:
        per_token_policy_gradient_loss : (B, L) float  — compute_policy_gradient_loss 的输出
        mask                           : (B, L) long/float  — 1 表示 response token，0 表示 prompt/padding
        loss_normalization             : 聚合策略（见下方说明）
        normalization_constant         : "constant" 策略下的固定分母 Z（通常为 total_tokens 或 B*G*L）

    Returns:
        loss : () scalar float  — 聚合后的标量 loss，用于 .backward()

    两种规范化策略：
      "sequence" — 先对每条序列内的 response token 取平均，再对 batch 取平均。
                   所有序列贡献相等权重，序列长度差异不影响梯度方向。
      "constant" — 将所有 response token 的 loss 之和除以固定常数 Z。
                   Dr.GRPO 使用此策略，使绝对 loss 量级与批次大小和序列长度解耦。
    """
    mask_f = mask.float()
    masked_loss = per_token_policy_gradient_loss * mask_f

    if loss_normalization == "sequence":
        # 每条序列的 response token 数量（至少为 1，防止除零）
        seq_token_count = mask_f.sum(dim=1).clamp(min=1)
        # 序列平均 loss，再 batch 平均
        seq_loss = masked_loss.sum(dim=1) / seq_token_count
        loss = seq_loss.mean()

    elif loss_normalization == "constant":
        # 将 loss 总和除以固定常数 Z（通常 = B*G*L，由调用方传入）
        loss = masked_loss.sum() / normalization_constant

    else:
        raise ValueError(f"未知的 loss 规范化策略：{loss_normalization}")

    return loss


# ---------------------------------------------------------------------------
# 完整 GRPO 训练步骤（含梯度累积）
# ---------------------------------------------------------------------------

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[Tensor, dict]:
    """
    执行一次 GRPO 更新（前向 + 反向 + 优化器步），支持梯度累积。

    Args:
        model                      : 待训练的 causal LM
        tokenizer                  : 与 model 配套的 tokenizer
        optimizer                  : PyTorch 优化器（调用前无需 zero_grad，函数内部负责）
        gradient_accumulation_steps: 梯度累积步数 G；batch 被均分为 G 个 microbatch
        max_grad_norm              : 梯度裁剪阈值；None 表示不裁剪
        reward_fn                  : (response, ground_truth) → {"reward": float, "format_reward": float}
        repeated_prompts           : 长度为 N = n_prompts * G 的 prompt 列表
        rollout_responses          : 长度为 N 的模型采样响应列表
        repeated_ground_truths     : 长度为 N 的参考答案列表
        group_size                 : 每个 prompt 的 rollout 数量（即上方 G）
        （其余参数透传至 compute_group_normalized_rewards / compute_policy_gradient_loss /
         aggregate_loss_across_microbatch，含义见各函数 docstring）

    Returns:
        loss     : () scalar tensor  — 全 batch 的平均 loss（sequence 规范化）或总和/C（constant 规范化）
        metadata : dict，包含 grad_norm、mean_reward、mean_format_reward、reward_mean/std/max/min

    流程：
      1. 计算每条 rollout 的奖励
      2. 组内归一化奖励得到优势
      3. 分词（prompt + response）→ input_ids/labels/response_mask
      4. 将 batch 切分为 G 个 microbatch，逐 microbatch 前向 + 反向累积梯度
      5. 梯度裁剪（可选）→ optimizer.step() → zero_grad()
    """
    device = next(model.parameters()).device

    # ── 步骤 1：计算 rollout 奖励 ───────────────────────────────────────
    raw_rewards, reward_meta = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )

    # ── 步骤 2：组内归一化 → 优势 ────────────────────────────────────────
    advantages, _, reward_stats = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )
    advantages = advantages.to(device)

    # ── 步骤 3：分词 ─────────────────────────────────────────────────────
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)

    # ── 步骤 4-5：梯度累积训练 ───────────────────────────────────────────
    batch_size = len(repeated_prompts)
    microbatch_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad(set_to_none=True)
    total_loss_val = 0.0
    clip_fraction_val = None

    for step_i in range(gradient_accumulation_steps):
        s = step_i * microbatch_size
        e = s + microbatch_size

        mb_input_ids = input_ids[s:e]
        mb_labels = labels[s:e]
        mb_response_mask = response_mask[s:e]
        mb_advantages = advantages[s:e]

        mb_old_lp = old_log_probs[s:e].to(device) if old_log_probs is not None else None

        # 前向传播：获取当前策略的 per-token log 概率（需要梯度）
        lp_dict = get_response_log_probs(
            model=model,
            input_ids=mb_input_ids,
            labels=mb_labels,
            return_token_entropy=False,
        )
        policy_lp = lp_dict["log_probs"]
        # 你会发现这里和作为模型输出的rollout_responses重复了，明明可以复用。
        # 不这么做是因为两者往往不是同一时间生成的，比如rollout_responses是用户调用模型生成的答案，过几个月再训练时才用到。
        # 另外，后训练涉及多次更新模型参数，之前的log probs就不再准确了，所以只能重新计算。

        # 计算 per-token 策略梯度 loss
        per_token_loss, pg_meta = compute_policy_gradient_loss(
            raw_rewards_or_advantages=mb_advantages,
            policy_log_probs=policy_lp,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old_lp,
            cliprange=cliprange,
            response_mask=mb_response_mask,
        )
        if "clip_fraction" in pg_meta:
            clip_fraction_val = pg_meta["clip_fraction"]

        # 聚合为标量
        mb_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=mb_response_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        # 梯度缩放策略：
        #   sequence 规范化：mb_loss = mean(seq_losses)，需除以 G 使梯度等价于全 batch 平均
        #   constant 规范化：mb_loss = sum/C，已包含全 batch 正确缩放，无需再除 G
        if loss_normalization == "sequence":
            backward_loss = mb_loss / gradient_accumulation_steps #因为只做了backward没有归零，所以要除以外面的for循环步数G
        else:
            backward_loss = mb_loss
        backward_loss.backward()

        total_loss_val += mb_loss.detach().item() #只是为了打印，所以detach掉计算图

    # 返回真实批次 loss：sequence 取平均（除以 G），constant 直接返回总和/C
    if loss_normalization == "sequence":
        total_loss_val /= gradient_accumulation_steps

    # ── 步骤 6：梯度裁剪 + 优化器更新 ────────────────────────────────────
    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        ).item()
    # 这是 PyTorch 内置的梯度裁剪函数，做了两件事：
    # 1. 计算所有参数的总梯度范数（L2 norm），得到 grad_norm
    # 2. 如果 grad_norm > max_grad_norm，则按等比例缩放所有参数的梯度，使得新的范数等于 max_grad_norm
    # 这样可以防止梯度爆炸，保持训练稳定性。

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata = {
        "grad_norm": grad_norm,
        **reward_meta,
        **reward_stats,
    }
    if clip_fraction_val is not None:
        metadata["clip_fraction"] = clip_fraction_val
    return torch.tensor(total_loss_val), metadata
