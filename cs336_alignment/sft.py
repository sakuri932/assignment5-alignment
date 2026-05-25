"""
CS336 Assignment 5 — SFT 微批次训练工具
"""
from __future__ import annotations

import torch
from torch import Tensor


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    """
    对 tensor 中 mask=1 的元素按 dim 求和，再除以 normalize_constant。

    Args:
        tensor             : 任意形状的 float tensor
        mask               : 与 tensor 形状相同的 0/1 tensor（会被转为 float）
        dim                : 求和的维度；None 表示对全部元素求和
        normalize_constant : 求和结果的除数

    Returns:
        若 dim=None  : () scalar  — sum(tensor * mask) / normalize_constant
        若 dim 指定  : 沿 dim 缩减后的 tensor  — 形状与 tensor 相同但 dim 维度被消去

    mask=0 的位置贡献为零（乘以 0 而非跳过，因此输出形状与输入完全对应）。
    """
    masked = tensor * mask.float()
    if dim is None:
        total = masked.sum()
    else:
        total = masked.sum(dim=dim)
    return total / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict]:
    """
    SFT 微批次训练步骤：计算负对数似然损失并反向传播。

    Args:
        policy_log_probs          : (B, L) float，requires_grad=True  — 当前策略的 per-token log 概率
        response_mask             : (B, L) long/float  — 1 表示 response token，0 表示 prompt/padding
        gradient_accumulation_steps : 梯度累积步数 G（用于缩放梯度）
        normalize_constant        : 额外的归一化常数 C（默认 1.0）

    Returns:
        loss : () scalar float  — -sum(log_probs * mask) / (C * G * B)，已调用 .backward()
        {}   : 空 dict（保留接口一致性，便于将来扩展 metadata）

    损失公式：
        total_scale = C * G * B    （C=normalize_constant, G=gradient_accumulation_steps, B=batch_size）
        loss = -sum(policy_log_probs * response_mask) / total_scale

    不清零梯度，也不调用 optimizer.step()，由外层训练循环统一负责。
    调用 loss.backward() 后梯度会累积到 policy_log_probs 的叶子 tensor 上。
    """
    microbatch_size = policy_log_probs.shape[0]
    total_scale = normalize_constant * gradient_accumulation_steps * microbatch_size
    loss = -masked_normalize(policy_log_probs, response_mask, dim=None, normalize_constant=total_scale)
    loss.backward()

    return loss, {}
