"""
DPO 损失函数（HuggingFace Llama 版）

从 cs336_alignment/dpo.py 复制并修改。

修改说明：
  1. _response_log_prob_sum 改为直接调用 model(input_ids).logits
     （原版依赖 grpo.py 中基于 cs336_basics 自定义模型的 get_response_log_probs）
  2. 使用"无条件对数概率差抵消 prompt"技巧：
       log π(yw|x) - log π(yl|x) = log π(x⊕yw) - log π(x⊕yl)
     （prompt 的对数概率在差值中天然抵消，无需单独掩码）
  3. compute_per_instance_dpo_loss 新增 device_lm / device_ref 参数，
     支持两个模型分别放在不同设备（例如 GPU0 和 GPU1）
  4. 跨平台兼容：无 CUDA 特定操作

原始文件：cs336_alignment/dpo.py
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase

_ALPACA_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "alpaca_sft.prompt"
)
_ALPACA_TEMPLATE = _ALPACA_TEMPLATE_PATH.read_text(encoding="utf-8")


def _unconditional_log_prob_sum(
    model: torch.nn.Module,
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> Tensor:
    """计算文本所有 token 的无条件对数概率之和。

    实现原理（与主作业版本的差异）：
      - 原版：tokenize prompt 和 response 分开，用 response_mask 只算 response token
      - 本版：tokenize 完整文本（prompt+response+EOS），对所有 token 求和
        由于 DPO 损失只用 chosen - rejected 的差，prompt 部分会天然抵消

    Args:
        model    : HuggingFace AutoModelForCausalLM（调用方负责 eval/no_grad）
        text     : 完整字符串（alpaca前缀 + response + EOS字符串）
        tokenizer: 配套 tokenizer
        device   : 模型所在设备

    Returns:
        () 标量 float — Σ_t log p(x_t | x_{<t})，对文本所有 token 求和
    """
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    # ids: (1, L)，L 为 token 总数

    logits = model(ids).logits  # (1, L, vocab_size)

    # 取 log_softmax，计算每个位置的对数概率
    log_probs = F.log_softmax(logits, dim=-1)  # (1, L, vocab_size)

    # 目标 token：input[1:] 预测 input[0:-1]（语言模型的标准偏移）
    # log_probs[:, :-1, :] 对应位置 0..L-2 的预测
    # ids[:, 1:]          对应位置 1..L-1 的目标 token
    target = ids[:, 1:]                                        # (1, L-1)
    token_lp = log_probs[:, :-1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

    return token_lp.sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
    *,
    device_lm: torch.device | None = None,
    device_ref: torch.device | None = None,
) -> Tensor:
    """计算单条偏好数据的 DPO 损失（HuggingFace Llama 版）。

    与主作业 dpo.py 的差异：
      - 使用无条件对数概率（完整文本），而非有条件对数概率 + response_mask
      - 新增 device_lm / device_ref 参数，支持多 GPU 部署

    Args:
        lm               : 待训练模型（log 概率需要梯度）
        lm_ref           : 冻结参考模型（推理时自动 no_grad）
        tokenizer        : 两个模型共用的 tokenizer
        beta             : KL 惩罚系数 β
        prompt           : 对话 prompt（作为 Alpaca 模板的 {instruction}）
        response_chosen  : 偏好响应
        response_rejected: 被拒响应
        device_lm        : lm 所在设备（None 时自动从模型参数推断）
        device_ref       : lm_ref 所在设备（None 时自动从模型参数推断）

    Returns:
        () 标量 — DPO 损失，可直接 .backward()
    """
    # 推断设备
    if device_lm is None:
        device_lm = next(lm.parameters()).device
    if device_ref is None:
        device_ref = next(lm_ref.parameters()).device

    # 构建完整文本（alpaca前缀 + response + EOS）
    alpaca_prefix = _ALPACA_TEMPLATE.format(instruction=prompt, response="")
    eos = tokenizer.eos_token or ""
    text_chosen   = alpaca_prefix + response_chosen   + eos
    text_rejected = alpaca_prefix + response_rejected + eos

    # 参考模型：不需要梯度
    with torch.no_grad():
        lp_ref_chosen   = _unconditional_log_prob_sum(lm_ref, text_chosen,   tokenizer, device_ref)
        lp_ref_rejected = _unconditional_log_prob_sum(lm_ref, text_rejected, tokenizer, device_ref)

    # 训练模型：保留梯度
    lp_chosen   = _unconditional_log_prob_sum(lm, text_chosen,   tokenizer, device_lm)
    lp_rejected = _unconditional_log_prob_sum(lm, text_rejected, tokenizer, device_lm)

    # 将参考模型的值移到 lm 所在设备（可能不同 GPU）
    lp_ref_chosen   = lp_ref_chosen.to(device_lm)
    lp_ref_rejected = lp_ref_rejected.to(device_lm)

    log_ratio_chosen   = lp_chosen   - lp_ref_chosen
    log_ratio_rejected = lp_rejected - lp_ref_rejected

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss


def compute_dpo_reward_accuracy(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
    *,
    device_lm: torch.device | None = None,
    device_ref: torch.device | None = None,
) -> bool:
    """判断隐式奖励模型对当前样例的分类是否正确。

    分类正确 = chosen 的 log ratio 大于 rejected 的 log ratio。

    用于在验证集上追踪"分类准确率"。
    """
    if device_lm is None:
        device_lm = next(lm.parameters()).device
    if device_ref is None:
        device_ref = next(lm_ref.parameters()).device

    alpaca_prefix = _ALPACA_TEMPLATE.format(instruction=prompt, response="")
    eos = tokenizer.eos_token or ""
    text_chosen   = alpaca_prefix + response_chosen   + eos
    text_rejected = alpaca_prefix + response_rejected + eos

    with torch.no_grad():
        lp_ref_chosen   = _unconditional_log_prob_sum(lm_ref, text_chosen,   tokenizer, device_ref)
        lp_ref_rejected = _unconditional_log_prob_sum(lm_ref, text_rejected, tokenizer, device_ref)
        lp_chosen       = _unconditional_log_prob_sum(lm,     text_chosen,   tokenizer, device_lm)
        lp_rejected     = _unconditional_log_prob_sum(lm,     text_rejected, tokenizer, device_lm)

    ratio_chosen   = (lp_chosen   - lp_ref_chosen.to(device_lm)).item()
    ratio_rejected = (lp_rejected - lp_ref_rejected.to(device_lm)).item()
    return ratio_chosen > ratio_rejected
