"""
CS336 Assignment 5 — DPO（直接偏好优化）损失
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .grpo import tokenize_prompt_and_output, get_response_log_probs

_ALPACA_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "alpaca_sft.prompt"
_ALPACA_TEMPLATE = _ALPACA_TEMPLATE_PATH.read_text()


def _response_log_prob_sum(
    model: torch.nn.Module,
    prompt: str,
    response: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Tensor:
    """
    计算 model 对一条 response 的 token log 概率之和（含 EOS），使用 Alpaca 模板构造上下文。

    Args:
        model     : causal LM（调用方负责 no_grad / eval 等上下文）
        prompt    : 对话 prompt（将作为 Alpaca template 的 {instruction}）
        response  : 模型响应文本（log 概率计算目标，末尾自动追加 EOS token）
        tokenizer : 与 model 配套的 tokenizer，需含 eos_token_id

    Returns:
        () scalar float  — Σ_t log p(y_t | prefix, y_{<t})，对所有 response token（含 EOS）求和
    """
    # Alpaca 前缀：包含 header + "### Instruction:" + prompt + "### Response:"
    alpaca_prefix = _ALPACA_TEMPLATE.format(instruction=prompt, response="")

    p_ids: list[int] = tokenizer.encode(alpaca_prefix, add_special_tokens=False)
    r_ids: list[int] = tokenizer.encode(response, add_special_tokens=False)

    # 与 PackedSFTDataset 保持一致：response 末尾追加 EOS
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        r_ids = r_ids + [eos_id]

    full_ids = p_ids + r_ids
    n = len(full_ids)

    device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    labels    = torch.tensor([full_ids[1:],  ], dtype=torch.long, device=device)

    # response_mask: labels 中位于 [len(p_ids)-1, n-2] 的位置
    mask = torch.zeros(1, n - 1, device=device)
    mask[0, len(p_ids) - 1 : n - 1] = 1.0

    lp_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
    return (lp_dict["log_probs"] * mask).sum()


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    """
    计算单条偏好数据的 DPO 损失。

    Args:
        lm               : 待训练的 causal LM（log 概率需要梯度）
        lm_ref           : 冻结的参考 LM（推理时自动包裹 no_grad）
        tokenizer        : 与两个模型共用的 tokenizer
        beta             : KL 惩罚系数 β，控制训练策略偏离参考模型的程度
        prompt           : 对话 prompt（作为 Alpaca template 的 {instruction}）
        response_chosen  : 偏好响应（人类标注为更好的输出）
        response_rejected: 非偏好响应（人类标注为较差的输出）

    Returns:
        () scalar float  — DPO 损失，公式为：
            loss = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
            log_ratio_x = Σ_t log π_θ(y_t|·) - Σ_t log π_ref(y_t|·)  （对 response tokens 含 EOS 求和）

    调用方可直接对返回值调用 .backward() 进行反向传播。
    """
    # lm_ref 推理不需要梯度
    with torch.no_grad():
        lp_ref_chosen   = _response_log_prob_sum(lm_ref, prompt, response_chosen,   tokenizer)
        lp_ref_rejected = _response_log_prob_sum(lm_ref, prompt, response_rejected, tokenizer)

    lp_chosen   = _response_log_prob_sum(lm,   prompt, response_chosen,   tokenizer)
    lp_rejected = _response_log_prob_sum(lm,   prompt, response_rejected, tokenizer)

    log_ratio_chosen   = lp_chosen   - lp_ref_chosen
    log_ratio_rejected = lp_rejected - lp_ref_rejected

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss
