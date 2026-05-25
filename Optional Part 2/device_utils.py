"""
跨平台设备检测与模型加载工具。

支持环境：
  - macOS（CPU / Apple Silicon MPS）
  - Windows + NVIDIA GPU（CUDA）
  - Linux 服务器 + NVIDIA GPU（CUDA）
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

import torch


def get_device(device_override: str | None = None) -> torch.device:
    """返回当前环境中最优的计算设备。

    优先级：CUDA > MPS（Apple Silicon）> CPU。
    若指定 device_override，则直接使用指定设备。
    """
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_compute_dtype(device: torch.device) -> torch.dtype:
    """根据设备返回最合适的浮点精度。

    CUDA / MPS → bfloat16（与 FP32 同数值范围，无需 loss scaling）
    CPU        → float32（CPU 不支持 bfloat16 的低精度计算优化）
    """
    if device.type in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def get_attn_impl(device: torch.device) -> str:
    """返回该设备下应使用的注意力实现。

    FlashAttention-2 仅支持 NVIDIA CUDA 设备；
    MPS（Apple Silicon）和 CPU 使用标准 eager 实现。
    """
    return "flash_attention_2" if device.type == "cuda" else "eager"


def get_num_workers() -> int:
    """返回 DataLoader 安全的 num_workers 数。

    Windows 不支持 fork 多进程（会导致 RuntimeError），强制使用 0。
    Linux / macOS 使用 min(4, cpu_count)。
    """
    if platform.system() == "Windows":
        return 0
    return min(4, os.cpu_count() or 1)


def load_model_and_tokenizer(
    model_path: str | Path,
    device: torch.device | None = None,
    *,
    eval_mode: bool = True,
) -> tuple:
    """加载 HuggingFace 因果语言模型及其 tokenizer。

    Args:
        model_path : 模型目录路径（本地或 HuggingFace Hub ID）
        device     : 目标设备（None 时自动检测）
        eval_mode  : 为 True 时调用 model.eval()，关闭 dropout 等

    Returns:
        (model, tokenizer) 元组，model 已移至指定设备
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = get_device()
    dtype = get_compute_dtype(device)
    attn_impl = get_attn_impl(device)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)

    if eval_mode:
        model.eval()

    return model, tokenizer


def model_summary(device: torch.device) -> str:
    """返回当前设备和精度配置的简洁描述，用于日志输出。"""
    dtype = get_compute_dtype(device)
    attn = get_attn_impl(device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"CUDA ({gpu_name}, {vram_gb:.0f}GB) | {dtype} | attn={attn}"
    if device.type == "mps":
        return f"Apple MPS | {dtype} | attn={attn}"
    return f"CPU | {dtype} | attn={attn}"
