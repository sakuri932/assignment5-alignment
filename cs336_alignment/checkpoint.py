import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _select_attn_implementation(device: str) -> str:
    # flash_attention_2 仅在 CUDA + 已安装 flash_attn 包时可用；
    # 其余环境（CPU / Apple MPS / CUDA 未装 flash_attn）回退到 sdpa（PyTorch 内置高效注意力）。
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=_select_attn_implementation(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer
