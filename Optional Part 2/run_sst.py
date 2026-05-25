"""
SimpleSafetyTests 输出生成脚本（Problem sst_baseline / sst_sft）

用法：
  python run_sst.py \
      --model-path /data/a5-alignment/models/Llama-3.1-8B \
      --data-path  data/simple_safety_tests/simple_safety_tests.csv \
      --output     results/sst_baseline.jsonl \
      [--device cuda]  [--max-samples 100]

输出格式（与评估脚本兼容，JSON Lines）：
  每行一个 JSON 对象，至少含：
    prompts_final : 原始指令
    output        : 模型生成的响应
  以及原始 CSV 中的其他字段（id, harm_area, counter, category）。

评估命令（需在有 2 块 80GB+ GPU 的机器上运行）：
  uv run python scripts/evaluate_safety.py \\
      --input-path <output.jsonl> \\
      --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \\
      --num-gpus 2 \\
      --output-path <evaluated.jsonl>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device_utils import get_device, model_summary

_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "zero_shot_system_prompt.prompt"
)
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if _SYSTEM_PROMPT_PATH.exists() else ""

_STOP_STR = "# Query:"


def format_sst_prompt(instruction: str) -> str:
    """将 SST 指令包装为完整 prompt。"""
    if _SYSTEM_PROMPT:
        return _SYSTEM_PROMPT.format(instruction=instruction)
    return instruction


def load_sst_data(data_path: Path) -> list[dict]:
    """加载 SimpleSafetyTests 数据（CSV 格式）。"""
    examples = []
    with open(data_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(dict(row))
    return examples


try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


def _generate_vllm(model_path: str, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
    llm = LLM(model=model_path, dtype="bfloat16")
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, stop=_STOP_STR)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def _generate_hf(
    model_path: str,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 512,
) -> list[str]:
    from device_utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device, eval_mode=True)
    tokenizer.padding_side = "left"

    results = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if _STOP_STR in text:
                text = text[:text.index(_STOP_STR)]
            results.append(text.strip())
    return results


def run_generation(
    model_path: str,
    data_path: Path,
    output_path: Path,
    device: torch.device,
    max_samples: int | None = None,
    max_new_tokens: int = 512,
) -> dict:
    """生成 SST 输出，保存为 JSONL，返回指标。"""
    examples = load_sst_data(data_path)
    if max_samples:
        examples = examples[:max_samples]

    instructions = [ex["prompts_final"] for ex in examples]
    prompts = [format_sst_prompt(instr) for instr in instructions]

    print(f"[SST] 共 {len(examples)} 条指令，设备：{model_summary(device)}")

    t0 = time.time()
    if _VLLM_AVAILABLE and device.type == "cuda":
        print("[SST] 使用 vLLM 批量推理...")
        outputs = _generate_vllm(model_path, prompts, max_new_tokens)
    else:
        print("[SST] 使用 HuggingFace generate（逐条）...")
        outputs = _generate_hf(model_path, prompts, device, max_new_tokens)
    elapsed = time.time() - t0

    n = len(examples)
    metrics = {
        "n_total":     n,
        "throughput":  n / elapsed,
        "elapsed_sec": elapsed,
    }

    print(f"[SST] 吞吐量: {metrics['throughput']:.2f} 样例/秒")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex, output in zip(examples, outputs):
            record = {**ex, "output": output}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[SST] 预测结果已保存到 {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SimpleSafetyTests 输出生成")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path",  required=True)
    parser.add_argument("--output",     default="results/sst_baseline.jsonl")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    device = get_device(args.device)
    run_generation(
        model_path=args.model_path,
        data_path=Path(args.data_path),
        output_path=Path(args.output),
        device=device,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
