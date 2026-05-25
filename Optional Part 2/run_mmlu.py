"""
零样本 MMLU 评估脚本（Problem mmlu_baseline）

用法：
  python run_mmlu.py \
      --model-path /data/a5-alignment/models/Llama-3.1-8B \
      --data-path  /path/to/mmlu/test.jsonl \
      --output     results/mmlu_baseline.jsonl \
      [--device cuda]  [--max-samples 500]

跨平台：
  - CUDA（Linux/Windows NVIDIA）：使用 vLLM 批量推理（若安装），否则 HuggingFace generate
  - MPS（macOS Apple Silicon）：HuggingFace generate
  - CPU：HuggingFace generate（速度慢，仅用于调试）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# 将 Optional Part 2 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device_utils import get_device, get_compute_dtype, model_summary
from cs336_alignment.utils import parse_mmlu_response

# ── 系统 prompt（来自 PDF Section 2）──────────────────────────────────────
_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "zero_shot_system_prompt.prompt"
)
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if _SYSTEM_PROMPT_PATH.exists() else ""

# ── MMLU 格式化模板 ────────────────────────────────────────────────────────
_MMLU_QUERY_TEMPLATE = """\
Answer the following multiple choice question about {subject}. Respond with a single \
sentence of the form "The correct answer is _", filling the blank with the letter \
corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}
Answer:"""

_STOP_STR = "# Query:"   # 遇到此字符串时停止生成


def format_mmlu_prompt(example: dict) -> str:
    """将 MMLU 样例格式化为完整的模型输入 prompt。"""
    query = _MMLU_QUERY_TEMPLATE.format(
        subject=example["subject"],
        question=example["question"],
        opt_a=example["options"][0],
        opt_b=example["options"][1],
        opt_c=example["options"][2],
        opt_d=example["options"][3],
    )
    if _SYSTEM_PROMPT:
        return _SYSTEM_PROMPT.format(instruction=query)
    return query


def load_mmlu_data(data_path: Path) -> list[dict]:
    """加载 MMLU 数据（JSONL 格式）。"""
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ── 推理后端：尝试 vLLM，回退到 HuggingFace ──────────────────────────────

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


def _generate_vllm(model_path: str, prompts: list[str], max_new_tokens: int = 64) -> list[str]:
    """使用 vLLM 批量生成（CUDA 环境下推荐）。"""
    llm = LLM(model=model_path, dtype="bfloat16")
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, stop=_STOP_STR)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def _generate_hf(
    model_path: str,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 64,
) -> list[str]:
    """使用 HuggingFace generate 逐条生成（跨平台兼容）。"""
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
            # 去掉输入部分，只保留生成的 token
            gen_ids = out[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # 截断到 stop string
            if _STOP_STR in text:
                text = text[:text.index(_STOP_STR)]
            results.append(text.strip())
    return results


def run_evaluation(
    model_path: str,
    data_path: Path,
    output_path: Path,
    device: torch.device,
    max_samples: int | None = None,
    max_new_tokens: int = 64,
) -> dict:
    """运行 MMLU 零样本评估，返回评估指标字典。"""
    examples = load_mmlu_data(data_path)
    if max_samples:
        examples = examples[:max_samples]

    prompts = [format_mmlu_prompt(ex) for ex in examples]

    print(f"[MMLU] 共 {len(examples)} 条样例，设备：{model_summary(device)}")

    # 生成
    t0 = time.time()
    if _VLLM_AVAILABLE and device.type == "cuda":
        print("[MMLU] 使用 vLLM 批量推理...")
        outputs = _generate_vllm(model_path, prompts, max_new_tokens)
    else:
        print("[MMLU] 使用 HuggingFace generate（逐条）...")
        outputs = _generate_hf(model_path, prompts, device, max_new_tokens)
    elapsed = time.time() - t0

    # 解析 + 评分
    results = []
    n_correct = 0
    n_parseable = 0
    for ex, output in zip(examples, outputs):
        pred = parse_mmlu_response(ex, output)
        correct = (pred == ex["answer"]) if pred else False
        if pred:
            n_parseable += 1
        if correct:
            n_correct += 1
        results.append({**ex, "model_output": output, "predicted": pred, "correct": correct})

    n = len(examples)
    metrics = {
        "n_total":     n,
        "n_parseable": n_parseable,
        "n_failed":    n - n_parseable,
        "accuracy":    n_correct / n,
        "throughput":  n / elapsed,
        "elapsed_sec": elapsed,
    }

    print(f"[MMLU] 准确率: {metrics['accuracy']:.3f} | "
          f"无法解析: {metrics['n_failed']} | "
          f"吞吐量: {metrics['throughput']:.2f} 样例/秒")

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[MMLU] 结果已保存到 {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="零样本 MMLU 评估")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path",  required=True)
    parser.add_argument("--output",     default="results/mmlu_baseline.json")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = get_device(args.device)
    run_evaluation(
        model_path=args.model_path,
        data_path=Path(args.data_path),
        output_path=Path(args.output),
        device=device,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
