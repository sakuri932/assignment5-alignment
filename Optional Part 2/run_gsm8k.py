"""
零样本 GSM8K 评估脚本（Problem gsm8k_baseline）

用法：
  python run_gsm8k.py \
      --model-path /data/a5-alignment/models/Llama-3.1-8B \
      --data-path  /path/to/gsm8k/test.jsonl \
      --output     results/gsm8k_baseline.json \
      [--device cuda]  [--max-samples 500]

提示格式：
  {question}
  Answer:

解析方式：取模型输出中的最后一个数字作为预测答案。
金标准答案从 "#### N" 格式中提取。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device_utils import get_device, model_summary
from cs336_alignment.utils import parse_gsm8k_response

_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "zero_shot_system_prompt.prompt"
)
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if _SYSTEM_PROMPT_PATH.exists() else ""

_STOP_STR = "# Query:"

# 从 GSM8K 答案字段 "...#### 18" 中提取数字
_GOLD_ANS_RE = re.compile(r"####\s*([\d,]+)")


def _parse_gold(answer_str: str) -> str | None:
    """从 GSM8K gold answer 字段中提取数字字符串。"""
    m = _GOLD_ANS_RE.search(answer_str)
    if m:
        return m.group(1).replace(",", "")
    return None


def format_gsm8k_prompt(example: dict) -> str:
    """格式化 GSM8K 样例为完整 prompt。"""
    query = f"{example['question']}\nAnswer:"
    if _SYSTEM_PROMPT:
        return _SYSTEM_PROMPT.format(instruction=query)
    return query


def load_gsm8k_data(data_path: Path) -> list[dict]:
    """加载 GSM8K 数据（JSONL 格式）。"""
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


def _generate_vllm(model_path: str, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    llm = LLM(model=model_path, dtype="bfloat16")
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, stop=_STOP_STR)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def _generate_hf(
    model_path: str,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 256,
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


def run_evaluation(
    model_path: str,
    data_path: Path,
    output_path: Path,
    device: torch.device,
    max_samples: int | None = None,
    max_new_tokens: int = 256,
) -> dict:
    """运行 GSM8K 零样本评估，返回评估指标字典。"""
    examples = load_gsm8k_data(data_path)
    if max_samples:
        examples = examples[:max_samples]

    prompts = [format_gsm8k_prompt(ex) for ex in examples]

    print(f"[GSM8K] 共 {len(examples)} 条样例，设备：{model_summary(device)}")

    t0 = time.time()
    if _VLLM_AVAILABLE and device.type == "cuda":
        print("[GSM8K] 使用 vLLM 批量推理...")
        outputs = _generate_vllm(model_path, prompts, max_new_tokens)
    else:
        print("[GSM8K] 使用 HuggingFace generate（逐条）...")
        outputs = _generate_hf(model_path, prompts, device, max_new_tokens)
    elapsed = time.time() - t0

    results = []
    n_correct = 0
    n_parseable = 0
    for ex, output in zip(examples, outputs):
        pred = parse_gsm8k_response(ex, output)
        gold = _parse_gold(ex.get("answer", ""))
        correct = (pred == gold) if (pred is not None and gold is not None) else False
        if pred is not None:
            n_parseable += 1
        if correct:
            n_correct += 1
        results.append({
            "question": ex["question"],
            "answer":   ex["answer"],
            "gold_parsed": gold,
            "model_output": output,
            "predicted": pred,
            "correct": correct,
        })

    n = len(examples)
    metrics = {
        "n_total":     n,
        "n_parseable": n_parseable,
        "n_failed":    n - n_parseable,
        "accuracy":    n_correct / n,
        "throughput":  n / elapsed,
        "elapsed_sec": elapsed,
    }

    print(f"[GSM8K] 准确率: {metrics['accuracy']:.3f} | "
          f"无法解析: {metrics['n_failed']} | "
          f"吞吐量: {metrics['throughput']:.2f} 样例/秒")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[GSM8K] 结果已保存到 {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="零样本 GSM8K 评估")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path",  required=True)
    parser.add_argument("--output",     default="results/gsm8k_baseline.json")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
