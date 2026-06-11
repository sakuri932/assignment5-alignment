"""
Problem prompting_baselines (5 分): 在 GSM8K 上评测 OLMo-2-0425-1B 的三种提示性能。

三种提示:
  - question_only        零样本, \boxed{} 格式, 无 stop
  - r1_zero              零样本, <think>/<answer> 格式, stop="</answer>"
  - r1_zero_three_shot   三样本, <think>/<answer> 格式, stop="</answer>"

按题目要求: 温度 1.0, top-p 1.0, max_new_tokens=512。
按类别 (1)/(2)/(3) 统计, 并保存每类至少 10 个样例。

MPS 内存优化:
- 通过 PYTORCH_MPS_HIGH_WATERMARK_RATIO 把 MPS 池上限压到物理内存的 75%
- 每条推理后 empty_cache, 防止 KV cache 累积膨胀
- print 全部 flush=True, pipe/重定向时实时输出
- 增量写中间结果, 防止崩溃丢失
"""
from __future__ import annotations

import os
# 必须在 import torch 之前设置 MPS 内存水位:
#   HIGH = 池上限(0.85 → 物理内存的 85%, 留 15% 给系统避免触发 swap)
#   LOW  = 触发主动回收的阈值(必须 < HIGH; 0.5 → 池超过 50% 物理内存时积极释放空闲块)
# 同时设置, 否则 PyTorch 会因为 LOW(默认 1.4) > HIGH 报 "invalid low watermark ratio"
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import argparse
import functools
import json
import random
import time
from pathlib import Path

# 全局重定向 print: 默认带 flush, pipe/重定向时进度实时刷新
print = functools.partial(print, flush=True)  # noqa: A001

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "cs336_alignment" / "prompts"
GSM8K_TEST = ROOT / "data" / "gsm8k" / "test.jsonl"

PROMPT_CONFIGS = {
    "question_only": {
        "template_file": "question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
    },
    "r1_zero": {
        "template_file": "r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": "</answer>",
    },
    "r1_zero_three_shot": {
        "template_file": "r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": "</answer>",
    },
}


def parse_gsm8k_answer(answer_field: str) -> str:
    # GSM8K 答案末尾固定格式 "...\n#### N"
    return answer_field.rsplit("####", 1)[-1].strip()


def load_examples(n: int, seed: int) -> list[dict]:
    examples = [json.loads(l) for l in open(GSM8K_TEST)]
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n]


def categorize(format_r: float, answer_r: float) -> int:
    # 1: 格式对+答案对; 2: 格式对+答案错; 3: 格式错+答案错; 4: 格式错+答案对(极少见)
    if format_r == 1.0 and answer_r == 1.0:
        return 1
    if format_r == 1.0 and answer_r == 0.0:
        return 2
    if format_r == 0.0 and answer_r == 0.0:
        return 3
    return 4


def _empty_cache(device: str) -> None:
    """显式释放未占用的 MPS/CUDA 池内存, 防止 KV cache 在循环中累积。"""
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def generate_one(model, tok, prompt: str, stop: str | None, max_new_tokens: int, device: str) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tok.pad_token_id,
    )
    if stop is not None:
        gen_kwargs["stop_strings"] = [stop]
        gen_kwargs["tokenizer"] = tok
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # 安全兜底: 若模型在 stop 之后还多吐了 token(stop_strings 的边界处理),截断到 stop 之后
    if stop is not None and stop in text:
        text = text[: text.index(stop) + len(stop)]
    # 显式释放, 帮助 MPS 立即回收 KV cache 等临时张量
    del inputs, out
    _empty_cache(device)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="GSM8K 测试子集大小")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--model-path", default="/Users/admin/Documents/GitHub/CS336/OLMo-2-0425-1B")
    parser.add_argument("--out", default=str(ROOT / "extra_guidance" / "prompting_baselines_results.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} | n={args.n} | max_new_tokens={args.max_new_tokens}")

    t0 = time.time()
    model, tok = get_model_and_tokenizer(args.model_path, device)
    model.eval()
    _empty_cache(device)  # 加载完释放掉 CPU->MPS 搬运过程的临时分配
    print(f"[setup] 模型加载完成: {time.time()-t0:.1f}s")

    examples = load_examples(args.n, args.seed)
    ground_truths = [parse_gsm8k_answer(ex["answer"]) for ex in examples]
    questions = [ex["question"] for ex in examples]

    all_results: dict[str, dict] = {}

    for prompt_name, cfg in PROMPT_CONFIGS.items():
        template = (PROMPTS_DIR / cfg["template_file"]).read_text()
        reward_fn = cfg["reward_fn"]
        stop = cfg["stop"]

        per_example: list[dict] = []
        category_count = {1: 0, 2: 0, 3: 0, 4: 0}
        t_start = time.time()

        print(f"\n[{prompt_name}] 跑 {args.n} 条...")
        for i, (q, gt) in enumerate(zip(questions, ground_truths)):
            prompt = template.format(question=q)
            t_one = time.time()
            response = generate_one(model, tok, prompt, stop, args.max_new_tokens, device)
            elapsed_one = time.time() - t_one

            r = reward_fn(response, gt)
            cat = categorize(r["format_reward"], r["answer_reward"])
            category_count[cat] += 1

            per_example.append({
                "idx": i,
                "question": q,
                "ground_truth": gt,
                "response": response,
                "format_reward": r["format_reward"],
                "answer_reward": r["answer_reward"],
                "reward": r["reward"],
                "category": cat,
            })

            done = i + 1
            avg = (time.time() - t_start) / done
            eta = avg * (args.n - done)
            if done % 5 == 0 or done == args.n:
                print(f"  [{done:3d}/{args.n}] cat={cat} | {elapsed_one:.1f}s/ex | "
                      f"已用 {time.time()-t_start:.0f}s | ETA {eta:.0f}s | "
                      f"cum: 1={category_count[1]} 2={category_count[2]} 3={category_count[3]} 4={category_count[4]}")
                # 增量写中间结果, 防止崩溃丢失
                partial_out = Path(args.out).with_suffix(f".{prompt_name}.partial.json")
                partial_out.parent.mkdir(parents=True, exist_ok=True)
                with open(partial_out, "w") as f:
                    json.dump({"prompt": prompt_name, "done": done, "category_counts": category_count,
                               "per_example": per_example}, f, ensure_ascii=False)

        total = sum(category_count.values())
        all_results[prompt_name] = {
            "n_examples": total,
            "category_counts": category_count,
            "category_pct": {k: v / total for k, v in category_count.items()},
            "mean_format_reward": sum(e["format_reward"] for e in per_example) / total,
            "mean_answer_reward": sum(e["answer_reward"] for e in per_example) / total,
            "wall_time_sec": time.time() - t_start,
            "per_example": per_example,
        }
        print(f"[{prompt_name}] 完成: {dict(category_count)}, 准确率={category_count[1]/total:.1%}, 用时 {time.time()-t_start:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] 结果保存到 {args.out}")
    print(f"[done] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
