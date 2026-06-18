"""
单卡 GRPO 训练驱动（kong RTX 4090, 24GB）—— OLMo-2-0425-1B + LoRA + HF generate。

设计要点（为单张 24GB 卡适配，偏离 PDF 的双卡 vLLM 方案）：
  - LoRA 微调：基座 bf16 冻结，只训 LoRA adapter，AdamW 优化器状态极小，省下 ~12-18GB
  - HF model.generate() 在进程内生成 rollout（替代独立 vLLM GPU + NCCL 权重同步）
  - 生成与训练用同一套（LoRA 适配后的）权重 → 严格在策略（on-policy）
  - 缩小规模：n_prompts/group_size/max_new_tokens 都可调小

标准在策略 GRPO 配置：baseline=mean, advantage_normalizer=std,
importance_reweighting=none, loss_normalization=sequence。

用法：
  python grpo_train_kong.py --max-steps 150 --wandb-project grpo-olmo-kong
"""
from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import wandb

from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

_HERE = Path(__file__).resolve().parent
# 兼容两种布局：脚本与 cs336_alignment 同级（kong），或在 scripts/ 子目录（本地仓库）
for _cand in (_HERE, _HERE.parent):
    if (_cand / "cs336_alignment" / "prompts" / "r1_zero.prompt").exists():
        ROOT = _cand
        break
else:
    ROOT = _HERE
DEFAULT_MODEL = "/mnt/a/kong/workspace/ass5/OLMo-2-0425-1B"
DEFAULT_GSM8K = "/mnt/a/kong/workspace/ass5/data/gsm8k/train.jsonl"
DEFAULT_PROMPT = ROOT / "cs336_alignment" / "prompts" / "r1_zero.prompt"


def parse_gsm8k_answer(answer_field: str) -> str:
    # GSM8K 答案末尾固定 "...\n#### N"
    return answer_field.rsplit("####", 1)[-1].strip()


def load_gsm8k(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    return [{"question": r["question"], "answer": parse_gsm8k_answer(r["answer"])} for r in rows]


@torch.no_grad()
def generate_rollouts(model, tokenizer, prompts, group_size, max_new_tokens,
                      gen_prompt_chunk, temperature, top_p, device):
    """对每个 prompt 采样 group_size 条 rollout。
    返回展平列表，顺序为 [p0_g0..p0_g{G-1}, p1_g0.., ...]，与 GRPO 分组一致。"""
    model.eval()
    tokenizer.padding_side = "left"  # decoder-only 批量生成必须左 padding
    responses: list[str] = []
    for i in range(0, len(prompts), gen_prompt_chunk):
        chunk = prompts[i:i + gen_prompt_chunk]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=["</answer>"],
            tokenizer=tokenizer,
        )
        gen = out[:, enc.input_ids.shape[1]:]  # 去掉 prompt 前缀
        for g in gen:
            txt = tokenizer.decode(g, skip_special_tokens=True)
            # 保险：若 stop 后仍有多余内容，截断到 </answer>
            if "</answer>" in txt:
                txt = txt[: txt.index("</answer>") + len("</answer>")]
            responses.append(txt)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--gsm8k", default=DEFAULT_GSM8K)
    ap.add_argument("--prompt-template", default=str(DEFAULT_PROMPT))
    ap.add_argument("--n-prompts", type=int, default=8, help="每步采样的 prompt 数")
    ap.add_argument("--group-size", type=int, default=8, help="每个 prompt 的 rollout 数 G")
    ap.add_argument("--gen-prompt-chunk", type=int, default=2, help="生成时一次喂几个 prompt")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=16, help="梯度累积步数（切 microbatch）")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-dir", default="/mnt/a/kong/workspace/ass5/grpo_ckpt")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eval-sample-every", type=int, default=10, help="每隔几步打印一条样例")
    ap.add_argument("--wandb-project", default="grpo-olmo-kong")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    # ── 数据 + 模板 ────────────────────────────────────────────────
    data = load_gsm8k(args.gsm8k)
    template = Path(args.prompt_template).read_text()
    print(f"[setup] GSM8K train: {len(data)} 条 | 模板: {args.prompt_template}", flush=True)

    # ── 模型 + LoRA ────────────────────────────────────────────────
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = True  # 生成时用 KV cache

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    print(f"[setup] 模型+LoRA 加载完成 {time.time()-t0:.1f}s", flush=True)

    # ── wandb ──────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ── GRPO 主循环 ────────────────────────────────────────────────
    for step in range(1, args.max_steps + 1):
        t_step = time.time()
        # 采样一批 prompt
        batch = random.sample(data, args.n_prompts)
        questions = [b["question"] for b in batch]
        gts = [b["answer"] for b in batch]
        prompts = [template.format(question=q) for q in questions]

        # 生成 rollout（on-policy）
        t_gen = time.time()
        responses = generate_rollouts(
            model, tokenizer, prompts, args.group_size, args.max_new_tokens,
            args.gen_prompt_chunk, args.temperature, args.top_p, device,
        )
        gen_time = time.time() - t_gen

        # 展开成 N = n_prompts * G 的对齐列表
        repeated_prompts = [prompts[i] for i in range(len(prompts)) for _ in range(args.group_size)]
        repeated_gts = [gts[i] for i in range(len(gts)) for _ in range(args.group_size)]

        # GRPO 更新（标准在策略）
        model.train()
        t_train = time.time()
        loss, meta = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            reward_fn=r1_zero_reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            baseline="mean",
            advantage_normalizer="std",
            importance_reweighting_method="none",
            loss_normalization="sequence",
        )
        train_time = time.time() - t_train
        if device == "cuda":
            torch.cuda.empty_cache()

        gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        log = {
            "step": step,
            "loss": float(loss),
            "mean_reward": meta.get("mean_reward"),
            "mean_format_reward": meta.get("mean_format_reward"),
            "reward_std": meta.get("reward_std"),
            "grad_norm": meta.get("grad_norm"),
            "gen_time_s": gen_time,
            "train_time_s": train_time,
            "step_time_s": time.time() - t_step,
            "gpu_gb": gpu_gb,
        }
        print(f"[step {step:3d}] reward={log['mean_reward']:.3f} "
              f"fmt={log['mean_format_reward']:.3f} loss={log['loss']:.4f} "
              f"|grad|={log['grad_norm']:.2f} gen={gen_time:.0f}s train={train_time:.0f}s "
              f"gpu={gpu_gb:.1f}GB", flush=True)
        if use_wandb:
            wandb.log(log, step=step)

        # 周期性打印一条样例
        if step % args.eval_sample_every == 0:
            print(f"  [样例] Q: {questions[0][:80]}...", flush=True)
            print(f"  [样例] A(gt={gts[0]}): {responses[0][:200]}", flush=True)

        # 周期性保存 LoRA adapter
        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_dir = Path(args.save_dir) / f"step_{step}"
            model.save_pretrained(str(ckpt_dir))
            print(f"  [ckpt] LoRA adapter 已保存 → {ckpt_dir}", flush=True)

    if use_wandb:
        wandb.finish()
    print("[done] GRPO 训练完成", flush=True)


if __name__ == "__main__":
    main()
