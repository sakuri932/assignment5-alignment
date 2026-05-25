"""
SFT 训练脚本（Problem sft_script / sft）

在 UltraChat + SafetyTunedLlamas 混合数据上对 Llama 3.1 8B Base 进行指令微调。

用法（服务器）：
  python train_sft.py \
      --model-path  /data/a5-alignment/models/Llama-3.1-8B \
      --train-path  /data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz \
      --val-path    /data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz \
      --output-dir  /mnt/a/kong/sft_llama31_8b \
      --seq-length  512 \
      --batch-size  2 \
      --grad-accum  16 \
      --epochs      1 \
      --lr          2e-5 \
      --warmup-frac 0.03 \
      [--device cuda]  [--wandb-project cs336-sft]

推荐超参数（PDF Section 3.2.2）：
  lr=2e-5, 余弦衰减, 线性预热占总步数 3%
  有效批大小 = batch_size × grad_accum = 32 条序列
  seq_length=512, 1 epoch

说明：
  - 磁盘写入必须到 /mnt/a（见 server security 配置）
  - wandb API key 通过 ~/.netrc 读取，不写入代码
  - 跨平台：CUDA（服务器/Windows）和 MPS/CPU（macOS）均支持
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device_utils import get_device, model_summary
from sft_dataset import get_packed_sft_dataset, iterate_batches


# ── 学习率调度：线性预热 + 余弦衰减 ─────────────────────────────────────────

def get_cosine_with_warmup_lr_lambda(
    current_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """线性预热 + 余弦衰减的 LR 乘子。"""
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_lr_ratio, cosine)


# ── 验证损失 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_val_loss(
    model: torch.nn.Module,
    val_dataloader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """在验证集上计算平均 cross-entropy 损失（最多 max_batches 个批次）。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        logits = model(input_ids).logits               # (B, L, V)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break
    model.train()
    return total_loss / max(1, n_batches)


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"[SFT] 设备：{model_summary(device)}")

    # ── 加载模型和 tokenizer ──
    from device_utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device, eval_mode=False
    )
    model.train()

    # ── 数据集 ──
    print("[SFT] 加载训练集...")
    train_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=args.train_path,
        seq_length=args.seq_length,
        shuffle=True,
    )
    print(f"[SFT] 训练集：{len(train_dataset)} 条序列")

    val_dataset = None
    if args.val_path:
        print("[SFT] 加载验证集...")
        val_dataset = get_packed_sft_dataset(
            tokenizer=tokenizer,
            dataset_path=args.val_path,
            seq_length=args.seq_length,
            shuffle=False,
        )
        print(f"[SFT] 验证集：{len(val_dataset)} 条序列")

    train_loader = iterate_batches(train_dataset, args.batch_size, shuffle=True)
    val_loader   = iterate_batches(val_dataset, args.batch_size, shuffle=False) if val_dataset else None

    # ── 优化器和调度器 ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.epochs // args.grad_accum
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    print(f"[SFT] 总优化步数: {total_steps}，预热步数: {warmup_steps}")

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_cosine_with_warmup_lr_lambda(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )

    # ── 可选 wandb ──
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            print("[SFT] wandb 未安装，跳过日志记录")
            use_wandb = False

    # ── 训练 ──
    global_step = 0
    optimizer_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\n[SFT] Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            logits = model(input_ids).logits               # (B, L, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ) / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum    # 还原未除前的值

            global_step += 1

            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if optimizer_step % args.log_interval == 0:
                    avg_loss = epoch_loss / global_step
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(
                        f"  step {optimizer_step:>6d} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr_now:.2e} | "
                        f"{elapsed:.0f}s"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr_now,
                            "train/optimizer_step": optimizer_step,
                        })

        # ── 每个 epoch 结束：验证 ──
        if val_loader is not None:
            val_loss = evaluate_val_loss(model, val_loader, device, max_batches=100)
            print(f"[SFT] Epoch {epoch + 1} 结束 | 验证损失: {val_loss:.4f}")
            if use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "epoch": epoch + 1})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = Path(args.output_dir) / "best"
                model.save_pretrained(str(best_ckpt))
                tokenizer.save_pretrained(str(best_ckpt))
                print(f"[SFT] 最优模型已保存到 {best_ckpt}")

    # ── 保存最终模型 ──
    final_dir = Path(args.output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[SFT] 训练完成，最终模型已保存到 {final_dir}")

    if use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="SFT 训练：Llama 3.1 8B 指令微调")
    parser.add_argument("--model-path",   required=True, help="预训练模型目录")
    parser.add_argument("--train-path",   required=True, help="训练集 JSONL（支持 .gz）")
    parser.add_argument("--val-path",     default=None,  help="验证集 JSONL（支持 .gz），可选")
    parser.add_argument("--output-dir",   required=True, help="模型保存目录（服务器用 /mnt/a/...）")
    parser.add_argument("--seq-length",   type=int, default=512,   help="序列长度（默认 512）")
    parser.add_argument("--batch-size",   type=int, default=2,     help="每步批大小（默认 2）")
    parser.add_argument("--grad-accum",   type=int, default=16,    help="梯度累积步数（有效批=batch×accum，默认16→有效批32）")
    parser.add_argument("--epochs",       type=int, default=1,     help="训练轮数（默认 1）")
    parser.add_argument("--lr",           type=float, default=2e-5, help="峰值学习率（默认 2e-5）")
    parser.add_argument("--warmup-frac",  type=float, default=0.03, help="预热步数占总步数比例（默认 0.03）")
    parser.add_argument("--log-interval", type=int, default=10,    help="每多少优化步打印一次日志")
    parser.add_argument("--device",       default=None, help="设备（cuda/mps/cpu，默认自动检测）")
    parser.add_argument("--wandb-project", default="",  help="Weights & Biases 项目名（空字符串=不记录）")
    args = parser.parse_args()

    # 服务器安全：确保输出不写到根磁盘
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
