"""
DPO 训练脚本（Problem dpo_training）

在 Anthropic HH 偏好数据上对 SFT 模型进行 DPO 对齐。

用法（服务器，2 块 GPU）：
  python train_dpo.py \
      --model-path  /mnt/a/kong/sft_llama31_8b/final \
      --hh-data-dir /data/a5-alignment/hh \
      --output-dir  /mnt/a/kong/dpo_llama31_8b \
      --beta         0.1 \
      --batch-size   64 \
      --lr           1e-6 \
      --epochs       1 \
      [--device-lm  cuda:0]  [--device-ref cuda:1] \
      [--wandb-project cs336-dpo]

核心设计（PDF Section 5.4）：
  - 两块 GPU：device-lm 放训练模型，device-ref 放冻结参考模型
  - RMSprop 优化器（与原始 DPO 论文一致，避免 AdamW 的显存问题）
  - 梯度累积：有效批大小 = 1 × grad_accum（逐条处理）
  - 验证集（200 条）追踪隐式奖励分类准确率
  - 保存验证准确率最高时的模型

说明：
  - 磁盘写入必须到 /mnt/a（见 server security 配置）
  - wandb API key 通过 ~/.netrc 读取，不写入代码
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device_utils import get_device, model_summary
from data_hh import load_hh_dataset, split_train_val
from dpo_hf import compute_per_instance_dpo_loss, compute_dpo_reward_accuracy


def load_model_for_dpo(model_path: str, device: torch.device, freeze: bool = False):
    """加载模型到指定设备，freeze=True 时冻结所有参数（参考模型）。"""
    from device_utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device, eval_mode=freeze)
    if not freeze:
        model.train()
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    return model, tokenizer


# ── 验证：隐式奖励分类准确率 ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_reward_accuracy(
    lm,
    lm_ref,
    tokenizer,
    beta: float,
    val_examples,
    device_lm: torch.device,
    device_ref: torch.device,
    max_samples: int = 200,
) -> float:
    """计算验证集上隐式奖励模型对 chosen > rejected 的分类准确率。"""
    lm.eval()
    n_correct = 0
    n_total = min(len(val_examples), max_samples)
    for ex in val_examples[:n_total]:
        correct = compute_dpo_reward_accuracy(
            lm=lm, lm_ref=lm_ref,
            tokenizer=tokenizer,
            beta=beta,
            prompt=ex.instruction,
            response_chosen=ex.response_chosen,
            response_rejected=ex.response_rejected,
            device_lm=device_lm, device_ref=device_ref,
        )
        if correct:
            n_correct += 1
    lm.train()
    return n_correct / max(1, n_total)


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── 设备解析 ──
    device_lm  = torch.device(args.device_lm)  if args.device_lm  else get_device()
    device_ref = torch.device(args.device_ref) if args.device_ref else device_lm

    print(f"[DPO] 训练模型设备：{model_summary(device_lm)}")
    print(f"[DPO] 参考模型设备：{model_summary(device_ref)}")

    # ── 加载模型 ──
    print("[DPO] 加载训练模型...")
    lm, tokenizer = load_model_for_dpo(args.model_path, device_lm, freeze=False)
    print("[DPO] 加载参考模型（冻结）...")
    lm_ref, _ = load_model_for_dpo(args.model_path, device_ref, freeze=True)

    # ── 加载数据 ──
    print("[DPO] 加载 HH 数据集...")
    all_examples = load_hh_dataset(args.hh_data_dir)
    train_examples, val_examples = split_train_val(all_examples, val_size=200, seed=42)
    print(f"[DPO] 训练集: {len(train_examples)} 条，验证集: {len(val_examples)} 条")

    # ── 优化器（RMSprop，与原始 DPO 论文一致）──
    optimizer = torch.optim.RMSprop(lm.parameters(), lr=args.lr)

    # ── 可选 wandb ──
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            print("[DPO] wandb 未安装，跳过日志记录")
            use_wandb = False

    # ── 训练 ──
    best_val_acc = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n[DPO] Epoch {epoch + 1}/{args.epochs}")

        # 每个 epoch 随机打乱训练集
        import random
        rng = random.Random(epoch)
        epoch_data = train_examples[:]
        rng.shuffle(epoch_data)

        optimizer.zero_grad()
        running_loss = 0.0
        optimizer_step = 0
        t0 = time.time()

        for idx, ex in enumerate(epoch_data):
            loss = compute_per_instance_dpo_loss(
                lm=lm, lm_ref=lm_ref,
                tokenizer=tokenizer,
                beta=args.beta,
                prompt=ex.instruction,
                response_chosen=ex.response_chosen,
                response_rejected=ex.response_rejected,
                device_lm=device_lm, device_ref=device_ref,
            )
            # 梯度累积：对 batch_size 个样本的梯度求平均
            (loss / args.batch_size).backward()
            running_loss += loss.item()

            if (idx + 1) % args.batch_size == 0:
                torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if optimizer_step % args.log_interval == 0:
                    avg_loss = running_loss / (idx + 1)
                    elapsed = time.time() - t0
                    print(
                        f"  step {optimizer_step:>5d} | "
                        f"loss {avg_loss:.4f} | "
                        f"{elapsed:.0f}s"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "train/dpo_loss": avg_loss,
                            "optimizer_step": optimizer_step,
                        })

                # 定期验证
                if optimizer_step % args.val_interval == 0:
                    val_acc = evaluate_reward_accuracy(
                        lm, lm_ref, tokenizer, args.beta,
                        val_examples, device_lm, device_ref,
                    )
                    print(f"  [验证] 奖励分类准确率: {val_acc:.3f}")
                    if use_wandb:
                        import wandb
                        wandb.log({"val/reward_accuracy": val_acc, "optimizer_step": optimizer_step})

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_ckpt = output_dir / "best"
                        lm.save_pretrained(str(best_ckpt))
                        tokenizer.save_pretrained(str(best_ckpt))
                        print(f"  [验证] 新最优模型已保存（准确率 {val_acc:.3f}）→ {best_ckpt}")

        # epoch 结束时处理剩余梯度
        remaining = len(epoch_data) % args.batch_size
        if remaining > 0:
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # epoch 结束验证
        val_acc = evaluate_reward_accuracy(
            lm, lm_ref, tokenizer, args.beta,
            val_examples, device_lm, device_ref,
        )
        avg_loss = running_loss / len(epoch_data)
        print(
            f"[DPO] Epoch {epoch + 1} 结束 | "
            f"平均损失: {avg_loss:.4f} | "
            f"验证准确率: {val_acc:.3f}"
        )
        if use_wandb:
            import wandb
            wandb.log({"val/reward_accuracy": val_acc, "epoch": epoch + 1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = output_dir / "best"
            lm.save_pretrained(str(best_ckpt))
            tokenizer.save_pretrained(str(best_ckpt))
            print(f"[DPO] 新最优模型已保存（准确率 {val_acc:.3f}）→ {best_ckpt}")

    # ── 保存最终模型 ──
    final_dir = output_dir / "final"
    lm.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[DPO] 训练完成，最终模型已保存到 {final_dir}")
    print(f"[DPO] 验证集最优奖励分类准确率: {best_val_acc:.3f}")

    if use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="DPO 训练：Llama 3.1 8B 偏好对齐")
    parser.add_argument("--model-path",   required=True, help="SFT 模型目录（训练起点）")
    parser.add_argument("--hh-data-dir",  required=True, help="HH 数据目录（含 4 个 .jsonl.gz 文件）")
    parser.add_argument("--output-dir",   required=True, help="模型保存目录（服务器用 /mnt/a/...）")
    parser.add_argument("--beta",         type=float, default=0.1,  help="DPO KL 惩罚系数 β（默认 0.1）")
    parser.add_argument("--batch-size",   type=int,   default=64,   help="有效批大小（梯度累积步数，默认 64）")
    parser.add_argument("--lr",           type=float, default=1e-6, help="学习率（默认 1e-6）")
    parser.add_argument("--epochs",       type=int,   default=1,    help="训练轮数（默认 1）")
    parser.add_argument("--log-interval", type=int,   default=10,   help="每多少优化步打印一次训练日志")
    parser.add_argument("--val-interval", type=int,   default=50,   help="每多少优化步验证一次")
    parser.add_argument("--device-lm",   default=None, help="训练模型设备（默认 cuda:0 或自动检测）")
    parser.add_argument("--device-ref",  default=None, help="参考模型设备（默认与 device-lm 相同）")
    parser.add_argument("--wandb-project", default="", help="Weights & Biases 项目名（空=不记录）")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
