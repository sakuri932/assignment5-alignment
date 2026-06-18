#!/bin/bash
# kong 链式 GRPO 任务：等 Run A（3-shot，已在跑）结束后，自动启动 Run B（零样本大 batch）。
# 全程 nohup 脱离，SSH 断开也不影响。
set -u
cd /mnt/a/kong/workspace/ass5
export TMPDIR=/mnt/a/kong/tmp
export HF_HOME=/mnt/a/kong/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/mnt/a/kong/workspace/ass5

echo "[chain] $(date '+%F %T') 等待 Run A (3-shot) 结束..."
# 按 prompt 路径里的 three_shot 关键字匹配 Run A 的 python 进程
while pgrep -f "three_shot_gsm8k" >/dev/null 2>&1; do
  sleep 30
done
echo "[chain] $(date '+%F %T') Run A 已结束。等 10s 释放显存后启动 Run B..."
sleep 10

# Run B：零样本 r1_zero + 256 rollout 大 batch（n_prompts=16 × group_size=16）
# 单次生成/微批仍保持小尺寸（gen_prompt_chunk=2, grad_accum=64 → 微批=4），峰值显存与 Run A 相当
.venv/bin/python grpo_train_kong.py \
  --prompt-template /mnt/a/kong/workspace/ass5/cs336_alignment/prompts/r1_zero.prompt \
  --n-prompts 16 --group-size 16 --gen-prompt-chunk 2 --grad-accum 64 \
  --max-new-tokens 300 --lr 1e-5 --max-steps 150 \
  --eval-sample-every 10 --save-every 50 \
  --wandb-project grpo-olmo-kong --wandb-run-name zeroshot-bigbatch-grpo \
  > runB_zeroshot.log 2>&1

echo "[chain] $(date '+%F %T') Run B 已结束。全部完成。"
