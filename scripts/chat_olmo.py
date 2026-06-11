"""
OLMo-2-0425-1B 交互式 chat 脚本（流式输出 + MPS 内存管理）。

特性:
  - 逐 token 流式输出 (TextIteratorStreamer + 后台 generate 线程)
  - 多轮对话, 在 token id 层面拼接历史 (零重复 tokenize)
  - MPS 内存控制:
      * 启动设 watermark 防 swap
      * 每轮回答后 empty_cache 释放 KV cache 残留
      * 历史 token 超阈值自动滑窗 trim 一半
  - Ctrl+C 中断当前生成, /exit 退出, /clear 重置历史, /mem 查询状态

用法:
  python scripts/chat_olmo.py
  python scripts/chat_olmo.py --style raw           # 纯补全, 无对话模板
  python scripts/chat_olmo.py --temperature 0.9     # 更随机
  python scripts/chat_olmo.py --max-history 2048    # 调历史窗口
  python scripts/chat_olmo.py --keep-kv-cache       # 跨轮复用 KV cache, 跳过历史 prefill (内存代价大)
"""
from __future__ import annotations

import os
# 必须在 import torch 之前设, 把 MPS 内存池压在物理内存 85% 内, 避免 swap
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import argparse
import sys
import time
from threading import Thread

import torch
from transformers import TextIteratorStreamer

from cs336_alignment.checkpoint import get_model_and_tokenizer


DEFAULT_MODEL_PATH = "/Users/admin/Documents/GitHub/CS336/OLMo-2-0425-1B"


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _empty_cache(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _format_user_turn(text: str, style: str) -> str:
    """构造用户回合追加到历史末尾的文本片段。"""
    if style == "chat":
        return f"\nUser: {text}\nAssistant:"
    if style == "raw":
        return text  # 纯补全, 直接续写
    if style == "r1":
        # r1_zero 单轮风格 (仅首轮; 多轮在这种模板下意义不大, 但允许)
        return (
            "A conversation between User and Assistant. The User asks a question, "
            "and the Assistant solves it. The Assistant first thinks about the "
            "reasoning process in the mind and then provides the User with the "
            "answer. The reasoning process is enclosed within <think> </think> "
            "and the answer is enclosed within <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think> "
            "<answer> answer here </answer>.\n"
            f"User: {text}\nAssistant: <think>"
        )
    raise ValueError(f"未知 style: {style}")


def _stop_strings_for(style: str) -> list[str] | None:
    if style == "chat":
        # 防止模型自问自答出新的 "User:" 回合
        return ["\nUser:", "\n\nUser:"]
    if style == "r1":
        return ["</answer>"]
    return None  # raw: 不设, 跑到 max_new_tokens 为止


def _trim_history(history_ids: torch.Tensor, max_history: int) -> torch.Tensor:
    """滑窗 trim: 只保留最后 max_history // 2 个 token, 留出一半空间。"""
    keep = max_history // 2
    return history_ids[:, -keep:]


def chat_loop(model, tok, args, device: str) -> None:
    # 历史就是不断扩张的 token id 序列, 初始空
    history_ids = torch.empty((1, 0), dtype=torch.long, device=device)
    # KV cache (仅 --keep-kv-cache 模式下保留): 类型 transformers.cache_utils.Cache | None
    # None 表示下一轮 generate 走完整 prefill (首轮、/clear 后、trim 后)
    past_kv = None

    print()
    print("=" * 70)
    print(f"  OLMo-2-0425-1B chat  |  device={device}  |  style={args.style}"
          f"  |  kv_cache={'on' if args.keep_kv_cache else 'off'}")
    print(f"  命令: /exit 退出  /clear 清空历史  /mem 查询内存与历史长度")
    print(f"  Ctrl+C 中断当前生成 (不退出程序)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n> ").strip()
        except EOFError:
            print("\n[bye]")
            break
        except KeyboardInterrupt:
            print("\n[bye]")
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("[bye]")
            break
        if user_input in ("/clear", "/reset"):
            del history_ids
            history_ids = torch.empty((1, 0), dtype=torch.long, device=device)
            past_kv = None  # 同步清掉 KV cache 引用, 下一轮重新 prefill
            _empty_cache(device)
            print("[历史已清空]")
            continue
        if user_input == "/mem":
            hist_len = history_ids.shape[1]
            kv_state = "无" if past_kv is None else f"在 ({type(past_kv).__name__})"
            extra = ""
            if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
                alloc = torch.mps.current_allocated_memory() / (1024 ** 3)
                drv = torch.mps.driver_allocated_memory() / (1024 ** 3)
                extra = f" | MPS 已分配 {alloc:.2f} GB | 驱动总占 {drv:.2f} GB"
            print(f"[历史 {hist_len} token / 上限 {args.max_history} | KV cache {kv_state}{extra}]")
            continue

        # ── 拼接新一轮的输入 ──────────────────────────────────────────
        user_chunk = _format_user_turn(user_input, args.style)
        new_ids = tok(user_chunk, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # KV cache 复用判断: 启用复用 AND 已有 past_kv (非首轮、非 /clear 后、非 trim 后)
        use_cached_kv = args.keep_kv_cache and past_kv is not None
        if use_cached_kv:
            # 只传新 user 的 token, past_key_values 接续历史 K/V; 跳过历史 prefill
            input_ids = new_ids
        else:
            # 完整传入历史 + 新 user, 走全量 prefill
            input_ids = torch.cat([history_ids, new_ids], dim=1)

        # ── 流式生成 ─────────────────────────────────────────────────
        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        if use_cached_kv:
            gen_kwargs["past_key_values"] = past_kv
        stops = _stop_strings_for(args.style)
        if stops is not None:
            gen_kwargs["stop_strings"] = stops
            gen_kwargs["tokenizer"] = tok

        output_holder: list[torch.Tensor] = []

        def _run_generate():
            try:
                with torch.no_grad():
                    out = model.generate(**gen_kwargs)
                output_holder.append(out)
            except Exception as exc:  # 把错误推到主线程
                output_holder.append(exc)

        thread = Thread(target=_run_generate, daemon=True)
        thread.start()

        # 主线程从 streamer 逐 token 拿文本, 实时刷新到 stdout
        t_start = time.time()
        n_tokens = 0
        interrupted = False
        try:
            for chunk in streamer:
                print(chunk, end="", flush=True)
                # streamer 每次给的是一个 chunk (可能 1+ token), 估算用 split
                n_tokens += max(1, len(tok.encode(chunk, add_special_tokens=False)))
        except KeyboardInterrupt:
            interrupted = True
            print("\n[已中断]")
        elapsed = time.time() - t_start
        print()  # 收行

        thread.join(timeout=5.0)
        if interrupted:
            # 用户中断: 历史不更新, 但清掉残留
            _empty_cache(device)
            continue
        if output_holder and isinstance(output_holder[0], Exception):
            print(f"[generate 失败: {output_holder[0]}]", file=sys.stderr)
            _empty_cache(device)
            continue
        if not output_holder:
            print("[generate 未返回, 跳过本轮]", file=sys.stderr)
            _empty_cache(device)
            continue

        # ── 更新历史 ─────────────────────────────────────────────────
        # transformers 5.x: generate(return_dict_in_generate=True) 返回 GenerateDecoderOnlyOutput
        # .sequences: 长度 = input_ids.shape[1] + n_new_tokens
        # .past_key_values: Cache 对象 (仅当 use_cache=True 时存在)
        result = output_holder[0]
        full_sequences = result.sequences  # shape (1, ...)
        new_past_kv = getattr(result, "past_key_values", None)

        if use_cached_kv:
            # 复用模式: sequences 只含 (new_user_ids + assistant 输出), 要拼回完整历史
            history_ids = torch.cat([history_ids, full_sequences], dim=1).detach()
        else:
            # 全 prefill 模式: sequences 本身就是完整历史
            history_ids = full_sequences.detach()

        if args.keep_kv_cache:
            past_kv = new_past_kv  # 保留, 下一轮接续

        # 显示速度 + 长度
        speed = n_tokens / elapsed if elapsed > 0 else 0.0
        cache_marker = " (复用 KV)" if use_cached_kv else ""
        print(f"[~{speed:.1f} tok/s | 用时 {elapsed:.1f}s | 历史长度 {history_ids.shape[1]}{cache_marker}]")

        # ── 内存管理: 历史超阈值就滑窗 trim, 然后释放 KV cache ─────────
        if history_ids.shape[1] > args.max_history:
            old_len = history_ids.shape[1]
            history_ids = _trim_history(history_ids, args.max_history)
            # KV cache 复用模式下: trim 后 past_kv 与历史不再对齐, 丢弃, 下一轮重新 prefill
            if past_kv is not None:
                past_kv = None
                print(f"[历史 trim: {old_len} → {history_ids.shape[1]} token, KV cache 已重置, 下一轮重 prefill]")
            else:
                print(f"[历史 trim: {old_len} → {history_ids.shape[1]} token]")

        # 每轮回答后清理无引用的池块 (MPS 不会主动归还给系统)
        # past_kv 有 Python 引用时 empty_cache 不会动它, 不影响复用
        del new_ids, input_ids, full_sequences, result
        _empty_cache(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--style", choices=["chat", "raw", "r1"], default="chat",
                        help="对话模板: chat=User/Assistant (默认); raw=纯补全; r1=r1_zero 思维链")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=2048,
                        help="历史 token 上限, 超过则滑窗保留后一半")
    parser.add_argument("--keep-kv-cache", action="store_true",
                        help="跨轮复用 KV cache, 跳过历史 prefill 加速; 内存代价大, mac 16GB 上长对话可能 swap")
    args = parser.parse_args()

    device = _pick_device()
    print(f"[setup] device={device}, 加载模型 {args.model_path}")
    t0 = time.time()
    model, tok = get_model_and_tokenizer(args.model_path, device)
    model.eval()
    _empty_cache(device)
    print(f"[setup] 模型加载完成: {time.time() - t0:.1f}s")

    try:
        chat_loop(model, tok, args, device)
    finally:
        # 退出前最后一次清理
        _empty_cache(device)


if __name__ == "__main__":
    main()
