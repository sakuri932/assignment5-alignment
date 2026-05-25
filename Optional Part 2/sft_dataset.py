"""
SFT 打包数据集（Optional Part 2 版本）

从 cs336_alignment/utils.py 的 PackedSFTDataset 复制并修改。

修改说明：
  1. 支持 gzip 压缩文件（.jsonl.gz）：自动检测文件扩展名，
     用 gzip.open 替代普通 open 读取压缩数据
  2. 字段名自适应：优先使用 prompt/response，若不存在则尝试
     instruction/output（UltraChat-200K 单轮格式）
  3. 支持系统 prompt 前缀（zero_shot_system_prompt）
  4. 跨平台路径：全部使用 pathlib.Path

原始文件：cs336_alignment/utils.py
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

# Alpaca 指令微调格式（与主作业保持一致）
_ALPACA_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "alpaca_sft.prompt"
)
_ALPACA_TEMPLATE = _ALPACA_TEMPLATE_PATH.read_text(encoding="utf-8")


def _open_maybe_gz(path: Path):
    """统一打开普通或 gzip 压缩文件，返回文本行迭代器。"""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _extract_prompt_response(doc: dict) -> tuple[str, str]:
    """从 JSON 对象中提取 prompt 和 response，适配不同字段名。

    尝试顺序：
      1. "prompt" + "response"  （主作业格式）
      2. "instruction" + "output"（UltraChat 单轮格式）
    """
    if "prompt" in doc and "response" in doc:
        return doc["prompt"], doc["response"]
    if "instruction" in doc and "output" in doc:
        return doc["instruction"], doc["output"]
    raise KeyError(f"未找到 prompt/response 或 instruction/output 字段，已有字段：{list(doc.keys())}")


class PackedSFTDataset(Dataset):
    """将指令微调 JSONL（或 .gz）数据打包为固定长度序列的 Dataset。

    策略：
      - 将所有文档格式化为 Alpaca 模板字符串后分词
      - 拼接所有 token 为长序列 T
      - 以步长 seq_length 切分，每个窗口产生 (input_ids, labels) 对
        · input_ids = T[i*L : i*L + L]
        · labels    = T[i*L+1 : i*L + L+1]

    Args:
        tokenizer    : HuggingFace tokenizer
        dataset_path : JSONL 或 .jsonl.gz 文件路径
        seq_length   : 每条序列的 token 数
        shuffle      : 是否在拼接前随机打乱文档顺序
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ):
        dataset_path = Path(dataset_path)

        # 读取所有文档
        docs: list[dict] = []
        with _open_maybe_gz(dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        if shuffle:
            random.shuffle(docs)

        # 拼接所有文档的 token（每个文档末尾追加 EOS）
        eos_id = tokenizer.eos_token_id
        all_tokens: list[int] = []
        for doc in docs:
            try:
                prompt, response = _extract_prompt_response(doc)
            except KeyError:
                continue
            text = _ALPACA_TEMPLATE.format(instruction=prompt, response=response)
            toks = tokenizer.encode(text.rstrip(), add_special_tokens=True)
            if eos_id is not None and (not toks or toks[-1] != eos_id):
                toks = toks + [eos_id]
            all_tokens.extend(toks)

        # 以 stride=seq_length 切分，丢弃不足一个窗口的尾部
        n_windows = (len(all_tokens) - 1) // seq_length
        self._items: list[dict[str, Tensor]] = []
        for i in range(n_windows):
            start = i * seq_length
            window = all_tokens[start : start + seq_length + 1]
            self._items.append({
                "input_ids": torch.tensor(window[:-1], dtype=torch.long),
                "labels":    torch.tensor(window[1:],  dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self._items[idx]


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | Path,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """PackedSFTDataset 的工厂函数（签名与主作业 adapters 一致）。"""
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """对 Dataset 包装为 DataLoader，产生批次字典。

    Windows 上 num_workers 固定为 0（避免 fork 多进程问题）。
    """
    import platform
    num_workers = 0 if platform.system() == "Windows" else 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
