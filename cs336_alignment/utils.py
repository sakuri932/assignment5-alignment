"""
CS336 Assignment 5 — 工具函数

包含：
  parse_mmlu_response  — 从模型输出解析 MMLU 选项字母（A/B/C/D）
  parse_gsm8k_response — 从模型输出解析 GSM8K 数值答案（取最后一个数字）
  PackedSFTDataset     — 打包 SFT 数据集（固定长度序列）
  iterate_batches      — 从 Dataset 产生 batch 的迭代器
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

# alpaca 指令微调格式模板
_ALPACA_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "alpaca_sft.prompt"
_ALPACA_TEMPLATE = _ALPACA_TEMPLATE_PATH.read_text()


# ---------------------------------------------------------------------------
# MMLU / GSM8K 响应解析
# ---------------------------------------------------------------------------

def parse_mmlu_response(
    mmlu_example: dict,
    model_output: str,
) -> str | None:
    """
    从模型输出中解析 MMLU 选项字母（A/B/C/D）。

    策略：
      1. 在输出中搜索 "answer is X"、"answer: X"、"(X)"、"X." 等常见格式。
      2. 如果找到唯一且有效的选项字母，返回该字母。
      3. 否则返回 None。

    Args:
        mmlu_example: 包含 subject/question/options/answer 键的字典（未使用 options 内容，
                      仅使用 A/B/C/D 四个字母作为有效集合）
        model_output: 模型生成的文本
    """
    valid_options = {"A", "B", "C", "D"}

    # 模式 1：常见"答案是 X"格式
    pattern_answer_is = re.compile(
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-D])\b",
        re.IGNORECASE,
    )
    match = pattern_answer_is.search(model_output)
    if match:
        letter = match.group(1).upper()
        if letter in valid_options:
            return letter

    # 模式 2：选项格式 "(A)" 或 "A)" 或 "A."
    pattern_option = re.compile(r"\(?([A-D])\)?[.)]\s", re.IGNORECASE)
    matches = pattern_option.findall(model_output)
    if len(matches) == 1:
        letter = matches[0].upper()
        if letter in valid_options:
            return letter

    # 模式 3：孤立的大写字母（精确匹配单词边界）
    pattern_word = re.compile(r"\b([A-D])\b")
    all_letters = [m.upper() for m in pattern_word.findall(model_output) if m.upper() in valid_options]
    if len(set(all_letters)) == 1:
        return all_letters[0]

    return None


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    从 GSM8K 模型输出中解析预测的数值答案（取最后一个出现的数字）。

    策略：
      - 使用正则找到所有数字序列（包含可选的逗号分隔符，如 "1,000"）
      - 返回最后一个数字字符串（去掉逗号后的纯数字形式）
      - 若输出中无任何数字，返回 None

    Args:
        model_output: 模型生成的文本
    """
    # 匹配数字（含千位逗号分隔，如 1,000,000）
    numbers = re.findall(r"\d[\d,]*", model_output)
    if not numbers:
        return None

    # 取最后一个数字，去掉逗号
    last_number = numbers[-1].replace(",", "")
    return last_number


# ---------------------------------------------------------------------------
# 打包 SFT 数据集
# ---------------------------------------------------------------------------

class PackedSFTDataset(Dataset):
    """
    将指令微调数据（JSONL）打包为固定长度序列的数据集。

    策略：将所有文档格式化后分词，把所有 token 拼接成一条长序列 T，
    然后以步长 seq_length 滑窗切分（相邻窗口共享 1 个边界 token）：
      window i → T[i*L : i*L + L + 1]
      - input_ids = window[:-1]  (length L)
      - labels    = window[1:]   (length L)
    总共产生 floor((len(T) - 1) / seq_length) 个示例。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ):
        import json
        import random

        docs = []
        with open(dataset_path, encoding="utf-8") as f:
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
            text = _ALPACA_TEMPLATE.format(
                instruction=doc["prompt"],
                response=doc["response"],
            )
            # rstrip 去掉模板末尾换行，避免 ".\n" 合并为一个 BPE token
            toks = tokenizer.encode(text.rstrip(), add_special_tokens=True)
            if eos_id is not None and (not toks or toks[-1] != eos_id):
                toks = toks + [eos_id]
            all_tokens.extend(toks)

        # 以 stride=seq_length 滑窗切分
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
    """
    构建固定长度 SFT 数据集（PackedSFTDataset 的工厂函数）。
    """
    return PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        seq_length=seq_length,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------------
# 批次迭代器
# ---------------------------------------------------------------------------

def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    对 Dataset 包装为 DataLoader，产生 batch 字典。

    每个 batch 的 input_ids 和 labels 形状：(batch_size, seq_length)，
    最后一个 batch 可能小于 batch_size（drop_last=False）。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
