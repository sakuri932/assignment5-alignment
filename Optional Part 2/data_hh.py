"""
Anthropic HH（Helpful and Harmless）数据集加载与处理。

数据来源：Anthropic/hh-rlhf（HuggingFace）
服务器路径：/data/a5-alignment/hh/
文件列表：
  harmless-base.jsonl.gz
  helpful-base.jsonl.gz
  helpful-online.jsonl.gz
  helpful-rejection-sampled.jsonl.gz

每行 JSON 格式：
  {
    "chosen":   "\n\nHuman: ...\n\nAssistant: ...",
    "rejected":  "\n\nHuman: ...\n\nAssistant: ..."
  }

处理规则（按 PDF 要求）：
  - 忽略多轮对话（Human 出现超过一次，排除原始 prompt 中的首次）
  - 提取 instruction（第一条 Human 消息）
  - 提取 chosen / rejected response（对应助手的第一条消息）
  - 记录来源文件名
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HHExample:
    """一条 HH 偏好数据。"""
    instruction:       str   # 第一条人类消息（去掉 "Human:" 前缀后的纯文本）
    response_chosen:   str   # 偏好响应（助手的第一条消息）
    response_rejected: str   # 被拒响应
    source_file:       str   # 来源文件名（用于分析）


_HH_TURN_RE = re.compile(
    r"\n\nHuman:\s*(.*?)\n\nAssistant:\s*(.*?)(?=\n\nHuman:|\Z)",
    re.DOTALL,
)


def _parse_conversation(text: str) -> list[tuple[str, str]]:
    """将 HH 对话字符串解析为 [(human_msg, assistant_msg), ...] 列表。"""
    return [(h.strip(), a.strip()) for h, a in _HH_TURN_RE.findall(text)]


def _is_single_turn(turns: list[tuple[str, str]]) -> bool:
    """判断对话是否为单轮（Human 只说了一次话）。"""
    return len(turns) == 1


def load_hh_dataset(
    data_dir: str | Path,
    filenames: list[str] | None = None,
) -> list[HHExample]:
    """加载并处理 Anthropic HH 数据集。

    Args:
        data_dir  : 存放 .jsonl.gz 文件的目录（服务器上为 /data/a5-alignment/hh）
        filenames : 要加载的文件列表；None 则加载目录下所有 .jsonl.gz 文件

    Returns:
        HHExample 对象列表，已过滤多轮对话
    """
    data_dir = Path(data_dir)

    if filenames is None:
        filenames = sorted(p.name for p in data_dir.glob("*.jsonl.gz"))

    examples: list[HHExample] = []

    for fname in filenames:
        fpath = data_dir / fname
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chosen_text   = obj.get("chosen",   "")
                rejected_text = obj.get("rejected", "")

                chosen_turns   = _parse_conversation(chosen_text)
                rejected_turns = _parse_conversation(rejected_text)

                # 过滤多轮对话
                if not (_is_single_turn(chosen_turns) and _is_single_turn(rejected_turns)):
                    continue

                instruction       = chosen_turns[0][0]
                response_chosen   = chosen_turns[0][1]
                response_rejected = rejected_turns[0][1]

                # 跳过空字段
                if not (instruction and response_chosen and response_rejected):
                    continue

                examples.append(HHExample(
                    instruction=instruction,
                    response_chosen=response_chosen,
                    response_rejected=response_rejected,
                    source_file=fname,
                ))

    return examples


def split_train_val(
    examples: list[HHExample],
    val_size: int = 200,
    seed: int = 42,
) -> tuple[list[HHExample], list[HHExample]]:
    """将 HH 数据集切分为训练集和验证集。

    Args:
        examples : load_hh_dataset 的返回结果
        val_size : 验证集大小（PDF 推荐 200）
        seed     : 随机种子

    Returns:
        (train_examples, val_examples)
    """
    import random
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    val   = shuffled[:val_size]
    train = shuffled[val_size:]
    return train, val
