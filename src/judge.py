"""LLM 语义判断模块

当结构化输出提取失败（max_retry 耗尽）时，使用 LLM 判断模型的原始回答
与标准答案在语义上是否一致。
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from tqdm import tqdm

from src.llm import LLMClient

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """\
You are an impartial judge. Given a model's response and a gold (correct) answer, \
determine whether the model's response contains an answer that is semantically \
equivalent to the gold answer.

Focus ONLY on whether the final answer matches in meaning. Ignore formatting, \
extra explanation, or reasoning traces.

## Model Response
{raw_response}

## Gold Answer
{gold_answer}

## Question (for context)
{question}

Does the model's response contain an answer semantically equivalent to the gold answer?\
"""


class JudgeVerdict(BaseModel):
    """LLM judge 输出 schema — 布尔值约束"""
    is_correct: bool


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks so the judge sees only the final answer."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def judge_single(
    client: LLMClient,
    raw_response: str,
    gold_answer: str,
    question: str = "",
) -> bool:
    """判断单条原始回答是否语义等价于标准答案。

    Args:
        client: 用于 judge 的 LLMClient（通常低温度）
        raw_response: 被评测模型的完整原始输出
        gold_answer: 标准答案
        question: 原始问题（提供上下文，可为空）

    Returns:
        True 表示语义一致
    """
    cleaned = _strip_think_tags(raw_response)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        raw_response=cleaned,
        gold_answer=gold_answer,
        question=question,
    )
    result = client.generate_structure(prompt, JudgeVerdict, max_retry=3)
    return getattr(result, "is_correct", False)


def batch_judge(
    client: LLMClient,
    items: List[Dict[str, Any]],
) -> List[bool]:
    """批量语义判断。

    Args:
        client: judge LLMClient
        items: 每个元素是 {"raw_response": str, "gold_answer": str, "question": str}

    Returns:
        与 items 等长的布尔列表
    """
    if not items:
        return []

    with ThreadPoolExecutor(client.max_workers) as executor:
        futures = [
            executor.submit(
                judge_single,
                client,
                item["raw_response"],
                item["gold_answer"],
                item.get("question", ""),
            )
            for item in items
        ]

        results = []
        for future in tqdm(
            futures,
            total=len(futures),
            desc="LLM Judge",
            miniters=100,
        ):
            try:
                results.append(future.result())
            except Exception:
                logger.warning("[Judge] single judge call failed, treating as incorrect")
                results.append(False)

        return results
