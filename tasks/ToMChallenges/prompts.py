"""ToMChallenges prompts"""

from __future__ import annotations

from typing import Any, Dict

PROMPTS = {
    "zero_shot": (
        "Choose the correct answer from A or B for the following question:\n"
        "Question:\n"
        "{question_block}\n\n"
        "A. {option_a}\n"
        "B. {option_b}\n\n"
        "Answer with exactly one letter (A/B):"
    ),
}


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板"""
    return PROMPTS.get(method, PROMPTS["zero_shot"])


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """构建 prompt。"""
    story = row.get("Story") if isinstance(row.get("Story"), dict) else {}
    full_story = (story.get("full_story") if isinstance(story, dict) else "") or ""
    question = row.get("Question", "") or ""
    question_block = f"{str(full_story).strip()} {str(question).strip()}".strip()

    mcq = row.get("_mcq") or {}
    choices = mcq.get("choices") if isinstance(mcq.get("choices"), dict) else {}
    option_a = choices.get("A", "")
    option_b = choices.get("B", "")

    return template.format(
        question_block=question_block,
        option_a=option_a,
        option_b=option_b,
    )
