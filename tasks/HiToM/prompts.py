"""HiToM prompts：Higher-Order Theory of Mind，15 选项多选题。"""
from typing import Any, Dict


_VANILLA_TEMPLATE = (
    "Read the following story and answer the multiple-choice question.\n\n"
    "Story:\n{story}\n\n"
    "Question: {question}\n"
    "Choices:\n{choices}\n\n"
    "Note: You should assume the following. "
    "(1) An agent witnesses everything and every movement before exiting a location. "
    "(2) An agent A can infer another agent B's mental state only if A and B have been in the same location, "
    "or have private or public interactions. "
    "(3) Note that every agent tends to lie. What an agent A tells others doesn't affect A's actual belief. "
    "An agent tends to trust an agent that exited the room later than himself. "
    "The exit order is known to all agents. "
    "(4) Agents in private communications know that others won't hear them, "
    "but they know that anyone can hear any public claims.\n\n"
    "Output a single letter (A through O) as your answer."
)


def build_prompt(row: Dict[str, Any], method: str = "VANILLA", choices_str: str = None) -> str:
    """构建 HiToM 评测 prompt。

    Args:
        row: 数据行（含 Story.full_story, Question 字段）
        method: prompt 方法名
        choices_str: 已格式化的选项字符串（由 run.py 传入打乱后的选项）

    Returns:
        格式化的 prompt（用户内容）
    """
    story_field = row.get("Story") or {}
    if isinstance(story_field, dict):
        story = str(story_field.get("full_story", "")).strip()
    else:
        story = str(story_field).strip()

    question = str(row.get("Question", "")).strip()
    choices = choices_str or ""

    return _VANILLA_TEMPLATE.format(
        story=story,
        question=question,
        choices=choices,
    )
