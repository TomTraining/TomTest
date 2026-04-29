"""ExploreToM prompts"""

from typing import Any, Dict


PROMPTS = {
    "zero_shot": (
        "Read the story and answer the question.\n"
        "Output the answer JSON with a short phrase only. "
        "The answer should be the exact container/location name, without explanation.\n\n"
        "Story:\n{story}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
}


def _to_text(value: Any) -> str:
    """把 list / str / None 统一转成文本。"""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if x is not None)
    return str(value)


def build_prompt(row: Dict[str, Any], method: str = "zero_shot") -> str:
    """构建 ExploreToM prompt。"""
    template = PROMPTS.get(method, PROMPTS["zero_shot"])

    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}

    # ExploreToM 示例中 full_story 可能为空，summary 才是完整故事
    story = _to_text(story_info.get("summary")).strip()

    question = row.get("Question", "") or ""

    return template.format(
        story=story,
        question=question,
    )