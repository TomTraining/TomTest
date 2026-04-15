"""SimpleToM prompts."""
from typing import Any, Dict


SIMPLETOM_MCQA_SYSTEM = (
    "You are an expert at understanding human behavior and theory of mind. "
    "Use the transcript and options to choose the single best answer. "
    "Respond with structured output: the answer field must be exactly one "
    "letter A, B, or C matching one of the listed options."
)

PROMPTS = {
    "v2_generate": SIMPLETOM_MCQA_SYSTEM,
}


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板。"""
    return PROMPTS.get(method, SIMPLETOM_MCQA_SYSTEM)


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """拼接用户块。"""
    mcq = row["_mcq"]
    story_block = mcq["story"].strip()
    question = mcq["question"].strip()
    options = mcq["original_choices"]
    options_block = "\n".join(
        f"[{letter}] {options[letter]}"
        for letter in sorted(options.keys())
    )

    user = (
        f"# Transcript\n{story_block}\n\n"
        f"# Question\n{question}\n\n"
        f"# Options\n{options_block}"
    )
    return f"{template}\n\n{user}"
