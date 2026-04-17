"""RecToM prompts."""
from typing import Any, Dict

RECTOM_MULTI_LABEL_SYSTEM = (
    "You are an expert at understanding recommendation dialogues and multiple-choice reasoning. "
    "Read the transcript and the question carefully. Some questions have multiple correct options. "
    "Return only the option labels, not the option text. "
    "If more than one option is correct, include all correct labels in the answer list. "
    "If exactly one option is correct, return a one-element list. "
    "Be careful to only return the labels of the correct options, and do not include any incorrect options in the answer."
)

PROMPTS = {
    "multi_label_mcq": RECTOM_MULTI_LABEL_SYSTEM,
}


def get_template(method: str) -> str:
    return PROMPTS.get(method, RECTOM_MULTI_LABEL_SYSTEM)


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
    transcript = (story_info.get("full_story", "") or "").strip()
    question = str(row.get("Question", "") or "").strip()

    user = (
        f"# Transcript\n{transcript}\n\n"
        f"# Question\n{question}\n\n"
        "# Output Requirement\n"
        "Return JSON structured output where answer is a list of option labels."
    )
    return f"{template}\n\n{user}"
