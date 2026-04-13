"""IFEval prompts"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": "{prompt}",
}


def get_template(method: str) -> str:
    return PROMPTS.get(method, PROMPTS["zero_shot"])


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    return template.format(prompt=row["Question"])
