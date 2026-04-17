"""RecToM 数据集的输出 schema。"""
from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field, field_validator


class MultiLabelAnswer(BaseModel):
    """多标签多选题答案 schema。"""
    answer: List[str] = Field(
        default_factory=list,
        description="A list of option labels only, such as ['A'] or ['B', 'D'].",
    )

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, value):
        if value is None:
            return []

        if isinstance(value, str):
            items = re.findall(r"[A-Za-z][A-Za-z0-9_]*", value)
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]

        normalized = []
        seen = set()
        for item in items:
            token = str(item).strip().upper()
            if token and token not in seen:
                normalized.append(token)
                seen.add(token)
        return normalized


SCHEMAS = {
    "MultiLabelAnswer": MultiLabelAnswer,
}
