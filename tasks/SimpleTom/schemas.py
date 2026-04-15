from typing import Literal

from pydantic import BaseModel


class MCQAnswer(BaseModel):
    """多选题答案 schema（选项字母）。"""
    answer: Literal["A", "B", "C"]


SCHEMAS = {
    "MCQAnswer": MCQAnswer,
}
