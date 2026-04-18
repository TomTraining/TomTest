"""ToMChallenges 数据集的输出 schema（A/B 二选一）。"""

from typing import Literal

from pydantic import BaseModel


class MCQAnswer(BaseModel):
    """二选一答案 schema（选项字母）。"""

    answer: Literal["A", "B"]


SCHEMAS = {
    "MCQAnswer": MCQAnswer,
}
