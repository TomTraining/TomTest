"""HellaSwag 数据集的输出 schema"""

from typing import Literal

from pydantic import BaseModel


class MCQAnswer(BaseModel):
    """多选题答案 schema（选项字母）。"""

    answer: Literal["A", "B", "C", "D"]


SCHEMAS = {
    "MCQAnswer": MCQAnswer,
}
