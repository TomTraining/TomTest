"""Belief-R 数据集的输出 schema"""
from pydantic import BaseModel


class MCQAnswer(BaseModel):
    """多选题答案 schema（a/b/c）"""
    answer: str = ""


SCHEMAS = {
    "MCQAnswer": MCQAnswer,
}
