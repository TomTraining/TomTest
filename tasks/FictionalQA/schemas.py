"""FictionalQA 数据集的输出 schema"""
from pydantic import BaseModel


class MCQAnswer(BaseModel):
    """多选题答案 schema（A/B/C/D）"""
    answer: str = ""


SCHEMAS = {
    "MCQAnswer": MCQAnswer,
}
