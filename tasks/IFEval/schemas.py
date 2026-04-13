"""IFEval schemas"""
from pydantic import BaseModel


class OpenAnswer(BaseModel):
    """开放式文本答案 schema"""
    answer: str


SCHEMAS = {
    "OpenAnswer": OpenAnswer,
}
