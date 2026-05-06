"""HiToM metrics：整体准确率 + 按 Meta.order（0-4 阶）分层统计。"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics


def compute_metrics(
    predictions: List[str],
    gold_letters: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """整体准确率 + 按 Meta.order 分层的准确率。

    Args:
        predictions: 模型预测答案列表（字母 A-O，None 表示解析失败）
        gold_letters: 金标准答案列表（字母 A-O）
        data: 原始数据列表
        judge_client: 可选的 Judge LLM 客户端（HiToM 不使用）

    Returns:
        包含 accuracy, correct, total, by_order 和 per_sample_results 的字典
    """
    assert len(predictions) == len(data) == len(gold_letters), "predictions/data/gold 长度须一致"

    def is_correct_fn(pred: Any, gold: Any) -> bool:
        return bool(pred and pred == gold)

    sample_metrics = compute_sample_metrics(predictions, gold_letters, is_correct_fn)
    correct = sample_metrics["correct"]
    total = sample_metrics["total"]
    per_sample_results = sample_metrics["per_sample_results"]

    # 按 Meta.order 分层统计
    order_total: Dict[str, int] = defaultdict(int)
    order_correct: Dict[str, int] = defaultdict(int)

    for result, row in zip(per_sample_results, data):
        meta = row.get("Meta") or {}
        if isinstance(meta, dict):
            order = str(meta.get("order", "__missing__"))
        else:
            order = "__missing__"
        order_total[order] += 1
        if result["is_correct"]:
            order_correct[order] += 1

    by_order: Dict[str, float] = {}
    for k in sorted(order_total.keys()):
        by_order[k] = order_correct.get(k, 0) / order_total[k] if order_total[k] else 0.0

    order_counts = {k: order_total[k] for k in sorted(order_total.keys())}

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_order": by_order,
        "order_counts": order_counts,
        "per_sample_results": per_sample_results,
    }
