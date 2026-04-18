"""HellaSwag 数据集的 metrics：Overall / In-domain / Zero-shot 三类准确率。"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class _SplitStats(TypedDict):
    correct: int
    total: int


def _safe_div(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 HellaSwag 的 metrics（直接匹配选项字母）。

    - Overall Accuracy: 全量准确率
    - In-domain Accuracy: Meta.split_type == "indomain"
    - Zero-shot Accuracy: Meta.split_type == "zeroshot"
    """
    if len(predictions) != len(data):
        raise ValueError(f"predictions/data length mismatch: {len(predictions)} vs {len(data)}")

    total = len(predictions)
    correct = 0

    by_split_type: Dict[str, _SplitStats] = {}

    for pred, row in zip(predictions, data):
        mcq = row.get("_mcq") or {}
        gold = mcq.get("gold_letter")
        hit = bool(pred) and bool(gold) and pred == gold
        correct += int(hit)

        meta = row.get("Meta") if isinstance(row.get("Meta"), dict) else {}
        split_type = (meta.get("split_type") if isinstance(meta, dict) else None) or "unknown"
        split_type = str(split_type)

        if split_type not in by_split_type:
            by_split_type[split_type] = {"correct": 0, "total": 0}
        by_split_type[split_type]["total"] += 1
        if hit:
            by_split_type[split_type]["correct"] += 1

    accuracy = _safe_div(correct, total)

    by_split_type_acc = {k: _safe_div(v["correct"], v["total"]) for k, v in by_split_type.items()}
    split_type_counts = {k: v["total"] for k, v in by_split_type.items()}
    secondary_metrics = {f"by_split_type.{k}": v for k, v in by_split_type_acc.items()}

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **secondary_metrics,
        "by_split_type": by_split_type_acc,
        "split_type_counts": split_type_counts,
    }
