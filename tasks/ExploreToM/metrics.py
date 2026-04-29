"""ExploreToM 数据集的 metrics 计算"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


def _normalize(text: Any) -> str:
    """答案归一化：小写、去前缀、去引号、空白合并、空格转下划线、去首尾标点。"""
    if text is None:
        return ""

    s = str(text).strip().lower()

    for prefix in ("answer:", "answer", "ans:", "output:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "_")
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)

    return s


def normalize_answer(text: Any) -> str:
    return _normalize(text)


def _get_gold_list(row: Dict[str, Any]) -> List[str]:
    """从 Answer.Correct_Answer 中提取标准答案列表。"""
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    gold = answer_block.get("Correct_Answer", [])

    if isinstance(gold, list):
        return [str(g) for g in gold if g is not None]
    if gold is None:
        return []
    return [str(gold)]


def _get_first_or_unknown(value: Any) -> str:
    """Meta.dimension 可能是 list；这里取第一个非空值。"""
    if isinstance(value, list):
        return str(value[0]) if value else "unknown"
    if value is None or value == "":
        return "unknown"
    return str(value)


def _update_group(stats: Dict[str, Dict[str, int]], key: Any, correct: bool) -> None:
    key_str = _get_first_or_unknown(key)
    if key_str not in stats:
        stats[key_str] = {"correct": 0, "total": 0}
    stats[key_str]["total"] += 1
    if correct:
        stats[key_str]["correct"] += 1


def _flatten(group: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}.{k}": (v["correct"] / v["total"] if v["total"] else 0.0)
        for k, v in group.items()
    }


def _to_accuracy_dict(group: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    return {
        k: v["correct"] / v["total"]
        for k, v in group.items()
        if v["total"] > 0
    }


def _to_count_dict(group: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    return {k: v["total"] for k, v in group.items()}


def compute_metrics(
    predictions: List[str],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 ExploreToM 指标。

    - accuracy：预测短答案是否命中 Answer.Correct_Answer
    - by_dimension：按 Meta.dimension 统计
    - by_order：按 Meta.order 统计
    - by_task_type：按 Meta.task_type 统计
    - by_difficulty：按 Meta.difficulty 统计
    """

    pred_norm_list = []
    gold_norm_sets = []

    for pred, row in zip(predictions, data):
        if pred is None:
            pred_norm_list.append(None)
            gold_norm_sets.append(set())
            continue

        gold_list = _get_gold_list(row)
        pred_norm = _normalize(pred)
        gold_norm = {_normalize(g) for g in gold_list}

        pred_norm_list.append(pred_norm)
        gold_norm_sets.append(gold_norm)

    def is_correct_fn(pred_norm: Any, gold_norm: Set[str]) -> bool:
        return bool(pred_norm) and pred_norm in gold_norm if gold_norm else False

    if judge_client is not None:
        sample_metrics = compute_sample_metrics_with_llm(
            predictions,
            gold_answers,
            judge_client,
        )
    else:
        sample_metrics = compute_sample_metrics(
            pred_norm_list,
            gold_norm_sets,
            is_correct_fn,
        )

    correct = sample_metrics["correct"]
    total = sample_metrics["total"]
    per_sample_results = sample_metrics["per_sample_results"]

    by_dimension: Dict[str, Dict[str, int]] = {}
    by_difficulty: Dict[str, Dict[str, int]] = {}
    by_task_type: Dict[str, Dict[str, int]] = {}
    by_order: Dict[str, Dict[str, int]] = {}

    for sample_result, row in zip(per_sample_results, data):
        is_correct = sample_result["is_correct"]
        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}

        _update_group(by_dimension, meta.get("dimension", "unknown"), is_correct)
        _update_group(by_difficulty, meta.get("difficulty", "unknown") or "unknown", is_correct)
        _update_group(by_task_type, meta.get("task_type", "unknown") or "unknown", is_correct)
        _update_group(by_order, meta.get("order", "unknown"), is_correct)

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,

        **_flatten(by_dimension, "by_dimension"),
        **_flatten(by_difficulty, "by_difficulty"),
        **_flatten(by_task_type, "by_task_type"),
        **_flatten(by_order, "by_order"),

        "by_dimension": _to_accuracy_dict(by_dimension),
        "dimension_counts": _to_count_dict(by_dimension),

        "by_difficulty": _to_accuracy_dict(by_difficulty),
        "difficulty_counts": _to_count_dict(by_difficulty),

        "by_task_type": _to_accuracy_dict(by_task_type),
        "task_type_counts": _to_count_dict(by_task_type),

        "by_order": _to_accuracy_dict(by_order),
        "order_counts": _to_count_dict(by_order),

        "per_sample_results": per_sample_results,
    }