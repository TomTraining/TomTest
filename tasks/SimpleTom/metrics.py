"""SimpleToM metrics."""
from typing import Any, Dict, List


def _update_group(stats: Dict[str, Dict[str, int]], key: Any, correct: bool) -> None:
    key_str = str(key) if key not in (None, "") else "unknown"
    if key_str not in stats:
        stats[key_str] = {"correct": 0, "total": 0}
    stats[key_str]["total"] += 1
    if correct:
        stats[key_str]["correct"] += 1


def _flatten(group: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}.{key}": (value["correct"] / value["total"] if value["total"] else 0.0)
        for key, value in sorted(group.items())
    }


def compute_metrics(
    predictions: List[str],
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """计算整体准确率和按来源、维度、难度分组的准确率。"""
    gold_letters = [row["_mcq"]["gold_letter"] for row in data]

    correct = 0
    total = len(predictions)
    by_source: Dict[str, Dict[str, int]] = {}
    by_dimension: Dict[str, Dict[str, int]] = {}
    by_difficulty: Dict[str, Dict[str, int]] = {}

    for pred, gold, row in zip(predictions, gold_letters, data):
        hit = bool(pred and pred == gold)
        correct += int(hit)

        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        source = meta.get("dataset_source", "unknown")
        difficulty = meta.get("difficulty", "unknown")
        dims = meta.get("dimension", [])
        if isinstance(dims, list) and dims:
            dimension = dims[0]
        else:
            dimension = dims if dims else "unknown"

        _update_group(by_source, source, hit)
        _update_group(by_dimension, dimension, hit)
        _update_group(by_difficulty, difficulty, hit)

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **_flatten(by_source, "by_source"),
        **_flatten(by_dimension, "by_dimension"),
        **_flatten(by_difficulty, "by_difficulty"),
        "by_source": {
            key: (value["correct"] / value["total"] if value["total"] else 0.0)
            for key, value in sorted(by_source.items())
        },
        "source_counts": {key: value["total"] for key, value in sorted(by_source.items())},
        "by_dimension": {
            key: (value["correct"] / value["total"] if value["total"] else 0.0)
            for key, value in sorted(by_dimension.items())
        },
        "dimension_counts": {key: value["total"] for key, value in sorted(by_dimension.items())},
        "by_difficulty": {
            key: (value["correct"] / value["total"] if value["total"] else 0.0)
            for key, value in sorted(by_difficulty.items())
        },
        "difficulty_counts": {key: value["total"] for key, value in sorted(by_difficulty.items())},
    }
