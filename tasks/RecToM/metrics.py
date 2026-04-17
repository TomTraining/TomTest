"""RecToM metrics."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _normalize_labels(labels: Any) -> List[str]:
    if labels is None:
        return []
    if isinstance(labels, str):
        raw = [labels]
    elif isinstance(labels, (list, tuple, set)):
        raw = list(labels)
    else:
        raw = [labels]

    normalized = []
    seen = set()
    for label in raw:
        token = str(label).strip().upper()
        if token and token not in seen:
            normalized.append(token)
            seen.add(token)
    return sorted(normalized)


def _classify_prediction(pred_labels: Iterable[str], gold_labels: Iterable[str]) -> str:
    pred_set = set(_normalize_labels(list(pred_labels)))
    gold_set = set(_normalize_labels(list(gold_labels)))

    if pred_set == gold_set:
        return "full_correct"
    if pred_set.issubset(gold_set):
        return "partial_no_error"
    return "has_error"


def _init_group(stats: Dict[str, Dict[str, int]], key: Any) -> Dict[str, int]:
    key_str = str(key) if key not in (None, "") else "unknown"
    if key_str not in stats:
        stats[key_str] = {
            "full_correct": 0,
            "partial_no_error": 0,
            "has_error": 0,
            "total": 0,
        }
    return stats[key_str]


def _update_group(stats: Dict[str, Dict[str, int]], key: Any, category: str) -> None:
    group = _init_group(stats, key)
    group["total"] += 1
    group[category] += 1


def _flatten_group(stats: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in sorted(stats.items()):
        total = value["total"]
        flat[f"{prefix}_full_correct.{key}"] = value["full_correct"] / total if total else 0.0
        flat[f"{prefix}_partial_no_error.{key}"] = value["partial_no_error"] / total if total else 0.0
        flat[f"{prefix}_has_error.{key}"] = value["has_error"] / total if total else 0.0
    return flat


def _rate_dict(stats: Dict[str, Dict[str, int]], field: str) -> Dict[str, float]:
    return {
        key: (value[field] / value["total"] if value["total"] else 0.0)
        for key, value in sorted(stats.items())
    }


def compute_metrics(
    predictions: List[List[str]],
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total = len(data)
    full_correct = 0
    partial_no_error = 0
    has_error = 0

    by_source: Dict[str, Dict[str, int]] = {}
    by_num_gold: Dict[str, Dict[str, int]] = {}

    for pred, row in zip(predictions, data):
        answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
        gold = _normalize_labels(answer_block.get("Correct_Answer", []))
        category = _classify_prediction(pred, gold)

        if category == "full_correct":
            full_correct += 1
        elif category == "partial_no_error":
            partial_no_error += 1
        else:
            has_error += 1

        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        source = meta.get("datasource") or meta.get("dataset_source") or "unknown"
        _update_group(by_source, source, category)
        _update_group(by_num_gold, len(gold), category)

    accuracy = full_correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": full_correct,
        "total": total,
        "full_correct": full_correct,
        "partial_no_error": partial_no_error,
        "has_error": has_error,
        "full_correct_rate": full_correct / total if total else 0.0,
        "partial_no_error_rate": partial_no_error / total if total else 0.0,
        "has_error_rate": has_error / total if total else 0.0,
        **_flatten_group(by_source, "by_source"),
        **_flatten_group(by_num_gold, "by_num_gold"),
        "by_source_full_correct": _rate_dict(by_source, "full_correct"),
        "by_source_partial_no_error": _rate_dict(by_source, "partial_no_error"),
        "by_source_has_error": _rate_dict(by_source, "has_error"),
        "source_counts": {key: value["total"] for key, value in sorted(by_source.items())},
        "by_num_gold_full_correct": _rate_dict(by_num_gold, "full_correct"),
        "by_num_gold_partial_no_error": _rate_dict(by_num_gold, "partial_no_error"),
        "by_num_gold_has_error": _rate_dict(by_num_gold, "has_error"),
        "num_gold_counts": {key: value["total"] for key, value in sorted(by_num_gold.items())},
    }
