"""FictionalQA 数据集的 metrics 计算"""
from typing import Any, Dict, List, Tuple
import hashlib
import random
import re


def _stable_shuffle(items: List[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    items_copy = list(items)
    rng.shuffle(items_copy)
    return items_copy


def _get_ids(row: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    id_str = meta.get("id", "") or ""

    event_id = "unknown"
    doc_id = "unknown"
    style = meta.get("fiction_type", "") or "unknown"

    if "_style_" in id_str:
        event_id = id_str.split("_style_")[0]
        style_part = id_str.split("_style_")[1]
        style = style or style_part.split("_")[0]
    if "_question_" in id_str:
        doc_id = id_str.split("_question_")[0]
    elif id_str:
        doc_id = id_str

    return event_id, doc_id, style


def _build_options(row: Dict[str, Any]) -> Tuple[List[str], str]:
    """构建选项列表与标准答案字母（A/B/C/D）"""
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    correct_list = answer_block.get("Correct_Answer", [])
    wrong_list = answer_block.get("Wrong_Answer", [])

    correct = correct_list[0] if isinstance(correct_list, list) and correct_list else ""
    wrongs: List[str] = []
    if isinstance(wrong_list, list):
        wrongs = [str(w) for w in wrong_list]

    options = [str(correct)] + wrongs
    if len(options) < 4:
        options.extend([""] * (4 - len(options)))
    options = options[:4]

    event_id, doc_id, style = _get_ids(row)
    seed_src = f"{event_id}|{doc_id}|{style}"
    seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest(), 16)
    shuffled = _stable_shuffle(options, seed)

    gold_index = shuffled.index(correct) if correct in shuffled else 0
    gold_letter = "ABCD"[gold_index]

    return shuffled, gold_letter


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s


def normalize_pred(pred: Any) -> str:
    """将模型输出归一化为 A/B/C/D"""
    if pred is None:
        return ""
    s = str(pred).strip().lower()

    m = re.search(r"final\s*answer\s*\[\s*([abcd])\s*\]", s)
    if not m:
        m = re.search(r"final\s*answer\s*[:\-]?\s*([abcd])", s)
    if m:
        return m.group(1).upper()

    if s in {"a", "b", "c", "d"}:
        return s.upper()

    if s and s[0] in {"a", "b", "c", "d"}:
        return s[0].upper()

    return ""


def get_gold_label(row: Dict[str, Any]) -> str:
    """获取该样本的标准答案字母"""
    _, gold = _build_options(row)
    return gold


def map_pred_to_choice(pred: Any, row: Dict[str, Any]) -> str:
    """如果模型输出了选项文本，尝试映射为字母"""
    pred_norm = _normalize_text(pred)
    options, gold = _build_options(row)
    option_norms = [_normalize_text(o) for o in options]
    if pred_norm in option_norms:
        idx = option_norms.index(pred_norm)
        return "ABCD"[idx]
    return ""


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 FictionalQA 的 metrics

    - Overall Accuracy
    - Informed vs Blind Gap
    - Split-based Evaluation: Event / Document / Style
    """
    total = len(predictions)
    correct = 0

    by_event: Dict[str, Dict[str, int]] = {}
    by_doc: Dict[str, Dict[str, int]] = {}
    by_style: Dict[str, Dict[str, int]] = {}

    blind_values: List[float] = []

    for pred, row in zip(predictions, data):
        gold_label = get_gold_label(row)
        pred_label = normalize_pred(pred)
        if not pred_label:
            pred_label = map_pred_to_choice(pred, row)

        is_correct = pred_label == gold_label
        if is_correct:
            correct += 1

        event_id, doc_id, style = _get_ids(row)

        by_event.setdefault(event_id, {"correct": 0, "total": 0})
        by_event[event_id]["total"] += 1
        if is_correct:
            by_event[event_id]["correct"] += 1

        by_doc.setdefault(doc_id, {"correct": 0, "total": 0})
        by_doc[doc_id]["total"] += 1
        if is_correct:
            by_doc[doc_id]["correct"] += 1

        by_style.setdefault(style, {"correct": 0, "total": 0})
        by_style[style]["total"] += 1
        if is_correct:
            by_style[style]["correct"] += 1

        blind_val = None
        if isinstance(row.get("blind_grade_avg"), (int, float)):
            blind_val = float(row.get("blind_grade_avg"))
        else:
            meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
            if isinstance(meta.get("blind_grade_avg"), (int, float)):
                blind_val = float(meta.get("blind_grade_avg"))
        if blind_val is not None:
            blind_values.append(blind_val)

    accuracy = correct / total if total else 0

    def _avg_group(group: Dict[str, Dict[str, int]]) -> float:
        if not group:
            return 0
        vals = []
        for stats in group.values():
            if stats["total"]:
                vals.append(stats["correct"] / stats["total"])
        return sum(vals) / len(vals) if vals else 0

    event_split_acc = _avg_group(by_event)
    doc_split_acc = _avg_group(by_doc)
    style_split_acc = _avg_group(by_style)

    blind_avg = sum(blind_values) / len(blind_values) if blind_values else None
    informed_vs_blind_gap = (accuracy - blind_avg) if blind_avg is not None else None

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "informed_vs_blind_gap": informed_vs_blind_gap,
        "blind_avg": blind_avg,
        "event_split_acc": event_split_acc,
        "document_split_acc": doc_split_acc,
        "style_split_acc": style_split_acc,
        "event_split_details": {
            k: (v["correct"] / v["total"] if v["total"] else 0)
            for k, v in by_event.items()
        },
        "document_split_details": {
            k: (v["correct"] / v["total"] if v["total"] else 0)
            for k, v in by_doc.items()
        },
        "style_split_details": {
            k: (v["correct"] / v["total"] if v["total"] else 0)
            for k, v in by_style.items()
        },
        "event_counts": {k: v["total"] for k, v in by_event.items()},
        "document_counts": {k: v["total"] for k, v in by_doc.items()},
        "style_counts": {k: v["total"] for k, v in by_style.items()},
    }
