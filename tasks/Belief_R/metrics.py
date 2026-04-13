"""Belief-R 数据集的 metrics 计算"""
from typing import Any, Dict, List, Tuple
import re


def _build_options(row: Dict[str, Any]) -> Tuple[List[str], str]:
    """构建选项列表与标准答案字母。

    约定：
    - time_t+1 需要更新信念 => 正确答案放在 c
    - 其他情况 => 正确答案放在 a
    """
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    correct_list = answer_block.get("Correct_Answer", [])
    wrong_list = answer_block.get("Wrong_Answer", [])

    correct = correct_list[0] if isinstance(correct_list, list) and correct_list else ""
    wrongs: List[str] = []
    if isinstance(wrong_list, list):
        wrongs = [str(w) for w in wrong_list]
    if len(wrongs) < 2:
        wrongs.extend([""] * (2 - len(wrongs)))

    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    step = (meta.get("step", "") or "").lower()
    is_update = step in {"time_t+1", "time_t1", "t+1", "time_t_1"}

    if is_update:
        options = [wrongs[0], wrongs[1], correct]
        gold = "c"
    else:
        options = [correct, wrongs[0], wrongs[1]]
        gold = "a"

    return options, gold


def normalize_pred(pred: Any) -> str:
    """将模型输出归一化为 a/b/c"""
    if pred is None:
        return ""
    s = str(pred).strip().lower()

    # 优先匹配 "Final Answer [x]"
    m = re.search(r"final\s*answer\s*\[\s*([abc])\s*\]", s)
    if not m:
        m = re.search(r"final\s*answer\s*[:\-]?\s*([abc])", s)
    if m:
        return m.group(1)

    # 直接是 a/b/c
    if s in {"a", "b", "c"}:
        return s

    # 兜底：取首字母
    if s and s[0] in {"a", "b", "c"}:
        return s[0]

    return ""


def get_gold_label(row: Dict[str, Any]) -> str:
    """获取该样本的标准答案字母"""
    _, gold = _build_options(row)
    return gold


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 Belief-R 的 metrics

    - Overall Accuracy
    - BU-Acc: time_t+1 且标准答案为 c
    - BM-Acc: 其余样本（无需更新时保持原结论）
    - BREU: (BU-Acc + BM-Acc) / 2
    """
    total = len(predictions)
    correct = 0

    bu_total = 0
    bu_correct = 0
    bm_total = 0
    bm_correct = 0

    for pred, row in zip(predictions, data):
        pred_label = normalize_pred(pred)
        gold_label = get_gold_label(row)

        is_correct = pred_label == gold_label
        if is_correct:
            correct += 1

        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        step = (meta.get("step", "") or "").lower()
        is_bu = (step in {"time_t+1", "time_t1", "t+1", "time_t_1"} and gold_label == "c")
        if is_bu:
            bu_total += 1
            if is_correct:
                bu_correct += 1
        else:
            bm_total += 1
            if is_correct:
                bm_correct += 1

    accuracy = correct / total if total else 0
    bu_acc = bu_correct / bu_total if bu_total else 0
    bm_acc = bm_correct / bm_total if bm_total else 0
    breu = (bu_acc + bm_acc) / 2

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "BU-Acc": bu_acc,
        "BM-Acc": bm_acc,
        "BREU": breu,
        "bu_correct": bu_correct,
        "bu_total": bu_total,
        "bm_correct": bm_correct,
        "bm_total": bm_total,
    }
