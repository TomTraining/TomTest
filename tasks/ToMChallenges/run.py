"""ToMChallenges 评测脚本（结构化 MCQAnswer；二选一 A/B；支持 deterministic shuffle）。"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保从任意工作目录执行时都能 import src 与 tasks 下的数据集包
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tasks"))

from src import runner

from ToMChallenges.metrics import compute_metrics
from ToMChallenges.prompts import build_prompt, get_template

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _extract_ab_answers(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """提取 (correct, wrong)。不符合 1 正 + 1 误则返回 None。"""
    ans = row.get("Answer")
    ca = ans.get("Correct_Answer")
    wa = ans.get("Wrong_Answer")
    if len(ca) != 1 or len(wa) != 1:
        return None
    return str(ca[0]).strip(), str(wa[0]).strip()


def preprocess_mcq(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为每行注入 _mcq.original_choices 与 gold_letter（初始 A 为正确答案）。"""
    valid: List[Dict[str, Any]] = []
    skipped = 0

    for row in data:
        pair = _extract_ab_answers(row)
        if pair is None:
            skipped += 1
            continue
        correct, wrong = pair
        out = dict(row)
        out["_mcq"] = {
            "original_choices": {"A": correct, "B": wrong},
            "choices": {"A": correct, "B": wrong},
            "gold_letter": "A",
        }
        valid.append(out)

    if skipped:
        print(f"Warning: skipped {skipped} rows (expected 1 Correct_Answer + 1 Wrong_Answer).")
    if not valid:
        raise RuntimeError("没有可评测样本：数据需包含 Answer 且为 1 Correct_Answer + 1 Wrong_Answer。")
    return valid


def shuffle_ab_choices(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """对 A/B 两个选项做 deterministic shuffle，并同步 gold_letter。"""
    rng = random.Random(seed)
    swap = rng.random() < 0.5

    original = mcq["original_choices"]
    if not swap:
        return {**mcq, "choices": dict(original), "gold_letter": "A"}

    return {
        **mcq,
        "choices": {"A": original["B"], "B": original["A"]},
        "gold_letter": "B",
    }


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/ToMChallenges/config.yaml")
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    schema = dataset_config["schema"]
    prompt_method = dataset_config["default_prompt"]
    template = get_template(prompt_method)
    client = runner.create_llm_client(experiment_config["llm_config"])

    data = runner.load_and_limit_data(
        subset=dataset_config["subset"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} raw rows from {dataset_config['subset']}")
    data = preprocess_mcq(data)

    repeats = experiment_config["repeats"]
    n = len(data)
    print(f"MCQ samples: {n}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {repeats} (each with deterministic A/B shuffle)")

    all_prompts: List[str] = []
    repeat_data: List[List[Dict[str, Any]]] = []

    for i in range(repeats):
        rows_i: List[Dict[str, Any]] = []
        for j, row in enumerate(data):
            shuffled_mcq = shuffle_ab_choices(row["_mcq"], seed=42 * (i + 1) + j)
            out = dict(row)
            out["_mcq"] = shuffled_mcq
            rows_i.append(out)
            all_prompts.append(build_prompt(template, out))
        repeat_data.append(rows_i)

    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    all_predictions: List[List[str]] = []
    all_metrics: List[Dict[str, Any]] = []
    all_gold: List[List[str]] = []

    for i in range(repeats):
        start = i * n
        end = start + n
        rows_i = repeat_data[i]
        repeat_results = results[start:end]
        predictions = [r.answer for r in repeat_results]
        all_predictions.append(predictions)

        metrics = compute_metrics(predictions, rows_i)
        all_metrics.append(metrics)
        all_gold.append([row["_mcq"]["gold_letter"] for row in rows_i])

        print(
            f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, "
            f"Correct={metrics['correct']}/{metrics['total']}"
        )

    runner.save_common_results(
        dataset_name=dataset_config["dataset"],
        model=experiment_config["llm_config"]["model_name"],
        prompt_method=prompt_method,
        all_predictions=all_predictions,
        gold_answers=all_gold,
        all_metrics=all_metrics,
        results_path=experiment_config["results_path"],
        dataset_config=dataset_config,
        experiment_config=experiment_config,
    )

    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
