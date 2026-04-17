"""RecToM 评测脚本（多标签 MCQ）。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

TASKS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TASKS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TASKS_ROOT))

from src import runner
from RecToM.prompts import build_prompt, get_template
from RecToM.metrics import compute_metrics

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _normalize_prediction(prediction: Any) -> List[str]:
    if prediction is None:
        return []
    if isinstance(prediction, str):
        items = [prediction]
    elif isinstance(prediction, (list, tuple, set)):
        items = list(prediction)
    else:
        items = [prediction]

    normalized = []
    seen = set()
    for item in items:
        token = str(item).strip().upper()
        if token and token not in seen:
            normalized.append(token)
            seen.add(token)
    return sorted(normalized)


def _validate_row(row: Dict[str, Any]) -> bool:
    answer = row.get("Answer")
    if not isinstance(answer, dict):
        return False
    correct = answer.get("Correct_Answer")
    wrong = answer.get("Wrong_Answer")
    question = row.get("Question")
    return (
        isinstance(correct, list)
        and isinstance(wrong, list)
        and bool(correct)
        and isinstance(question, str)
        and question.strip() != ""
    )


def preprocess_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [row for row in data if _validate_row(row)]
    skipped = len(data) - len(valid)
    if skipped:
        print(f"Warning: skipped {skipped} invalid rows.")
    if not valid:
        raise RuntimeError("没有可评测样本：RecToM 数据需要包含 Question 和 Answer.{Correct_Answer, Wrong_Answer}。")
    return valid


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/RecToM/config.yaml")
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
    data = preprocess_data(data)
    print(f"Valid samples: {len(data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {experiment_config['repeats']}")

    prompts = [build_prompt(template, row) for row in data]
    all_prompts = prompts * experiment_config["repeats"]

    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    all_predictions: List[List[List[str]]] = []
    all_metrics: List[Dict[str, Any]] = []
    gold_answers = [
        sorted(str(x).strip().upper() for x in row["Answer"]["Correct_Answer"] if str(x).strip())
        for row in data
    ]

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        repeat_results = results[start:end]
        predictions = [_normalize_prediction(getattr(result, "answer", [])) for result in repeat_results]
        all_predictions.append(predictions)

        metrics = compute_metrics(predictions, data)
        all_metrics.append(metrics)
        print(
            f"Run {i+1}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"FullCorrect={metrics['full_correct']}/{metrics['total']}, "
            f"PartialNoError={metrics['partial_no_error']}, "
            f"HasError={metrics['has_error']}"
        )

    runner.save_common_results(
        dataset_name=dataset_config["dataset"],
        model=experiment_config["llm_config"]["model_name"],
        prompt_method=prompt_method,
        all_predictions=all_predictions,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        results_path=experiment_config["results_path"],
        dataset_config=dataset_config,
        experiment_config=experiment_config,
    )

    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(data))


if __name__ == "__main__":
    main()
