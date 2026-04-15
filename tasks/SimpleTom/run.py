"""SimpleToM 评测（结构化 MCQAnswer，3 选 1）。"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TASKS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TASKS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TASKS_ROOT))

from src import runner
from SimpleTom.metrics import compute_metrics
from SimpleTom.prompts import build_prompt, get_template

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _format_background(background: Any) -> str:
    if isinstance(background, str):
        return background.strip()
    if isinstance(background, Iterable):
        parts = [str(item).strip() for item in background if str(item).strip()]
        return "\n".join(parts)
    return ""


def _story_to_prompt_text(story: Dict[str, Any]) -> str:
    parts: List[str] = []
    full_story = str(story.get("full_story", "")).strip()
    summary = str(story.get("summary", "")).strip()
    background = _format_background(story.get("background", []))

    if full_story:
        parts.append(full_story)
    if summary:
        parts.append(f"Summary: {summary}")
    if background:
        parts.append(f"Background: {background}")

    return "\n".join(parts).strip()


def build_mcq_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将统一格式样本转成 3 选 1 结构。"""
    story = row.get("Story")
    answer = row.get("Answer")
    if not isinstance(story, dict) or not isinstance(answer, dict):
        return None

    correct = answer.get("Correct_Answer")
    wrong = answer.get("Wrong_Answer")
    if not isinstance(correct, list) or not isinstance(wrong, list):
        return None
    if len(correct) != 1 or len(wrong) != 1:
        return None

    correct_text = str(correct[0]).strip()
    wrong_text = str(wrong[0]).strip()
    if not correct_text or not wrong_text:
        return None

    letters = ["A", "B", "C"]
    texts = [correct_text, wrong_text, "Empty"]
    original_choices = {letters[i]: texts[i] for i in range(3)}

    return {
        "story": _story_to_prompt_text(story),
        "question": str(row.get("Question", "")).strip(),
        "original_choices": original_choices,
        "gold_letter": "A",
    }


def preprocess_mcq(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    skipped = 0

    for row in data:
        mcq = build_mcq_from_row(row)
        if mcq is None:
            skipped += 1
            continue
        out = dict(row)
        out["_mcq"] = mcq
        valid.append(out)

    if skipped:
        print(f"Warning: skipped {skipped} rows (expected 1 Correct_Answer + 1 Wrong_Answer).")
    if not valid:
        raise RuntimeError("没有可评测样本：数据需包含 1 个正确答案和 1 个错误答案。")
    return valid


def shuffle_mcq_options(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """打乱 3 个选项，并同步更新 gold_letter。"""
    rng = random.Random(seed)
    letters = sorted(mcq["original_choices"].keys())
    texts = [mcq["original_choices"][letter] for letter in letters]
    old_gold_idx = letters.index(mcq["gold_letter"])

    indices = list(range(len(letters)))
    rng.shuffle(indices)

    new_choices: Dict[str, str] = {}
    new_gold = mcq["gold_letter"]
    for new_pos, old_idx in enumerate(indices):
        new_choices[letters[new_pos]] = texts[old_idx]
        if old_idx == old_gold_idx:
            new_gold = letters[new_pos]

    return {**mcq, "original_choices": new_choices, "gold_letter": new_gold}


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/SimpleTom/config.yaml")
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
    print(f"MCQ samples: {len(data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {repeats} (each with different option shuffle)")

    all_prompts: List[str] = []
    repeat_data: List[List[Dict[str, Any]]] = []

    for i in range(repeats):
        shuffled_rows: List[Dict[str, Any]] = []
        for j, row in enumerate(data):
            shuffled_mcq = shuffle_mcq_options(row["_mcq"], seed=101 * (i + 1) + j)
            shuffled_row = dict(row)
            shuffled_row["_mcq"] = shuffled_mcq
            shuffled_rows.append(shuffled_row)
            all_prompts.append(build_prompt(template, shuffled_row))
        repeat_data.append(shuffled_rows)

    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    n = len(data)
    all_predictions: List[List[str]] = []
    all_metrics: List[Dict[str, Any]] = []
    all_gold: List[List[str]] = []

    for i in range(repeats):
        start = i * n
        end = start + n
        repeat_results = results[start:end]
        rows = repeat_data[i]
        predictions = [getattr(result, "answer", "") for result in repeat_results]
        all_predictions.append(predictions)

        metrics = compute_metrics(predictions, rows)
        all_metrics.append(metrics)
        all_gold.append([row["_mcq"]["gold_letter"] for row in rows])
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

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
