"""ToMBench 评测脚本（基于结构化输出）"""
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from src import judge as judge_module

from ToMBench.prompts import get_template, build_prompt
from ToMBench.metrics import compute_metrics

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _is_correct(pred: str, gold: str) -> bool:
    return pred == gold


def main():
    dataset_config = runner.load_dataset_config("tasks/ToMBench/config.yaml")
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    schema = dataset_config["schema"]
    prompt_method = dataset_config["default_prompt"]
    template = get_template(prompt_method)
    client = runner.create_llm_client(experiment_config["llm_config"])

    badcase_enabled = experiment_config["badcase_enabled"]
    enable_judge = experiment_config["enable_llm_judge"]
    judge_client = None
    if enable_judge:
        judge_client = runner.create_llm_client(experiment_config["judge_config"])

    data = runner.load_and_limit_data(
        subset=dataset_config["subset"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} samples from {dataset_config['subset']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {experiment_config['repeats']}")

    prompts = [build_prompt(template, row) for row in data]
    all_prompts = prompts * experiment_config["repeats"]

    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    gold_answers = [row['Answer']['Correct Answer'][0] for row in data]

    all_predictions: List[List[str]] = []
    all_metrics: List[Dict[str, Any]] = []
    all_metrics_with_judge: List[Dict[str, Any]] = []
    all_badcases: List[Dict[str, Any]] = []

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        repeat_results = results[start:end]
        predictions = [r.answer for r in repeat_results]
        all_predictions.append(predictions)

        metrics = compute_metrics(predictions, data)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

        # --- LLM Judge 兜底 ---
        judge_verdicts = None
        if judge_client:
            failed_items = [
                {
                    "raw_response": r.raw_response,
                    "gold_answer": gold,
                    "question": prompt,
                }
                for r, gold, prompt in zip(repeat_results, gold_answers, prompts)
                if not r.extraction_success
            ]
            if failed_items:
                judge_results = judge_module.batch_judge(judge_client, failed_items)
                judge_verdicts_full: List[bool] = []
                ji = 0
                for r in repeat_results:
                    if not r.extraction_success:
                        judge_verdicts_full.append(judge_results[ji])
                        ji += 1
                    else:
                        judge_verdicts_full.append(False)
                judge_verdicts = judge_verdicts_full

                corrected = runner.build_corrected_predictions(
                    predictions, repeat_results, judge_results, gold_answers,
                )
                metrics_j = compute_metrics(corrected, data)
                all_metrics_with_judge.append(metrics_j)
                print(
                    f"  [Judge] Accuracy={metrics_j['accuracy']:.4f}, "
                    f"Recovered={metrics_j['correct'] - metrics['correct']}"
                )
            else:
                all_metrics_with_judge.append(metrics)

        # --- Bad case 收集 ---
        if badcase_enabled:
            bcs = runner.collect_badcases(
                repeat_results, predictions, gold_answers, prompts,
                dataset_config["dataset"], _is_correct,
                repeat_idx=i, judge_verdicts=judge_verdicts,
            )
            all_badcases.extend(bcs)

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
        badcases=all_badcases if badcase_enabled else None,
        all_metrics_with_judge=all_metrics_with_judge if enable_judge else None,
    )

    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(gold_answers))


if __name__ == "__main__":
    main()
