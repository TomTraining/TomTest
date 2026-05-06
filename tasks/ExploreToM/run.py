"""ExploreToM 评测脚本（基于结构化输出）"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT_DIR / "tasks"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(TASKS_DIR))

from src import runner
from ExploreToM.prompts import build_prompt
from ExploreToM.metrics import compute_metrics


def extract_gold_answers(data):
    """提取标准答案：取 Answer.Correct_Answer 的第一个，用于保存结果和可选 judge。"""
    golds = []

    for row in data:
        answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
        correct_list = answer_block.get("Correct_Answer", [])

        if isinstance(correct_list, list) and correct_list:
            golds.append(correct_list[0])
        elif isinstance(correct_list, list):
            golds.append("")
        elif correct_list is None:
            golds.append("")
        else:
            golds.append(str(correct_list))

    return golds


def main():
    dataset_config = runner.load_dataset_config("tasks/ExploreToM/config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="experiment_config.yaml")
    args = parser.parse_args()

    experiment_config = runner.load_experiment_config(args.experiment_config)
    print(f"Experiment config: {args.experiment_config}")

    prompt_method = dataset_config["method"]
    schema = runner.load_schema(dataset_config["schema"])

    client = runner.create_llm_client(
        experiment_config["llm_config"],
        dataset_config,
    )
    judge_client = runner.create_judge_client(
        experiment_config["judge_config"],
        dataset_config,
    )

    data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} samples from {dataset_config['path']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Schema: {dataset_config['schema']}")
    print(f"Repeats: {experiment_config['repeats']}")

    gold_answers = extract_gold_answers(data)

    prompts = [build_prompt(row, prompt_method) for row in data]
    all_prompts = [prompts for _ in range(experiment_config["repeats"])]

    flat_prompts = [p for repeat_prompts in all_prompts for p in repeat_prompts]
    print(f"Running inference ({len(flat_prompts)} prompts)...")

    results = client.batch_generate_structure(flat_prompts, schema)

    all_metrics = []
    all_results = []

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)

        repeat_results = results[start:end]
        all_results.append(repeat_results)

        predictions = [r.content.answer if r.content else None for r in repeat_results]

        metrics = compute_metrics(
            predictions=predictions,
            gold_answers=gold_answers,
            data=data,
            judge_client=judge_client,
        )
        all_metrics.append(metrics)

        print(
            f"Run {i + 1}: Accuracy={metrics['accuracy']:.4f}, "
            f"Correct={metrics['correct']}/{metrics['total']}"
        )

    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    runner.print_summary_stats(
        all_metrics,
        experiment_config["repeats"],
        len(gold_answers),
    )


if __name__ == "__main__":
    main()