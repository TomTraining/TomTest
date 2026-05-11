"""HiToM 评测（结构化 MCQAnswer15）。15 选项（1 正 + 14 误），每 repeat 随机打乱选项顺序。"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from src.llm.client import LLMResponse
from HiToM.prompts import build_prompt
from HiToM.metrics import compute_metrics

LETTERS = list("ABCDEFGHIJKLMNO")  # 15 letters A-O


def build_choices_str(choices: Dict[str, str]) -> str:
    """将字母->文本映射格式化为多行选项字符串。"""
    return "\n".join(f"{letter}. {choices[letter]}" for letter in sorted(choices.keys()))


def shuffle_options(row: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """为一条样本打乱 15 个选项，返回 (choices_dict, gold_letter)。"""
    ans = row.get("Answer") or {}
    ca = ans.get("Correct_Answer", [])
    wa = ans.get("Wrong_Answer", [])
    correct = str(ca[0]).strip() if ca else ""
    wrong = [str(x).strip() for x in wa]

    # 合并：正确答案 + 14 个错误答案
    all_options = [correct] + wrong
    # 保证恰好 15 个（不足则截断或保持）
    all_options = all_options[:15]

    rng = random.Random(seed)
    indices = list(range(len(all_options)))
    rng.shuffle(indices)

    choices: Dict[str, str] = {}
    gold_letter = LETTERS[0]
    for new_pos, old_idx in enumerate(indices):
        letter = LETTERS[new_pos]
        choices[letter] = all_options[old_idx]
        if old_idx == 0:  # 正确答案原始索引为 0
            gold_letter = letter

    return choices, gold_letter


def preprocess_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤掉不满足 1 正 + 14 误格式的样本。"""
    valid = []
    skipped = 0
    for row in data:
        ans = row.get("Answer") or {}
        ca = ans.get("Correct_Answer", [])
        wa = ans.get("Wrong_Answer", [])
        if not isinstance(ca, list) or not isinstance(wa, list):
            skipped += 1
            continue
        if len(ca) != 1 or len(wa) != 14:
            skipped += 1
            continue
        valid.append(row)
    if skipped:
        print(f"Warning: skipped {skipped} rows (expected 1 Correct_Answer + 14 Wrong_Answer).")
    if not valid:
        raise RuntimeError("没有可评测样本：数据需为 HiToM 标准字段与 1+14 选项。")
    return valid


preprocess_mcq = preprocess_data


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/HiToM/config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="experiment_config.yaml")
    args = parser.parse_args()
    experiment_config = runner.load_experiment_config(args.experiment_config)
    print(f"Experiment config: {args.experiment_config}")

    prompt_method = dataset_config["method"]
    schema = runner.load_schema(dataset_config["schema"])

    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)
    judge_client = runner.create_judge_client(experiment_config["judge_config"], dataset_config)

    data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} raw rows from {dataset_config['path']}")
    data = preprocess_data(data)

    repeats = experiment_config["repeats"]
    print(f"Valid samples: {len(data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Schema: {dataset_config['schema']}")
    print(f"Repeats: {repeats} (each with different option shuffle)")

    all_prompts: List[List[str]] = []
    all_gold: List[List[str]] = []

    for i in range(repeats):
        repeat_prompts: List[str] = []
        repeat_gold: List[str] = []
        for j, row in enumerate(data):
            choices, gold_letter = shuffle_options(row, seed=42 * (i + 1) + j)
            choices_str = build_choices_str(choices)
            prompt = build_prompt(row, prompt_method, choices_str=choices_str)
            repeat_prompts.append(prompt)
            repeat_gold.append(gold_letter)
        all_prompts.append(repeat_prompts)
        all_gold.append(repeat_gold)

    flat_prompts = [p for repeat_prompts in all_prompts for p in repeat_prompts]
    print(f"Running inference ({len(flat_prompts)} prompts)...")
    results = client.batch_generate_structure(flat_prompts, schema)

    n = len(data)
    all_metrics: List[Dict[str, Any]] = []
    all_results: List[List[LLMResponse]] = []

    for i in range(repeats):
        start = i * n
        end = start + n
        repeat_results = results[start:end]
        all_results.append(repeat_results)
        predictions = [r.content.answer if r.content else None for r in repeat_results]
        gold_letters = all_gold[i]

        metrics = compute_metrics(predictions, gold_letters, data, judge_client)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=all_gold,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
