"""统一评测入口 & 可复用评测工具

提供三层使用方式：
1. python run_all.py                → 运行当前 experiment_config.yaml 下的所有数据集
2. scripts/run_single_model.py      → 指定 model yaml，运行所有数据集
3. scripts/run_single_dataset.py    → 指定 dataset yaml，在所有 model 上运行
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DATASETS = [
    "ToMBench",
    "Tomato",
    "ToMQA",
    "ToMi",
]

EXPERIMENT_CONFIG = Path("experiment_config.yaml")
MODEL_CONFIGS_DIR = Path("experiment_configs")


# ---------------------------------------------------------------------------
# 可复用工具函数
# ---------------------------------------------------------------------------


def apply_config(yaml_path: str) -> None:
    """将指定 yaml 复制为 experiment_config.yaml 供各 task run.py 读取。"""
    src = Path(yaml_path)
    if not src.exists():
        raise FileNotFoundError(f"配置文件不存在: {src}")
    shutil.copy2(src, EXPERIMENT_CONFIG)
    print(f"[config] {src} → {EXPERIMENT_CONFIG}")


def discover_model_configs(configs_dir: str = str(MODEL_CONFIGS_DIR)) -> List[Path]:
    """发现 experiment_configs/ 下所有 model yaml（按文件名排序）。"""
    d = Path(configs_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"模型配置目录不存在: {d}")
    yamls = sorted(p for p in d.iterdir() if p.suffix in (".yaml", ".yml") and p.is_file())
    if not yamls:
        raise RuntimeError(f"{d} 中没有找到任何 yaml 配置文件")
    return yamls


def get_model_name(yaml_path: Path) -> str:
    """从 experiment config yaml 中提取 model_name。"""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("llm", {}).get("model_name", yaml_path.stem)


def get_dataset_name(yaml_path: str) -> str:
    """从 dataset config yaml 中提取 dataset 名称。"""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    name = cfg.get("dataset")
    if not name:
        raise ValueError(f"yaml 中缺少 'dataset' 字段: {yaml_path}")
    return name


def run_dataset(dataset: str) -> bool:
    """运行指定数据集的评测脚本。"""
    project_root = Path(__file__).resolve().parent
    run_script = Path(f"tasks/{dataset}/run.py")
    if not (project_root / run_script).exists():
        print(f"[{dataset}] run.py not found, skipping.")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {dataset}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(
            [sys.executable, str(run_script)],
            check=True,
            capture_output=False,
            cwd=str(project_root),
            env=env,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{dataset}] Error: {e}")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"[{dataset}] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_datasets(datasets: Optional[List[str]] = None) -> Dict[str, bool]:
    """依次运行指定数据集列表（默认全部），返回 {dataset: success}。"""
    if datasets is None:
        datasets = DATASETS
    results = {}
    for ds in datasets:
        results[ds] = run_dataset(ds)
    return results


# ---------------------------------------------------------------------------
# 默认入口：运行所有数据集
# ---------------------------------------------------------------------------


def main():
    results = run_datasets()

    print(f"\n{'='*60}")
    print("All datasets completed.")
    for ds, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {ds}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
