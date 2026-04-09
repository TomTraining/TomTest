#!/usr/bin/env bash
set -euo pipefail
# 最终 SCRIPT_DIR 就是脚本文件所在的绝对路径目录,
# 项目的根目录,pwd 输出其绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOMTEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_PATH="${TOMTEST_DIR}/data/tomi_test_processed.json"
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="not-needed"
MODEL="Qwen3-4B"
OUTPUT_DIR="${TOMTEST_DIR}/result/tomi_processed_runs"
RUN_NAME="$(date +tomi_processed_%Y%m%d_%H%M%S)"
SAMPLE_SIZE="0"
TEMPERATURE="0.01"
TOP_P="1.0"
MAX_TOKENS="32"
SAVE_INTERVAL="500"
FORMAT_RETRY="1"

print_help() {
  cat <<'USAGE'
Run ToMi processed JSON evaluation on local vLLM.

Usage:
  bash scripts/run_tomi_test_processed.sh [options]

Options:
  --model <name>          Served model name on vLLM (default: Qwen3-4B)
  --base-url <url>        OpenAI-compatible endpoint (default: http://127.0.0.1:8000/v1)
  --api-key <key>         API key (default: not-needed)
  --data <path>           Dataset json path (default: TomTest/data/tomi_test_processed.json)
  --output-dir <path>     Output directory (default: TomTest/result/tomi_processed_runs)
  --run-name <name>       Output file prefix (default: tomi_processed_YYYYmmdd_HHMMSS)
  --sample-size <int>     0 means full dataset (default: 0)
  --temperature <float>   Sampling temperature (default: 0.0)
  --top-p <float>         Top-p (default: 1.0)
  --max-tokens <int>      Max generation tokens (default: 32)
  --save-interval <int>   Intermediate save interval (default: 500)
  --format-retry          Retry once if <answer> tag is missing (default: on)
  --no-format-retry       Disable format retry
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --save-interval)
      SAVE_INTERVAL="$2"
      shift 2
      ;;
    --format-retry)
      FORMAT_RETRY="1"
      shift
      ;;
    --no-format-retry)
      FORMAT_RETRY="0"
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "[ERR] Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

if [[ ! -f "$DATA_PATH" ]]; then
  echo "[ERR] dataset not found: $DATA_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  python "${SCRIPT_DIR}/retest_tomi_local_vllm.py"
  --data "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --base-url "$BASE_URL"
  --api-key "$API_KEY"
  --model "$MODEL"
  --sample-size "$SAMPLE_SIZE"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-tokens "$MAX_TOKENS"
  --save-interval "$SAVE_INTERVAL"
)

if [[ "$FORMAT_RETRY" == "1" ]]; then
  cmd+=(--format-retry)
fi

echo "[INFO] Running ToMi processed evaluation"
echo "[INFO] data=$DATA_PATH"
echo "[INFO] model=$MODEL"
echo "[INFO] base_url=$BASE_URL"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] run_name=$RUN_NAME"

"${cmd[@]}"
