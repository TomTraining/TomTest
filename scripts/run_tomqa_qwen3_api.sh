#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/zhangfutong/Documents/ICT/科研/Social-TOM/home/xujy/TomTest"

MODEL_NAME="${MODEL_NAME:-qwen/qwen3-8b}"
MODEL_TAG="${MODEL_TAG:-$MODEL_NAME}"
MODEL_TAG_SAFE="${MODEL_TAG//\//_}"
API_URL="${API_URL:-https://openrouter.ai/api/v1}"
API_KEY="${API_KEY:-${OPENROUTER_API_KEY:-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}}"
PROMPT_NAME="${PROMPT_NAME:-BigToM Standard}"
RESULT_DIR="${RESULT_DIR:-$ROOT/result/${MODEL_TAG_SAFE}/ToMQA}"

if [[ -z "$API_KEY" ]]; then
  echo "[ERR] API_KEY 不能为空，请先设置，例如：" >&2
  echo "      export API_KEY='your_openrouter_api_key'" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"

PYTHONUNBUFFERED=1 python "$ROOT/run.py" \
  --dataset-root "$ROOT/TomDatasets" \
  --prompt-dir "$ROOT/prompt" \
  --result-dir "$RESULT_DIR" \
  --model "$MODEL_NAME" \
  --model-tag "$MODEL_TAG" \
  --api-url "$API_URL" \
  --api-key "$API_KEY" \
  --dataset-filter ToMQA \
  --split-filter test \
  --prompt-names "$PROMPT_NAME" \
  --judge-open-datasets \
  --shuffle-repeats 1 \
  "$@" \
  2>&1 | tee "$RESULT_DIR/run.log"
