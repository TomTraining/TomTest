#!/usr/bin/env bash
# 仅使用 GPU 6/7 跑评测，并在数据集阶段之间动态切换：
# - 非 judge 数据集：2 worker（GPU 6 + GPU 7）
# - judge 数据集：1 worker（GPU 6）+ 1 judge（GPU 7）
#
# 默认数据集顺序下，通常会表现为：
# - SocialIQA / ToMBench / Tomato：双 worker
# - ToMi / ToMQA：单 worker + 常驻 judge
#
# 说明：
# - `experiment.log` / `run.log` / `vllm_serve.log` 会覆盖旧日志。
# - `baseline.txt` 与 `results_table.md` 由 run.py 追加写入，保留历史结果。

set -euo pipefail

TOMTEST="${TOMTEST:-/home/xujy/TomTest}"
DATASET_ROOT="${DATASET_ROOT:-$TOMTEST/TomDatasets}"
PROMPT_DIR="${PROMPT_DIR:-$TOMTEST/prompt}"
PROMPT_NAMES="${PROMPT_NAMES:-State-First ToM|Question-Targeted ToM}"
VLLM_EXTRA="${VLLM_EXTRA:-}"
EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS:-}"
DATASET_LIST="${DATASET_LIST:-SocialIQA ToMBench Tomato ToMi ToMQA}"
JUDGE_OPEN_DATASETS="${JUDGE_OPEN_DATASETS:-ToMi ToMQA}"
GPU_IDS_LIST="${GPU_IDS_LIST:-6 7}"
PORTS_LIST="${PORTS_LIST:-8006 8007}"
MODEL_TAGS="${MODEL_TAGS:-}"
MODEL_DATASET_SPECS="${MODEL_DATASET_SPECS:-}"

read -r -a DATASETS <<< "$DATASET_LIST"
read -r -a JUDGE_DATASETS_ARR <<< "$JUDGE_OPEN_DATASETS"
read -r -a GPU_IDS <<< "$GPU_IDS_LIST"
read -r -a PORTS <<< "$PORTS_LIST"
IFS='|' read -r -a PROMPT_NAMES_ARR <<< "$PROMPT_NAMES"

# 与 ToM-baseline/run.py 中默认路径一致（本地权重）
QWEN06="/DATA/zhanghy/xai/alignment/SELOR/learning-from-rationales/PRETRAINEDMODEL/Qwen3-0.6B"
QWEN4B="/data/yugx/LongBench/simple_tune/Qwen3-4B"
QWEN8B="/data/yugx/LongBench/simple_tune/Qwen3-8B"
GEMMA="/DATA/xujy/models/gemma-3-4b-it"
LLAMA="/DATA/zhanghy/xai/alignment/SELOR/learning-from-rationales/PRETRAINEDMODEL/Llama-3.1-8B-Instruct"

JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-8B}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$QWEN8B}"
JUDGE_GPU_ID="${JUDGE_GPU_ID:-7}"
JUDGE_PORT="${JUDGE_PORT:-8010}"
JUDGE_API_URL="${JUDGE_API_URL:-http://127.0.0.1:${JUDGE_PORT}/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-not-needed}"

ALL_MODELS=(
  "$QWEN06|Qwen3-0.6B|Qwen3-0.6B"
  "$QWEN4B|Qwen3-4B|Qwen3-4B"
  "$QWEN8B|Qwen3-8B|Qwen3-8B"
  "$GEMMA|gemma-3-4b-it|gemma-3-4b-it"
  "$LLAMA|Meta-Llama-3.1-8B-Instruct|Meta-Llama-3.1-8B-Instruct"
)
LOCAL_MODELS=()

cd "$TOMTEST"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERR] 数据集目录不存在: $DATASET_ROOT"
  exit 1
fi

if [[ "${#GPU_IDS[@]}" -ne "${#PORTS[@]}" ]]; then
  echo "[ERR] GPU_IDS_LIST 与 PORTS_LIST 数量不一致。"
  exit 1
fi

if [[ ! -d "$JUDGE_MODEL_PATH" ]]; then
  echo "[ERR] judge 模型路径不存在: $JUDGE_MODEL_PATH"
  exit 1
fi

PROMPT_LABEL="$(printf '%s, ' "${PROMPT_NAMES_ARR[@]}")"
PROMPT_LABEL="${PROMPT_LABEL%, }"

declare -a SLOT_PID
declare -a SLOT_TAG
declare -a ACTIVE_SLOTS
declare -a SEGMENT_MODES
declare -a SEGMENT_DATASETS
declare -a SEGMENT_TASKS

FAILED=0
NEXT_LOCAL_IDX=0
COMPLETED_MODELS=0
TOTAL_MODELS="${#LOCAL_MODELS[@]}"
JUDGE_VLLM_PID=""
JUDGE_SLOT="-1"
CURRENT_SEGMENT_MODE=""
CURRENT_SEGMENT_DATASET_STR=""

for slot in "${!GPU_IDS[@]}"; do
  SLOT_PID[$slot]=""
  SLOT_TAG[$slot]=""
  if [[ "${GPU_IDS[$slot]}" == "$JUDGE_GPU_ID" ]]; then
    JUDGE_SLOT="$slot"
  fi
done

if [[ "$JUDGE_SLOT" == "-1" ]]; then
  echo "[ERR] JUDGE_GPU_ID=$JUDGE_GPU_ID 不在 GPU_IDS_LIST=[$GPU_IDS_LIST] 中。"
  exit 1
fi

cleanup_all() {
  if [[ -n "${JUDGE_VLLM_PID:-}" ]]; then
    kill "$JUDGE_VLLM_PID" 2>/dev/null || true
    wait "$JUDGE_VLLM_PID" 2>/dev/null || true
  fi
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup_all EXIT INT TERM

wait_vllm_port() {
  local port="$1"
  local tag="$2"
  local n=0
  echo "[INFO] [$tag] 等待 vLLM 监听 :${port} …"
  until curl -s -o /dev/null --connect-timeout 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null; do
    sleep 3
    n=$((n + 1))
    if [[ "$n" -gt 120 ]]; then
      echo "[ERR] [$tag] vLLM 启动超时（port=${port}）。"
      return 1
    fi
  done
  echo "[INFO] [$tag] vLLM 已就绪。"
}

dataset_needs_judge() {
  local dataset="$1"
  local judged
  for judged in "${JUDGE_DATASETS_ARR[@]}"; do
    if [[ "$dataset" == "$judged" ]]; then
      return 0
    fi
  done
  return 1
}

build_model_tasks() {
  local item model_path served_name tag dataset_list spec entry wanted
  local -a model_tag_filters=() raw_specs=()
  declare -A spec_map=()

  LOCAL_MODELS=()

  if [[ -n "$MODEL_TAGS" ]]; then
    read -r -a model_tag_filters <<< "$MODEL_TAGS"
  fi

  if [[ -n "$MODEL_DATASET_SPECS" ]]; then
    IFS=';' read -r -a raw_specs <<< "$MODEL_DATASET_SPECS"
    for spec in "${raw_specs[@]}"; do
      [[ -z "$spec" ]] && continue
      tag="${spec%%=*}"
      dataset_list="${spec#*=}"
      dataset_list="${dataset_list//,/ }"
      spec_map["$tag"]="$dataset_list"
    done
  fi

  for item in "${ALL_MODELS[@]}"; do
    IFS='|' read -r model_path served_name tag <<< "$item"

    if (( ${#model_tag_filters[@]} > 0 )); then
      local matched=0
      for wanted in "${model_tag_filters[@]}"; do
        if [[ "$tag" == "$wanted" ]]; then
          matched=1
          break
        fi
      done
      if (( matched == 0 )); then
        continue
      fi
    fi

    if [[ -n "$MODEL_DATASET_SPECS" ]]; then
      if [[ -z "${spec_map[$tag]+x}" ]]; then
        continue
      fi
      dataset_list="${spec_map[$tag]}"
    else
      dataset_list="$DATASET_LIST"
    fi

    entry="$model_path|$served_name|$tag|$dataset_list"
    LOCAL_MODELS+=("$entry")
  done
}

build_dataset_segments() {
  local dataset mode last_mode="" current=""
  SEGMENT_MODES=()
  SEGMENT_DATASETS=()

  for dataset in "${DATASETS[@]}"; do
    if dataset_needs_judge "$dataset"; then
      mode="judge"
    else
      mode="workers"
    fi

    if [[ -z "$last_mode" ]]; then
      last_mode="$mode"
      current="$dataset"
      continue
    fi

    if [[ "$mode" == "$last_mode" ]]; then
      current+=" $dataset"
    else
      SEGMENT_MODES+=("$last_mode")
      SEGMENT_DATASETS+=("$current")
      last_mode="$mode"
      current="$dataset"
    fi
  done

  if [[ -n "$last_mode" ]]; then
    SEGMENT_MODES+=("$last_mode")
    SEGMENT_DATASETS+=("$current")
  fi
}

build_segment_tasks() {
  local segment_str="$1"
  local item model_path served_name tag model_dataset_str overlap dataset wanted
  local -a segment_datasets model_datasets

  read -r -a segment_datasets <<< "$segment_str"
  SEGMENT_TASKS=()

  for item in "${LOCAL_MODELS[@]}"; do
    IFS='|' read -r model_path served_name tag model_dataset_str <<< "$item"
    read -r -a model_datasets <<< "$model_dataset_str"
    overlap=""

    for dataset in "${segment_datasets[@]}"; do
      for wanted in "${model_datasets[@]}"; do
        if [[ "$dataset" == "$wanted" ]]; then
          if [[ -z "$overlap" ]]; then
            overlap="$dataset"
          else
            overlap+=" $dataset"
          fi
          break
        fi
      done
    done

    if [[ -n "$overlap" ]]; then
      SEGMENT_TASKS+=("$model_path|$served_name|$tag|$overlap")
    fi
  done
}

stop_judge_service() {
  if [[ -n "${JUDGE_VLLM_PID:-}" ]]; then
    kill "$JUDGE_VLLM_PID" 2>/dev/null || true
    wait "$JUDGE_VLLM_PID" 2>/dev/null || true
    JUDGE_VLLM_PID=""
  fi
}

start_judge_service() {
  local judge_root="$TOMTEST/result/_judge_service/primary"
  local judge_log="$judge_root/vllm_serve.log"

  if [[ -n "${JUDGE_VLLM_PID:-}" ]] && kill -0 "$JUDGE_VLLM_PID" 2>/dev/null; then
    return 0
  fi

  mkdir -p "$judge_root"
  echo "========== [judge:$JUDGE_MODEL] GPU=$JUDGE_GPU_ID PORT=$JUDGE_PORT =========="
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU_ID" vllm serve "$JUDGE_MODEL_PATH" \
    --port "$JUDGE_PORT" \
    --served-model-name "$JUDGE_MODEL" \
    $VLLM_EXTRA \
    >"$judge_log" 2>&1 &
  JUDGE_VLLM_PID=$!

  wait_vllm_port "$JUDGE_PORT" "judge:$JUDGE_MODEL"
  echo "[INFO] [judge:$JUDGE_MODEL] 常驻 judge 服务已启动：$JUDGE_API_URL"
}

prepare_segment_mode() {
  local mode="$1"
  local slot
  ACTIVE_SLOTS=()

  case "$mode" in
    workers)
      stop_judge_service
      for slot in "${!GPU_IDS[@]}"; do
        ACTIVE_SLOTS+=("$slot")
      done
      ;;
    judge)
      start_judge_service
      for slot in "${!GPU_IDS[@]}"; do
        if [[ "$slot" != "$JUDGE_SLOT" ]]; then
          ACTIVE_SLOTS+=("$slot")
        fi
      done
      if (( ${#ACTIVE_SLOTS[@]} == 0 )); then
        echo "[ERR] judge 模式下没有可用 worker GPU。"
        exit 1
      fi
      ;;
    *)
      echo "[ERR] 未知阶段模式: $mode"
      exit 1
      ;;
  esac
}

run_one_local_worker() {
  local model_path="$1"
  local served_name="$2"
  local tag="$3"
  local gpu="$4"
  local port="$5"
  local slot="$6"
  local dataset_list_str="$7"

  if [[ ! -d "$model_path" ]]; then
    echo "[WARN] 跳过（路径不存在）: $model_path ($tag)"
    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))
    return 0
  fi

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    local model_root="$TOMTEST/result/${tag}"
    local api_local="http://127.0.0.1:${port}/v1"
    local serve_log="$model_root/vllm_serve.log"
    local dataset result_dir run_log
    local -a segment_datasets
    read -r -a segment_datasets <<< "$dataset_list_str"

    mkdir -p "$model_root"

    echo "========== [$tag] GPU=$gpu PORT=$port MODE=$CURRENT_SEGMENT_MODE DATASETS=[$dataset_list_str] =========="
    # shellcheck disable=SC2086
    vllm serve "$model_path" \
      --port "$port" \
      --served-model-name "$served_name" \
      $VLLM_EXTRA \
      >"$serve_log" 2>&1 &
    local vllm_pid=$!

    stop_local() {
      kill "$vllm_pid" 2>/dev/null || true
      wait "$vllm_pid" 2>/dev/null || true
    }
    trap stop_local EXIT

    wait_vllm_port "$port" "$tag"

    for dataset in "${segment_datasets[@]}"; do
      result_dir="$model_root/$dataset"
      run_log="$result_dir/run.log"
      mkdir -p "$result_dir"
      echo "[INFO] [$tag] 开始数据集 $dataset -> result/${tag}/${dataset}/"
      # shellcheck disable=SC2086
      python run.py \
        --dataset-root "$DATASET_ROOT" \
        --prompt-dir "$PROMPT_DIR" \
        --result-dir "$result_dir" \
        --dataset-filter "$dataset" \
        --prompt-names "${PROMPT_NAMES_ARR[@]}" \
        --model "$served_name" \
        --model-tag "$tag" \
        --api-url "$api_local" \
        --api-key not-needed \
        --judge-model "$JUDGE_MODEL" \
        --judge-api-url "$JUDGE_API_URL" \
        --judge-api-key "$JUDGE_API_KEY" \
        --judge-open-datasets "${JUDGE_DATASETS_ARR[@]}" \
        $EXTRA_RUN_ARGS \
        >"$run_log" 2>&1
      echo "[INFO] [$tag] 完成数据集 $dataset -> result/${tag}/${dataset}/"
    done

    echo "[INFO] [$tag] 阶段完成 -> result/${tag}/"
  ) &

  SLOT_PID[$slot]="$!"
  SLOT_TAG[$slot]="$tag"
}

start_next_model_for_slot() {
  local slot="$1"
  local model_path served_name tag dataset_list_str

  if (( NEXT_LOCAL_IDX >= TOTAL_MODELS )); then
    return 1
  fi

  IFS='|' read -r model_path served_name tag dataset_list_str <<< "${SEGMENT_TASKS[$NEXT_LOCAL_IDX]}"
  run_one_local_worker "$model_path" "$served_name" "$tag" "${GPU_IDS[$slot]}" "${PORTS[$slot]}" "$slot" "$dataset_list_str"
  NEXT_LOCAL_IDX=$((NEXT_LOCAL_IDX + 1))
}

check_judge_health() {
  if [[ -n "${JUDGE_VLLM_PID:-}" ]] && ! kill -0 "$JUDGE_VLLM_PID" 2>/dev/null; then
    echo "[ERR] [judge:$JUDGE_MODEL] judge 服务已退出。"
    FAILED=1
    return 1
  fi
}

check_finished_slots() {
  local slot pid status
  for slot in "${ACTIVE_SLOTS[@]}"; do
    pid="${SLOT_PID[$slot]:-}"
    if [[ -z "$pid" ]] || kill -0 "$pid" 2>/dev/null; then
      continue
    fi

    set +e
    wait "$pid"
    status=$?
    set -e

    if [[ "$status" -ne 0 ]]; then
      echo "[ERR] [${SLOT_TAG[$slot]}] 任务失败，退出码=$status"
      FAILED=1
    fi

    SLOT_PID[$slot]=""
    SLOT_TAG[$slot]=""
    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))

    start_next_model_for_slot "$slot" || true
  done
}

run_segment() {
  local mode="$1"
  local dataset_str="$2"
  local slot

  CURRENT_SEGMENT_MODE="$mode"
  CURRENT_SEGMENT_DATASET_STR="$dataset_str"
  build_segment_tasks "$dataset_str"
  NEXT_LOCAL_IDX=0
  COMPLETED_MODELS=0
  TOTAL_MODELS="${#SEGMENT_TASKS[@]}"

  if (( TOTAL_MODELS == 0 )); then
    echo "[INFO] ===== 跳过空阶段: mode=$mode | datasets=[$dataset_str] ====="
    return 0
  fi

  for slot in "${!GPU_IDS[@]}"; do
    SLOT_PID[$slot]=""
    SLOT_TAG[$slot]=""
  done

  prepare_segment_mode "$mode"
  echo "[INFO] ===== 阶段切换: mode=$mode | datasets=[$dataset_str] ====="

  for slot in "${ACTIVE_SLOTS[@]}"; do
    start_next_model_for_slot "$slot" || true
  done

  while (( COMPLETED_MODELS < TOTAL_MODELS )); do
    sleep 5
    check_judge_health || break
    check_finished_slots
  done
}

build_model_tasks

TOTAL_MODELS="${#LOCAL_MODELS[@]}"
if (( TOTAL_MODELS == 0 )); then
  echo "[ERR] 没有匹配到任何模型任务。请检查 MODEL_TAGS 或 MODEL_DATASET_SPECS。"
  exit 1
fi

build_dataset_segments

for idx in "${!SEGMENT_MODES[@]}"; do
  run_segment "${SEGMENT_MODES[$idx]}" "${SEGMENT_DATASETS[$idx]}"
  if [[ "$FAILED" -ne 0 ]]; then
    break
  fi
done

stop_judge_service

if [[ "$FAILED" -ne 0 ]]; then
  echo "[DONE] 已结束，但至少有一个模型任务失败。请检查各 result/<模型名>/<数据集>/run.log 与 result/<模型名>/vllm_serve.log。"
  exit 1
fi

echo "[DONE] ${PROMPT_LABEL} 已按动态 2-worker / 1-worker+judge 模式跑完。结果在 $TOMTEST/result/<模型名>/<数据集>/"
