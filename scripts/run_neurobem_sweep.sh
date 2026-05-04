#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

MODELS=(mlp gru tcn tcnlstm grutcn)
HISTORIES=(1 10 20 50)
EPOCHS=10000
UNROLL_LENGTH=50
SEED=10
NUM_WORKERS=4
GPU_ID=0
WANDB_MODE_PREF="auto"
SMOKE=0
SWEEP_ROOT=""
LIMIT_TRAIN_BATCHES="1.0"
LIMIT_VAL_BATCHES="1.0"
LIMIT_PREDICT_BATCHES="0"

usage() {
  cat <<'USAGE'
Usage: scripts/run_neurobem_sweep.sh [options]

Run the NeuroBEM full-state sweep on one GPU.

Options:
  --smoke                  Run a one-epoch MLP/H=1 smoke test only
  --sweep-root DIR         Output sweep root directory
  --epochs N               Max epochs for each run (default: 10000)
  --wandb-mode MODE        auto, online, or offline (default: auto)
  -h, --help               Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke)
      SMOKE=1
      shift
      ;;
    --sweep-root)
      [ "$#" -ge 2 ] || { echo "error: --sweep-root requires a value" >&2; exit 2; }
      SWEEP_ROOT="$2"
      shift 2
      ;;
    --epochs)
      [ "$#" -ge 2 ] || { echo "error: --epochs requires a value" >&2; exit 2; }
      EPOCHS="$2"
      shift 2
      ;;
    --wandb-mode)
      [ "$#" -ge 2 ] || { echo "error: --wandb-mode requires a value" >&2; exit 2; }
      WANDB_MODE_PREF="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$SMOKE" -eq 1 ]; then
  MODELS=(mlp)
  HISTORIES=(1)
  EPOCHS=1
  LIMIT_TRAIN_BATCHES="8"
  LIMIT_VAL_BATCHES="4"
  LIMIT_PREDICT_BATCHES="4"
fi

if [ -z "$SWEEP_ROOT" ]; then
  suffix="$(date +%Y%m%d-%H%M%S)"
  if [ "$SMOKE" -eq 1 ]; then
    SWEEP_ROOT="$REPO_ROOT/resources/experiments/neurobem_fullstate_smoke_$suffix"
  else
    SWEEP_ROOT="$REPO_ROOT/resources/experiments/neurobem_fullstate_$suffix"
  fi
fi

mkdir -p "$SWEEP_ROOT/logs"
REPORT="$SWEEP_ROOT/RUN_REPORT.md"
RESULTS_CSV="$SWEEP_ROOT/horizon_results.csv"
PLOTS_DIR="$SWEEP_ROOT/horizon_curves"

BATCH_SIZES=(64 32 16)
ACCUM_STEPS=(1 2 4)

cd "$SCRIPT_DIR"

append_report() {
  printf '%s\n' "$*" >> "$REPORT"
}

init_report() {
  cat > "$REPORT" <<REPORT
# NeuroBEM Full-State Sweep Report

- Sweep root: \`$SWEEP_ROOT\`
- Started at: \`$(date -Is)\`
- Models: \`${MODELS[*]}\`
- History lengths: \`${HISTORIES[*]}\`
- Unroll length: \`$UNROLL_LENGTH\`
- Max epochs: \`$EPOCHS\`
- Seed: \`$SEED\`
- OOM fallback: \`64x1 -> 32x2 -> 16x4\`
- W&B preference: \`$WANDB_MODE_PREF\`
- Batch limits: train=\`$LIMIT_TRAIN_BATCHES\`, val=\`$LIMIT_VAL_BATCHES\`, predict=\`$LIMIT_PREDICT_BATCHES\`

## Implementation Notes
- Added early stopping and gradient accumulation support in training.
- Added fixed experiment paths so eval reads the intended checkpoint.
- Eval computes horizon metrics batch-by-batch instead of storing the full test set in memory.
- Smoke mode limits train/validation/eval batches; formal runs are not batch-limited.
- Results are aggregated into one CSV and four horizon-curve PNGs.

## Events

REPORT
}

write_status() {
  local path="$1"
  local status="$2"
  local failure_reason="$3"
  local retry_count="$4"
  local batch_size="$5"
  local accumulate="$6"
  local wandb_mode="$7"
  local train_log="$8"
  local eval_log="$9"

  "$PYTHON" - "$path" "$status" "$failure_reason" "$retry_count" \
    "$batch_size" "$accumulate" "$wandb_mode" "$train_log" "$eval_log" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "failure_reason": sys.argv[3],
    "retry_count": int(sys.argv[4]),
    "batch_size": int(sys.argv[5]) if sys.argv[5] else None,
    "accumulate_grad_batches": int(sys.argv[6]) if sys.argv[6] else None,
    "effective_batch_size": (
        int(sys.argv[5]) * int(sys.argv[6]) if sys.argv[5] and sys.argv[6] else None
    ),
    "wandb_mode": sys.argv[7],
    "train_log": sys.argv[8],
    "eval_log": sys.argv[9],
}
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w") as file:
    json.dump(payload, file, indent=2)
PY
}

aggregate_results() {
  "$PYTHON" "$SCRIPT_DIR/aggregate_horizon_results.py" \
    --experiments-root "$SWEEP_ROOT" \
    --output "$RESULTS_CSV" \
    --plots-dir "$PLOTS_DIR" \
    --include-missing
}

wandb_logged_in() {
  if ! command -v wandb >/dev/null 2>&1; then
    return 1
  fi
  local status_output
  if ! status_output="$(wandb status 2>/dev/null)"; then
    return 1
  fi
  if echo "$status_output" | grep -q '"api_key": null'; then
    return 1
  fi
  echo "$status_output" | grep -q '"api_key":' || return 1
  return 0
}

initial_wandb_mode() {
  case "$WANDB_MODE_PREF" in
    online)
      echo "online"
      ;;
    offline)
      echo "offline"
      ;;
    auto)
      if wandb_logged_in; then
        echo "online"
      else
        echo "offline"
      fi
      ;;
    *)
      echo "error: --wandb-mode must be auto, online, or offline" >&2
      exit 2
      ;;
  esac
}

is_oom_log() {
  grep -Eiq 'out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory' "$1"
}

is_wandb_log() {
  grep -Eiq 'api_key not configured|wandb.errors|CommError|Network error|network|timed out|timeout|403|401' "$1"
}

run_train_attempt() {
  local model="$1"
  local history="$2"
  local exp_dir="$3"
  local batch_size="$4"
  local accumulate="$5"
  local wandb_mode="$6"
  local train_log="$7"

  WANDB_MODE="$wandb_mode" WANDB_SILENT=true "$PYTHON" train.py \
    --dataset neurobemfullstate \
    --predictor_type full_state \
    --model_type "$model" \
    --history_length "$history" \
    --unroll_length "$UNROLL_LENGTH" \
    --epochs "$EPOCHS" \
    --early_stopping True \
    --early_stopping_patience 300 \
    --early_stopping_min_delta 1e-5 \
    --batch_size "$batch_size" \
    --accumulate_grad_batches "$accumulate" \
    --num_workers "$NUM_WORKERS" \
    --limit_train_batches "$LIMIT_TRAIN_BATCHES" \
    --limit_val_batches "$LIMIT_VAL_BATCHES" \
    --limit_predict_batches "$LIMIT_PREDICT_BATCHES" \
    --accelerator cuda \
    --gpu_id "$GPU_ID" \
    --seed "$SEED" \
    --eval_horizons 1,10,25,50 \
    --limit_predict_batches "$LIMIT_PREDICT_BATCHES" \
    --wandb_mode "$wandb_mode" \
    --experiment_path "$exp_dir" \
    > "$train_log" 2>&1
}

run_eval_attempt() {
  local exp_dir="$1"
  local wandb_mode="$2"
  local eval_log="$3"
  local eval_batch_size="$4"

  WANDB_MODE="$wandb_mode" WANDB_SILENT=true "$PYTHON" eval.py \
    --dataset neurobemfullstate \
    --predictor_type full_state \
    --accelerator cuda \
    --gpu_id "$GPU_ID" \
    --eval_batch_size "$eval_batch_size" \
    --eval_horizons 1,10,25,50 \
    --wandb_mode "$wandb_mode" \
    --experiment_path "$exp_dir" \
    > "$eval_log" 2>&1
}

run_config() {
  local model="$1"
  local history="$2"
  local exp_name="${model}_H${history}_F${UNROLL_LENGTH}_seed${SEED}"
  local exp_dir="$SWEEP_ROOT/$exp_name"
  local status_path="$exp_dir/status.json"
  local retry_count=0
  local train_success=0
  local eval_success=0
  local failure_reason=""
  local train_log=""
  local eval_log=""
  local used_batch=""
  local used_accumulate=""
  local wandb_mode

  mkdir -p "$exp_dir/logs"
  wandb_mode="$(initial_wandb_mode)"
  append_report ""
  append_report "### $exp_name"
  append_report "- Started: \`$(date -Is)\`"
  append_report "- Initial W&B mode: \`$wandb_mode\`"

  for idx in "${!BATCH_SIZES[@]}"; do
    local batch_size="${BATCH_SIZES[$idx]}"
    local accumulate="${ACCUM_STEPS[$idx]}"
    local attempt_done=0

    while [ "$attempt_done" -eq 0 ]; do
      retry_count=$((retry_count + 1))
      train_log="$exp_dir/logs/train_attempt_${retry_count}_b${batch_size}_a${accumulate}_${wandb_mode}.log"
      append_report "- Train attempt $retry_count: batch=\`$batch_size\`, accumulate=\`$accumulate\`, wandb=\`$wandb_mode\`"

      set +e
      run_train_attempt "$model" "$history" "$exp_dir" "$batch_size" "$accumulate" "$wandb_mode" "$train_log"
      local exit_code=$?
      set -e

      if [ "$exit_code" -eq 0 ]; then
        train_success=1
        used_batch="$batch_size"
        used_accumulate="$accumulate"
        append_report "- Train success: batch=\`$batch_size\`, accumulate=\`$accumulate\`, wandb=\`$wandb_mode\`"
        attempt_done=1
        break
      fi

      if [ "$wandb_mode" = "online" ] && is_wandb_log "$train_log"; then
        failure_reason="W&B online failed; retried offline"
        append_report "- W&B online failure detected; retrying this attempt with \`WANDB_MODE=offline\`."
        wandb_mode="offline"
        continue
      fi

      if is_oom_log "$train_log"; then
        failure_reason="CUDA OOM at batch_size=$batch_size accumulate=$accumulate"
        append_report "- CUDA OOM detected; moving to smaller micro batch."
        attempt_done=1
        break
      fi

      failure_reason="train failed; see $train_log"
      append_report "- Train failed for non-OOM reason. See \`$train_log\`."
      write_status "$status_path" "failed" "$failure_reason" "$retry_count" \
        "${used_batch:-$batch_size}" "${used_accumulate:-$accumulate}" "$wandb_mode" "$train_log" ""
      aggregate_results || true
      return 1
    done

    if [ "$train_success" -eq 1 ]; then
      break
    fi
  done

  if [ "$train_success" -ne 1 ]; then
    failure_reason="resource limit after OOM fallbacks"
    append_report "- Failed after all OOM fallbacks."
    write_status "$status_path" "failed" "$failure_reason" "$retry_count" \
      "${used_batch:-16}" "${used_accumulate:-4}" "$wandb_mode" "$train_log" ""
    aggregate_results || true
    return 1
  fi

  local eval_exit=1
  for eval_batch in "$used_batch" 32 16; do
    eval_log="$exp_dir/logs/eval_${wandb_mode}_b${eval_batch}.log"
    append_report "- Eval attempt: batch=\`$eval_batch\`, wandb=\`$wandb_mode\`"
    set +e
    run_eval_attempt "$exp_dir" "$wandb_mode" "$eval_log" "$eval_batch"
    eval_exit=$?
    set -e

    if [ "$eval_exit" -eq 0 ]; then
      break
    fi
    if [ "$wandb_mode" = "online" ] && is_wandb_log "$eval_log"; then
      failure_reason="W&B online eval failed; retried offline"
      append_report "- W&B online eval failure detected; retrying eval offline."
      wandb_mode="offline"
      continue
    fi
    if is_oom_log "$eval_log"; then
      failure_reason="eval OOM at batch_size=$eval_batch"
      append_report "- Eval OOM detected; trying smaller eval batch."
      continue
    fi
    break
  done

  if [ "$eval_exit" -eq 0 ]; then
    eval_success=1
    append_report "- Eval success."
  else
    failure_reason="eval failed; see $eval_log"
    append_report "- Eval failed. See \`$eval_log\`."
  fi

  if [ "$eval_success" -eq 1 ]; then
    write_status "$status_path" "success" "$failure_reason" "$retry_count" \
      "$used_batch" "$used_accumulate" "$wandb_mode" "$train_log" "$eval_log"
    aggregate_results
    return 0
  fi

  write_status "$status_path" "failed" "$failure_reason" "$retry_count" \
    "$used_batch" "$used_accumulate" "$wandb_mode" "$train_log" "$eval_log"
  aggregate_results || true
  return 1
}

init_report
append_report "- Code directory: \`$REPO_ROOT\`"
append_report "- Python: \`$($PYTHON --version 2>&1)\`"

failed=0
total=0
for model in "${MODELS[@]}"; do
  for history in "${HISTORIES[@]}"; do
    total=$((total + 1))
    if ! run_config "$model" "$history"; then
      failed=$((failed + 1))
    fi
  done
done

aggregate_results || true
append_report ""
append_report "## Final Summary"
append_report "- Finished at: \`$(date -Is)\`"
append_report "- Total configs: \`$total\`"
append_report "- Failed configs: \`$failed\`"
append_report "- Results CSV: \`$RESULTS_CSV\`"
append_report "- Horizon curves: \`$PLOTS_DIR\`"

if [ "$failed" -gt 0 ]; then
  exit 1
fi
