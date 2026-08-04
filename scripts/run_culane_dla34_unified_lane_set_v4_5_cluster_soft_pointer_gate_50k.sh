#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
POINTER_CONFIG="${POINTER_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_5_cluster_soft_pointer.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-50000}"
TRAIN_STEPS="${TRAIN_STEPS:-15000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2500}"
SEEDS="${SEEDS:-3407}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GRAD_AUDIT="${RUN_GRAD_AUDIT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TRAJECTORY_EVAL="${RUN_TRAJECTORY_EVAL:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-0.75}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_5_cluster_soft_pointer_gate_50k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_5_cluster_soft_pointer_gate_50k}"
ARM_NAME="${ARM_NAME:-cluster_soft_pointer}"

if (( CHECKPOINT_INTERVAL < 1 )); then
  echo "CHECKPOINT_INTERVAL must be positive for the V4.5 trajectory" >&2
  exit 1
fi
if (( TRAIN_STEPS % CHECKPOINT_INTERVAL != 0 )); then
  echo "TRAIN_STEPS must be divisible by CHECKPOINT_INTERVAL" >&2
  exit 1
fi

"${PYTHON}" - "${POINTER_CONFIG}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
selection = cfg["model"]["structured_query"]["set_selection"]
loss = cfg["loss"]
training = cfg["training"]
checks = {
    "cluster_soft_randomized_teacher": selection.get("pointer_teacher_mode")
    == "cluster_soft_randomized",
    "representable_min_020": abs(
        float(selection.get("pointer_cluster_representable_min", -1.0)) - 0.20
    ) < 1e-12,
    "quality_delta_010": abs(
        float(selection.get("pointer_cluster_quality_delta", -1.0)) - 0.10
    ) < 1e-12,
    "temperature_003": abs(
        float(selection.get("pointer_cluster_temperature", -1.0)) - 0.03
    ) < 1e-12,
    "fixed_lane_row_grid": selection.get("row_grid_mode") == "fixed_rows",
    "teacher_only": float(selection.get("pointer_free_rollout_weight", 0.0)) == 0.0,
    "max_quality_unary": loss.get("pointer_unary_target_mode") == "max_quality",
    "unary_weight_010": abs(float(loss.get("pointer_quality_weight", -1.0)) - 0.10)
    < 1e-12,
    "standard_bce_unary": float(loss.get("set_selection_focal_beta", -1.0)) == 0.0,
    "geometry_detached": bool(selection.get("detach_geometry_features", False)),
    "compact_delta_checkpoints": bool(training.get("checkpoint_model_prefixes")),
    "optimizer_state_saved": bool(training.get("checkpoint_include_optimizer", False)),
}
print(checks)
if not all(checks.values()):
    raise SystemExit("V4.5 cluster-soft teacher contract failed")
PY

POINTER_CONFIG="${POINTER_CONFIG}" \
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
SOURCE_ITERATION="${SOURCE_ITERATION}" \
TRAIN_STEPS="${TRAIN_STEPS}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
SEEDS="${SEEDS}" \
DATA_ROOT="${DATA_ROOT}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
METRIC_WORKERS="${METRIC_WORKERS}" \
MAX_BATCHES="${MAX_BATCHES}" \
AMP_DTYPE="${AMP_DTYPE}" \
RUN_TRAIN="${RUN_TRAIN}" \
RUN_GRAD_AUDIT="${RUN_GRAD_AUDIT}" \
RUN_EVAL="${RUN_EVAL}" \
MIN_FREE_GB="${MIN_FREE_GB}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CACHE_ROOT="${CACHE_ROOT}" \
ARM_NAME="${ARM_NAME}" \
EXPERIMENT_LABEL="V4.5 GT-cluster soft randomized-teacher pointer gate" \
  bash scripts/run_culane_dla34_unified_lane_set_v4_3_pointer_stop_gate_50k.sh

if [[ "${RUN_EVAL}" != "1" || "${RUN_TRAJECTORY_EVAL}" != "1" ]]; then
  exit 0
fi

sample_count=$((EVAL_BATCH_SIZE * MAX_BATCHES))
end_iteration=$((SOURCE_ITERATION + TRAIN_STEPS))
gradient_report="${OUTPUT_ROOT}/gradient_contract.json"
for seed in ${SEEDS}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  output_dir="${seed_root}/${ARM_NAME}"
  report_dir="${seed_root}/reports"
  source_report="${report_dir}/source_v4_uniform${sample_count}.json"
  trajectory_args=()

  for ((iteration=SOURCE_ITERATION + CHECKPOINT_INTERVAL; iteration<=end_iteration; iteration+=CHECKPOINT_INTERVAL)); do
    tag="$(printf '%07d' "${iteration}")"
    checkpoint="${output_dir}/iter_${tag}.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing V4.5 trajectory checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    if (( iteration == end_iteration )); then
      report="${report_dir}/${ARM_NAME}_uniform${sample_count}.json"
    else
      report="${report_dir}/trajectory_iter_${tag}_uniform${sample_count}.json"
      "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
        --config "${POINTER_CONFIG}" \
        --checkpoint "${checkpoint}" \
        --dataset-root "${DATA_ROOT}" \
        --split val \
        --device "${DEVICE}" \
        --cache-dir "${CACHE_ROOT}/seed_${seed}/trajectory_${tag}" \
        --max-batches "${MAX_BATCHES}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --metric-workers "${METRIC_WORKERS}" \
        --sample-strategy uniform \
        --stage main \
        --top-k 4 \
        --iou-thresholds 0.50 0.75 \
        --near-min-iou 0.30 \
        --line-width 30 \
        --min-valid-rows 5 \
        --hard-diversity-distances 20 \
        --mmr-sigmas 20 \
        --mmr-penalties 0.50 \
        --output-json "${report}"
    fi
    trajectory_args+=("${iteration}=${report}")
  done

  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_5_pointer_trajectory \
    --source "${source_report}" \
    --gradient-audit "${gradient_report}" \
    --trajectory "${trajectory_args[@]}" \
    --output-json "${report_dir}/v4_5_trajectory_summary.json"
done
