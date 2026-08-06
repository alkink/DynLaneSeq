#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CONFIG="${SOURCE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_5_cluster_soft_pointer.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v4_5_pointer_gate_geometry100k/seed_3407/cluster_soft_pointer/iter_0105000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-105000}"
POINTER_CONFIG="${POINTER_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_6_remaining_cluster_mixture_pointer.yaml}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2500}"
SEEDS="${SEEDS:-3407}"
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
RUN_POLICY_AUDIT="${RUN_POLICY_AUDIT:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-0.75}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_6_remaining_cluster_mixture_gate_105k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_6_remaining_cluster_mixture_gate_105k}"
ARM_NAME="${ARM_NAME:-remaining_cluster_mixture}"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import torch
try:
    payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(sys.argv[1], map_location="cpu")
print(int(payload.get("iteration", -1)))
PY
}

for required in "${SOURCE_CONFIG}" "${SOURCE_CHECKPOINT}" "${POINTER_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V4.6 artefact: ${required}" >&2
    exit 1
  fi
done
if (( SOURCE_ITERATION != 105000 )); then
  echo "V4.6 requires the selected 100k geometry + 5k pointer source." >&2
  exit 1
fi
if (( TRAIN_STEPS != 5000 || CHECKPOINT_INTERVAL != 2500 )); then
  echo "V4.6 is predeclared as 5k steps with checkpoints every 2.5k." >&2
  exit 1
fi
actual_source_iteration="$(checkpoint_iteration "${SOURCE_CHECKPOINT}")"
if (( actual_source_iteration != SOURCE_ITERATION )); then
  echo "Expected source iteration ${SOURCE_ITERATION}, found ${actual_source_iteration}" >&2
  exit 1
fi

"${PYTHON}" - "${SOURCE_CONFIG}" "${POINTER_CONFIG}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

source = load_config(sys.argv[1])
target = load_config(sys.argv[2])
source_selection = source["model"]["structured_query"]["set_selection"]
target_selection = target["model"]["structured_query"]["set_selection"]
checks = {
    "source_is_v4_5": source_selection.get("pointer_teacher_mode")
    == "cluster_soft_randomized",
    "target_is_remaining_mixture": target_selection.get("pointer_teacher_mode")
    == "cluster_soft_remaining_mixture",
    "single_model_factor": {
        key: value
        for key, value in source_selection.items()
        if key != "pointer_teacher_mode"
    }
    == {
        key: value
        for key, value in target_selection.items()
        if key != "pointer_teacher_mode"
    },
    "loss_contract_unchanged": source["loss"] == target["loss"],
    "teacher_only": float(target_selection.get("pointer_free_rollout_weight", 0.0))
    == 0.0,
    "geometry_detached": bool(target_selection.get("detach_geometry_features", False)),
    "max_quality_unary": target["loss"].get("pointer_unary_target_mode")
    == "max_quality",
    "unary_weight_010": abs(
        float(target["loss"].get("pointer_quality_weight", -1.0)) - 0.10
    )
    < 1e-12,
    "stop_weight_100": abs(
        float(target["loss"].get("pointer_stop_weight", -1.0)) - 1.0
    )
    < 1e-12,
    "checkpoint_interval_2500": int(
        target["training"].get("checkpoint_interval", -1)
    )
    == 2500,
    "optimizer_state_saved": bool(
        target["training"].get("checkpoint_include_optimizer", False)
    ),
    "compact_delta_checkpoints": bool(
        target["training"].get("checkpoint_model_prefixes")
    ),
}
print(checks)
if not all(checks.values()):
    raise SystemExit("V4.6 remaining-cluster mixture contract failed")
PY

if [[ "${RUN_TRAIN}" == "1" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}" "${MIN_FREE_GB}" <<'PY'
import shutil
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(path).free
minimum = float(sys.argv[2]) * 1024 ** 3
print({"checkpoint_filesystem_free_gib": round(free / 1024 ** 3, 2)})
if free < minimum:
    raise SystemExit(
        f"only {free / 1024 ** 3:.2f} GiB free; V4.6 requires "
        f"at least {minimum / 1024 ** 3:.2f} GiB"
    )
PY
else
  mkdir -p "${OUTPUT_ROOT}"
fi

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
echo "V4.6 teacher-only remaining-cluster mixture pointer gate"
echo "source: ${SOURCE_CHECKPOINT}"
echo "logical iterations: ${SOURCE_ITERATION} -> ${END_ITERATION}"
echo "checkpoints: 107500 and 110000"
echo "one training arm per seed: ${SEEDS}"

gradient_report="${OUTPUT_ROOT}/gradient_contract.json"
if [[ "${RUN_GRAD_AUDIT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_2_gradient_contract \
    --config "${POINTER_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --expected-source-iteration "${SOURCE_ITERATION}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 2 \
    --num-workers "${NUM_WORKERS}" \
    --amp-dtype "${AMP_DTYPE}" \
    --output-json "${gradient_report}"
elif [[ ! -f "${gradient_report}" ]]; then
  echo "Missing gradient audit while RUN_GRAD_AUDIT=0: ${gradient_report}" >&2
  exit 1
fi

for seed in ${SEEDS}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  output_dir="${seed_root}/${ARM_NAME}"
  report_dir="${seed_root}/reports"
  mkdir -p "${output_dir}" "${report_dir}"
  final_checkpoint="${output_dir}/iter_${END_TAG}.pt"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    if [[ -f "${final_checkpoint}" ]] && \
       (( $(checkpoint_iteration "${final_checkpoint}") == END_ITERATION )); then
      echo "[SKIP] existing seed=${seed} checkpoint ${final_checkpoint}"
    else
      "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
        --config "${POINTER_CONFIG}" \
        --dataset-root "${DATA_ROOT}" \
        --device "${DEVICE}" \
        --output-dir "${output_dir}" \
        --max-iters "${TRAIN_STEPS}" \
        --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
        --seed "${seed}" \
        --batch-size "${BATCH_SIZE}" \
        --grad-accum "${GRAD_ACCUM}" \
        --seg-aux-amp-dtype "${AMP_DTYPE}" \
        --init-from "${SOURCE_CHECKPOINT}" \
        --init-iteration "${SOURCE_ITERATION}" \
        --checkpoint-base "${SOURCE_CHECKPOINT}" \
        2>&1 | tee "${output_dir}/train.log"
    fi
    if [[ ! -f "${final_checkpoint}" ]] || \
       (( $(checkpoint_iteration "${final_checkpoint}") != END_ITERATION )); then
      echo "V4.6 failed to produce ${final_checkpoint}" >&2
      exit 1
    fi
  fi

  if [[ "${RUN_EVAL}" != "1" ]]; then
    continue
  fi
  source_report="${report_dir}/source_iter_0105000_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
  source_policy="${report_dir}/source_iter_0105000_policy_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${POINTER_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_ROOT}/source" \
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
    --output-json "${source_report}"

  if [[ "${RUN_POLICY_AUDIT}" == "1" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_5_pointer_stop_representative \
      --config "${POINTER_CONFIG}" \
      --checkpoint "${SOURCE_CHECKPOINT}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --device "${DEVICE}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" \
      --max-batches "${MAX_BATCHES}" \
      --sample-strategy uniform \
      --amp-dtype none \
      --iou-thresholds 0.50 0.75 \
      --line-width 30 \
      --min-valid-rows 5 \
      --top-k 4 \
      --output-json "${source_policy}"
  fi

  trajectory_args=()
  policy_args=()
  for ((iteration=SOURCE_ITERATION + CHECKPOINT_INTERVAL; iteration<=END_ITERATION; iteration+=CHECKPOINT_INTERVAL)); do
    tag="$(printf '%07d' "${iteration}")"
    checkpoint="${output_dir}/iter_${tag}.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing V4.6 trajectory checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    report="${report_dir}/${ARM_NAME}_iter_${tag}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
    policy="${report_dir}/${ARM_NAME}_iter_${tag}_policy_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${POINTER_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_ROOT}/seed_${seed}/${ARM_NAME}_${tag}" \
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
    trajectory_args+=("${iteration}=${report}")

    if [[ "${RUN_POLICY_AUDIT}" == "1" ]]; then
      "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_5_pointer_stop_representative \
        --config "${POINTER_CONFIG}" \
        --checkpoint "${checkpoint}" \
        --dataset-root "${DATA_ROOT}" \
        --split val \
        --device "${DEVICE}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --metric-workers "${METRIC_WORKERS}" \
        --max-batches "${MAX_BATCHES}" \
        --sample-strategy uniform \
        --amp-dtype none \
        --iou-thresholds 0.50 0.75 \
        --line-width 30 \
        --min-valid-rows 5 \
        --top-k 4 \
        --output-json "${policy}"
      policy_args+=("${iteration}=${policy}")
    fi
  done

  if [[ "${RUN_POLICY_AUDIT}" == "1" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_6_remaining_cluster_mixture_gate \
      --source "${source_report}" \
      --source-policy "${source_policy}" \
      --gradient-audit "${gradient_report}" \
      --trajectory "${trajectory_args[@]}" \
      --policy-trajectory "${policy_args[@]}" \
      --output-json "${report_dir}/v4_6_summary.json"
  fi
done

echo "V4.6 remaining-cluster mixture gate completed under ${OUTPUT_ROOT}"
