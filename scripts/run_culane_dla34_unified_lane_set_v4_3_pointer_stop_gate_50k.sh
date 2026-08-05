#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CONFIG="${SOURCE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt}"
POINTER_CONFIG="${POINTER_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_3_pointer_stop.yaml}"
SOURCE_ITERATION="${SOURCE_ITERATION:-50000}"
TRAIN_STEPS="${TRAIN_STEPS:-15000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
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
MIN_FREE_GB="${MIN_FREE_GB:-0.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_3_pointer_stop_gate_50k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_3_pointer_stop_gate_50k}"
ARM_NAME="${ARM_NAME:-pointer_stop}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-V4.3 sequential pointer + STOP gate}"

if [[ ! -f "${SOURCE_CONFIG}" || ! -f "${SOURCE_CHECKPOINT}" || ! -f "${POINTER_CONFIG}" ]]; then
  echo "Missing V4 source or V4.3 pointer artefact" >&2
  exit 1
fi
if (( TRAIN_STEPS < 1 )); then
  echo "TRAIN_STEPS must be positive" >&2
  exit 1
fi

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

actual_source_iteration="$(checkpoint_iteration "${SOURCE_CHECKPOINT}")"
if (( actual_source_iteration != SOURCE_ITERATION )); then
  echo "Expected source iteration ${SOURCE_ITERATION}, found ${actual_source_iteration}" >&2
  exit 1
fi

"${PYTHON}" - "${POINTER_CONFIG}" "${CHECKPOINT_INTERVAL}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config
cfg = load_config(sys.argv[1])
checkpoint_interval = int(sys.argv[2])
selection = cfg["model"]["structured_query"]["set_selection"]
loss = cfg["loss"]
training = cfg["training"]
checks = {
    "sequential_pointer": selection.get("candidate_interaction") == "sequential_pointer",
    "geometry_detached": bool(selection.get("detach_geometry_features", False)),
    "four_decisions": int(selection.get("pointer_max_selections", 0)) == 4,
    "pointer_loss_only": float(loss.get("w_pointer_selection", 0.0)) > 0.0 and float(loss.get("w_set_selection", 0.0)) == 0.0,
    "pointer_deployment": cfg["postprocess"].get("score_mode") == "pointer",
    "checkpoint_interval_matches": int(training.get("checkpoint_interval", -1)) == checkpoint_interval,
    "checkpoint_state_policy": (
        bool(training.get("checkpoint_include_optimizer", True))
        if checkpoint_interval > 0
        else not bool(training.get("checkpoint_include_optimizer", True))
    ),
    "selector_trainable": "structured_query_head.set_selection_head" in training.get("trainable_parameter_prefixes", []),
}
print(checks)
if not all(checks.values()):
    raise SystemExit("V4.3 pointer config contract failed")
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
        f"only {free / 1024 ** 3:.2f} GiB free; pointer checkpoint policy requires "
        f"at least {minimum / 1024 ** 3:.2f} GiB"
    )
PY
else
  mkdir -p "${OUTPUT_ROOT}"
  echo "[SKIP] disk gate (RUN_TRAIN=0)"
fi

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
echo "${EXPERIMENT_LABEL}"
echo "source: ${SOURCE_CHECKPOINT}"
echo "logical iterations: ${SOURCE_ITERATION} -> ${END_ITERATION}"
echo "one training arm per seed; seeds: ${SEEDS}"

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
  echo "Missing gradient report while RUN_GRAD_AUDIT=0: ${gradient_report}" >&2
  exit 1
fi

for seed in ${SEEDS}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  output_dir="${seed_root}/${ARM_NAME}"
  report_dir="${seed_root}/reports"
  checkpoint="${output_dir}/iter_${END_TAG}.pt"
  mkdir -p "${output_dir}" "${report_dir}"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    if [[ -f "${checkpoint}" ]] && (( $(checkpoint_iteration "${checkpoint}") == END_ITERATION )); then
      echo "[SKIP] existing seed=${seed} checkpoint ${checkpoint}"
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
    if [[ ! -f "${checkpoint}" ]] || (( $(checkpoint_iteration "${checkpoint}") != END_ITERATION )); then
      echo "V4.3 failed to produce ${checkpoint}" >&2
      exit 1
    fi
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    source_report="${report_dir}/source_v4_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
    pointer_report="${report_dir}/${ARM_NAME}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${SOURCE_CONFIG}" \
      --checkpoint "${SOURCE_CHECKPOINT}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_ROOT}/seed_${seed}/source" \
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
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${POINTER_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_ROOT}/seed_${seed}/pointer" \
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
      --output-json "${pointer_report}"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_3_pointer_gate \
      --source "${source_report}" \
      --pointer "${pointer_report}" \
      --gradient-audit "${gradient_report}" \
      --output-json "${report_dir}/summary.json"
  fi
done

echo "${EXPERIMENT_LABEL} completed under ${OUTPUT_ROOT}"
