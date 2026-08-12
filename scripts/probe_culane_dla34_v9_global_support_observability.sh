#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v9_slot_owned_treatment_225k_to227k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/diagnostics/v9_slot_owned_geometry_gate_225k/generalization/treatment/iter_0227000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v9_global_support_observability_227k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/v9_global_support_observability_227k}"
TRAIN_IMAGES="${TRAIN_IMAGES:-4096}"
EARLY_STOP_IMAGES="${EARLY_STOP_IMAGES:-512}"
VAL_IMAGES="${VAL_IMAGES:-256}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
PATIENCE="${PATIENCE:-4}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SEEDS="${SEEDS:-3407 5419}"

for required in "${CONFIG}" "${CHECKPOINT}" "${DATA_ROOT}/list/train_gt.txt" "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing global-support observability input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}"

read -r -a seed_args <<< "${SEEDS}"
"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_v9_global_support_observability \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  --cache-dir "${CACHE_ROOT}" \
  --reuse-cache \
  --train-images "${TRAIN_IMAGES}" \
  --early-stop-images "${EARLY_STOP_IMAGES}" \
  --val-images "${VAL_IMAGES}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --seeds "${seed_args[@]}" \
  --output-json "${OUTPUT_ROOT}/v9_global_support_observability_summary.json" \
  2>&1 | tee "${OUTPUT_ROOT}/probe.log"
