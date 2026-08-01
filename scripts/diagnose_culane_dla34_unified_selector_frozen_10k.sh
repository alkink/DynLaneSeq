#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/diagnostics/dla34_rowref_unified_selection_gate_10k}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${CHECKPOINT_DIR}/iter_0010000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/dla34_unified_selector_frozen_10k}"
TRAIN_STEPS="${TRAIN_STEPS:-2500}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-64}"
TRAIN_CACHE_IMAGES="${TRAIN_CACHE_IMAGES:-2048}"
VAL_CACHE_IMAGES="${VAL_CACHE_IMAGES:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_BATCHES="${MAX_BATCHES:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
REUSE_CACHE="${REUSE_CACHE:-0}"

TRAJECTORY_CHECKPOINTS=(
  "${CHECKPOINT_DIR}/iter_0002500.pt"
  "${CHECKPOINT_DIR}/iter_0005000.pt"
  "${CHECKPOINT_DIR}/iter_0007500.pt"
  "${CHECKPOINT_DIR}/iter_0010000.pt"
)

for path in "${TRAJECTORY_CHECKPOINTS[@]}" "${SOURCE_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing checkpoint: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing dataset root: ${DATA_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.analyze_unified_selector_ownership_stability \
  --config "${CONFIG}" \
  --checkpoints "${TRAJECTORY_CHECKPOINTS[@]}" \
  --dataset-root "${DATA_ROOT}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --max-batches "${MAX_BATCHES}" \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}" \
  --output-json "${OUTPUT_DIR}/ownership_stability_uniform64.json"

REUSE_ARGS=()
if [[ "${REUSE_CACHE}" == "1" ]]; then
  REUSE_ARGS+=(--reuse-cache)
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_frozen_unified_selector \
  --config "${CONFIG}" \
  --checkpoint "${SOURCE_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --cache-batch-size "${CACHE_BATCH_SIZE}" \
  --train-cache-images "${TRAIN_CACHE_IMAGES}" \
  --val-cache-images "${VAL_CACHE_IMAGES}" \
  --train-cache "${OUTPUT_DIR}/train_features_${TRAIN_CACHE_IMAGES}.pt" \
  --val-cache "${OUTPUT_DIR}/val_features_${VAL_CACHE_IMAGES}.pt" \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}" \
  --train-steps "${TRAIN_STEPS}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --output-json "${OUTPUT_DIR}/frozen_selector_probe.json" \
  --save-probe "${OUTPUT_DIR}/frozen_selector_probe.pt" \
  --save-best-checkpoint "${OUTPUT_DIR}/best_frozen_selector_model.pt" \
  "${REUSE_ARGS[@]}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_frozen_selector_diagnosis \
  --ownership-json "${OUTPUT_DIR}/ownership_stability_uniform64.json" \
  --selector-json "${OUTPUT_DIR}/frozen_selector_probe.json" \
  --output-json "${OUTPUT_DIR}/summary.json"

echo "Diagnosis completed: ${OUTPUT_DIR}/summary.json"
