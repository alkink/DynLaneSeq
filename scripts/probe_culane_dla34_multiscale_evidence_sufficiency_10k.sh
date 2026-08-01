#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/diagnostics/dla34_rowref_unified_selection_gate_10k}"
CKPT="${CKPT:-${CHECKPOINT_DIR}/iter_0010000.pt}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostics/dla34_unified_selector_frozen_10k}"
TRAIN_CACHE="${TRAIN_CACHE:-${CACHE_DIR}/train_features_2048.pt}"
VAL_CACHE="${VAL_CACHE:-${CACHE_DIR}/val_features_256.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/dla34_multiscale_evidence_sufficiency_10k}"

for path in "${CONFIG}" "${CKPT}" "${TRAIN_CACHE}" "${VAL_CACHE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required diagnostic input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_multiscale_evidence_sufficiency \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --train-cache "${TRAIN_CACHE}" \
  --val-cache "${VAL_CACHE}" \
  --train-steps "${TRAIN_STEPS:-1000}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-2}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --amp-dtype "${AMP_DTYPE:-bfloat16}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-3}" \
  --curve-samples "${CURVE_SAMPLES:-20}" \
  --train-eval-images "${TRAIN_EVAL_IMAGES:-256}" \
  --val-eval-images "${VAL_EVAL_IMAGES:-256}" \
  --output-json "${OUTPUT_DIR}/multiscale_evidence_sufficiency.json" \
  --save-probe "${OUTPUT_DIR}/multiscale_evidence_sufficiency.pt"

echo "Frozen module-compatibility audit completed:"
echo "  ${OUTPUT_DIR}/multiscale_evidence_sufficiency.json"
echo "  ${OUTPUT_DIR}/multiscale_evidence_sufficiency.pt"
