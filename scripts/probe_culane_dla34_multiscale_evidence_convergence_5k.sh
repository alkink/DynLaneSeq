#!/usr/bin/env bash
set -euo pipefail

# This is intentionally a clean 5k restart of the frozen-readout probe.  The
# earlier 1k probe checkpoint contains no optimizer state, so resuming its
# weights with a reset AdamW state would not be a controlled convergence test.

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/diagnostics/dla34_rowref_unified_selection_gate_10k}"
CKPT="${CKPT:-${CHECKPOINT_DIR}/iter_0010000.pt}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostics/dla34_unified_selector_frozen_10k}"
TRAIN_CACHE="${TRAIN_CACHE:-${CACHE_DIR}/train_features_2048.pt}"
VAL_CACHE="${VAL_CACHE:-${CACHE_DIR}/val_features_256.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/dla34_multiscale_evidence_convergence_5k}"
OUTPUT_JSON="${OUTPUT_DIR}/multiscale_evidence_convergence_5k.json"
OUTPUT_PROBE="${OUTPUT_DIR}/multiscale_evidence_convergence_5k.pt"

for path in "${CONFIG}" "${CKPT}" "${TRAIN_CACHE}" "${VAL_CACHE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required diagnostic input: ${path}" >&2
    exit 1
  fi
done

if [[ "${TRAIN_STEPS:-5000}" -ne 5000 ]]; then
  echo "This controlled protocol requires TRAIN_STEPS=5000." >&2
  exit 1
fi

if [[ -e "${OUTPUT_JSON}" || -e "${OUTPUT_PROBE}" ]]; then
  if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "Diagnostic output already exists under ${OUTPUT_DIR}." >&2
    echo "Set ALLOW_OVERWRITE=1 only if replacing that complete run is intentional." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"

echo "frozen multi-scale evidence convergence audit"
echo "checkpoint: ${CKPT}"
echo "train cache: ${TRAIN_CACHE}"
echo "validation cache: ${VAL_CACHE}"
echo "training: clean restart, 5000 optimizer steps"
echo "output: ${OUTPUT_DIR}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_multiscale_evidence_sufficiency \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --train-cache "${TRAIN_CACHE}" \
  --val-cache "${VAL_CACHE}" \
  --train-steps 5000 \
  --batch-size "${BATCH_SIZE:-4}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --amp-dtype "${AMP_DTYPE:-bfloat16}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-3}" \
  --curve-samples "${CURVE_SAMPLES:-20}" \
  --train-eval-images "${TRAIN_EVAL_IMAGES:-256}" \
  --val-eval-images "${VAL_EVAL_IMAGES:-256}" \
  --seed "${SEED:-3407}" \
  --output-json "${OUTPUT_JSON}" \
  --save-probe "${OUTPUT_PROBE}"

echo "Frozen 5k convergence audit completed:"
echo "  ${OUTPUT_JSON}"
echo "  ${OUTPUT_PROBE}"
