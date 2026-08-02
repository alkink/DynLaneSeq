#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k/iter_0025000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
GRADIENT_IMAGES="${GRADIENT_IMAGES:-2}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.20 0.30}"
TOP_K="${TOP_K:-4}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/unified_lane_set_v3_training_contract.json}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "Unified lane-set frozen training-contract audit"
echo "checkpoint: ${CKPT}"
echo "sample: $((EVAL_BATCH_SIZE * MAX_BATCHES)) images (before final partial-batch clipping)"
echo "gradient images: ${GRADIENT_IMAGES}"
read -r -a score_threshold_values <<< "${SCORE_THRESHOLDS}"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.analyze_unified_lane_set_training_contract \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --gradient-images "${GRADIENT_IMAGES}" \
  --sample-strategy uniform \
  --amp-dtype "${AMP_DTYPE}" \
  --top-k "${TOP_K}" \
  --score-thresholds "${score_threshold_values[@]}" \
  --iou-thresholds 0.50 0.75 \
  --line-width 30 \
  --min-valid-rows 5 \
  --row-visibility-thresh 0 \
  --output-json "${OUTPUT_JSON}"
