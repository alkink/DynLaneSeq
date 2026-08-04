#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_3_pointer_stop.yaml}"
CKPT="${CKPT:-outputs/diagnostics/unified_lane_set_v4_3_pointer_stop_gate_50k/seed_3407/pointer_stop/iter_0065000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-none}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/unified_lane_set_v4_3_pointer_full_val_forensics.json}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing V4.3 checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "V4.3 full-validation pointer/STOP forensics"
echo "No training and no candidate cache will be written."
echo "checkpoint: ${CKPT}"
echo "output: ${OUTPUT_JSON}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_3_pointer_full_validation \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device cuda \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --max-batches 0 \
  --amp-dtype "${AMP_DTYPE}" \
  --iou-thresholds 0.50 0.75 \
  --line-width 30 \
  --min-valid-rows 5 \
  --top-k 4 \
  --output-json "${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_JSON%.json}.log"
