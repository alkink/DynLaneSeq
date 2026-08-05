#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_5_cluster_soft_pointer.yaml}"
CKPT="${CKPT:-outputs/diagnostics/unified_lane_set_v4_5_pointer_gate_geometry100k/seed_3407/cluster_soft_pointer/iter_0105000.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/unified_lane_set_v4_5_pointer_geometry100k_iter105_stop_representative_forensics.json}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
MAX_BATCHES="${MAX_BATCHES:-0}"
AMP_DTYPE="${AMP_DTYPE:-none}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Missing compact pointer checkpoint: ${CKPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSON}")"

echo "V4.5 zero-training STOP + representative audit"
echo "checkpoint: ${CKPT}"
echo "split: full validation"
echo "counterfactuals: forced continuation, prefix-extension oracle, local replacement, first-divergence reroll"
echo "output: ${OUTPUT_JSON}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_5_pointer_stop_representative \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --amp-dtype "${AMP_DTYPE}" \
  --iou-thresholds 0.50 0.75 \
  --line-width 30 \
  --min-valid-rows 5 \
  --top-k 4 \
  --output-json "${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_JSON%.json}.log"
