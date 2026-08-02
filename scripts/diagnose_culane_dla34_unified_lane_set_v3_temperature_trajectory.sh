#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
ITERATIONS="${ITERATIONS:-25000 50000 75000}"
TEMPERATURES="${TEMPERATURES:-0.5 1.0 2.0 4.0 8.0 16.0}"
RESULT_DIR="${RESULT_DIR:-outputs/diagnostics/unified_lane_set_v3_temperature_trajectory}"

mkdir -p "${RESULT_DIR}"
read -r -a temperature_values <<< "${TEMPERATURES}"
reports=()
for iteration in ${ITERATIONS}; do
  checkpoint="${CHECKPOINT_DIR}/iter_$(printf '%07d' "${iteration}").pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  report="${RESULT_DIR}/temperature_iter_$(printf '%07d' "${iteration}")_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
  echo "===== row-logit temperature audit: iteration ${iteration} ====="
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_row_distribution_decoding \
    --config "${CONFIG}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --device "${DEVICE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-batches "${MAX_BATCHES}" \
    --sample-strategy uniform \
    --temperatures "${temperature_values[@]}" \
    --local-mode-radii 4 \
    --line-width 30 \
    --iou-thresholds 0.50 0.70 \
    --amp-dtype "${AMP_DTYPE}" \
    --output-json "${report}"
  reports+=("${report}")
done

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.summarize_row_distribution_temperature_trajectory \
  --inputs "${reports[@]}" \
  --output-json "${RESULT_DIR}/summary.json"

echo "Temperature trajectory completed."
