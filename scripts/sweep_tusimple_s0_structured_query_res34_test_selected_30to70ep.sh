#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/TuSimple}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep}"
RESULT_DIR="${RESULT_DIR:-outputs/tusimple_test_selected_sweeps/res34_30to70ep}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60}"
QUALITY_POWERS="${QUALITY_POWERS:-0.25 0.50 0.75 1.00}"

CHECKPOINTS=(
  "${OUTPUT_DIR}/iter_0006810.pt"
  "${OUTPUT_DIR}/iter_0009080.pt"
  "${OUTPUT_DIR}/iter_0011350.pt"
  "${OUTPUT_DIR}/iter_0013620.pt"
  "${OUTPUT_DIR}/iter_0015890.pt"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

read -r -a SCORE_ARGS <<< "${SCORE_THRESHOLDS}"
read -r -a QUALITY_ARGS <<< "${QUALITY_POWERS}"

python -u -m dynlaneseq_eg.tools.sweep_tusimple_thresholds \
  --config "${CONFIG}" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --dataset-root "${DATA_ROOT}" \
  --split test \
  --device cuda \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --score-thresholds "${SCORE_ARGS[@]}" \
  --quality-powers "${QUALITY_ARGS[@]}" \
  --cache-dir "${RESULT_DIR}/cache" \
  --reuse-cache \
  --output-json "${RESULT_DIR}/sweep.json" \
  --output-csv "${RESULT_DIR}/sweep.csv"
