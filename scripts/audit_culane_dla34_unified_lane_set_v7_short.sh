#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_278k.yaml}"
CKPT="${CKPT:?Set CKPT to a V7 checkpoint}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_JSON="${OUTPUT_JSON:-${CKPT%.pt}_uniform256.json}"
CACHE_DIR="${CACHE_DIR:-${CKPT%.pt}_uniform256_cache}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device cuda \
  --cache-dir "${CACHE_DIR}" \
  --max-batches "${MAX_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}" \
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
  --output-json "${OUTPUT_JSON}"
