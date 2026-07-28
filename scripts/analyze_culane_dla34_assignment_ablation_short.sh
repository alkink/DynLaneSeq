#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_CONFIG="${BASE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:?Set CANDIDATE_CONFIG}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:?Set CANDIDATE_CHECKPOINT}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
OUTPUT_JSON="${OUTPUT_JSON:?Set OUTPUT_JSON}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-none}"

python -u -m dynlaneseq_eg.tools.analyze_cross_backbone_error_overlap \
  --r34-config "$BASE_CONFIG" \
  --r34-checkpoint "$BASE_CHECKPOINT" \
  --dla34-config "$CANDIDATE_CONFIG" \
  --dla34-checkpoint "$CANDIDATE_CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-batches "$MAX_BATCHES" \
  --sample-strategy uniform \
  --line-width 30 \
  --iou-thresholds 0.5 0.7 \
  --amp-dtype "$AMP_DTYPE" \
  --output-json "$OUTPUT_JSON"
