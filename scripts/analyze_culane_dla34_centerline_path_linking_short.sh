#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:?Set PROBE_CHECKPOINT to the frozen-P2 centerline probe}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/centerline_path_linking/dla34_225k_uniform64.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

python -u -m dynlaneseq_eg.tools.analyze_centerline_path_linking \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --probe-checkpoint "$PROBE_CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-batches "$MAX_BATCHES" \
  --num-paths 8 \
  --max-step-bins 4 8 16 \
  --transition-penalties 0.0 0.05 0.1 \
  --suppression-radius-bins 8 \
  --line-width 30 \
  --amp-dtype "$AMP_DTYPE" \
  --output-json "$OUTPUT_JSON"
