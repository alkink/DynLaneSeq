#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/query_conditioned_dense_curve_probe.json}"
SAVE_PROBE="${SAVE_PROBE:-/tmp/query_conditioned_dense_curve_probe.pt}"

TRAIN_STEPS="${TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
STATE_SOURCE="${STATE_SOURCE:-final}"

python -m dynlaneseq_eg.tools.probe_query_conditioned_dense_curve \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --train-steps "$TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-max-batches "$EVAL_MAX_BATCHES" \
  --sample-strategy uniform \
  --num-workers "$NUM_WORKERS" \
  --hidden-dim 64 \
  --evidence-width 400 \
  --state-source "$STATE_SOURCE" \
  --learning-rate 1e-3 \
  --line-width 30.0 \
  --amp-dtype "$AMP_DTYPE" \
  --save-probe "$SAVE_PROBE" \
  --output-json "$OUTPUT_JSON"
