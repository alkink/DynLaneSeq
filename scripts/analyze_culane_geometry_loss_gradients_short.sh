#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:?Set CONFIG to the experiment yaml}"
CKPT="${CKPT:?Set CKPT to the checkpoint}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/geometry_loss_gradients.json}"
DEVICE="${DEVICE:-cpu}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_BATCHES="${MAX_BATCHES:-8}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"

python -m dynlaneseq_eg.tools.analyze_geometry_loss_gradients \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --device "$DEVICE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-batches "$MAX_BATCHES" \
  --sample-strategy "$SAMPLE_STRATEGY" \
  --output-json "$OUTPUT_JSON"
