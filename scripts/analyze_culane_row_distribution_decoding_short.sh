#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:?Set CONFIG to the experiment yaml}"
CKPT="${CKPT:?Set CKPT to the checkpoint}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/row_distribution_decoding.json}"
DEVICE="${DEVICE:-cpu}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_BATCHES="${MAX_BATCHES:-32}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"

python -m dynlaneseq_eg.tools.analyze_row_distribution_decoding \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --device "$DEVICE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-batches "$MAX_BATCHES" \
  --sample-strategy "$SAMPLE_STRATEGY" \
  --temperatures 0.25 0.5 0.75 1.0 1.5 \
  --local-mode-radii 2 4 8 16 \
  --line-width 30 \
  --iou-thresholds 0.5 0.7 \
  --output-json "$OUTPUT_JSON"
