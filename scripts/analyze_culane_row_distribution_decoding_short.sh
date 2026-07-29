#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:?Set CONFIG to the experiment yaml}"
CKPT="${CKPT:?Set CKPT to the checkpoint}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/dla34_225k_row_distribution_decoding_uniform64.json}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"
AMP_DTYPE="${AMP_DTYPE:-none}"

python -u -m dynlaneseq_eg.tools.analyze_row_distribution_decoding \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --device "$DEVICE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-batches "$MAX_BATCHES" \
  --sample-strategy "$SAMPLE_STRATEGY" \
  --amp-dtype "$AMP_DTYPE" \
  --temperatures 0.25 0.5 0.75 1.0 1.5 \
  --local-mode-radii 2 4 8 16 \
  --line-width 30 \
  --iou-thresholds 0.5 0.7 \
  --output-json "$OUTPUT_JSON"
