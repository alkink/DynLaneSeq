#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b4x4_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/culane_s0_structured_query_res34_b${BATCH_SIZE}x${GRAD_ACCUM}_1600x640_bins800_fpn256_l4_dfl_50ep}"

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --output-dir "${OUTPUT_DIR}"
