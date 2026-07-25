#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_balanced_detail_fpn256_l4_dfl_50ep.yaml}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-dataset}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}" \
  --grad-accum "${GRAD_ACCUM}"

