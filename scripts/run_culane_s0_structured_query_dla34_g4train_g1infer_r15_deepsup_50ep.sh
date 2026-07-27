#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_g4train_g1infer_b4x4_1600x640_bins800_fpn256_l4_dfl_r15_deepsup_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_slots32_g4train_g1infer_b4x4_1600x640_bins800_fpn256_l4_dfl_r15_deepsup_50ep}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-dataset}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
MAX_ITERS="${MAX_ITERS:-0}"

EXTRA_ARGS=()
if (( MAX_ITERS > 0 )); then
  EXTRA_ARGS+=(--max-iters "${MAX_ITERS}")
fi

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}" \
  --grad-accum "${GRAD_ACCUM}" \
  "${EXTRA_ARGS[@]}"
