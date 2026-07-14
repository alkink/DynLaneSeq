#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONHASHSEED="${PYTHONHASHSEED:-3407}"

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_no_intra_seed3407_278k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_no_intra_seed3407_278k}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

EXTRA_ARGS=()
if [[ -n "${MAX_ITERS:-}" ]]; then
  EXTRA_ARGS+=(--max-iters "${MAX_ITERS}")
fi

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --output-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  "${EXTRA_ARGS[@]}"
