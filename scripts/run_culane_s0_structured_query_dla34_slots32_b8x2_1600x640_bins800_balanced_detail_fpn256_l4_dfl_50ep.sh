#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_balanced_detail_fpn256_l4_dfl_50ep.yaml}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-dataset}"
OUT_DIR="${OUT_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
COMPILE_MODE="${COMPILE_MODE:-default}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-default}"

TRAIN_ARGS=(
  --config "${CONFIG}"
  --device "${DEVICE}"
  --dataset-root "${DATA_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}"
  --grad-accum "${GRAD_ACCUM}"
  --compile-mode "${COMPILE_MODE}"
  --attention-backend "${ATTENTION_BACKEND}"
)

if [[ -n "${OUT_DIR}" ]]; then
  TRAIN_ARGS+=(--output-dir "${OUT_DIR}")
fi

if [[ "${COMPILE_MODEL}" == "1" || "${COMPILE_MODEL}" == "true" ]]; then
  TRAIN_ARGS+=(--compile-model)
else
  TRAIN_ARGS+=(--no-compile-model)
fi

python -u -m dynlaneseq_eg.tools.train "${TRAIN_ARGS[@]}"
