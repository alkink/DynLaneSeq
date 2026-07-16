#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CONFIG:?Set CONFIG to a TuSimple backbone config}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/TuSimple}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_ITERS="${MAX_ITERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

ARGS=(
  --config "${CONFIG}"
  --device "${DEVICE}"
  --dataset-root "${DATA_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
)
if [[ "${MAX_ITERS}" -gt 0 ]]; then
  ARGS+=(--max-iters "${MAX_ITERS}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

python -u -m dynlaneseq_eg.tools.train "${ARGS[@]}"
