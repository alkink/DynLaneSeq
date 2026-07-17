#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CONFIG:?Set CONFIG to a CurveLanes backbone config}"
: "${RESUME:?Set RESUME to a CurveLanes checkpoint}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/Datasets/CurveLanes/Curvelanes}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

START_ITER="$(python -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("iteration", 0)))' "${RESUME}")"
TARGET_ITERS="$(python -c 'import sys; from dynlaneseq_eg.config import load_config; print(int(load_config(sys.argv[1])["training"]["max_iters"]))' "${CONFIG}")"
if (( START_ITER >= TARGET_ITERS )); then
  echo "checkpoint already reached target: start_iter=${START_ITER}, target_iters=${TARGET_ITERS}"
  exit 0
fi
REMAINING_ITERS=$((TARGET_ITERS - START_ITER))

echo "config: ${CONFIG}"
echo "resume: ${RESUME}"
echo "start_iter: ${START_ITER}"
echo "target_iters_total: ${TARGET_ITERS}"
echo "remaining_iters_this_run: ${REMAINING_ITERS}"
echo "batch_size: ${BATCH_SIZE}"
echo "grad_accum: ${GRAD_ACCUM}"

ARGS=(
  --config "${CONFIG}"
  --device "${DEVICE}"
  --dataset-root "${DATA_ROOT}"
  --resume "${RESUME}"
  --max-iters "${REMAINING_ITERS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

python -u -m dynlaneseq_eg.tools.train "${ARGS[@]}"
