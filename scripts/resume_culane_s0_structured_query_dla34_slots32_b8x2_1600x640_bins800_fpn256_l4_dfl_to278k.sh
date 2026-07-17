#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep}"
RESUME="${RESUME:-${OUT_DIR}/last.pt}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-dataset}"
TARGET_ITERS="${TARGET_ITERS:-278000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

if [[ ! -f "${RESUME}" ]]; then
  echo "Missing resume checkpoint: ${RESUME}" >&2
  exit 1
fi

START_ITER="$(python -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("iteration", 0)))' "${RESUME}")"
REMAINING_ITERS=$((TARGET_ITERS - START_ITER))
if (( REMAINING_ITERS <= 0 )); then
  echo "checkpoint already reached target: start_iter=${START_ITER}, target_iters=${TARGET_ITERS}"
  exit 0
fi

echo "config: ${CONFIG}"
echo "resume: ${RESUME}"
echo "start_iter: ${START_ITER}"
echo "target_iters_total: ${TARGET_ITERS}"
echo "remaining_iters_this_run: ${REMAINING_ITERS}"

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --resume "${RESUME}" \
  --max-iters "${REMAINING_ITERS}" \
  --output-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}"
