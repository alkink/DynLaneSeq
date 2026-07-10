#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_semantic_instance_p4p5_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_semantic_instance_p4p5_50ep}"
RESUME="${RESUME:-${OUT_DIR}/iter_0050000.pt}"
TARGET_ITERS="${TARGET_ITERS:-278000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

if [[ ! -f "${RESUME}" ]]; then
  echo "Missing resume checkpoint: ${RESUME}" >&2
  echo "Set RESUME=/path/to/iter_XXXXXXX.pt or check OUT_DIR." >&2
  exit 1
fi

START_ITER="$(
  python -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu"); print(int(p.get("iteration", 0)))' "${RESUME}"
)"
REMAINING_ITERS=$(( TARGET_ITERS - START_ITER ))

if (( REMAINING_ITERS <= 0 )); then
  echo "Checkpoint iteration (${START_ITER}) is already >= TARGET_ITERS (${TARGET_ITERS}). Nothing to do." >&2
  exit 0
fi

echo "config: ${CONFIG}"
echo "out_dir: ${OUT_DIR}"
echo "resume: ${RESUME}"
echo "start_iter: ${START_ITER}"
echo "target_iters_total: ${TARGET_ITERS}"
echo "remaining_iters_this_run: ${REMAINING_ITERS}"
echo "batch_size: ${BATCH_SIZE}"
echo "grad_accum: ${GRAD_ACCUM}"

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --resume "${RESUME}" \
  --max-iters "${REMAINING_ITERS}" \
  --output-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}"
