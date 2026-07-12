#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_bntrain_step40_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_bntrain_step40_50ep}"
TARGET_ITERS="${TARGET_ITERS:-278000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

if [[ -z "${RESUME:-}" ]]; then
  shopt -s nullglob
  checkpoints=("${OUT_DIR}"/iter_*.pt)
  shopt -u nullglob
  if (( ${#checkpoints[@]} == 0 )); then
    echo "No iter_*.pt checkpoints found in ${OUT_DIR}" >&2
    echo "Set RESUME=/path/to/checkpoint.pt explicitly." >&2
    exit 1
  fi
  RESUME="${checkpoints[$(( ${#checkpoints[@]} - 1 ))]}"
fi

if [[ ! -f "${RESUME}" ]]; then
  echo "Missing resume checkpoint: ${RESUME}" >&2
  exit 1
fi

START_ITER="$(
  python -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu"); print(int(p.get("iteration", 0)))' "${RESUME}"
)"

if (( START_ITER <= 0 )); then
  echo "Refusing to resume: checkpoint iteration is ${START_ITER} for ${RESUME}" >&2
  exit 1
fi

REMAINING_ITERS=$(( TARGET_ITERS - START_ITER ))
if (( REMAINING_ITERS <= 0 )); then
  echo "Checkpoint iteration (${START_ITER}) is already >= TARGET_ITERS (${TARGET_ITERS}). Nothing to do."
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
