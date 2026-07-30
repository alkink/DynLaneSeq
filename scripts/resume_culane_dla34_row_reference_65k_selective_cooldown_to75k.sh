#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/dla34_rowref_from65k_selective_cooldown_10k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
TARGET_ITERATION=75000

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Refusing unmatched effective batch size: $((BATCH_SIZE * GRAD_ACCUM)); expected 16." >&2
  exit 1
fi

RESUME="${RESUME:-$(
  find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' 2>/dev/null \
    | sort -V \
    | tail -n 1
)}"
if [[ -z "${RESUME}" || ! -f "${RESUME}" ]]; then
  echo "No selective-cooldown checkpoint found in ${OUT_DIR}." >&2
  exit 1
fi

CURRENT_ITERATION="$(
  "${PYTHON}" -c '
import sys
import torch
payload = torch.load(sys.argv[1], map_location="cpu")
print(int(payload.get("iteration", -1)))
' "${RESUME}"
)"
if (( CURRENT_ITERATION < 65000 || CURRENT_ITERATION > TARGET_ITERATION )); then
  echo "Unexpected resume iteration ${CURRENT_ITERATION}; expected [65000, ${TARGET_ITERATION}]." >&2
  exit 1
fi
if (( CURRENT_ITERATION == TARGET_ITERATION )); then
  echo "Selective cooldown already complete: ${RESUME}"
  exit 0
fi
REMAINING_ITERS=$((TARGET_ITERATION - CURRENT_ITERATION))

echo "Resuming selective cooldown from ${CURRENT_ITERATION} to ${TARGET_ITERATION}"
echo "Checkpoint: ${RESUME}"
echo "Remaining optimizer steps: ${REMAINING_ITERS}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --max-iters "${REMAINING_ITERS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}" \
  --resume "${RESUME}"
