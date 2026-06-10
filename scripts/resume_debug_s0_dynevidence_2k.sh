#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_dynevidence_2k.yaml}"
RESUME="${RESUME:-outputs/debug_s0_dynevidence_2k/iter_0009000.pt}"
MAX_ITERS="${MAX_ITERS:-24000}"

if [[ ! -f "${RESUME}" ]]; then
  echo "Missing resume checkpoint: ${RESUME}" >&2
  exit 1
fi

python -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --resume "${RESUME}" \
  --max-iters "${MAX_ITERS}"
