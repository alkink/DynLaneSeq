#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
INIT_FROM="${INIT_FROM:-outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing S0 checkpoint: ${INIT_FROM}" >&2
  echo "Download or copy the strong S0 75k checkpoint before running S1." >&2
  exit 1
fi

python -m dynlaneseq_eg.tools.train \
  --config dynlaneseq_eg/configs/culane_s1_res34_strong_b16_from_s0_75k.yaml \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
