#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s2_residual_structured_query_res34_b16_from_s1.yaml}"
INIT_FROM="${INIT_FROM:-outputs/culane_s1_residual_structured_query_res34_b16_from_s0/last.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing S1 structured checkpoint: ${INIT_FROM}" >&2
  echo "Finish S1 structured training or set INIT_FROM=/path/to/checkpoint.pt." >&2
  exit 1
fi

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
