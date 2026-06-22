#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s2_residual_structured_query_csr_res34_b16_from_s1_frozen_50ep.yaml}"
INIT_FROM="${INIT_FROM:-outputs/culane_s1_residual_structured_query_res34_b16_from_s0_50ep_frozen/iter_0050000.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing frozen S1 checkpoint: ${INIT_FROM}" >&2
  echo "Set INIT_FROM=/path/to/frozen_s1_checkpoint.pt." >&2
  exit 1
fi

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
