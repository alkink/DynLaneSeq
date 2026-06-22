#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s2_residual_structured_query_2k_from_s1.yaml}"
INIT_FROM="${INIT_FROM:-outputs/debug_s1_residual_structured_query_2k_init_structured/last.pt}"

python -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
