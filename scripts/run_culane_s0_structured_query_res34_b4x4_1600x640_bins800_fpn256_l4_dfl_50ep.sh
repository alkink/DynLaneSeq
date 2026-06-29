#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b4x4_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}"
