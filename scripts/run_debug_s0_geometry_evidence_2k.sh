#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_geometry_evidence_2k.yaml}"

python -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}"
