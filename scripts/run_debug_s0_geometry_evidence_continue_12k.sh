#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_geometry_evidence_2k_continue_12k.yaml}"
INIT_FROM="${INIT_FROM:-outputs/debug_s0_strong_2k_giou/last.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing S0 init checkpoint: ${INIT_FROM}" >&2
  exit 1
fi

python -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
