#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s2_residual_geometry_evidence_2k_init_giou.yaml}"
INIT_FROM="${INIT_FROM:-outputs/debug_s1_residual_geometry_evidence_2k_init_giou/last.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing S1 geometry checkpoint: ${INIT_FROM}" >&2
  echo "Run scripts/run_debug_s1_residual_geometry_evidence_2k_init_giou.sh first, or set INIT_FROM." >&2
  exit 1
fi

python -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
