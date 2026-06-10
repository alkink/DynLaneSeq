#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
INIT_FROM="${INIT_FROM:-outputs/debug_s2_residual_strong_2k_init_giou/last.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing local 2k S2 checkpoint: ${INIT_FROM}" >&2
  echo "Run/debug-copy outputs/debug_s2_residual_strong_2k_init_giou/last.pt before this ablation." >&2
  exit 1
fi

python -m dynlaneseq_eg.tools.train \
  --config dynlaneseq_eg/configs/debug/culane_s3_qualitycal_noexist_lineiou_2k.yaml \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
