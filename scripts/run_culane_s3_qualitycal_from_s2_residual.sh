#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
INIT_FROM="${INIT_FROM:-outputs/culane_s2_res34_strong_b16_from_s1_residual/iter_0075000.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing S2 checkpoint: ${INIT_FROM}" >&2
  echo "Finish S2 training or copy/download the intended S2 checkpoint before running S3 QualityCal." >&2
  exit 1
fi

python -m dynlaneseq_eg.tools.train \
  --config dynlaneseq_eg/configs/culane_s3_active_corridor_qualitycal_res34_strong_b16_from_s2_residual.yaml \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
