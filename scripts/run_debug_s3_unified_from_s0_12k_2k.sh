#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s3_unified_active_corridor_qualitycal_structured_query_2k_from_s0_12k.yaml}"
DEVICE="${DEVICE:-cuda}"
INIT_FROM="${INIT_FROM:-outputs/debug_s0_structured_query_2k/last.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing init checkpoint: ${INIT_FROM}" >&2
  exit 1
fi

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}"
