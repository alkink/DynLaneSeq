#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_igar_geometry_only_25k.yaml}"
INIT_FROM="${INIT_FROM:-outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt}"
MAX_ITERS="${MAX_ITERS:-0}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing S0 structured checkpoint: ${INIT_FROM}" >&2
  echo "Set INIT_FROM=/path/to/checkpoint.pt." >&2
  exit 1
fi

ARGS=()
if [[ "${MAX_ITERS}" != "0" ]]; then
  ARGS+=(--max-iters "${MAX_ITERS}")
fi

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}" \
  "${ARGS[@]}"
