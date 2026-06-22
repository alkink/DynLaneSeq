#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT_FROM="${INIT_FROM:-outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt}"
CONFIG="dynlaneseq_eg/configs/culane_igsr_coarse_to_fine_frozen_25k.yaml"
MAX_ITERS="${MAX_ITERS:-25000}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing initialization checkpoint: ${INIT_FROM}" >&2
  exit 1
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}" \
  --max-iters "${MAX_ITERS}"

