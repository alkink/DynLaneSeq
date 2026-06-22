#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="dynlaneseq_eg/configs/culane_igsr_joint_controlled_25k.yaml"
RESUME="${RESUME:-outputs/culane_igsr_joint_controlled_25k/iter_0025000.pt}"
ADDITIONAL_ITERS="${ADDITIONAL_ITERS:-253000}"

if [[ ! -f "${RESUME}" ]]; then
  echo "Missing 25k gate checkpoint: ${RESUME}" >&2
  exit 1
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --resume "${RESUME}" \
  --max-iters "${ADDITIONAL_ITERS}"
