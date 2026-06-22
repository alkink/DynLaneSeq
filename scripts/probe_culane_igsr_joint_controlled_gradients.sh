#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_igsr_joint_controlled_25k.yaml}"
INIT_FROM="${INIT_FROM:-outputs/culane_igsr_evidence_tower_frozen_100kimg/last.pt}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing initialization checkpoint: ${INIT_FROM}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m dynlaneseq_eg.tools.probe_grad_flow \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}" \
  --loss-mode total
