#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="dynlaneseq_eg/configs/culane_igsr_joint_controlled_25k.yaml"
INIT_FROM="${INIT_FROM:-outputs/culane_igsr_evidence_tower_frozen_100kimg/last.pt}"
MAX_ITERS="${MAX_ITERS:-25000}"

if [[ ! -f "${INIT_FROM}" ]]; then
  echo "Missing warmed evidence-tower checkpoint: ${INIT_FROM}" >&2
  echo "Run scripts/run_culane_igsr_evidence_tower_frozen_100kimg.sh first or set INIT_FROM." >&2
  exit 1
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --init-from "${INIT_FROM}" \
  --max-iters "${MAX_ITERS}"
