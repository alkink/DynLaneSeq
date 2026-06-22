#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_igsr_gate_geometry_authoritative_2k.yaml}"
CKPT="${CKPT:-outputs/debug_igsr_gate_geometry_authoritative_2k/last.pt}"
EVAL_LIST="${EVAL_LIST:-dataset/list/test_2k.txt}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/active_corridor_diagnostics/oracle_discrete_offset_2k.json}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_active_corridor_oracle_offset \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --list-path "${EVAL_LIST}" \
  --device "${DEVICE}" \
  --cache-dir outputs/candidate_diagnostics/cache \
  --reuse-cache \
  --score-thresholds 0.40 0.50 0.55 \
  --quality-power 0.0 \
  --top-k 4 \
  --iou-thresholds 0.3 0.5 0.7 \
  --output-json "${OUTPUT_JSON}"
