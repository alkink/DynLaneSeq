#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_geometry_evidence_2k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/debug_s0_geometry_evidence_2k}"
DEVICE="${DEVICE:-cuda}"
QUALITY_POWER="${QUALITY_POWER:-0.0}"
CKPT_NAME="${CKPT_NAME:-last.pt}"

if [[ -n "${SCORE_THRESH:-}" ]]; then
  SCORE_THRESHOLDS=("${SCORE_THRESH}")
else
  SCORE_THRESHOLDS=("0.40" "0.45" "0.50" "0.55")
fi

ckpt="${OUT_DIR}/${CKPT_NAME}"
if [[ ! -f "${ckpt}" ]]; then
  echo "Missing checkpoint: ${ckpt}" >&2
  exit 1
fi

ckpt_label="${CKPT_NAME%.pt}"
q_label="${QUALITY_POWER//./p}"
for thresh in "${SCORE_THRESHOLDS[@]}"; do
  thr_label="${thresh//./p}"
  pred_dir="${OUT_DIR}/sweep_${ckpt_label}_thr${thr_label}_q${q_label}"
  echo
  echo "== checkpoint=${CKPT_NAME} score_thresh=${thresh} quality_score_power=${QUALITY_POWER} =="
  python -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${CONFIG}" \
    --checkpoint "${ckpt}" \
    --split val \
    --device "${DEVICE}" \
    --score-thresh "${thresh}" \
    --quality-score-power "${QUALITY_POWER}" \
    --pred-dir "${pred_dir}" \
    --sequential
done
