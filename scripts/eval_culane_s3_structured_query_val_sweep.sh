#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s3_active_corridor_qualitycal_structured_query_res34_b16_from_s2.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s3_active_corridor_qualitycal_structured_query_res34_b16_from_s2}"
CKPT_NAME="${CKPT_NAME:-last.pt}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"

SCORE_THRESHOLDS=(${SCORE_THRESHOLDS:-0.35 0.40 0.45 0.50})
QUALITY_POWERS=(${QUALITY_POWERS:-0.25 0.5 0.75})

ckpt="${OUT_DIR}/${CKPT_NAME}"
if [[ ! -f "${ckpt}" ]]; then
  echo "Missing checkpoint: ${ckpt}" >&2
  exit 1
fi

ckpt_label="${CKPT_NAME%.pt}"
for qpow in "${QUALITY_POWERS[@]}"; do
  q_label="${qpow//./p}"
  for thresh in "${SCORE_THRESHOLDS[@]}"; do
    thr_label="${thresh//./p}"
    pred_dir="${OUT_DIR}/sweep_${SPLIT}_${ckpt_label}_thr${thr_label}_q${q_label}"
    echo
    echo "== split=${SPLIT} checkpoint=${CKPT_NAME} score_thresh=${thresh} quality_score_power=${qpow} =="
    python -m dynlaneseq_eg.tools.evaluate_culane \
      --config "${CONFIG}" \
      --checkpoint "${ckpt}" \
      --split "${SPLIT}" \
      --device "${DEVICE}" \
      --score-thresh "${thresh}" \
      --quality-score-power "${qpow}" \
      --pred-dir "${pred_dir}"
  done
done
