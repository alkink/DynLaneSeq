#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s3_active_corridor_qualitycal_res34_strong_b16_from_s2_residual.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s3_active_corridor_qualitycal_res34_strong_b16_from_s2_residual}"
CKPT_NAME="${CKPT_NAME:-iter_0075000.pt}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"

SCORE_THRESHOLDS=(${SCORE_THRESHOLDS:-0.45 0.50 0.55 0.60})
QUALITY_POWERS=(${QUALITY_POWERS:-0.0 0.5 1.0})

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
