#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s3_active_corridor_qualitycal_oracle_coarse_2k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/debug_s3_active_corridor_qualitycal_oracle_coarse_2k}"
CKPT="${CKPT:-outputs/debug_s3_active_corridor_qualitycal_depthwise_residual_strong_2k_init_giou/last.pt}"
DEVICE="${DEVICE:-cuda}"
CKPT_NAME="${CKPT_NAME:-}"
QUALITY_POWER="${QUALITY_POWER:-0.5}"

if [[ -n "${CKPT_NAME}" ]]; then
  CKPT="${OUT_DIR}/${CKPT_NAME}"
fi

if [[ -n "${SCORE_THRESH:-}" ]]; then
  SCORE_THRESHOLDS=("${SCORE_THRESH}")
else
  SCORE_THRESHOLDS=("0.10" "0.30" "0.50" "0.70")
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

ckpt_label="$(basename "${CKPT%.pt}")"
q_label="${QUALITY_POWER//./p}"
for thresh in "${SCORE_THRESHOLDS[@]}"; do
  thr_label="${thresh//./p}"
  pred_dir="${OUT_DIR}/oracle_${ckpt_label}_thr${thr_label}_q${q_label}"
  echo
  echo "== oracle_coarse checkpoint=${CKPT} score_thresh=${thresh} quality_score_power=${QUALITY_POWER} =="
  python -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --split val \
    --device "${DEVICE}" \
    --score-thresh "${thresh}" \
    --quality-score-power "${QUALITY_POWER}" \
    --pred-dir "${pred_dir}" \
    --sequential
done
