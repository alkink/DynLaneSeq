#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s3_active_corridor_qualitycal_res34_strong_b16_from_s2_residual.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s3_active_corridor_qualitycal_res34_strong_b16_from_s2_residual}"
CKPT_NAME="${CKPT_NAME:-iter_0075000.pt}"
DEVICE="${DEVICE:-cuda}"
SCORE_THRESH="${SCORE_THRESH:-0.55}"
QUALITY_POWER="${QUALITY_POWER:-0.5}"
SPLIT="${SPLIT:-test}"
CATEGORIES="${CATEGORIES:---categories}"

ckpt="${OUT_DIR}/${CKPT_NAME}"
if [[ ! -f "${ckpt}" ]]; then
  echo "Missing checkpoint: ${ckpt}" >&2
  exit 1
fi

ckpt_label="${CKPT_NAME%.pt}"
thr_label="${SCORE_THRESH//./p}"
q_label="${QUALITY_POWER//./p}"
pred_dir="${OUT_DIR}/culane_pred_${SPLIT}_${ckpt_label}_thr${thr_label}_q${q_label}"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${ckpt}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --pred-dir "${pred_dir}" \
  ${CATEGORIES}
