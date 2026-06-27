#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-test}"
SCORE_THRESH="${SCORE_THRESH:-0.55}"
QUALITY_POWER="${QUALITY_POWER:-0.5}"
CKPT="${CKPT:-outputs/culane_s3_qualitycal_noexist_lineiou_shadow_res34_strong_b16_from_s2_residual/iter_0075000.pt}"
PRED_DIR="${PRED_DIR:-outputs/culane_s3_qualitycal_noexist_lineiou_shadow_${SPLIT}_thr${SCORE_THRESH}_q${QUALITY_POWER}}"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config dynlaneseq_eg/configs/culane_s3_qualitycal_noexist_lineiou_shadow_res34_strong_b16_from_s2_residual.yaml \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --pred-dir "${PRED_DIR}" \
  --categories
