#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CKPT="${CKPT:-outputs/debug_s3_active_corridor_qualitycal_curveaware_2k/last.pt}"
SCORE_THRESH="${SCORE_THRESH:-0.55}"
QUALITY_POWER="${QUALITY_POWER:-0.5}"
PRED_DIR="${PRED_DIR:-outputs/debug_s3_active_corridor_qualitycal_curveaware_2k/culane_pred_val_thr${SCORE_THRESH}_q${QUALITY_POWER}}"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config dynlaneseq_eg/configs/debug/culane_s3_active_corridor_qualitycal_curveaware_2k.yaml \
  --checkpoint "${CKPT}" \
  --split val \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --pred-dir "${PRED_DIR}" \
  --sequential
