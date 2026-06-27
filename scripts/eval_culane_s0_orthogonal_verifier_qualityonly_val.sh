#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_orthogonal_verifier_qualityonly_25k.yaml}"
CKPT="${CKPT:-outputs/culane_s0_orthogonal_verifier_qualityonly_25k/iter_0002500.pt}"
DEVICE="${DEVICE:-cuda}"
SCORE_THRESH="${SCORE_THRESH:-0.40}"
QUALITY_POWER="${QUALITY_POWER:-0.25}"
PRED_DIR="${PRED_DIR:-outputs/culane_s0_orthogonal_verifier_qualityonly_25k/val_eval}"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --list-path dataset/list/val.txt \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --pred-dir "${PRED_DIR}"

