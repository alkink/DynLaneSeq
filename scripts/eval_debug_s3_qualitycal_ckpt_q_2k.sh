#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s3_active_corridor_qualitycal_depthwise_residual_strong_2k_init_giou.yaml}"
OUT_DIR="${OUT_DIR:-outputs/debug_s3_active_corridor_qualitycal_depthwise_residual_strong_2k_init_giou}"
DEVICE="${DEVICE:-cuda}"
SCORE_THRESH="${SCORE_THRESH:-0.5}"

CHECKPOINTS=(
  "iter_0006000.pt"
  "iter_0009000.pt"
  "last.pt"
)

QUALITY_POWERS=(
  "0.0"
  "0.5"
  "1.0"
)

for ckpt_name in "${CHECKPOINTS[@]}"; do
  ckpt="${OUT_DIR}/${ckpt_name}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    continue
  fi
  ckpt_label="${ckpt_name%.pt}"
  for qpow in "${QUALITY_POWERS[@]}"; do
    q_label="${qpow//./p}"
    pred_dir="${OUT_DIR}/sweep_${ckpt_label}_thr${SCORE_THRESH//./p}_q${q_label}"
    echo
    echo "== checkpoint=${ckpt_name} score_thresh=${SCORE_THRESH} quality_score_power=${qpow} =="
    python -m dynlaneseq_eg.tools.evaluate_culane \
      --config "${CONFIG}" \
      --checkpoint "${ckpt}" \
      --split val \
      --device "${DEVICE}" \
      --score-thresh "${SCORE_THRESH}" \
      --quality-score-power "${qpow}" \
      --pred-dir "${pred_dir}" \
      --sequential
  done
done
