#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_igsr_joint_controlled_25k.yaml}"
CKPT="${CKPT:-outputs/culane_igsr_joint_controlled_25k/iter_0100000.pt}"
OUT_DIR="${OUT_DIR:-outputs/culane_igsr_joint_controlled_25k/test_eval_100k}"
DEVICE="${DEVICE:-cuda}"
SCORE_THRESH="${SCORE_THRESH:-0.55}"
QUALITY_POWER="${QUALITY_POWER:-0.0}"
SPLIT="${SPLIT:-test}"
CATEGORIES="${CATEGORIES:---categories}"
OUTPUT_STAGE="${OUTPUT_STAGE:-final}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

thr_label="${SCORE_THRESH//./p}"
pred_dir="${OUT_DIR}/${OUTPUT_STAGE}_thr${thr_label}_q0p0"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --pred-dir "${pred_dir}" \
  --output-stage "${OUTPUT_STAGE}" \
  ${CATEGORIES}
