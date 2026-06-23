#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_50ep.yaml"
CKPT="outputs/culane_s0_structured_query_res34_b16_50ep/iter_0100000.pt"
OUT_DIR="outputs/culane_s0_structured_query_res34_b16_50ep/test_eval_100k"
DEVICE="${DEVICE:-cuda}"
SCORE_THRESH="${SCORE_THRESH:-0.50}"
QUALITY_POWER="${QUALITY_POWER:-0.0}"
SPLIT="${SPLIT:-test}"
CATEGORIES="${CATEGORIES:---categories}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

thr_label="${SCORE_THRESH//./p}"
pred_dir="${OUT_DIR}/main_thr${thr_label}_q0p0"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --pred-dir "${pred_dir}" \
  ${CATEGORIES}
