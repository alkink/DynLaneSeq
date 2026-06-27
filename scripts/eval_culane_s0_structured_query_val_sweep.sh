#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_res34_b16}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"

CKPT_NAMES=(${CKPT_NAMES:-iter_0025000.pt iter_0050000.pt iter_0075000.pt iter_0090000.pt last.pt})
SCORE_THRESHOLDS=(${SCORE_THRESHOLDS:-0.40 0.45 0.50 0.55})
QUALITY_POWERS=(${QUALITY_POWERS:-0.0})

for ckpt_name in "${CKPT_NAMES[@]}"; do
  ckpt="${OUT_DIR}/${ckpt_name}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping missing checkpoint: ${ckpt}" >&2
    continue
  fi

  ckpt_label="${ckpt_name%.pt}"
  for qpow in "${QUALITY_POWERS[@]}"; do
    q_label="${qpow//./p}"
    for thresh in "${SCORE_THRESHOLDS[@]}"; do
      thr_label="${thresh//./p}"
      pred_dir="${OUT_DIR}/sweep_${SPLIT}_${ckpt_label}_thr${thr_label}_q${q_label}"
      echo
      echo "== split=${SPLIT} checkpoint=${ckpt_name} score_thresh=${thresh} quality_score_power=${qpow} =="
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
done
