#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE="${STAGE:-s3}"
DEVICE="${DEVICE:-cuda}"
CKPT_NAME="${CKPT_NAME:-last.pt}"

case "${STAGE}" in
  s0)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_structured_query_sota96_2k.yaml}"
    OUT_DIR="${OUT_DIR:-outputs/debug_s0_structured_query_sota96_2k}"
    QUALITY_POWER="${QUALITY_POWER:-0.0}"
    ;;
  s0_continue)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_structured_query_sota96_2k_continue_12k.yaml}"
    OUT_DIR="${OUT_DIR:-outputs/debug_s0_structured_query_sota96_2k_continue_12k}"
    QUALITY_POWER="${QUALITY_POWER:-0.0}"
    ;;
  s1)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s1_residual_structured_query_sota96_2k_init_structured.yaml}"
    OUT_DIR="${OUT_DIR:-outputs/debug_s1_residual_structured_query_sota96_2k_init_structured}"
    QUALITY_POWER="${QUALITY_POWER:-0.0}"
    ;;
  s2)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s2_residual_structured_query_sota96_2k_from_s1.yaml}"
    OUT_DIR="${OUT_DIR:-outputs/debug_s2_residual_structured_query_sota96_2k_from_s1}"
    QUALITY_POWER="${QUALITY_POWER:-0.0}"
    ;;
  s3)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s3_active_corridor_qualitycal_structured_query_sota96_2k_from_s2.yaml}"
    OUT_DIR="${OUT_DIR:-outputs/debug_s3_active_corridor_qualitycal_structured_query_sota96_2k_from_s2}"
    QUALITY_POWER="${QUALITY_POWER:-0.5}"
    ;;
  *)
    echo "Unknown STAGE='${STAGE}'. Use one of: s0 s0_continue s1 s2 s3" >&2
    exit 1
    ;;
esac

if [[ -n "${SCORE_THRESH:-}" ]]; then
  SCORE_THRESHOLDS=("${SCORE_THRESH}")
else
  SCORE_THRESHOLDS=("0.40" "0.45" "0.50" "0.55")
fi

ckpt="${OUT_DIR}/${CKPT_NAME}"
if [[ ! -f "${ckpt}" ]]; then
  echo "Missing checkpoint: ${ckpt}" >&2
  exit 1
fi

ckpt_label="${CKPT_NAME%.pt}"
q_label="${QUALITY_POWER//./p}"
for thresh in "${SCORE_THRESHOLDS[@]}"; do
  thr_label="${thresh//./p}"
  pred_dir="${OUT_DIR}/sweep_${ckpt_label}_thr${thr_label}_q${q_label}"
  echo
  echo "== stage=${STAGE} checkpoint=${CKPT_NAME} score_thresh=${thresh} quality_score_power=${QUALITY_POWER} =="
  python -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${CONFIG}" \
    --checkpoint "${ckpt}" \
    --split val \
    --device "${DEVICE}" \
    --score-thresh "${thresh}" \
    --quality-score-power "${QUALITY_POWER}" \
    --pred-dir "${pred_dir}" \
    --sequential
done
