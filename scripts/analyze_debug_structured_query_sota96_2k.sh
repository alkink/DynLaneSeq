#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE="${STAGE:-s3}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"
TOP_K="${TOP_K:-0}"
RANK_BY="${RANK_BY:-none}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.3 0.5 0.7}"
MAX_BATCHES="${MAX_BATCHES:-0}"
PYTHON="${PYTHON:-python}"

case "${STAGE}" in
  s0)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_structured_query_sota96_2k.yaml}"
    CKPT="${CKPT:-outputs/debug_s0_structured_query_sota96_2k/last.pt}"
    ;;
  s0_continue)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_structured_query_sota96_2k_continue_12k.yaml}"
    CKPT="${CKPT:-outputs/debug_s0_structured_query_sota96_2k_continue_12k/last.pt}"
    ;;
  s1)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s1_residual_structured_query_sota96_2k_init_structured.yaml}"
    CKPT="${CKPT:-outputs/debug_s1_residual_structured_query_sota96_2k_init_structured/last.pt}"
    ;;
  s2)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s2_residual_structured_query_sota96_2k_from_s1.yaml}"
    CKPT="${CKPT:-outputs/debug_s2_residual_structured_query_sota96_2k_from_s1/last.pt}"
    ;;
  s3)
    CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s3_active_corridor_qualitycal_structured_query_sota96_2k_from_s2.yaml}"
    CKPT="${CKPT:-outputs/debug_s3_active_corridor_qualitycal_structured_query_sota96_2k_from_s2/last.pt}"
    ;;
  *)
    echo "Unknown STAGE='${STAGE}'. Use one of: s0 s0_continue s1 s2 s3" >&2
    exit 1
    ;;
esac

"${PYTHON}" -m dynlaneseq_eg.tools.analyze_proposal_recall \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --top-k "${TOP_K}" \
  --rank-by "${RANK_BY}" \
  --line-width "${LINE_WIDTH}" \
  --min-valid-rows "${MIN_VALID_ROWS}" \
  --iou-thresholds ${IOU_THRESHOLDS} \
  --max-batches "${MAX_BATCHES}"
