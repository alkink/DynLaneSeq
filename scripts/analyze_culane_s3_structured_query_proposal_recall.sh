#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s3_active_corridor_qualitycal_structured_query_res34_b16_from_s2.yaml}"
CKPT="${CKPT:-outputs/culane_s3_active_corridor_qualitycal_structured_query_res34_b16_from_s2/last.pt}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-0}"
RANK_BY="${RANK_BY:-none}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.3 0.5 0.7}"
MAX_BATCHES="${MAX_BATCHES:-0}"
PYTHON="${PYTHON:-python}"

EXTRA_ARGS=()
if [[ "${CATEGORIES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--categories)
fi
if [[ "${SKIP_OVERALL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-overall)
fi

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
  --max-batches "${MAX_BATCHES}" \
  "${EXTRA_ARGS[@]}"
