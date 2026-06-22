#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_res34_b16}"
CKPT_NAME="${CKPT_NAME:-last.pt}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-0}"
RANK_BY="${RANK_BY:-none}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.3 0.5 0.7}"
MAX_BATCHES="${MAX_BATCHES:-0}"
PYTHON="${PYTHON:-python}"

CKPT="${CKPT:-${OUT_DIR}/${CKPT_NAME}}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
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
  --max-batches "${MAX_BATCHES}"
