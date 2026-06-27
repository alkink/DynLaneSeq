#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_dynamic_proposal_2k_continue_12k.yaml}"
CKPT="${CKPT:-outputs/debug_s0_dynamic_proposal_2k_continue_12k/last.pt}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-0}"
RANK_BY="${RANK_BY:-none}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.3 0.5 0.7}"
MAX_BATCHES="${MAX_BATCHES:-0}"
PYTHON="${PYTHON:-python}"

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
