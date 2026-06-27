#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s3_unified_active_corridor_qualitycal_structured_query_2k_from_s0_12k.yaml}"
CKPT="${CKPT:-outputs/debug_s3_unified_active_corridor_qualitycal_structured_query_2k_from_s0_12k/last.pt}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.3 0.5 0.7}"
TOP_K="${TOP_K:-4}"
QUALITY_POWERS="${QUALITY_POWERS:-0.0 0.25 0.5}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
MAX_BATCHES="${MAX_BATCHES:-0}"

# shellcheck disable=SC2086
python -m dynlaneseq_eg.tools.analyze_frozen_vs_joint \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --iou-thresholds $IOU_THRESHOLDS \
  --top-k "${TOP_K}" \
  --quality-powers $QUALITY_POWERS \
  --line-width "${LINE_WIDTH}" \
  --min-valid-rows "${MIN_VALID_ROWS}" \
  --max-batches "${MAX_BATCHES}"
