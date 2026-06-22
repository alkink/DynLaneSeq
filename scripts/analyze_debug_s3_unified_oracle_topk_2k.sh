#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s3_unified_active_corridor_qualitycal_structured_query_2k_from_s0_12k.yaml}"
CKPT="${CKPT:-outputs/debug_s3_unified_active_corridor_qualitycal_structured_query_2k_from_s0_12k/last.pt}"
SPLIT="${SPLIT:-val}"
EVAL_LIST="${EVAL_LIST:-dataset/list/test_2k.txt}"
DEVICE="${DEVICE:-cuda}"
TOP_K_VALUES="${TOP_K_VALUES:-4 6 8}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.3 0.5 0.7}"
QUALITY_POWERS="${QUALITY_POWERS:-0.0 0.25 0.5}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.40 0.50 0.55}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
MAX_BATCHES="${MAX_BATCHES:-0}"
CACHE_DIR="${CACHE_DIR:-outputs/candidate_diagnostics/cache}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/candidate_diagnostics/unified_oracle_topk.json}"
REUSE_CACHE="${REUSE_CACHE:-0}"
EXACT_POSTPROCESS="${EXACT_POSTPROCESS:-1}"

EXTRA_ARGS=()
if [[ "$REUSE_CACHE" == "1" ]]; then EXTRA_ARGS+=(--reuse-cache); fi
if [[ "$EXACT_POSTPROCESS" == "1" ]]; then EXTRA_ARGS+=(--exact-postprocess); fi

# shellcheck disable=SC2086
python -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --list-path "${EVAL_LIST}" \
  --device "${DEVICE}" \
  --top-k-values $TOP_K_VALUES \
  --iou-thresholds $IOU_THRESHOLDS \
  --quality-powers $QUALITY_POWERS \
  --score-thresholds $SCORE_THRESHOLDS \
  --line-width "${LINE_WIDTH}" \
  --min-valid-rows "${MIN_VALID_ROWS}" \
  --max-batches "${MAX_BATCHES}" \
  --cache-dir "${CACHE_DIR}" \
  --output-json "${OUTPUT_JSON}" \
  "${EXTRA_ARGS[@]}"
