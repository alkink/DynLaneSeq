#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CONFIG:?Set CONFIG to the model config}"
: "${CKPT:?Set CKPT to the checkpoint}"

EVAL_LIST="${EVAL_LIST:-dataset/list/test_2k.txt}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/candidate_diagnostics}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_ROOT}/cache}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.40 0.50 0.55}"
QUALITY_POWERS="${QUALITY_POWERS:-0.0 0.25 0.5}"
TOP_K_VALUES="${TOP_K_VALUES:-4 6 8}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.3 0.5 0.7}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
MAX_BATCHES="${MAX_BATCHES:-0}"
REUSE_CACHE="${REUSE_CACHE:-0}"
EXACT_POSTPROCESS="${EXACT_POSTPROCESS:-1}"
RUN_TRANSITIONS="${RUN_TRANSITIONS:-0}"

TAG="${TAG:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)}"
OUT_DIR="${OUTPUT_ROOT}/${TAG}"
mkdir -p "$OUT_DIR" "$CACHE_DIR"

COMMON_ARGS=(
  --config "$CONFIG"
  --checkpoint "$CKPT"
  --split "$SPLIT"
  --list-path "$EVAL_LIST"
  --device "$DEVICE"
  --cache-dir "$CACHE_DIR"
  --line-width "$LINE_WIDTH"
  --min-valid-rows "$MIN_VALID_ROWS"
  --max-batches "$MAX_BATCHES"
)
if [[ "$REUSE_CACHE" == "1" ]]; then
  COMMON_ARGS+=(--reuse-cache)
fi
EXACT_ARGS=()
if [[ "$EXACT_POSTPROCESS" == "1" ]]; then
  EXACT_ARGS+=(--exact-postprocess)
fi

# shellcheck disable=SC2086
python -m dynlaneseq_eg.tools.analyze_score_distribution \
  "${COMMON_ARGS[@]}" \
  --score-thresholds $SCORE_THRESHOLDS \
  --quality-powers $QUALITY_POWERS \
  --top-k-values $TOP_K_VALUES \
  --iou-thresh 0.5 \
  --output-json "$OUT_DIR/score_distribution.json" \
  "${EXACT_ARGS[@]}"

# The first command has populated the cache, so reuse it unconditionally.
# shellcheck disable=SC2086
python -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --split "$SPLIT" \
  --list-path "$EVAL_LIST" \
  --device "$DEVICE" \
  --cache-dir "$CACHE_DIR" \
  --reuse-cache \
  --score-thresholds $SCORE_THRESHOLDS \
  --quality-powers $QUALITY_POWERS \
  --top-k-values $TOP_K_VALUES \
  --iou-thresholds $IOU_THRESHOLDS \
  --line-width "$LINE_WIDTH" \
  --min-valid-rows "$MIN_VALID_ROWS" \
  --max-batches "$MAX_BATCHES" \
  --output-json "$OUT_DIR/oracle_topk.json" \
  "${EXACT_ARGS[@]}"

if [[ "$RUN_TRANSITIONS" == "1" ]]; then
  # shellcheck disable=SC2086
  python -m dynlaneseq_eg.tools.analyze_stage_transitions \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --list-path "$EVAL_LIST" \
    --device "$DEVICE" \
    --cache-dir "$CACHE_DIR" \
    --reuse-cache \
    --score-thresholds $SCORE_THRESHOLDS \
    --quality-powers $QUALITY_POWERS \
    --top-k-values $TOP_K_VALUES \
    --iou-thresh 0.5 \
    --line-width "$LINE_WIDTH" \
    --min-valid-rows "$MIN_VALID_ROWS" \
    --max-batches "$MAX_BATCHES" \
    --output-json "$OUT_DIR/stage_transitions.json" \
    "${EXACT_ARGS[@]}"
fi

echo "diagnostics: $OUT_DIR"
