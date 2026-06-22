#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${STAGE_CONFIGS:?Set STAGE_CONFIGS to a space-separated config list}"
: "${STAGE_CKPTS:?Set STAGE_CKPTS to a space-separated checkpoint list}"
: "${STAGE_NAMES:?Set STAGE_NAMES to a space-separated stage-name list}"

EVAL_LIST="${EVAL_LIST:-dataset/list/test_2k.txt}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/candidate_diagnostics}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_ROOT}/cache}"
TAG="${TAG:-stage_chain_2k}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.40 0.50 0.55}"
QUALITY_POWERS="${QUALITY_POWERS:-0.0 0.25 0.5}"
TOP_K_VALUES="${TOP_K_VALUES:-4}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
MAX_BATCHES="${MAX_BATCHES:-0}"
REUSE_CACHE="${REUSE_CACHE:-0}"
EXACT_POSTPROCESS="${EXACT_POSTPROCESS:-1}"

# shellcheck disable=SC2206
CONFIG_ARGS=($STAGE_CONFIGS)
# shellcheck disable=SC2206
CKPT_ARGS=($STAGE_CKPTS)
# shellcheck disable=SC2206
NAME_ARGS=($STAGE_NAMES)

if [[ ${#CONFIG_ARGS[@]} -ne ${#CKPT_ARGS[@]} || ${#CONFIG_ARGS[@]} -ne ${#NAME_ARGS[@]} ]]; then
  echo "STAGE_CONFIGS, STAGE_CKPTS, and STAGE_NAMES must have equal lengths" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/$TAG" "$CACHE_DIR"
EXTRA_ARGS=()
if [[ "$REUSE_CACHE" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-cache)
fi
if [[ "$EXACT_POSTPROCESS" == "1" ]]; then
  EXTRA_ARGS+=(--exact-postprocess)
fi

# shellcheck disable=SC2086
python -m dynlaneseq_eg.tools.analyze_stage_transitions \
  --stage-configs "${CONFIG_ARGS[@]}" \
  --stage-checkpoints "${CKPT_ARGS[@]}" \
  --stage-names "${NAME_ARGS[@]}" \
  --split "$SPLIT" \
  --list-path "$EVAL_LIST" \
  --device "$DEVICE" \
  --cache-dir "$CACHE_DIR" \
  --score-thresholds $SCORE_THRESHOLDS \
  --quality-powers $QUALITY_POWERS \
  --top-k-values $TOP_K_VALUES \
  --iou-thresh 0.5 \
  --line-width "$LINE_WIDTH" \
  --min-valid-rows "$MIN_VALID_ROWS" \
  --max-batches "$MAX_BATCHES" \
  --output-json "$OUTPUT_ROOT/$TAG/stage_transitions.json" \
  "${EXTRA_ARGS[@]}"

echo "diagnostics: $OUTPUT_ROOT/$TAG"
