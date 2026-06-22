#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Single-checkpoint mode (default): analyzes coarse vs final within S3 model.
# Multi-checkpoint mode: set STAGE_CONFIGS and STAGE_CKPTS as space-separated lists.

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s3_unified_active_corridor_qualitycal_structured_query_2k_from_s0_12k.yaml}"
CKPT="${CKPT:-outputs/debug_s3_unified_active_corridor_qualitycal_structured_query_2k_from_s0_12k/last.pt}"
SPLIT="${SPLIT:-val}"
EVAL_LIST="${EVAL_LIST:-dataset/list/test_2k.txt}"
DEVICE="${DEVICE:-cuda}"
IOU_THRESH="${IOU_THRESH:-0.5}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.40 0.50 0.55}"
QUALITY_POWERS="${QUALITY_POWERS:-0.0 0.25 0.5}"
TOP_K_VALUES="${TOP_K_VALUES:-4}"
LINE_WIDTH="${LINE_WIDTH:-30.0}"
MIN_VALID_ROWS="${MIN_VALID_ROWS:-5}"
MAX_BATCHES="${MAX_BATCHES:-0}"
CACHE_DIR="${CACHE_DIR:-outputs/candidate_diagnostics/cache}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/candidate_diagnostics/unified_stage_transitions.json}"
REUSE_CACHE="${REUSE_CACHE:-0}"
EXACT_POSTPROCESS="${EXACT_POSTPROCESS:-1}"

# Multi-checkpoint mode (optional — leave empty for single-checkpoint mode)
# Example:
#   STAGE_CONFIGS="configs/.../s0.yaml configs/.../s1.yaml configs/.../s3.yaml"
#   STAGE_CKPTS="outputs/s0/last.pt outputs/s1/last.pt outputs/s3/last.pt"
#   STAGE_NAMES="S0 S1 S3"
STAGE_CONFIGS="${STAGE_CONFIGS:-}"
STAGE_CKPTS="${STAGE_CKPTS:-}"
STAGE_NAMES="${STAGE_NAMES:-}"

EXTRA_ARGS=()
if [[ -n "${STAGE_CONFIGS}" && -n "${STAGE_CKPTS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(--stage-configs $STAGE_CONFIGS)
  # shellcheck disable=SC2206
  EXTRA_ARGS+=(--stage-checkpoints $STAGE_CKPTS)
  if [[ -n "${STAGE_NAMES}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(--stage-names $STAGE_NAMES)
  fi
else
  EXTRA_ARGS+=(--config "${CONFIG}" --checkpoint "${CKPT}")
fi
if [[ "$REUSE_CACHE" == "1" ]]; then EXTRA_ARGS+=(--reuse-cache); fi
if [[ "$EXACT_POSTPROCESS" == "1" ]]; then EXTRA_ARGS+=(--exact-postprocess); fi

python -m dynlaneseq_eg.tools.analyze_stage_transitions \
  --split "${SPLIT}" \
  --list-path "${EVAL_LIST}" \
  --device "${DEVICE}" \
  --iou-thresh "${IOU_THRESH}" \
  --score-thresholds $SCORE_THRESHOLDS \
  --quality-powers $QUALITY_POWERS \
  --top-k-values $TOP_K_VALUES \
  --line-width "${LINE_WIDTH}" \
  --min-valid-rows "${MIN_VALID_ROWS}" \
  --max-batches "${MAX_BATCHES}" \
  --cache-dir "${CACHE_DIR}" \
  --output-json "${OUTPUT_JSON}" \
  "${EXTRA_ARGS[@]}"
