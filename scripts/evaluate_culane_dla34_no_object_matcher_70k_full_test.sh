#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_no_object_matcher_5k.yaml}"
CKPT="${CKPT:-outputs/diagnostics/dla34_rowref_from65k_no_object_matcher_5k/iter_0070000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
SCORE_THRESH="${SCORE_THRESH:-0.15}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
AMP_DTYPE="${AMP_DTYPE:-none}"

CONFIG="${CONFIG}" \
DATA_ROOT="${DATA_ROOT}" \
CKPT="${CKPT}" \
SCORE_THRESH="${SCORE_THRESH}" \
QUALITY_POWER="${QUALITY_POWER}" \
TOP_K="${TOP_K}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
AMP_DTYPE="${AMP_DTYPE}" \
IOU_THRESHOLDS="0.5 0.75" \
CATEGORIES=--categories \
  bash scripts/eval_culane_dla34_row_reference_full_test.sh
