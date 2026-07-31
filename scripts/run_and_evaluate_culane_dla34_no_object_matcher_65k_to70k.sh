#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"

DATA_ROOT="${DATA_ROOT}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}" \
  bash scripts/run_culane_dla34_row_reference_65k_no_object_matcher_5k.sh

DATA_ROOT="${DATA_ROOT}" \
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
MAX_BATCHES="${MAX_BATCHES}" \
  bash scripts/evaluate_culane_dla34_no_object_matcher_70k_gate.sh
