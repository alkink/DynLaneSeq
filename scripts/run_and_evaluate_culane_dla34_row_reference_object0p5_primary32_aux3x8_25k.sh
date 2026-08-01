#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"

DATA_ROOT="${DATA_ROOT}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
  bash scripts/run_culane_dla34_row_reference_object0p5_primary32_aux3x8_fromscratch_25k.sh

DATA_ROOT="${DATA_ROOT}" \
DEVICE="${DEVICE}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
MAX_BATCHES="${MAX_BATCHES}" \
  bash scripts/evaluate_culane_dla34_row_reference_object0p5_primary32_aux3x8_25k_gate.sh

