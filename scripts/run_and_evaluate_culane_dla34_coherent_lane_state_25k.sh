#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
AUTO_RESUME="${AUTO_RESUME:-1}" \
  bash scripts/run_culane_dla34_coherent_lane_state_fromscratch_25k.sh

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
MAX_BATCHES="${MAX_BATCHES:-64}" \
AMP_DTYPE="${AMP_DTYPE:-bfloat16}" \
  bash scripts/evaluate_culane_dla34_coherent_lane_state_25k_gate.sh
