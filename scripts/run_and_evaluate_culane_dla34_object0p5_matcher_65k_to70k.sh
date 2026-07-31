#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"

CANDIDATE_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_object0p5_matcher_5k.yaml"
CANDIDATE_OUT_DIR="outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_5k"
CANDIDATE_CHECKPOINT="${CANDIDATE_OUT_DIR}/iter_0070000.pt"
GATE_OUTPUT_DIR="outputs/diagnostics/object0p5_matcher_70k_gate"

DATA_ROOT="${DATA_ROOT}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}" \
CONFIG="${CANDIDATE_CONFIG}" \
OUT_DIR="${CANDIDATE_OUT_DIR}" \
EXPECTED_LAMBDA_OBJ=0.5 \
  bash scripts/run_culane_dla34_row_reference_65k_no_object_matcher_5k.sh

DATA_ROOT="${DATA_ROOT}" \
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-}" \
CANDIDATE_CONFIG="${CANDIDATE_CONFIG}" \
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT}" \
OUTPUT_DIR="${GATE_OUTPUT_DIR}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
MAX_BATCHES="${MAX_BATCHES}" \
REUSE_CACHE="${REUSE_CACHE:-1}" \
  bash scripts/evaluate_culane_dla34_no_object_matcher_70k_gate.sh
