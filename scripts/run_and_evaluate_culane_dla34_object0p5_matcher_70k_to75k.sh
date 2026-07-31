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
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache}"

CONTROL_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml"
CANDIDATE_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_object0p5_matcher_10k.yaml"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_selective_cooldown_10k/iter_0075000.pt}"
CANDIDATE_OUT_DIR="${CANDIDATE_OUT_DIR:-outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_10k}"
CANDIDATE_CHECKPOINT="${CANDIDATE_OUT_DIR}/iter_0075000.pt"
RESULT_DIR="${RESULT_DIR:-outputs/diagnostics/object0p5_matcher_75k_pr_frontier}"

if [[ ! -f "${CONTROL_CHECKPOINT}" ]]; then
  echo "Missing matched 75k control checkpoint: ${CONTROL_CHECKPOINT}" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT}" \
DEVICE="${DEVICE}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}" \
CONFIG="${CANDIDATE_CONFIG}" \
OUT_DIR="${CANDIDATE_OUT_DIR}" \
  bash scripts/run_culane_dla34_row_reference_object0p5_matcher_70k_to75k.sh

DATA_ROOT="${DATA_ROOT}" \
DEVICE="${DEVICE}" \
CACHE_DIR="${CACHE_DIR}" \
CACHE_ONLY=0 \
CONTROL_CONFIG="${CONTROL_CONFIG}" \
CANDIDATE_CONFIG="${CANDIDATE_CONFIG}" \
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT}" \
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT}" \
OUTPUT_DIR="${RESULT_DIR}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
MAX_BATCHES="${MAX_BATCHES}" \
  bash scripts/evaluate_culane_dla34_object0p5_matcher_70k_cached_pr_frontier.sh

echo "Decisive 75k comparison: ${RESULT_DIR}/summary.json"
