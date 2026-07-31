#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-4}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
METRIC_CHUNKSIZE="${METRIC_CHUNKSIZE:-64}"
AMP_DTYPE="${AMP_DTYPE:-none}"
RUN_CONTROL="${RUN_CONTROL:-1}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_selective_cooldown_10k/iter_0075000.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_object0p5_matcher_10k.yaml}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_10k/iter_0075000.pt}"

# These settings were frozen from the matched uniform-256 validation PR gate.
# The test set is evaluated once per model; this script performs no test sweep.
CONTROL_SCORE_THRESH="${CONTROL_SCORE_THRESH:-0.30}"
CONTROL_QUALITY_POWER="${CONTROL_QUALITY_POWER:-0.50}"
CANDIDATE_SCORE_THRESH="${CANDIDATE_SCORE_THRESH:-0.20}"
CANDIDATE_QUALITY_POWER="${CANDIDATE_QUALITY_POWER:-0.25}"

for path in \
  "${CONTROL_CONFIG}" \
  "${CONTROL_CHECKPOINT}" \
  "${CANDIDATE_CONFIG}" \
  "${CANDIDATE_CHECKPOINT}"
do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required paired-test input: ${path}" >&2
    exit 1
  fi
done

run_full_test() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local score_thresh="$4"
  local quality_power="$5"

  echo
  echo "===== ${label} ====="
  echo "Checkpoint: ${checkpoint}"
  echo "Frozen validation setting: score=${score_thresh}, quality=${quality_power}"

  CONFIG="${config}" \
  CKPT="${checkpoint}" \
  DATA_ROOT="${DATA_ROOT}" \
  DEVICE="${DEVICE}" \
  SCORE_THRESH="${score_thresh}" \
  QUALITY_POWER="${quality_power}" \
  TOP_K=4 \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS}" \
  EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR}" \
  METRIC_WORKERS="${METRIC_WORKERS}" \
  METRIC_CHUNKSIZE="${METRIC_CHUNKSIZE}" \
  AMP_DTYPE="${AMP_DTYPE}" \
  IOU_THRESHOLDS="0.5 0.75" \
  CATEGORIES=--categories \
    bash scripts/eval_culane_dla34_row_reference_full_test.sh
}

case "${RUN_CONTROL}" in
  1|true|TRUE|yes|YES)
    run_full_test \
      "75k control (lambda_obj=2.0)" \
      "${CONTROL_CONFIG}" \
      "${CONTROL_CHECKPOINT}" \
      "${CONTROL_SCORE_THRESH}" \
      "${CONTROL_QUALITY_POWER}"
    ;;
  0|false|FALSE|no|NO)
    echo "RUN_CONTROL=0: skipping the matched control full test."
    ;;
  *)
    echo "RUN_CONTROL must be 0/1 (or true/false), got: ${RUN_CONTROL}" >&2
    exit 1
    ;;
esac

run_full_test \
  "75k candidate (lambda_obj=0.5)" \
  "${CANDIDATE_CONFIG}" \
  "${CANDIDATE_CHECKPOINT}" \
  "${CANDIDATE_SCORE_THRESH}" \
  "${CANDIDATE_QUALITY_POWER}"

echo
echo "Paired full tests completed. No test-time parameter sweep was performed."
