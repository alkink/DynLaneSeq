#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_BASELINE_INIT="${RUN_BASELINE_INIT:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_SCRATCH_CONTINUE="${RUN_SCRATCH_CONTINUE:-1}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"

if [[ "${RUN_BASELINE_INIT}" == "1" ]]; then
  echo
  echo "== V2.1 baseline-init continue: old S0 12k -> geometry evidence 12k =="
  bash scripts/run_debug_s0_geometry_evidence_continue_12k.sh
  if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
    bash scripts/eval_debug_s0_geometry_evidence_continue_12k.sh
  fi
fi

if [[ "${RUN_SCRATCH}" == "1" ]]; then
  echo
  echo "== V2.1 scratch 12k =="
  bash scripts/run_debug_s0_geometry_evidence_2k.sh
  if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
    bash scripts/eval_debug_s0_geometry_evidence_2k.sh
  fi
fi

if [[ "${RUN_SCRATCH_CONTINUE}" == "1" ]]; then
  echo
  echo "== V2.1 scratch 12k -> V2.1 continue 12k =="
  bash scripts/run_debug_s0_geometry_evidence_fromscratch_continue_12k.sh
  if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
    bash scripts/eval_debug_s0_geometry_evidence_fromscratch_continue_12k.sh
  fi
fi
