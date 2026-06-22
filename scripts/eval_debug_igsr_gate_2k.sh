#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_LIST="${EVAL_LIST:-dataset/list/test_2k.txt}"
DEVICE="${DEVICE:-cuda}"
EXPERIMENTS="${EXPERIMENTS:-s0_continue geometry_frozen geometry_joint selection_joint}"
REUSE_CACHE="${REUSE_CACHE:-1}"

config_for() {
  case "$1" in
    s0_continue) echo "dynlaneseq_eg/configs/debug/culane_igsr_gate_s0_continue_2k.yaml" ;;
    geometry_frozen) echo "dynlaneseq_eg/configs/debug/culane_igsr_gate_geometry_frozen_2k.yaml" ;;
    geometry_joint) echo "dynlaneseq_eg/configs/debug/culane_igsr_gate_geometry_joint_2k.yaml" ;;
    selection_joint) echo "dynlaneseq_eg/configs/debug/culane_igsr_gate_selection_joint_2k.yaml" ;;
    *) echo "Unknown experiment: $1" >&2; exit 2 ;;
  esac
}

for experiment in $EXPERIMENTS; do
  config="$(config_for "$experiment")"
  ckpt="outputs/debug_igsr_gate_${experiment}_2k/last.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint for $experiment: $ckpt" >&2
    exit 1
  fi
  echo "== Diagnosing $experiment =="
  run_transitions=1
  if [[ "$experiment" == "s0_continue" ]]; then
    run_transitions=0
  fi
  CONFIG="$config" \
  CKPT="$ckpt" \
  EVAL_LIST="$EVAL_LIST" \
  DEVICE="$DEVICE" \
  REUSE_CACHE="$REUSE_CACHE" \
  RUN_TRANSITIONS="$run_transitions" \
  TAG="igsr_gate_${experiment}_2k" \
  bash scripts/run_candidate_diagnostics_2k.sh
done
