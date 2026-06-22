#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_LIST="${EVAL_LIST:-dataset/list/test_2k.txt}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_BATCHES="${MAX_BATCHES:-0}"
OUT_ROOT="${OUT_ROOT:-outputs/active_corridor_diagnostics}"

run_one() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  echo
  echo "== Active Corridor evidence diagnosis: ${name} =="
  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_active_corridor_evidence \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --split val \
    --list-path "${EVAL_LIST}" \
    --device "${DEVICE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-batches "${MAX_BATCHES}" \
    --shuffle-modes offset_reverse lane_roll \
    --output-json "${OUT_ROOT}/${name}.json"
}

run_one \
  geometry_frozen \
  dynlaneseq_eg/configs/debug/culane_igsr_gate_geometry_frozen_2k.yaml \
  outputs/debug_igsr_gate_geometry_frozen_2k/last.pt

run_one \
  geometry_joint \
  dynlaneseq_eg/configs/debug/culane_igsr_gate_geometry_joint_2k.yaml \
  outputs/debug_igsr_gate_geometry_joint_2k/last.pt

run_one \
  selection_joint \
  dynlaneseq_eg/configs/debug/culane_igsr_gate_selection_joint_2k.yaml \
  outputs/debug_igsr_gate_selection_joint_2k/last.pt
