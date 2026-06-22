#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
INIT_FROM="${INIT_FROM:-outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt}"
MAX_ITERS="${MAX_ITERS:-12000}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXPERIMENTS="${EXPERIMENTS:-s0_continue geometry_frozen geometry_joint selection_joint}"

if [[ ! -f "$INIT_FROM" ]]; then
  echo "Missing initialization checkpoint: $INIT_FROM" >&2
  exit 1
fi

config_for() {
  case "$1" in
    s0_continue) echo "dynlaneseq_eg/configs/debug/culane_igsr_gate_s0_continue_2k.yaml" ;;
    geometry_frozen) echo "dynlaneseq_eg/configs/debug/culane_igsr_gate_geometry_frozen_2k.yaml" ;;
    geometry_joint) echo "dynlaneseq_eg/configs/debug/culane_igsr_gate_geometry_joint_2k.yaml" ;;
    selection_joint) echo "dynlaneseq_eg/configs/debug/culane_igsr_gate_selection_joint_2k.yaml" ;;
    *) echo "Unknown experiment: $1" >&2; exit 2 ;;
  esac
}

out_for() {
  echo "outputs/debug_igsr_gate_$1_2k"
}

for experiment in $EXPERIMENTS; do
  config="$(config_for "$experiment")"
  out_dir="$(out_for "$experiment")"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$out_dir/last.pt" ]]; then
    echo "== $experiment already complete; reusing $out_dir/last.pt =="
    continue
  fi
  echo "== Training $experiment from $INIT_FROM =="
  python -u -m dynlaneseq_eg.tools.train \
    --config "$config" \
    --device "$DEVICE" \
    --init-from "$INIT_FROM" \
    --max-iters "$MAX_ITERS"
done
