#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_4_set_oracle_pointer.yaml}" \
CKPT="${CKPT:-outputs/diagnostics/unified_lane_set_v4_4_set_oracle_pointer_gate_50k/seed_3407/set_oracle_pointer/iter_0065000.pt}" \
MODEL_LABEL="V4.4 set-oracle pointer" \
  bash scripts/eval_culane_dla34_unified_lane_set_v4_3_pointer_full_val.sh
