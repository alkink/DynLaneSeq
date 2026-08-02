#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k/iter_0030000.pt}"
export SOURCE_ITERATION="${SOURCE_ITERATION:-30000}"
export GATE_STEPS="${GATE_STEPS:-5000}"
export TARGET_ITERATION="${TARGET_ITERATION:-35000}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v3_rescue_30k_to35k}"

exec bash scripts/run_culane_dla34_unified_lane_set_v3_rescue_25k_to30k.sh "$@"
