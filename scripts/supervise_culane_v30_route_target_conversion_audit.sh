#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq_v30_joint_field
export PYTHON="/venv/clrernet/bin/python"

ROOT="outputs/diagnostics/v30_field_only_exact_pair_30k_to35k/stage_autopsy"
exec bash scripts/run_culane_v30_route_target_conversion_audit.sh \
  >>"$ROOT/route_target_conversion_strict075.log" 2>&1
