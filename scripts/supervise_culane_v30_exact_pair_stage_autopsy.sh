#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq_v30_joint_field

export PYTHON="/venv/clrernet/bin/python"
export DATA_ROOT="/workspace/CULane"
export DEVICE="cuda"
export EXACT_ROOT="/workspace/DynLaneSeq_v30_joint_field/outputs/diagnostics/v30_field_only_exact_pair_30k_to35k"

exec bash scripts/run_culane_v30_exact_pair_stage_autopsy.sh
