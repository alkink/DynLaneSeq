#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq_v30_joint_field

export PYTHON="/venv/clrernet/bin/python"
export DATA_ROOT="/workspace/CULane"
export DEVICE="cuda"

exec bash scripts/run_culane_v31_selection_bridge_stage_autopsy.sh
