#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq_v30_joint_field

export PYTHON="/venv/clrernet/bin/python"
export DATA_ROOT="/workspace/CULane"
export DEVICE="cuda"
export SOURCE_CHECKPOINT="/workspace/DynLaneSeq_v30_joint_field/outputs/source_checkpoints/v7_resume_safe_iter_0030000.pt"
export CONTROL_ROOT="/workspace/DynLaneSeq_v30_joint_field/outputs/diagnostics/v30_field_only_exact_pair_30k_to35k"
export OUTPUT_ROOT="/workspace/DynLaneSeq_v30_joint_field/outputs/diagnostics/v31_selection_gradient_bridge_exact_pair_30k_to35k"

exec bash scripts/run_culane_v31_selection_gradient_bridge_30k_to35k.sh
