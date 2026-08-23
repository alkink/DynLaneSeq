#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq_v30_joint_field

export PYTHON="/venv/clrernet/bin/python"
export DATA_ROOT="/workspace/CULane"
export DEVICE="cuda"
export SOURCE_CHECKPOINT="/workspace/DynLaneSeq_v30_joint_field/outputs/source_checkpoints/v7_resume_safe_iter_0030000.pt"
export OUTPUT_ROOT="/workspace/DynLaneSeq_v30_joint_field/outputs/diagnostics/v30_joint_slot_field_30k_to35k"

exec bash scripts/run_culane_v30_joint_slot_field_30k_to35k.sh
