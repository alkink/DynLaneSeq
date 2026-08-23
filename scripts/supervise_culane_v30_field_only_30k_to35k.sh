#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq_v30_joint_field

export PYTHON="/venv/clrernet/bin/python"
export DATA_ROOT="/workspace/CULane"
export DEVICE="cuda"
export SOURCE_CHECKPOINT="/workspace/DynLaneSeq_v30_joint_field/outputs/source_checkpoints/v7_resume_safe_iter_0030000.pt"
export TREATMENT_CONFIG="dynlaneseq_eg/configs/culane_v30_joint_slot_field_35k_route_residual_off.yaml"
export OUTPUT_ROOT="/workspace/DynLaneSeq_v30_joint_field/outputs/diagnostics/v30_field_only_30k_to35k"
export EXPERIMENT_LABEL="V30 field-only causal arm, route residual disabled during training and inference"
export CONTRACT_MODE="field_only"

exec bash scripts/run_culane_v30_joint_slot_field_30k_to35k.sh
