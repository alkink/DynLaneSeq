#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v23
python=/venv/clrernet/bin/python
output=${project}/outputs/diagnostics/v23_ordered_slot_cost_volume_gate
resume=${output}/resume_latest.pt
endpoint=${output}/v23_gate_endpoint.pt

if [[ -f "${endpoint}" ]]; then
  echo "V23 endpoint already exists: ${endpoint}"
  exit 0
fi

args=(
  -m dynlaneseq_eg.tools.train_v23_ordered_slot_cost_volume
  --config dynlaneseq_eg/configs/culane_v23_ordered_slot_cost_volume_gate.yaml
  --v7-checkpoint /workspace/DynLaneSeq/outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt
  --v22-checkpoint /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/train/lane_field_endpoint.pt
  --dataset-root /workspace/CULane
  --output-dir "${output}"
  --device cuda
  # A measured 12-worker control was slower because full-resolution workers
  # increased IPC/cache pressure. Four gave the best sustained throughput.
  --num-workers 4
  --mode gate
  --log-interval 25
  --resume-interval 500
)
if [[ -f "${resume}" ]]; then
  args+=(--resume "${resume}")
fi

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec "${python}" "${args[@]}"
