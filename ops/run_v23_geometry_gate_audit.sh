#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v23
python=/venv/clrernet/bin/python
train_output=${project}/outputs/diagnostics/v23_ordered_slot_cost_volume_gate
official_output=${project}/outputs/diagnostics/v23_ordered_slot_cost_volume_official_val
audit_output=${project}/outputs/diagnostics/v23_geometry_gate_raw_student_audit
report=${audit_output}/raw_student_gate_audit.json

if [[ -f "${report}" ]]; then
  echo "V23 raw-student gate audit already exists: ${report}"
  exit 0
fi

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec "${python}" -m dynlaneseq_eg.tools.audit_v23_geometry_gate \
  --config dynlaneseq_eg/configs/culane_v23_ordered_slot_cost_volume_gate.yaml \
  --checkpoint "${train_output}/v23_gate_endpoint.pt" \
  --training-report "${train_output}/training_report.json" \
  --official-val-report "${official_output}/official_val_report.json" \
  --official-val-prediction-root "${official_output}/predictions" \
  --dataset-root /workspace/CULane \
  --wrong-image-list /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.txt \
  --wrong-image-report /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.json \
  --output-dir "${audit_output}" \
  --device cuda \
  --eval-batch-size 8 \
  --num-workers 2 \
  --metric-workers 20 \
  --metric-chunksize 32 \
  --log-interval 100
