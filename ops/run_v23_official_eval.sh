#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v23
python=/venv/clrernet/bin/python
train_output=${project}/outputs/diagnostics/v23_ordered_slot_cost_volume_gate
eval_output=${project}/outputs/diagnostics/v23_ordered_slot_cost_volume_official_val
report=${eval_output}/official_val_report.json

if [[ -f "${report}" ]]; then
  echo "V23 official validation report already exists: ${report}"
  exit 0
fi

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec "${python}" -m dynlaneseq_eg.tools.evaluate_v23_official \
  --config dynlaneseq_eg/configs/culane_v23_ordered_slot_cost_volume_gate.yaml \
  --checkpoint "${train_output}/v23_gate_endpoint.pt" \
  --training-report "${train_output}/training_report.json" \
  --dataset-root /workspace/CULane \
  --wrong-image-list /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.txt \
  --wrong-image-report /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.json \
  --output-dir "${eval_output}" \
  --device cuda \
  --eval-batch-size 8 \
  --num-workers 2 \
  --metric-workers 20 \
  --metric-chunksize 32 \
  --log-interval 100
