#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
train_output=${project}/outputs/diagnostics/v25_g0_direct_four_lane_objects_1ep
eval_output=${project}/outputs/diagnostics/v25_g0_g1_official_val
report=${eval_output}/g0_g1_official_val_report.json

if [[ -f "${report}" ]]; then
  echo "V25 G0/G1 report already exists: ${report}"
  exit 0
fi
if [[ ! -f "${train_output}/v25_g0_endpoint.pt" ]]; then
  echo "V25 G0 endpoint does not exist yet" >&2
  exit 2
fi

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${python}" -m dynlaneseq_eg.tools.evaluate_v25_g0_and_path_gate \
  --config dynlaneseq_eg/configs/culane_v25_g0_direct_four_lane_objects_1ep.yaml \
  --checkpoint "${train_output}/v25_g0_endpoint.pt" \
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
