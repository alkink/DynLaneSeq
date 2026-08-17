#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
g0=${project}/outputs/diagnostics/v25_g0_direct_four_lane_objects_1ep
v7_predictions=/workspace/DynLaneSeq_v23/outputs/diagnostics/v23_ordered_slot_cost_volume_official_val/predictions/source_v7
output=${project}/outputs/diagnostics/v25_v7_top3_union_oracle

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -f "${output}/v7_top3_union_oracle_report.json" ]]; then
  echo "V25 exact-V7/top3 union-oracle report already exists"
  exit 0
fi

exec "${python}" -m dynlaneseq_eg.tools.evaluate_v25_v7_top3_union_oracle \
  --config dynlaneseq_eg/configs/culane_v25_g0_direct_four_lane_objects_1ep.yaml \
  --checkpoint "${g0}/v25_g0_endpoint.pt" \
  --dataset-root /workspace/CULane \
  --v7-prediction-dir "${v7_predictions}" \
  --output-dir "${output}" \
  --device cuda \
  --eval-batch-size 8 \
  --num-workers 2 \
  --metric-workers 20 \
  --metric-chunksize 32 \
  --oracle-workers 8 \
  --log-interval 100
