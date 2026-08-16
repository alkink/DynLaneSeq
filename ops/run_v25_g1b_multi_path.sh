#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
g0=${project}/outputs/diagnostics/v25_g0_direct_four_lane_objects_1ep
g0_report=${project}/outputs/diagnostics/v25_g0_g1_official_val/g0_g1_official_val_report.json
output=${project}/outputs/diagnostics/v25_g1b_multi_path_capacity

while [[ ! -f "${g0_report}" ]]; do
  echo "Waiting for V25 G0/G1 report..."
  sleep 60
done

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -f "${output}/g1b_multi_path_capacity_report.json" ]]; then
  echo "V25 G1B report already exists"
  exit 0
fi

exec "${python}" -m dynlaneseq_eg.tools.evaluate_v25_g1b_multi_path_capacity \
  --config dynlaneseq_eg/configs/culane_v25_g0_direct_four_lane_objects_1ep.yaml \
  --checkpoint "${g0}/v25_g0_endpoint.pt" \
  --dataset-root /workspace/CULane \
  --output-dir "${output}" \
  --device cuda \
  --eval-batch-size 8 \
  --num-workers 2 \
  --metric-workers 20 \
  --metric-chunksize 32 \
  --log-interval 100

