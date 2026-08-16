#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
output=${project}/outputs/diagnostics/v25_g0_direct_four_lane_objects_1ep
resume=${output}/resume_latest.pt
endpoint=${output}/v25_g0_endpoint.pt

if [[ -f "${endpoint}" ]]; then
  echo "V25 G0 endpoint already exists: ${endpoint}"
  exit 0
fi

args=(
  -m dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects
  --config dynlaneseq_eg/configs/culane_v25_g0_direct_four_lane_objects_1ep.yaml
  --dataset-root /workspace/CULane
  --output-dir "${output}"
  --device cuda
  --num-workers 2
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYNLANESEQ_DLA34_WEIGHTS=/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth
exec "${python}" "${args[@]}"
