#!/usr/bin/env bash
set -euo pipefail

project=${DYNLANESEQ_PROJECT:-/workspace/DynLaneSeq}
dataset=${CULANE_ROOT:-/workspace/CULane}
python=${DYNLANESEQ_PYTHON:-/venv/clrernet/bin/python}

root=${project}/outputs/diagnostics/v33_primary_aux_sufficiency_gate
output=${root}/gradient_interaction/v33_gradient_interaction_32pairs_3way.json

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$(dirname "${output}")"

if [[ -f "${output}" ]]; then
  echo "V33-GI output already exists: ${output}"
  exit 0
fi

"${python}" -m dynlaneseq_eg.tools.audit_v33_primary_aux_gradient_interaction \
  --config dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml \
  --parent-checkpoint "${root}/g0_primary_parent_1ep/v25_g0_endpoint.pt" \
  --trained-checkpoint "${root}/arm_b_primary_plus_aux_training_only/component_endpoint.pt" \
  --dataset-root "${dataset}" \
  --output-json "${output}" \
  --device cuda \
  --num-workers 4 \
  --batch-size 4 \
  --num-pairs 32 \
  --data-start-iteration 0 \
  --relative-step-scales 1e-5 3e-5 1e-4

echo "V33-GI complete: ${output}"
