#!/usr/bin/env bash
set -euo pipefail

project=${DYNLANESEQ_PROJECT:-/workspace/DynLaneSeq}
dataset=${CULANE_ROOT:-/workspace/CULane}
python=${DYNLANESEQ_PYTHON:-/venv/clrernet/bin/python}

root=${project}/outputs/diagnostics/v38_direct_primary_maturity
train=${root}/train
smoke=${root}/smoke_batch16
eval_dir=${root}/official_val_50k
config=dynlaneseq_eg/configs/culane_v38_direct_primary_maturity_50k.yaml
endpoint=${train}/iter_0050000.pt
report=${eval_dir}/v38_direct_primary_50k_official_val.json
v7_reference=${project}/outputs/diagnostics/v30_field_only_vs_v7_exact_35k_to50k/reports/v7_exact_full_val/metrics.json

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYNLANESEQ_DLA34_WEIGHTS=${DYNLANESEQ_DLA34_WEIGHTS:-/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth}

mkdir -p "${root}" "${train}" "${smoke}" "${eval_dir}"

if [[ ! -f "${smoke}/smoke_endpoint.pt" ]]; then
  "${python}" -m dynlaneseq_eg.tools.train_v38_direct_primary_maturity \
    --config "${config}" \
    --dataset-root "${dataset}" \
    --output-dir "${smoke}" \
    --device cuda \
    --num-workers 6 \
    --mode smoke \
    --smoke-steps 2 \
    --log-interval 1
fi

if [[ ! -f "${endpoint}" ]]; then
  args=(
    -m dynlaneseq_eg.tools.train_v38_direct_primary_maturity
    --config "${config}"
    --dataset-root "${dataset}"
    --output-dir "${train}"
    --device cuda
    --num-workers 6
    --mode scientific
    --log-interval 25
    --checkpoint-interval 2500
  )
  if [[ -f "${train}/resume_latest.pt" ]]; then
    args+=(--resume "${train}/resume_latest.pt")
  fi
  "${python}" "${args[@]}"
fi

if [[ ! -f "${report}" ]]; then
  "${python}" -m dynlaneseq_eg.tools.evaluate_v38_direct_primary_maturity \
    --config "${config}" \
    --checkpoint "${endpoint}" \
    --training-report "${train}/training_report.json" \
    --v7-reference-metrics "${v7_reference}" \
    --dataset-root "${dataset}" \
    --output-dir "${eval_dir}" \
    --device cuda \
    --eval-batch-size 16 \
    --num-workers 6 \
    --metric-workers 20 \
    --metric-chunksize 64 \
    --log-interval 100
fi

echo "V38 complete: ${report}"
