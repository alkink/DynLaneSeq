#!/usr/bin/env bash
set -euo pipefail

project=${DYNLANESEQ_PROJECT:-/workspace/DynLaneSeq}
dataset=${CULANE_ROOT:-/workspace/CULane}
python=${DYNLANESEQ_PYTHON:-/venv/clrernet/bin/python}

root=${project}/outputs/diagnostics/v39_pattern_query_initialization
patterns=${root}/artifacts/train_curve_patterns_4x16x160.json
source=${project}/outputs/diagnostics/v38_direct_primary_maturity/train/iter_0050000.pt
control_cfg=dynlaneseq_eg/configs/culane_v39_pqi_control_65k.yaml
treatment_cfg=dynlaneseq_eg/configs/culane_v39_pqi_treatment_65k.yaml
control_dir=${root}/control_65k
treatment_dir=${root}/treatment_65k
control_endpoint=${control_dir}/iter_0065000.pt
treatment_endpoint=${treatment_dir}/iter_0065000.pt
official=${root}/official_val_65k
v7_reference=${project}/outputs/diagnostics/v30_field_only_vs_v7_exact_35k_to50k/reports/v7_exact_full_val/metrics.json

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYNLANESEQ_DLA34_WEIGHTS=${DYNLANESEQ_DLA34_WEIGHTS:-/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth}

mkdir -p "${root}/artifacts" "${root}/smoke_control" "${root}/smoke_treatment" \
  "${control_dir}" "${treatment_dir}" "${official}"

if [[ ! -f "${source}" ]]; then
  echo "Missing fixed V38 50K source: ${source}" >&2
  exit 1
fi

if [[ ! -f "${patterns}" ]]; then
  "${python}" -m dynlaneseq_eg.tools.build_v39_train_curve_patterns \
    --config "${treatment_cfg}" \
    --dataset-root "${dataset}" \
    --output-json "${patterns}" \
    --pattern-count 16 \
    --max-images 20000 \
    --kmeans-iters 25 \
    --seed 3407
fi

if [[ ! -f "${root}/smoke_control/smoke_endpoint.pt" ]]; then
  "${python}" -m dynlaneseq_eg.tools.train_v39_pattern_query_initialization \
    --config "${control_cfg}" \
    --dataset-root "${dataset}" \
    --source-checkpoint "${source}" \
    --output-dir "${root}/smoke_control" \
    --device cuda --num-workers 6 --mode smoke --smoke-steps 2 --log-interval 1
fi

if [[ ! -f "${root}/smoke_treatment/smoke_endpoint.pt" ]]; then
  "${python}" -m dynlaneseq_eg.tools.train_v39_pattern_query_initialization \
    --config "${treatment_cfg}" \
    --dataset-root "${dataset}" \
    --source-checkpoint "${source}" \
    --pattern-bank "${patterns}" \
    --output-dir "${root}/smoke_treatment" \
    --device cuda --num-workers 6 --mode smoke --smoke-steps 2 --log-interval 1
fi

run_arm() {
  local arm=$1
  local config=$2
  local output=$3
  local endpoint=$4
  shift 4
  if [[ -f "${endpoint}" ]]; then
    return
  fi
  local args=(
    -m dynlaneseq_eg.tools.train_v39_pattern_query_initialization
    --config "${config}"
    --dataset-root "${dataset}"
    --source-checkpoint "${source}"
    --output-dir "${output}"
    --device cuda
    --num-workers 6
    --mode scientific
    --log-interval 25
    --checkpoint-interval 2500
  )
  args+=("$@")
  if [[ -f "${output}/resume_latest.pt" ]]; then
    args+=(--resume "${output}/resume_latest.pt")
  fi
  echo "Starting V39 ${arm}"
  "${python}" "${args[@]}"
}

run_arm control "${control_cfg}" "${control_dir}" "${control_endpoint}"
run_arm treatment "${treatment_cfg}" "${treatment_dir}" "${treatment_endpoint}" \
  --pattern-bank "${patterns}"

pair_report=${official}/v39_official_pair.json
if [[ ! -f "${pair_report}" ]]; then
  "${python}" -m dynlaneseq_eg.tools.evaluate_v39_pattern_query_pair \
    --control-config "${control_cfg}" \
    --control-checkpoint "${control_endpoint}" \
    --control-training-report "${control_dir}/training_report.json" \
    --treatment-config "${treatment_cfg}" \
    --treatment-checkpoint "${treatment_endpoint}" \
    --treatment-training-report "${treatment_dir}/training_report.json" \
    --v7-reference-metrics "${v7_reference}" \
    --dataset-root "${dataset}" \
    --output-dir "${official}" \
    --device cuda --eval-batch-size 16 --num-workers 6 \
    --metric-workers 20 --metric-chunksize 64 --log-interval 100
fi

for arm in control treatment; do
  config=${control_cfg}
  checkpoint=${control_endpoint}
  if [[ "${arm}" == treatment ]]; then
    config=${treatment_cfg}
    checkpoint=${treatment_endpoint}
  fi
  autopsy_dir=${root}/autopsy_${arm}_65k
  autopsy_json=${autopsy_dir}/v39_${arm}_65k_autopsy.json
  if [[ ! -f "${autopsy_json}" ]]; then
    "${python}" -m dynlaneseq_eg.tools.audit_v38_direct_primary_autopsy \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --v38-report "${official}/${arm}/official_val.json" \
      --dataset-root "${dataset}" \
      --output-dir "${autopsy_dir}" \
      --cache-dir "${root}/cache_${arm}_65k" \
      --device cuda --eval-batch-size 16 --num-workers 6 \
      --official-workers 12 --channels-last --reuse-cache \
      --expected-iteration 65000 \
      --experiment-label "V39 ${arm} 65K" \
      --output-stem "v39_${arm}_65k_autopsy"
  fi
done

summary=${root}/v39_pattern_query_gate_65k.json
if [[ ! -f "${summary}" ]]; then
  "${python}" -m dynlaneseq_eg.tools.summarize_v39_pattern_query_gate \
    --official-pair "${pair_report}" \
    --control-autopsy "${root}/autopsy_control_65k/v39_control_65k_autopsy.json" \
    --treatment-autopsy "${root}/autopsy_treatment_65k/v39_treatment_65k_autopsy.json" \
    --output-json "${summary}"
fi

echo "V39 complete: ${summary}"
