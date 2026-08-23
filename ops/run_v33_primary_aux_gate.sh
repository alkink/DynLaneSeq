#!/usr/bin/env bash
set -euo pipefail

project=${DYNLANESEQ_PROJECT:-/workspace/DynLaneSeq}
dataset=${CULANE_ROOT:-/workspace/CULane}
python=${DYNLANESEQ_PYTHON:-/venv/clrernet/bin/python}

root=${project}/outputs/diagnostics/v33_primary_aux_sufficiency_gate
parent=${root}/g0_primary_parent_1ep
arm_a=${root}/arm_a_primary_only
arm_b=${root}/arm_b_primary_plus_aux_training_only
arm_c=${root}/arm_c_primary_plus_aux_memory
eval_ab=${root}/eval_primary_vs_aux
eval_bc=${root}/eval_aux_vs_memory
summary=${root}/v33_primary_aux_summary.json
wrong_dir=${root}/controls
wrong_list=${wrong_dir}/official_val_cross_clip_wrong.txt
wrong_report=${wrong_dir}/official_val_cross_clip_wrong.json

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYNLANESEQ_DLA34_WEIGHTS=${DYNLANESEQ_DLA34_WEIGHTS:-/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth}

mkdir -p "${root}" "${wrong_dir}"

if [[ ! -f "${wrong_report}" ]]; then
  "${python}" -m dynlaneseq_eg.tools.build_cross_clip_derangement \
    --input-list "${dataset}/list/val.txt" \
    --output-list "${wrong_list}" \
    --output-json "${wrong_report}" \
    --seed 3407
fi

parent_endpoint=${parent}/v25_g0_endpoint.pt
if [[ ! -f "${parent_endpoint}" ]]; then
  args=(
    -m dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects
    --config dynlaneseq_eg/configs/culane_v25_g0_direct_four_lane_objects_1ep.yaml
    --dataset-root "${dataset}"
    --output-dir "${parent}"
    --device cuda
    --num-workers 6
    --mode gate
    --log-interval 25
    --resume-interval 500
  )
  if [[ -f "${parent}/resume_latest.pt" ]]; then
    args+=(--resume "${parent}/resume_latest.pt")
  fi
  "${python}" "${args[@]}"
fi

run_arm() {
  local name=$1
  local config=$2
  local output=$3
  local advanced=$4
  if [[ -f "${output}/component_endpoint.pt" ]]; then
    echo "V33 ${name} endpoint already exists"
    return
  fi
  local args=(
    -m dynlaneseq_eg.tools.train_v25_component_gate
    --config "${config}"
    --init-checkpoint "${parent_endpoint}"
    --expected-init-iteration 11110
    --dataset-root "${dataset}"
    --output-dir "${output}"
    --component-name "${name}"
    --device cuda
    --num-workers 6
    --mode gate
    --log-interval 25
    --resume-interval 250
    --reset-runtime-rng-after-init
  )
  if [[ "${advanced}" == 1 ]]; then
    args+=(--allow-advanced-init)
  fi
  if [[ -f "${output}/resume_latest.pt" ]]; then
    args+=(--resume "${output}/resume_latest.pt")
  fi
  "${python}" "${args[@]}"
}

run_arm \
  v33_primary_only \
  dynlaneseq_eg/configs/culane_v33_primary_control_025ep.yaml \
  "${arm_a}" \
  0
run_arm \
  v33_primary_plus_aux_training_only \
  dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml \
  "${arm_b}" \
  1
run_arm \
  v33_primary_plus_aux_memory \
  dynlaneseq_eg/configs/culane_v33_primary_aux_memory_025ep.yaml \
  "${arm_c}" \
  1

evaluate_pair() {
  local control_config=$1
  local control_checkpoint=$2
  local control_report=$3
  local control_name=$4
  local treatment_config=$5
  local treatment_checkpoint=$6
  local treatment_report=$7
  local treatment_name=$8
  local output=$9
  local report_name=${10}
  local experiment_name=${11}
  if [[ -f "${output}/${report_name}" ]]; then
    echo "V33 evaluation already exists: ${output}/${report_name}"
    return
  fi
  "${python}" -m dynlaneseq_eg.tools.evaluate_v25_g2_component_pair \
    --control-config "${control_config}" \
    --control-checkpoint "${control_checkpoint}" \
    --control-report "${control_report}" \
    --control-component "${control_name}" \
    --treatment-config "${treatment_config}" \
    --treatment-checkpoint "${treatment_checkpoint}" \
    --treatment-report "${treatment_report}" \
    --treatment-component "${treatment_name}" \
    --experiment-name "${experiment_name}" \
    --report-filename "${report_name}" \
    --dataset-root "${dataset}" \
    --wrong-image-list "${wrong_list}" \
    --wrong-image-report "${wrong_report}" \
    --output-dir "${output}" \
    --device cuda \
    --eval-batch-size 8 \
    --num-workers 6 \
    --metric-workers 20 \
    --metric-chunksize 32 \
    --log-interval 100
}

evaluate_pair \
  dynlaneseq_eg/configs/culane_v33_primary_control_025ep.yaml \
  "${arm_a}/component_endpoint.pt" \
  "${arm_a}/training_report.json" \
  v33_primary_only \
  dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml \
  "${arm_b}/component_endpoint.pt" \
  "${arm_b}/training_report.json" \
  v33_primary_plus_aux_training_only \
  "${eval_ab}" \
  primary_vs_aux_official_val.json \
  "V33 primary-only versus training-only auxiliary proposals"

evaluate_pair \
  dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml \
  "${arm_b}/component_endpoint.pt" \
  "${arm_b}/training_report.json" \
  v33_primary_plus_aux_training_only \
  dynlaneseq_eg/configs/culane_v33_primary_aux_memory_025ep.yaml \
  "${arm_c}/component_endpoint.pt" \
  "${arm_c}/training_report.json" \
  v33_primary_plus_aux_memory \
  "${eval_bc}" \
  aux_vs_memory_official_val.json \
  "V33 training-only auxiliary proposals versus proposal-memory fusion"

"${python}" -m dynlaneseq_eg.tools.summarize_v33_primary_aux_gate \
  --primary-vs-aux-report "${eval_ab}/primary_vs_aux_official_val.json" \
  --aux-vs-memory-report "${eval_bc}/aux_vs_memory_official_val.json" \
  --output "${summary}"

echo "V33 complete: ${summary}"

