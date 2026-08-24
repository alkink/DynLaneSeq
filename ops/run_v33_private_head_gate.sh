#!/usr/bin/env bash
set -euo pipefail

project=${DYNLANESEQ_PROJECT:-/workspace/DynLaneSeq}
dataset=${CULANE_ROOT:-/workspace/CULane}
python=${DYNLANESEQ_PYTHON:-/venv/clrernet/bin/python}

source_root=${project}/outputs/diagnostics/v33_primary_aux_sufficiency_gate
root=${project}/outputs/diagnostics/v33_private_head_firewall_gate
parent=${source_root}/g0_primary_parent_1ep/v25_g0_endpoint.pt
arm_a=${source_root}/arm_a_primary_only
arm_b=${source_root}/arm_b_primary_plus_aux_training_only
private=${root}/private_proposal_memory_pretrain
arm_d=${root}/arm_d_pretrained_head_joint
gradient_audit=${root}/pretrain_gradient_audit_32pairs.json
eval_ad=${root}/eval_a_vs_d
eval_bd=${root}/eval_b_vs_d
summary=${root}/v33_private_head_summary.json
wrong_list=${source_root}/controls/official_val_cross_clip_wrong.txt
wrong_report=${source_root}/controls/official_val_cross_clip_wrong.json

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYNLANESEQ_DLA34_WEIGHTS=${DYNLANESEQ_DLA34_WEIGHTS:-/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth}

mkdir -p "${root}"
for required in \
  "${parent}" \
  "${arm_a}/component_endpoint.pt" \
  "${arm_a}/training_report.json" \
  "${arm_b}/component_endpoint.pt" \
  "${arm_b}/training_report.json" \
  "${wrong_list}" \
  "${wrong_report}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing prerequisite: ${required}" >&2
    exit 2
  fi
done

private_endpoint=${private}/private_head_endpoint.pt
if [[ ! -f "${private_endpoint}" ]]; then
  args=(
    -m dynlaneseq_eg.tools.train_v33_private_auxiliary_head
    --config dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml
    --parent-checkpoint "${parent}"
    --expected-parent-iteration 11110
    --dataset-root "${dataset}"
    --output-dir "${private}"
    --device cuda
    --num-workers 6
    --mode gate
    --private-steps 2778
    --data-start-iteration 0
    --log-interval 25
    --resume-interval 250
  )
  if [[ -f "${private}/private_resume_latest.pt" ]]; then
    args+=(--resume "${private}/private_resume_latest.pt")
  fi
  "${python}" "${args[@]}"
fi

if [[ ! -f "${gradient_audit}" ]]; then
  "${python}" -m dynlaneseq_eg.tools.audit_v33_primary_aux_gradient_interaction \
    --config dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml \
    --parent-checkpoint "${parent}" \
    --trained-checkpoint "${private_endpoint}" \
    --dataset-root "${dataset}" \
    --output-json "${gradient_audit}" \
    --device cuda \
    --num-workers 4 \
    --batch-size 4 \
    --num-pairs 32 \
    --data-start-iteration 0 \
    --expected-parent-iteration 11110 \
    --expected-trained-iteration 11110
fi

if [[ ! -f "${arm_d}/component_endpoint.pt" ]]; then
  args=(
    -m dynlaneseq_eg.tools.train_v25_component_gate
    --config dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml
    --init-checkpoint "${parent}"
    --expected-init-iteration 11110
    --allow-advanced-init
    --advanced-private-checkpoint "${private_endpoint}"
    --advanced-private-prefix detector.proposal_memory
    --dataset-root "${dataset}"
    --output-dir "${arm_d}"
    --component-name v33_pretrained_head_joint
    --device cuda
    --num-workers 6
    --mode gate
    --log-interval 25
    --resume-interval 250
    --reset-runtime-rng-after-init
  )
  if [[ -f "${arm_d}/resume_latest.pt" ]]; then
    args+=(--resume "${arm_d}/resume_latest.pt")
  fi
  "${python}" "${args[@]}"
fi

evaluate_pair() {
  local control_config=$1
  local control_checkpoint=$2
  local control_report=$3
  local control_component=$4
  local output=$5
  local report_name=$6
  local experiment=$7
  if [[ -f "${output}/${report_name}" ]]; then
    echo "V33-PH evaluation already exists: ${output}/${report_name}"
    return
  fi
  "${python}" -m dynlaneseq_eg.tools.evaluate_v25_g2_component_pair \
    --control-config "${control_config}" \
    --control-checkpoint "${control_checkpoint}" \
    --control-report "${control_report}" \
    --control-component "${control_component}" \
    --treatment-config dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml \
    --treatment-checkpoint "${arm_d}/component_endpoint.pt" \
    --treatment-report "${arm_d}/training_report.json" \
    --treatment-component v33_pretrained_head_joint \
    --experiment-name "${experiment}" \
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
  "${eval_ad}" \
  a_vs_d_official_val.json \
  "V33-PH primary-only versus pretrained-private-head joint"

evaluate_pair \
  dynlaneseq_eg/configs/culane_v33_primary_aux_only_025ep.yaml \
  "${arm_b}/component_endpoint.pt" \
  "${arm_b}/training_report.json" \
  v33_primary_plus_aux_training_only \
  "${eval_bd}" \
  b_vs_d_official_val.json \
  "V33-PH random-head joint versus pretrained-private-head joint"

"${python}" -m dynlaneseq_eg.tools.summarize_v33_private_head_gate \
  --a-vs-d-report "${eval_ad}/a_vs_d_official_val.json" \
  --b-vs-d-report "${eval_bd}/b_vs_d_official_val.json" \
  --private-training-report "${private}/private_training_report.json" \
  --pretrain-gradient-audit "${gradient_audit}" \
  --output "${summary}"

echo "V33-PH complete: ${summary}"

