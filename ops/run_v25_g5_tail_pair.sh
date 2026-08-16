#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
g4_report=${project}/outputs/diagnostics/v25_g4_denoising_official_val/g4_official_val_report.json
control_parent=${project}/outputs/diagnostics/v25_g4_denoising_control_025ep
treatment_parent=${project}/outputs/diagnostics/v25_g4_denoising_treatment_025ep
control=${project}/outputs/diagnostics/v25_g5_tail_control_025ep
treatment=${project}/outputs/diagnostics/v25_g5_tail_treatment_025ep
evaluation=${project}/outputs/diagnostics/v25_g5_tail_official_val

while [[ ! -f "${g4_report}" ]]; do
  echo "Waiting for V25 G4 paired report..."
  sleep 60
done

read -r parent parent_iteration < <("${python}" - "${g4_report}" "${control_parent}" "${treatment_parent}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
parent = sys.argv[3] if report["gate"]["passed"] else sys.argv[2]
training = json.load(open(parent + "/training_report.json", "r", encoding="utf-8"))
print(parent, int(training["final_iteration"]))
PY
)

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYNLANESEQ_DLA34_WEIGHTS=/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth

run_arm() {
  local name=$1
  local config=$2
  local output=$3
  local resume=${output}/resume_latest.pt
  if [[ -f "${output}/component_endpoint.pt" ]]; then
    echo "V25 ${name} endpoint already exists"
    return
  fi
  local args=(
    -m dynlaneseq_eg.tools.train_v25_component_gate
    --config "${config}"
    --init-checkpoint "${parent}/component_endpoint.pt"
    --expected-init-iteration "${parent_iteration}"
    --dataset-root /workspace/CULane
    --output-dir "${output}"
    --component-name "${name}"
    --device cuda
    --num-workers 2
    --mode gate
    --log-interval 25
    --resume-interval 250
  )
  if [[ -f "${resume}" ]]; then
    args+=(--resume "${resume}")
  fi
  "${python}" "${args[@]}"
}

run_arm \
  g5_tail_control \
  dynlaneseq_eg/configs/culane_v25_g5_tail_control_025ep.yaml \
  "${control}"
run_arm \
  g5_tail_treatment \
  dynlaneseq_eg/configs/culane_v25_g5_tail_treatment_025ep.yaml \
  "${treatment}"

if [[ -f "${evaluation}/g5_official_val_report.json" ]]; then
  echo "V25 G5 evaluation already exists"
  exit 0
fi

exec "${python}" -m dynlaneseq_eg.tools.evaluate_v25_g5_tail_pair \
  --control-config dynlaneseq_eg/configs/culane_v25_g5_tail_control_025ep.yaml \
  --control-checkpoint "${control}/component_endpoint.pt" \
  --control-report "${control}/training_report.json" \
  --treatment-config dynlaneseq_eg/configs/culane_v25_g5_tail_treatment_025ep.yaml \
  --treatment-checkpoint "${treatment}/component_endpoint.pt" \
  --treatment-report "${treatment}/training_report.json" \
  --dataset-root /workspace/CULane \
  --output-dir "${evaluation}" \
  --device cuda \
  --eval-batch-size 4 \
  --num-workers 2 \
  --metric-workers 20 \
  --metric-chunksize 32 \
  --log-interval 100

