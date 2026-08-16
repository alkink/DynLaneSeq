#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
g2_report=${project}/outputs/diagnostics/v25_g2_official_val/g2_official_val_report.json
parent=${project}/outputs/diagnostics/v25_g2_image_ownership_025ep
control=${project}/outputs/diagnostics/v25_g4_denoising_control_025ep
treatment=${project}/outputs/diagnostics/v25_g4_denoising_treatment_025ep
evaluation=${project}/outputs/diagnostics/v25_g4_denoising_official_val

while [[ ! -f "${g2_report}" ]]; do
  echo "Waiting for V25 G2 paired report..."
  sleep 60
done

# Keep the single GPU queue serial. G4 is an independent component gate built
# from the fixed G2 treatment endpoint, but it starts only after G3 releases
# the device regardless of whether the optional G3 treatment passes.
while supervisorctl status v25_g3_dual_energy 2>/dev/null | grep -q RUNNING; do
  echo "Waiting for V25 G3 queue slot..."
  sleep 60
done

parent_iteration=$("${python}" - "${parent}/training_report.json" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1], "r", encoding="utf-8"))["final_iteration"]))
PY
)

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYNLANESEQ_DLA34_WEIGHTS=/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth

smoke=${project}/outputs/diagnostics/v25_g4_denoising_smoke
if [[ ! -f "${smoke}/component_smoke.pt" ]]; then
  "${python}" -m dynlaneseq_eg.tools.train_v25_component_gate \
    --config dynlaneseq_eg/configs/culane_v25_g4_denoising_treatment_025ep.yaml \
    --init-checkpoint "${parent}/component_endpoint.pt" \
    --expected-init-iteration "${parent_iteration}" \
    --dataset-root /workspace/CULane \
    --output-dir "${smoke}" \
    --component-name g4_denoising_smoke \
    --device cuda \
    --num-workers 2 \
    --mode smoke \
    --smoke-steps 1 \
    --log-interval 1
fi

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
  g4_denoising_control \
  dynlaneseq_eg/configs/culane_v25_g4_denoising_control_025ep.yaml \
  "${control}"
run_arm \
  g4_denoising_treatment \
  dynlaneseq_eg/configs/culane_v25_g4_denoising_treatment_025ep.yaml \
  "${treatment}"

if [[ -f "${evaluation}/g4_official_val_report.json" ]]; then
  echo "V25 G4 evaluation already exists"
  exit 0
fi

exec "${python}" -m dynlaneseq_eg.tools.evaluate_v25_g4_denoising_pair \
  --control-config dynlaneseq_eg/configs/culane_v25_g4_denoising_control_025ep.yaml \
  --control-checkpoint "${control}/component_endpoint.pt" \
  --control-report "${control}/training_report.json" \
  --treatment-config dynlaneseq_eg/configs/culane_v25_g4_denoising_treatment_025ep.yaml \
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

