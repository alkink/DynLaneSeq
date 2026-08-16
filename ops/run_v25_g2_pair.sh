#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
g0=${project}/outputs/diagnostics/v25_g0_direct_four_lane_objects_1ep
g0_eval=${project}/outputs/diagnostics/v25_g0_g1_official_val/g0_g1_official_val_report.json
control=${project}/outputs/diagnostics/v25_g2_control_025ep
treatment=${project}/outputs/diagnostics/v25_g2_image_ownership_025ep
evaluation=${project}/outputs/diagnostics/v25_g2_official_val

while [[ ! -f "${g0_eval}" ]]; do
  echo "Waiting for V25 G0/G1 report..."
  sleep 60
done

"${python}" - "${g0_eval}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if not report["g0_mechanism_gate"]["passed"]:
    raise SystemExit("V25 G0 mechanism gate failed; G2 is not authorized")
if not report["g1_hard_path_gate"]["passed"]:
    raise SystemExit("V25 G1 path gate failed; G2 is not authorized")
PY

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
    --init-checkpoint "${g0}/v25_g0_endpoint.pt"
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
  g2_control \
  dynlaneseq_eg/configs/culane_v25_g2_control_025ep.yaml \
  "${control}"
run_arm \
  g2_image_ownership \
  dynlaneseq_eg/configs/culane_v25_g2_image_ownership_025ep.yaml \
  "${treatment}"

if [[ -f "${evaluation}/g2_official_val_report.json" ]]; then
  echo "V25 G2 evaluation already exists"
  exit 0
fi

exec "${python}" -m dynlaneseq_eg.tools.evaluate_v25_g2_component_pair \
  --control-config dynlaneseq_eg/configs/culane_v25_g2_control_025ep.yaml \
  --control-checkpoint "${control}/component_endpoint.pt" \
  --control-report "${control}/training_report.json" \
  --treatment-config dynlaneseq_eg/configs/culane_v25_g2_image_ownership_025ep.yaml \
  --treatment-checkpoint "${treatment}/component_endpoint.pt" \
  --treatment-report "${treatment}/training_report.json" \
  --dataset-root /workspace/CULane \
  --wrong-image-list /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.txt \
  --wrong-image-report /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.json \
  --output-dir "${evaluation}" \
  --device cuda \
  --eval-batch-size 8 \
  --num-workers 2 \
  --metric-workers 20 \
  --metric-chunksize 32 \
  --log-interval 100
