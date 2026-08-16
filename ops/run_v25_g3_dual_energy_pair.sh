#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
g2_report=${project}/outputs/diagnostics/v25_g2_official_val/g2_official_val_report.json
g1b_report=${project}/outputs/diagnostics/v25_g1b_multi_path_capacity/g1b_multi_path_capacity_report.json
parent=${project}/outputs/diagnostics/v25_g2_image_ownership_025ep
control=${project}/outputs/diagnostics/v25_g3_dual_energy_control_025ep
treatment=${project}/outputs/diagnostics/v25_g3_dual_energy_treatment_025ep
evaluation=${project}/outputs/diagnostics/v25_g3_dual_energy_official_val

while [[ ! -f "${g2_report}" ]]; do
  echo "Waiting for V25 G2 paired report..."
  sleep 60
done

while [[ ! -f "${g1b_report}" ]]; do
  echo "Waiting for V25 G1B multi-path report..."
  sleep 60
done

"${python}" - "${g2_report}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if not report["gate"]["passed"]:
    raise SystemExit("V25 G2 failed; G3 dual-energy continuation is not authorized")
PY

"${python}" - "${g1b_report}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if not report["gate"]["passed"]:
    raise SystemExit("V25 G1B failed; multi-hypothesis G3 is not authorized")
PY

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
    --allow-advanced-init
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
  g3_dual_energy_control \
  dynlaneseq_eg/configs/culane_v25_g3_dual_energy_control_025ep.yaml \
  "${control}"
run_arm \
  g3_dual_energy_treatment \
  dynlaneseq_eg/configs/culane_v25_g3_dual_energy_treatment_025ep.yaml \
  "${treatment}"

if [[ -f "${evaluation}/g3_official_val_report.json" ]]; then
  echo "V25 G3 evaluation already exists"
  exit 0
fi

exec "${python}" -m dynlaneseq_eg.tools.evaluate_v25_g2_component_pair \
  --control-config dynlaneseq_eg/configs/culane_v25_g3_dual_energy_control_025ep.yaml \
  --control-checkpoint "${control}/component_endpoint.pt" \
  --control-report "${control}/training_report.json" \
  --control-component g3_dual_energy_control \
  --treatment-config dynlaneseq_eg/configs/culane_v25_g3_dual_energy_treatment_025ep.yaml \
  --treatment-checkpoint "${treatment}/component_endpoint.pt" \
  --treatment-report "${treatment}/training_report.json" \
  --treatment-component g3_dual_energy_treatment \
  --experiment-name "V25 G3 dual-energy multi-hypothesis paired gate" \
  --report-filename g3_official_val_report.json \
  --dataset-root /workspace/CULane \
  --wrong-image-list /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.txt \
  --wrong-image-report /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.json \
  --output-dir "${evaluation}" \
  --device cuda \
  --eval-batch-size 4 \
  --num-workers 2 \
  --metric-workers 20 \
  --metric-chunksize 32 \
  --log-interval 100
