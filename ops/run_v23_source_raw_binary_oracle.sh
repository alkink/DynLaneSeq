#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v23
output=${project}/outputs/diagnostics/v23_geometry_gate_raw_student_audit/source_raw_binary_oracle.json

if [[ -f "${output}" ]]; then
  echo "V23 source/raw binary oracle already exists: ${output}"
  exit 0
fi

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
exec /venv/clrernet/bin/python -m dynlaneseq_eg.tools.analyze_v23_source_raw_binary_oracle \
  --dataset-root /workspace/CULane \
  --source-predictions "${project}/outputs/diagnostics/v23_ordered_slot_cost_volume_official_val/predictions/source_v7" \
  --raw-predictions "${project}/outputs/diagnostics/v23_geometry_gate_raw_student_audit/predictions/raw_student_correct_image" \
  --output-json "${output}" \
  --workers 20 \
  --chunksize 32
