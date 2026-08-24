#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq

PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/DynLaneSeq/outputs/diagnostics/v36_assignment_posterior_contract}"

mkdir -p "${OUTPUT_DIR}"

exec "${PYTHON_BIN}" -m dynlaneseq_eg.tools.audit_v36_assignment_posterior_contract \
  --config dynlaneseq_eg/configs/culane_v34_temporal_observability_v7_225k.yaml \
  --target-cache /workspace/DynLaneSeq/outputs/diagnostics/v34_temporal_candidate_observability/target_main_official_iou.pt \
  --pairs-json /workspace/DynLaneSeq/outputs/diagnostics/v34_temporal_candidate_observability/v34_temporal_observability.json \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint-iteration 225000 \
  --bootstrap-reps 2000
