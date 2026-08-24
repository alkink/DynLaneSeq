#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq

PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/DynLaneSeq/outputs/diagnostics/v37_deep_supervision_trajectory}"

mkdir -p "${OUTPUT_DIR}"

exec "${PYTHON_BIN}" -m dynlaneseq_eg.tools.audit_v37_deep_supervision_trajectory \
  --config dynlaneseq_eg/configs/culane_v34_temporal_observability_v7_225k.yaml \
  --checkpoint /workspace/DynLaneSeq/outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt \
  --dataset-root /workspace/CULane \
  --replay-list /workspace/DynLaneSeq/outputs/diagnostics/v34_temporal_candidate_observability/temporal_union_val.txt \
  --target-cache /workspace/DynLaneSeq/outputs/diagnostics/v34_temporal_candidate_observability/target_main_official_iou.pt \
  --v36-json /workspace/DynLaneSeq/outputs/diagnostics/v36_assignment_posterior_contract/v36_assignment_posterior_contract.json \
  --output-dir "${OUTPUT_DIR}" \
  --eval-batch-size 8 \
  --num-workers 6 \
  --gradient-images-per-fold 64 \
  --bootstrap-reps 2000 \
  --amp-dtype bf16
