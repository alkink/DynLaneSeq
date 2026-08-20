#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v28_refined_belief_gate.yaml}"
GATE_ROOT="${GATE_ROOT:-${REPO_ROOT}/outputs/diagnostics/v28_refined_belief_gate/scientific_gate_6k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/diagnostics/v28_refined_belief_gate/switch_confidence_audit}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.audit_v28_switch_confidence \
  --config "${CONFIG}" \
  --arm-b-checkpoint "${GATE_ROOT}/arm_b/v28_gate_endpoint.pt" \
  --arm-b-training-report "${GATE_ROOT}/arm_b/training_report.json" \
  --arm-c-router-checkpoint "${GATE_ROOT}/arm_c/v28_gate_router_only.pt" \
  --arm-c-training-report "${GATE_ROOT}/arm_c/training_report.json" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --log-interval 100
