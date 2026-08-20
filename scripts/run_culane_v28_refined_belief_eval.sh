#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v28_refined_belief_gate.yaml}"
GATE_ROOT="${GATE_ROOT:-${REPO_ROOT}/outputs/diagnostics/v28_refined_belief_gate/scientific_gate_6k}"
CONTRACT_ROOT="${CONTRACT_ROOT:-${REPO_ROOT}/outputs/diagnostics/v28_refined_belief_gate/eval_contract}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/diagnostics/v28_refined_belief_gate/official_val_b_c}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"

cd "${REPO_ROOT}"
mkdir -p "${CONTRACT_ROOT}" "${OUTPUT_DIR}"

WRONG_LIST="${CONTRACT_ROOT}/val_cross_clip_wrong.txt"
WRONG_REPORT="${CONTRACT_ROOT}/val_cross_clip_wrong.json"
if [[ ! -f "${WRONG_LIST}" || ! -f "${WRONG_REPORT}" ]]; then
  "${PYTHON_BIN}" -m dynlaneseq_eg.tools.build_cross_clip_derangement \
    --input-list "${DATASET_ROOT}/list/val.txt" \
    --output-list "${WRONG_LIST}" \
    --output-json "${WRONG_REPORT}" \
    --seed 3407
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.evaluate_v28_refined_belief_gate \
  --config "${CONFIG}" \
  --arm-b-checkpoint "${GATE_ROOT}/arm_b/v28_gate_endpoint.pt" \
  --arm-b-training-report "${GATE_ROOT}/arm_b/training_report.json" \
  --arm-c-router-checkpoint "${GATE_ROOT}/arm_c/v28_gate_router_only.pt" \
  --arm-c-training-report "${GATE_ROOT}/arm_c/training_report.json" \
  --dataset-root "${DATASET_ROOT}" \
  --wrong-image-list "${WRONG_LIST}" \
  --wrong-image-report "${WRONG_REPORT}" \
  --output-dir "${OUTPUT_DIR}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --metric-chunksize 32 \
  --log-interval 100

