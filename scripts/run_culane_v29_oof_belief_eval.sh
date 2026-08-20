#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v28_refined_belief_gate.yaml}"
ROOT="${ROOT:-${REPO_ROOT}/outputs/diagnostics/v29_oof_rbf_gate}"
FOLD_CONTRACT="${FOLD_CONTRACT:-${ROOT}/folds/fold_contract.json}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${ROOT}/supports}"
GATE_ROOT="${GATE_ROOT:-${ROOT}/belief_gate_6k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/official_val}"
CONTRACT_ROOT="${CONTRACT_ROOT:-${ROOT}/eval_contract}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${CONTRACT_ROOT}"

WRONG_LIST="${CONTRACT_ROOT}/val_cross_clip_wrong.txt"
WRONG_REPORT="${CONTRACT_ROOT}/val_cross_clip_wrong.json"
if [[ ! -f "${WRONG_LIST}" || ! -f "${WRONG_REPORT}" ]]; then
  "${PYTHON_BIN}" -m dynlaneseq_eg.tools.build_cross_clip_derangement \
    --input-list "${DATASET_ROOT}/list/val.txt" \
    --output-list "${WRONG_LIST}" \
    --output-json "${WRONG_REPORT}" \
    --seed 3407
fi

run_direction() {
  local support_fold="$1"
  local belief_fold="$2"
  local direction="support_${support_fold}_to_fold_${belief_fold}"
  local direction_gate="${GATE_ROOT}/${direction}"
  local direction_output="${OUTPUT_ROOT}/${direction}"
  local support_checkpoint="${SUPPORT_ROOT}/support_fold_${support_fold}/iter_0112500.pt"
  local support_report="${SUPPORT_ROOT}/support_fold_${support_fold}/support_training_report.json"

  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.evaluate_v28_refined_belief_gate \
    --config "${CONFIG}" \
    --arm-b-checkpoint "${direction_gate}/arm_b/v28_gate_endpoint.pt" \
    --arm-b-training-report "${direction_gate}/arm_b/training_report.json" \
    --arm-c-router-checkpoint "${direction_gate}/arm_c/v28_gate_router_only.pt" \
    --arm-c-training-report "${direction_gate}/arm_c/training_report.json" \
    --dataset-root "${DATASET_ROOT}" \
    --wrong-image-list "${WRONG_LIST}" \
    --wrong-image-report "${WRONG_REPORT}" \
    --oof-fold "${belief_fold}" \
    --train-list-contract "${FOLD_CONTRACT}" \
    --expected-v7-checkpoint "${support_checkpoint}" \
    --support-training-report "${support_report}" \
    --output-dir "${direction_output}/paired" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --metric-chunksize 32 \
    --log-interval 100

  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.audit_v28_switch_confidence \
    --config "${CONFIG}" \
    --arm-b-checkpoint "${direction_gate}/arm_b/v28_gate_endpoint.pt" \
    --arm-b-training-report "${direction_gate}/arm_b/training_report.json" \
    --arm-c-router-checkpoint "${direction_gate}/arm_c/v28_gate_router_only.pt" \
    --arm-c-training-report "${direction_gate}/arm_c/training_report.json" \
    --dataset-root "${DATASET_ROOT}" \
    --oof-fold "${belief_fold}" \
    --train-list-contract "${FOLD_CONTRACT}" \
    --expected-v7-checkpoint "${support_checkpoint}" \
    --support-training-report "${support_report}" \
    --output-dir "${direction_output}/switch_confidence" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --log-interval 100
}

run_direction a b
run_direction b a

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.summarize_v29_oof_belief_gate \
  --direction-a-to-b "${OUTPUT_ROOT}/support_a_to_fold_b" \
  --direction-b-to-a "${OUTPUT_ROOT}/support_b_to_fold_a" \
  --output "${OUTPUT_ROOT}/v29_oof_gate_summary.json"
