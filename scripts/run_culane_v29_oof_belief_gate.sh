#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v28_refined_belief_gate.yaml}"
ROOT="${ROOT:-${REPO_ROOT}/outputs/diagnostics/v29_oof_rbf_gate}"
FOLD_CONTRACT="${FOLD_CONTRACT:-${ROOT}/folds/fold_contract.json}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${ROOT}/supports}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/belief_gate_6k}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
RESUME_INTERVAL="${RESUME_INTERVAL:-500}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${SUPPORT_ROOT}/support_training_summary.json" ]]; then
  echo "V29 OOF support endpoints are incomplete; refusing belief training." >&2
  exit 2
fi

run_arm() {
  local support_fold="$1"
  local belief_fold="$2"
  local arm="$3"
  local arm_lower
  arm_lower="$(printf '%s' "${arm}" | tr '[:upper:]' '[:lower:]')"
  local direction="support_${support_fold}_to_fold_${belief_fold}"
  local output_dir="${OUTPUT_ROOT}/${direction}/arm_${arm_lower}"
  local support_checkpoint="${SUPPORT_ROOT}/support_fold_${support_fold}/iter_0112500.pt"
  local train_list="${ROOT}/folds/fold_${belief_fold}_train.txt"
  local resume_args=()

  if [[ ! -f "${support_checkpoint}" ]]; then
    echo "Missing support endpoint: ${support_checkpoint}" >&2
    exit 2
  fi
  if [[ -f "${output_dir}/v28_gate_endpoint.pt" ]]; then
    echo "V29 ${direction} arm ${arm} endpoint exists; preserving it."
    return
  fi
  if [[ -f "${output_dir}/resume_latest.pt" ]]; then
    resume_args=(--resume "${output_dir}/resume_latest.pt")
  fi

  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.train_v28_refined_belief_router \
    --config "${CONFIG}" \
    --v7-checkpoint "${support_checkpoint}" \
    --dataset-root "${DATASET_ROOT}" \
    --train-list "${train_list}" \
    --train-list-contract "${FOLD_CONTRACT}" \
    --oof-fold "${belief_fold}" \
    --output-dir "${output_dir}" \
    --arm "${arm}" \
    --mode gate \
    --num-workers "${NUM_WORKERS}" \
    --log-interval "${LOG_INTERVAL}" \
    --resume-interval "${RESUME_INTERVAL}" \
    "${resume_args[@]}"
}

# Each direction uses a support V7 that never saw the belief-training fold.
# One 16-GiB GPU runs all arms sequentially to preserve the paired contract.
run_arm a b B
run_arm a b C
run_arm b a B
run_arm b a C

echo "V29 two-direction OOF B/C belief endpoints completed."
