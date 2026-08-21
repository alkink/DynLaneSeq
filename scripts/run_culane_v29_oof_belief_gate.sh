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
SUPPORT_ITERATION="${SUPPORT_ITERATION:-112500}"
DIRECTION_PAIRS="${DIRECTION_PAIRS:-a:b b:a}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

run_arm() {
  local support_fold="$1"
  local belief_fold="$2"
  local arm="$3"
  local arm_lower
  arm_lower="$(printf '%s' "${arm}" | tr '[:upper:]' '[:lower:]')"
  local direction="support_${support_fold}_to_fold_${belief_fold}"
  local output_dir="${OUTPUT_ROOT}/${direction}/arm_${arm_lower}"
  local support_tag
  support_tag="$(printf '%07d' "${SUPPORT_ITERATION}")"
  local support_checkpoint="${SUPPORT_ROOT}/support_fold_${support_fold}/iter_${support_tag}.pt"
  local support_report="${SUPPORT_ROOT}/support_fold_${support_fold}/support_training_report.json"
  local train_list="${ROOT}/folds/fold_${belief_fold}_train.txt"
  local resume_args=()

  if [[ ! -f "${support_checkpoint}" ]]; then
    echo "Missing support endpoint: ${support_checkpoint}" >&2
    exit 2
  fi
  if [[ ! -f "${support_report}" ]]; then
    echo "Missing support training report: ${support_report}" >&2
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
read -r -a direction_pairs <<< "${DIRECTION_PAIRS}"
for pair in "${direction_pairs[@]}"; do
  IFS=: read -r support_fold belief_fold <<< "${pair}"
  if [[ -z "${support_fold}" || -z "${belief_fold}" ]]; then
    echo "Invalid DIRECTION_PAIRS entry: ${pair}" >&2
    exit 2
  fi
  run_arm "${support_fold}" "${belief_fold}" B
  run_arm "${support_fold}" "${belief_fold}" C
done

echo "V29 requested OOF B/C belief endpoints completed: ${DIRECTION_PAIRS}."
