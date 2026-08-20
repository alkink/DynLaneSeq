#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v29_oof_v7_support.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/diagnostics/v29_oof_rbf_gate}"
FOLD_ROOT="${FOLD_ROOT:-${OUTPUT_ROOT}/folds}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${OUTPUT_ROOT}/supports}"
NUM_WORKERS="${NUM_WORKERS:-4}"

cd "${REPO_ROOT}"
mkdir -p "${FOLD_ROOT}" "${SUPPORT_ROOT}"

if [[ ! -f "${FOLD_ROOT}/fold_contract.json" ]]; then
  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.build_v29_oof_folds \
    --train-list "${DATASET_ROOT}/list/train.txt" \
    --train-gt-list "${DATASET_ROOT}/list/train_gt.txt" \
    --output-dir "${FOLD_ROOT}" \
    --seed 3407
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.train_v29_oof_supports \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --fold-contract "${FOLD_ROOT}/fold_contract.json" \
  --output-root "${SUPPORT_ROOT}" \
  --device cuda \
  --num-workers "${NUM_WORKERS}"
