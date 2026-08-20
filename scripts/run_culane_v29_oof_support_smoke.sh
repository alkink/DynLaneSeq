#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v29_oof_v7_support.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/diagnostics/v29_oof_rbf_gate}"
FOLD_ROOT="${FOLD_ROOT:-${OUTPUT_ROOT}/folds}"
SMOKE_ROOT="${SMOKE_ROOT:-${OUTPUT_ROOT}/support_smoke_fold_a}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"

cd "${REPO_ROOT}"
mkdir -p "${FOLD_ROOT}" "${SMOKE_ROOT}"

if [[ ! -f "${FOLD_ROOT}/fold_contract.json" ]]; then
  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.build_v29_oof_folds \
    --train-list "${DATASET_ROOT}/list/train.txt" \
    --train-gt-list "${DATASET_ROOT}/list/train_gt.txt" \
    --output-dir "${FOLD_ROOT}" \
    --seed 3407
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device cuda \
  --dataset-root "${DATASET_ROOT}" \
  --train-list "${FOLD_ROOT}/fold_a_train_gt.txt" \
  --output-dir "${SMOKE_ROOT}" \
  --max-iters "${SMOKE_STEPS}" \
  --checkpoint-interval 0 \
  --num-workers "${NUM_WORKERS}" \
  --resume-safe-data true

echo "V29 fold-support smoke completed at ${SMOKE_STEPS} optimizer steps."
