#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDLSTR_ROOT="${CONDLSTR_ROOT:?Set CONDLSTR_ROOT to the CondLSTR checkout}"
DATA_PARENT="${DATA_PARENT:?Set DATA_PARENT so DATA_PARENT/culane is the official CULane root}"
CULANE_ROOT="${DATA_PARENT}/culane"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/external/condlstr_culane_official}"
CULANE_EVALUATOR="${CULANE_EVALUATOR:?Set CULANE_EVALUATOR to the official CULane evaluate binary}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="${VERSION:-official_train_val_v1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-3407}"

mkdir -p "${OUTPUT_ROOT}"

if git -C "${CONDLSTR_ROOT}" apply --unidiff-zero --check "${PROJECT_ROOT}/tools/condlstr/condlstr_official_culane.patch" >/dev/null 2>&1; then
  git -C "${CONDLSTR_ROOT}" apply --unidiff-zero "${PROJECT_ROOT}/tools/condlstr/condlstr_official_culane.patch"
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.prepare_condlstr_culane_official \
  --dataset-root "${CULANE_ROOT}" \
  --train-list "${CULANE_ROOT}/list/train.txt" \
  --val-list "${CULANE_ROOT}/list/val.txt" \
  --version "${VERSION}" \
  --workers "${NUM_WORKERS}" \
  --manifest "${OUTPUT_ROOT}/dataset_protocol.json"

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.smoke_condlstr_culane_official \
  --condlstr-root "${CONDLSTR_ROOT}" \
  --dataset-root "${CULANE_ROOT}" \
  --version "${VERSION}"

cd "${CONDLSTR_ROOT}"
"${PYTHON_BIN}" -u tools/train.py \
  -a CondLSTR2DRes34 \
  -d culane \
  -v "${VERSION}" \
  -c 1 \
  -t lane_det_2d \
  --data-dir "${DATA_PARENT}" \
  --logs-dir "${OUTPUT_ROOT}/train" \
  -b "${BATCH_SIZE}" \
  -j "${NUM_WORKERS}" \
  -e "${EPOCHS}" \
  --eval-epoch "$((EPOCHS - 1))" \
  --seed "${SEED}" \
  -p amp

"${PYTHON_BIN}" -u tools/test.py \
  -a CondLSTR2DRes34 \
  -d culane \
  -v "${VERSION}" \
  -c 1 \
  -t lane_det_2d \
  --data-dir "${DATA_PARENT}" \
  --logs-dir "${OUTPUT_ROOT}/train" \
  --test-dir "${OUTPUT_ROOT}/raw_results" \
  --split val \
  -b "${BATCH_SIZE}" \
  -j "${NUM_WORKERS}" \
  -p amp \
  --seed "${SEED}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.convert_condlstr_culane_results \
  --results "${OUTPUT_ROOT}/raw_results/results.pkl" \
  --val-list "${CULANE_ROOT}/list/val.txt" \
  --output-dir "${OUTPUT_ROOT}/predictions" \
  --manifest "${OUTPUT_ROOT}/conversion_manifest.json"

for IOU in 0.50 0.75; do
  "${CULANE_EVALUATOR}" \
    -a "${CULANE_ROOT}/" \
    -d "${OUTPUT_ROOT}/predictions/" \
    -i "${CULANE_ROOT}/" \
    -l "${CULANE_ROOT}/list/val.txt" \
    -w 30 -t "${IOU}" -c 1640 -r 590 -f 1 -p 1 \
    -o "${OUTPUT_ROOT}/official_iou${IOU}.txt"
done
