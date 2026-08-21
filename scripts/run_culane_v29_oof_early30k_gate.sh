#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
ROOT="${ROOT:-${REPO_ROOT}/outputs/diagnostics/v29_oof_rbf_gate}"
FOLD_CONTRACT="${FOLD_CONTRACT:-${ROOT}/folds/fold_contract.json}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${ROOT}/supports}"
SUPPORT_CONFIG="${SUPPORT_CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v29_oof_v7_support.yaml}"
BELIEF_CONFIG="${BELIEF_CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v28_refined_belief_gate.yaml}"
EARLY_ROOT="${EARLY_ROOT:-${ROOT}/early30k}"
SUPPORT_ITERATION=30000
SUPPORT_TAG=0030000
SUPPORT_CHECKPOINT="${SUPPORT_ROOT}/support_fold_a/iter_${SUPPORT_TAG}.pt"
SUPPORT_REPORT="${SUPPORT_ROOT}/support_fold_a/support_training_report.json"
BANK_ROOT="${EARLY_ROOT}/support_a_bank_fold_b_uniform1024"
GATE_ROOT="${EARLY_ROOT}/belief_gate_6k"
EVAL_ROOT="${EARLY_ROOT}/official_val"

cd "${REPO_ROOT}"
mkdir -p "${BANK_ROOT}" "${GATE_ROOT}" "${EVAL_ROOT}"

if [[ ! -f "${SUPPORT_CHECKPOINT}" ]]; then
  echo "Missing fixed 30K support checkpoint: ${SUPPORT_CHECKPOINT}" >&2
  exit 2
fi

# Materialize a hash-bound report for this predeclared early endpoint. The
# existing checkpoint is preserved; no further optimizer step is taken here.
"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.train_v29_oof_supports \
  --config "${SUPPORT_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --fold-contract "${FOLD_CONTRACT}" \
  --output-root "${SUPPORT_ROOT}" \
  --device cuda \
  --num-workers 4 \
  --folds a \
  --endpoint-iteration "${SUPPORT_ITERATION}" \
  --summary-name support_fold_a_early30k_summary.json

# Capacity is checked only on Fold B, which support A never saw. Official
# validation and test remain closed at this preflight stage.
if [[ ! -f "${BANK_ROOT}/oracle_topk.json" ]]; then
  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
    --config "${SUPPORT_CONFIG}" \
    --checkpoint "${SUPPORT_CHECKPOINT}" \
    --split train \
    --list-path "${ROOT}/folds/fold_b_train.txt" \
    --dataset-root "${DATASET_ROOT}" \
    --top-k-values 4 \
    --iou-thresholds 0.50 0.75 \
    --operating-points 0.0:0.0 \
    --fixed-points-only \
    --nms-distance-thresh-px 0 \
    --max-batches 256 \
    --eval-batch-size 4 \
    --num-workers 4 \
    --official-iou-workers 12 \
    --sample-strategy uniform \
    --cache-dir "${BANK_ROOT}/cache" \
    --exact-postprocess \
    --output-json "${BANK_ROOT}/oracle_topk.json"
fi

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.summarize_v29_early_support_bank \
  --oracle-report "${BANK_ROOT}/oracle_topk.json" \
  --output "${BANK_ROOT}/bank_gate_summary.json"

"${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("passed") is True else 3)' \
  "${BANK_ROOT}/bank_gate_summary.json"

# First and only the A->B direction. A pass authorizes, but does not itself
# claim, reverse-direction replication.
ROOT="${ROOT}" \
SUPPORT_ROOT="${SUPPORT_ROOT}" \
OUTPUT_ROOT="${GATE_ROOT}" \
CONFIG="${BELIEF_CONFIG}" \
SUPPORT_ITERATION="${SUPPORT_ITERATION}" \
DIRECTION_PAIRS="a:b" \
bash "${REPO_ROOT}/scripts/run_culane_v29_oof_belief_gate.sh"

ROOT="${ROOT}" \
SUPPORT_ROOT="${SUPPORT_ROOT}" \
GATE_ROOT="${GATE_ROOT}" \
OUTPUT_ROOT="${EVAL_ROOT}" \
CONFIG="${BELIEF_CONFIG}" \
SUPPORT_ITERATION="${SUPPORT_ITERATION}" \
DIRECTION_PAIRS="a:b" \
bash "${REPO_ROOT}/scripts/run_culane_v29_oof_belief_eval.sh"

echo "V29 early30K A-to-B gate completed without opening the test set."
echo "${EVAL_ROOT}/support_a_to_fold_b/direction_summary.json"
