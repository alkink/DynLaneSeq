#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/dla34_rowref_from65k_selective_cooldown_10k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-70000 75000}"
SCORE_THRESH="${SCORE_THRESH:-0.15}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
AMP_DTYPE="${AMP_DTYPE:-none}"

for iteration in ${CHECKPOINT_ITERS}; do
  checkpoint_tag="$(printf '%07d' "${iteration}")"
  checkpoint="${OUT_DIR}/iter_${checkpoint_tag}.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing selective-cooldown checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  CONFIG="${CONFIG}" \
  DATA_ROOT="${DATA_ROOT}" \
  CKPT="${checkpoint}" \
  SCORE_THRESH="${SCORE_THRESH}" \
  QUALITY_POWER="${QUALITY_POWER}" \
  TOP_K="${TOP_K}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  AMP_DTYPE="${AMP_DTYPE}" \
  IOU_THRESHOLDS="0.5 0.75" \
  CATEGORIES=--categories \
    bash scripts/eval_culane_dla34_row_reference_full_test.sh
done

echo "Fixed-score full tests completed with score=${SCORE_THRESH}, quality=${QUALITY_POWER}."
