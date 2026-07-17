#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-dataset}"
SCORE_THRESH="${SCORE_THRESH:-0.30}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CATEGORIES="${CATEGORIES:---categories}"

CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
SCORE_TAG="${SCORE_THRESH/./p}"
QUALITY_TAG="${QUALITY_POWER/./p}"
PRED_DIR="${PRED_DIR:-${CKPT_DIR}/test_eval_${CKPT_TAG}_thr${SCORE_TAG}_q${QUALITY_TAG}_nms20p0}"
mkdir -p "${PRED_DIR}"

EXTRA_ARGS=()
if [[ -n "${CATEGORIES}" ]]; then
  EXTRA_ARGS+=(${CATEGORIES})
fi

python -u -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split test \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --top-k "${TOP_K}" \
  --nms-distance-thresh-px 20.0 \
  --nms-min-overlap-points 5 \
  --pred-dir "${PRED_DIR}" \
  --output-txt "${PRED_DIR}/metrics.txt" \
  --output-json "${PRED_DIR}/metrics.json" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${PRED_DIR}/eval.log"
