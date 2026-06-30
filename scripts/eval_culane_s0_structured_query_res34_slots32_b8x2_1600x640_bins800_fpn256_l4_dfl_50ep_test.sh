#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0050000.pt}"
DEVICE="${DEVICE:-cuda}"
SCORE_THRESH="${SCORE_THRESH:-0.40}"
QUALITY_POWER="${QUALITY_POWER:-0.25}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"
CATEGORIES="${CATEGORIES:-}"

CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
SCORE_TAG="${SCORE_THRESH/./p}"
QUALITY_TAG="${QUALITY_POWER/./p}"
NMS_TAG="${NMS_DISTANCE_THRESH_PX/./p}"
PRED_ROOT="${PRED_ROOT:-${CKPT_DIR}}"
PRED_DIR="${PRED_DIR:-${PRED_ROOT}/test_eval_${CKPT_TAG}_thr${SCORE_TAG}_q${QUALITY_TAG}_nms${NMS_TAG}}"
LOG_FILE="${LOG_FILE:-${PRED_DIR}/eval.log}"
RESULT_TXT="${RESULT_TXT:-${PRED_DIR}/metrics.txt}"
RESULT_JSON="${RESULT_JSON:-${PRED_DIR}/metrics.json}"

mkdir -p "${PRED_DIR}"
EXTRA_ARGS=()
if [[ "${SKIP_WRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-write)
fi
if [[ -n "${CATEGORIES}" ]]; then
  EXTRA_ARGS+=(${CATEGORIES})
fi

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split test \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --top-k "${TOP_K}" \
  --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
  --nms-min-overlap-points "${NMS_MIN_OVERLAP_POINTS}" \
  --pred-dir "${PRED_DIR}" \
  --output-txt "${RESULT_TXT}" \
  --output-json "${RESULT_JSON}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
