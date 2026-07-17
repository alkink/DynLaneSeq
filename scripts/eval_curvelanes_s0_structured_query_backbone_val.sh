#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CONFIG:?Set CONFIG to a CurveLanes backbone config}"
: "${CKPT:?Set CKPT to the checkpoint path}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/Datasets/CurveLanes/Curvelanes}"
SPLIT="${SPLIT:-val}"
SCORE_THRESH="${SCORE_THRESH:-0.30}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
TOP_K="${TOP_K:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"

CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
SCORE_TAG="${SCORE_THRESH/./p}"
QUALITY_TAG="${QUALITY_POWER/./p}"
NMS_TAG="${NMS_DISTANCE_THRESH_PX/./p}"
RESULT_DIR="${RESULT_DIR:-${CKPT_DIR}/${SPLIT}_eval_${CKPT_TAG}_thr${SCORE_TAG}_q${QUALITY_TAG}_nms${NMS_TAG}_topk${TOP_K}}"
PREDICTION_DIR="${PREDICTION_DIR:-${RESULT_DIR}/predictions}"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/eval.log}"
RESULT_TXT="${RESULT_TXT:-${RESULT_DIR}/metrics.txt}"
RESULT_JSON="${RESULT_JSON:-${RESULT_DIR}/metrics.json}"

mkdir -p "${RESULT_DIR}"
EXTRA_ARGS=()
if [[ "${SKIP_WRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-write)
fi

python -u -m dynlaneseq_eg.tools.evaluate_curvelanes \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --prediction-dir "${PREDICTION_DIR}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --top-k "${TOP_K}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
  --nms-min-overlap-points "${NMS_MIN_OVERLAP_POINTS}" \
  --output-txt "${RESULT_TXT}" \
  --output-json "${RESULT_JSON}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
