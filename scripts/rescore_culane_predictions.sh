#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_1024x384_bins512_fpn256_l4_dfl_50ep.yaml}"
SPLIT="${SPLIT:-val}"
PRED_DIR="${PRED_DIR:?Set PRED_DIR to an existing CULane prediction directory.}"
DEVICE="${DEVICE:-cpu}"
WIDTH="${WIDTH:-30}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5}"
SEQUENTIAL="${SEQUENTIAL:-}"
CATEGORIES="${CATEGORIES:-}"

LOG_FILE="${LOG_FILE:-${PRED_DIR}/rescore_${SPLIT}.log}"
RESULT_TXT="${RESULT_TXT:-${PRED_DIR}/rescore_${SPLIT}_metrics.txt}"
RESULT_JSON="${RESULT_JSON:-${PRED_DIR}/rescore_${SPLIT}_metrics.json}"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --pred-dir "${PRED_DIR}" \
  --skip-write \
  --width "${WIDTH}" \
  --iou-thresholds ${IOU_THRESHOLDS} \
  --output-txt "${RESULT_TXT}" \
  --output-json "${RESULT_JSON}" \
  ${SEQUENTIAL} \
  ${CATEGORIES} \
  2>&1 | tee "${LOG_FILE}"
