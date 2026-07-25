#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_balanced_detail_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_balanced_detail_fpn256_l4_dfl_50ep/iter_0225000.pt}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-dataset}"
SCORE_THRESH="${SCORE_THRESH:-0.30}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-4}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
METRIC_CHUNKSIZE="${METRIC_CHUNKSIZE:-64}"
AMP_DTYPE="${AMP_DTYPE:-none}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
LEGACY_INFERENCE="${LEGACY_INFERENCE:-0}"
REUSE_PREDICTIONS="${REUSE_PREDICTIONS:-0}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5}"
CATEGORIES="${CATEGORIES:---categories}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

read -r -a IOU_ARGS <<< "${IOU_THRESHOLDS}"
CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
SCORE_TAG="${SCORE_THRESH/./p}"
QUALITY_TAG="${QUALITY_POWER/./p}"
NMS_TAG="${NMS_DISTANCE_THRESH_PX/./p}"
MODE_TAG="_fast_tf32"
if [[ "${AMP_DTYPE}" != "none" ]]; then
  MODE_TAG="_fast_amp${AMP_DTYPE}"
fi
if [[ "${COMPILE_MODEL}" == "1" ]]; then
  MODE_TAG+="_compile"
fi
PRED_DIR="${PRED_DIR:-${CKPT_DIR}/test_eval_${CKPT_TAG}_thr${SCORE_TAG}_q${QUALITY_TAG}_nms${NMS_TAG}${MODE_TAG}}"
mkdir -p "${PRED_DIR}"

EXTRA_ARGS=(--no-pretrained-init --amp-dtype "${AMP_DTYPE}")
if [[ -n "${CATEGORIES}" ]]; then
  EXTRA_ARGS+=(${CATEGORIES})
fi
if [[ "${COMPILE_MODEL}" == "1" ]]; then
  EXTRA_ARGS+=(--compile-model)
fi
if [[ "${LEGACY_INFERENCE}" == "1" ]]; then
  EXTRA_ARGS+=(--legacy-inference)
fi
if [[ "${REUSE_PREDICTIONS}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-write)
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
  --eval-num-workers "${EVAL_NUM_WORKERS}" \
  --eval-prefetch-factor "${EVAL_PREFETCH_FACTOR}" \
  --metric-workers "${METRIC_WORKERS}" \
  --metric-chunksize "${METRIC_CHUNKSIZE}" \
  --top-k "${TOP_K}" \
  --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
  --nms-min-overlap-points "${NMS_MIN_OVERLAP_POINTS}" \
  --iou-thresholds "${IOU_ARGS[@]}" \
  --pred-dir "${PRED_DIR}" \
  --output-txt "${PRED_DIR}/metrics.txt" \
  --output-json "${PRED_DIR}/metrics.json" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${PRED_DIR}/eval.log"

