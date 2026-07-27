#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_g4train_g1infer_b4x4_1600x640_bins800_fpn256_l4_dfl_r15_deepsup_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_slots32_g4train_g1infer_b4x4_1600x640_bins800_fpn256_l4_dfl_r15_deepsup_50ep/iter_0025000.pt}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-dataset}"
SPLIT="${SPLIT:-test}"
# The initial 0.20 default is selected on the historical ResNet-34 validation
# candidate cache for the quality-only, single-group protocol. Re-select on
# validation if the new run changes quality calibration materially.
SCORE_THRESH="${SCORE_THRESH:-0.20}"
SCORE_MODE="${SCORE_MODE:-quality}"
QUALITY_POWER="${QUALITY_POWER:-1.0}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-4}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
METRIC_CHUNKSIZE="${METRIC_CHUNKSIZE:-64}"
AMP_DTYPE="${AMP_DTYPE:-none}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
REUSE_PREDICTIONS="${REUSE_PREDICTIONS:-0}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-0.0}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5 0.75}"
CATEGORIES="${CATEGORIES:---categories}"
if [[ "${SPLIT}" != "test" ]]; then
  CATEGORIES=""
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

read -r -a IOU_ARGS <<< "${IOU_THRESHOLDS}"
CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
SCORE_TAG="${SCORE_THRESH/./p}"
QUALITY_TAG="${QUALITY_POWER/./p}"
PRECISION_TAG="tf32"
if [[ "${AMP_DTYPE}" != "none" ]]; then
  PRECISION_TAG="amp${AMP_DTYPE}"
fi
MODE_TAG="_g1_${SCORE_MODE}_q${QUALITY_TAG}_nonms_fast_${PRECISION_TAG}"
if [[ "${COMPILE_MODEL}" == "1" ]]; then
  MODE_TAG+="_compile"
fi
PRED_DIR="${PRED_DIR:-${CKPT_DIR}/${SPLIT}_eval_${CKPT_TAG}_thr${SCORE_TAG}${MODE_TAG}}"
mkdir -p "${PRED_DIR}"

EXTRA_ARGS=(--no-pretrained-init --amp-dtype "${AMP_DTYPE}")
if [[ -n "${CATEGORIES}" ]]; then
  EXTRA_ARGS+=(${CATEGORIES})
fi
if [[ "${COMPILE_MODEL}" == "1" ]]; then
  EXTRA_ARGS+=(--compile-model)
fi
if [[ "${REUSE_PREDICTIONS}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-write)
fi

python -u -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --score-mode "${SCORE_MODE}" \
  --quality-score-power "${QUALITY_POWER}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-num-workers "${EVAL_NUM_WORKERS}" \
  --eval-prefetch-factor "${EVAL_PREFETCH_FACTOR}" \
  --metric-workers "${METRIC_WORKERS}" \
  --metric-chunksize "${METRIC_CHUNKSIZE}" \
  --top-k "${TOP_K}" \
  --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
  --iou-thresholds "${IOU_ARGS[@]}" \
  --pred-dir "${PRED_DIR}" \
  --output-txt "${PRED_DIR}/metrics.txt" \
  --output-json "${PRED_DIR}/metrics.json" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${PRED_DIR}/eval.log"
