#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_3_pointer_stop.yaml}"
CKPT="${CKPT:-outputs/diagnostics/unified_lane_set_v4_3_pointer_stop_gate_50k/seed_3407/pointer_stop/iter_0065000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-4}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
METRIC_CHUNKSIZE="${METRIC_CHUNKSIZE:-64}"
AMP_DTYPE="${AMP_DTYPE:-none}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5 0.75}"
MODEL_LABEL="${MODEL_LABEL:-V4.3 pointer + STOP}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing ${MODEL_LABEL} checkpoint: ${CKPT}" >&2
  exit 1
fi

read -r -a IOU_ARGS <<< "${IOU_THRESHOLDS}"
CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
MODE_TAG="fp32"
if [[ "${AMP_DTYPE}" != "none" ]]; then
  MODE_TAG="amp${AMP_DTYPE}"
fi
PRED_DIR="${PRED_DIR:-${CKPT_DIR}/val_eval_${CKPT_TAG}_pointer_stop_${MODE_TAG}}"
mkdir -p "${PRED_DIR}"

echo "${MODEL_LABEL} full CULane validation"
echo "checkpoint: ${CKPT}"
echo "prediction directory: ${PRED_DIR}"
echo "deployment: learned STOP, maximum four lanes, no threshold, no NMS"

"${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --score-mode pointer \
  --score-thresh 0.0 \
  --quality-score-power 0.0 \
  --top-k 4 \
  --nms-distance-thresh-px 0.0 \
  --nms-min-overlap-points 5 \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-num-workers "${EVAL_NUM_WORKERS}" \
  --eval-prefetch-factor "${EVAL_PREFETCH_FACTOR}" \
  --metric-workers "${METRIC_WORKERS}" \
  --metric-chunksize "${METRIC_CHUNKSIZE}" \
  --iou-thresholds "${IOU_ARGS[@]}" \
  --pred-dir "${PRED_DIR}" \
  --output-txt "${PRED_DIR}/metrics.txt" \
  --output-json "${PRED_DIR}/metrics.json" \
  --no-pretrained-init \
  --amp-dtype "${AMP_DTYPE}" \
  2>&1 | tee "${PRED_DIR}/eval.log"
