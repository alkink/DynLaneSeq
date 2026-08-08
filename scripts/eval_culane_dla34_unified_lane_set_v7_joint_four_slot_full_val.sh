#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_278k.yaml}"
CKPT="${CKPT:?Set CKPT to a V7 checkpoint}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-none}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing V7 config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Missing V7 checkpoint: ${CKPT}" >&2
  exit 1
fi

checkpoint_name="$(basename "${CKPT}" .pt)"
checkpoint_dir="$(dirname "${CKPT}")"
output_dir="${OUTPUT_DIR:-${checkpoint_dir}/val_eval_${checkpoint_name}_four_slot_fp32}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --dataset-root "${DATA_ROOT}" \
  --device cuda \
  --score-mode four_slot \
  --score-thresh 0.0 \
  --quality-score-power 0.0 \
  --top-k 4 \
  --nms-distance-thresh-px 0.0 \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --iou-thresholds 0.5 0.75 \
  --pred-dir "${output_dir}/predictions" \
  --output-txt "${output_dir}/metrics.txt" \
  --output-json "${output_dir}/metrics.json" \
  --no-pretrained-init \
  --amp-dtype "${AMP_DTYPE}"
