#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/reference_guided_p2_update}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-${CHECKPOINT_ROOT}/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
DEVICE="${DEVICE:-cuda}"
TRAIN_STEPS="${TRAIN_STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
ANCHOR_LAYER="${ANCHOR_LAYER:-2}"
TRAIN_GROUP_MODE="${TRAIN_GROUP_MODE:-all}"
SEED="${SEED:-3407}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/r34_225k_l${ANCHOR_LAYER}_reference_p2.json}"
SAVE_PROBES="${SAVE_PROBES:-${OUTPUT_ROOT}/r34_225k_l${ANCHOR_LAYER}_reference_p2.pt}"

for path in "${CONFIG}" "${CKPT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing CULane root: ${DATA_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

python -u -m dynlaneseq_eg.tools.probe_reference_guided_p2_update \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --anchor-layer "${ANCHOR_LAYER}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --hidden-dim 64 \
  --offsets-px -64 -32 -16 -8 0 8 16 32 64 \
  --smooth-l1-beta-px 3.0 \
  --point-loss-weight 2.0 \
  --line-iou-loss-weight 1.0 \
  --line-width 30.0 \
  --train-group-mode "${TRAIN_GROUP_MODE}" \
  --eval-max-batches "${EVAL_MAX_BATCHES}" \
  --amp-dtype "${AMP_DTYPE}" \
  --log-interval 25 \
  --save-probes "${SAVE_PROBES}" \
  --output-json "${OUTPUT_JSON}"
