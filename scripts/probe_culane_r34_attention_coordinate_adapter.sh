#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/attention_coordinate_adapter}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-${CHECKPOINT_ROOT}/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
TRAIN_STEPS="${TRAIN_STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
ATTENTION_LAYERS="${ATTENTION_LAYERS:-3 4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
MAX_UPDATE_PX="${MAX_UPDATE_PX:-48}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TRAIN_GROUP_MODE="${TRAIN_GROUP_MODE:-all}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/r34_225k.json}"
SAVE_PROBES="${SAVE_PROBES:-${OUTPUT_ROOT}/r34_225k_probes.pt}"

mkdir -p "$(dirname "${OUTPUT_JSON}")" "$(dirname "${SAVE_PROBES}")"

# shellcheck disable=SC2086
"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_attention_coordinate_adapter \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --eval-max-batches "${EVAL_MAX_BATCHES}" \
  --amp-dtype "${AMP_DTYPE}" \
  --attention-layers ${ATTENTION_LAYERS} \
  --hidden-dim "${HIDDEN_DIM}" \
  --max-update-px "${MAX_UPDATE_PX}" \
  --learning-rate "${LEARNING_RATE}" \
  --train-group-mode "${TRAIN_GROUP_MODE}" \
  --save-probes "${SAVE_PROBES}" \
  --output-json "${OUTPUT_JSON}"
