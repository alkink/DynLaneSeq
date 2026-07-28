#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-${CHECKPOINT_ROOT}/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-${CHECKPOINT_ROOT}/diagnostics/position_transport/dla34_225k.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_BATCHES="${MAX_BATCHES:-16}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
TARGET_LAYER="${TARGET_LAYER:-3}"
LAYER_MODE="${LAYER_MODE:-cascade}"
CORRIDOR_RADIUS_PX="${CORRIDOR_RADIUS_PX:-16}"
BIAS_STRENGTH="${BIAS_STRENGTH:-2.0}"
FEEDBACK_SCALES="${FEEDBACK_SCALES:-0.5 1.0 2.0}"
FUSION_ALPHAS="${FUSION_ALPHAS:-0.1 0.25 0.5 0.75 1.0}"
ROLL_COLUMNS="${ROLL_COLUMNS:-50}"
POSITION_MODES="${POSITION_MODES:-zero reverse shuffle roll_left roll_right}"
SKIP_POSITION_SENSITIVITY="${SKIP_POSITION_SENSITIVITY:-0}"
SKIP_CODE_FEEDBACK="${SKIP_CODE_FEEDBACK:-0}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

EXTRA_ARGS=()
if [[ "${SKIP_POSITION_SENSITIVITY}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-position-sensitivity)
fi
if [[ "${SKIP_CODE_FEEDBACK}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-code-feedback)
fi

# shellcheck disable=SC2086
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_position_transport \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --sample-strategy "${SAMPLE_STRATEGY}" \
  --amp-dtype "${AMP_DTYPE}" \
  --target-layer "${TARGET_LAYER}" \
  --layer-mode "${LAYER_MODE}" \
  --corridor-radius-px "${CORRIDOR_RADIUS_PX}" \
  --bias-strength "${BIAS_STRENGTH}" \
  --feedback-scales ${FEEDBACK_SCALES} \
  --fusion-alphas ${FUSION_ALPHAS} \
  --roll-columns "${ROLL_COLUMNS}" \
  --position-modes ${POSITION_MODES} \
  "${EXTRA_ARGS[@]}" \
  --output-json "${OUTPUT_JSON}"
