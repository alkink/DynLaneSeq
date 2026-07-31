#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_joint_set_selection_5k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_joint_set_selection_5k/iter_0070000.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/dla34_joint_selection_assignment_gradient_conflict_70k_uniform64.json}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-32}"
GRADIENT_BATCHES="${GRADIENT_BATCHES:-4}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing joint-selection checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

echo "Assignment audit images: $((EVAL_BATCH_SIZE * MAX_BATCHES))"
echo "Gradient audit images: $((EVAL_BATCH_SIZE * GRADIENT_BATCHES))"
echo "Checkpoint: ${CHECKPOINT}"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.analyze_selection_assignment_gradient_conflict \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --gradient-batches "${GRADIENT_BATCHES}" \
  --sample-strategy uniform \
  --amp-dtype "${AMP_DTYPE}" \
  --line-width 30 \
  --min-valid-rows 5 \
  --official-win-margin 0.02 \
  --output-json "${OUTPUT_JSON}"
