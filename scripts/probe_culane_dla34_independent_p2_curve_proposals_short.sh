#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CKPT="${CKPT:-/home/alki/projects/DynLaneSeq/outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
TRAIN_STEPS="${TRAIN_STEPS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/independent_p2_curve_proposals}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/dla34_225k_uniform64.json}"
SAVE_PROBE="${SAVE_PROBE:-${OUTPUT_ROOT}/dla34_225k_probe.pt}"
LOAD_PROBE="${LOAD_PROBE:-}"

mkdir -p "${OUTPUT_ROOT}"

ARGS=(
  -m dynlaneseq_eg.tools.probe_independent_p2_curve_proposals
  --config "${CONFIG}"
  --checkpoint "${CKPT}"
  --dataset-root "${DATA_ROOT}"
  --train-steps "${TRAIN_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --eval-max-batches "${EVAL_MAX_BATCHES}"
  --sample-strategy uniform
  --num-workers "${NUM_WORKERS}"
  --amp-dtype "${AMP_DTYPE}"
  --save-probe "${SAVE_PROBE}"
  --output-json "${OUTPUT_JSON}"
)

if [[ -n "${LOAD_PROBE}" ]]; then
  ARGS+=(--load-probe "${LOAD_PROBE}")
fi

"${PYTHON}" "${ARGS[@]}"
