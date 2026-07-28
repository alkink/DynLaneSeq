#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/frozen_p2_discovery}"

mkdir -p "${OUTPUT_ROOT}"

run_probe() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"

  "${PYTHON}" -m dynlaneseq_eg.tools.probe_frozen_p2_lane_discovery \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --train-steps "${TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --eval-max-batches "${EVAL_MAX_BATCHES}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --amp-dtype "${AMP_DTYPE}" \
    --save-probe "${OUTPUT_ROOT}/${name}_probe.pt" \
    --output-json "${OUTPUT_ROOT}/${name}.json"
}

run_probe \
  r34_225k \
  dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml \
  "${CHECKPOINT_ROOT}/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt"

run_probe \
  dla34_225k \
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml \
  "${CHECKPOINT_ROOT}/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt"
