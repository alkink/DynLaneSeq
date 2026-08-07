#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-32}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
SEED="${SEED:-3407}"

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_assignment_10k_to25k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/diagnostics/unified_lane_set_v5_1_shared_trunk_gate/seed_3407/assignment_fork/iter_0025000.pt}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v5_ownership_graph_and_slot_probe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v5_four_slot_parameter_matched}"
TRAIN_CACHE="${TRAIN_CACHE:-${CACHE_ROOT}/train_4096.pt}"
VAL_CACHE="${VAL_CACHE:-${CACHE_ROOT}/val_256.pt}"
REPORT="${OUTPUT_ROOT}/four_slot_vs_parameter_matched_32_uniform256.json"

for artifact in "${CONFIG}" "${CHECKPOINT}" "${TRAIN_CACHE}" "${VAL_CACHE}"; do
  if [[ ! -f "${artifact}" ]]; then
    echo "Missing required comparison artifact: ${artifact}" >&2
    echo "This comparison is cache-only and will not rerun detector inference." >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"

echo "V5 frozen-structure comparison"
echo "seed: ${SEED}"
echo "control: parameter-matched 32-query set scorer"
echo "candidate model: four final object slots"

"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_v5_four_slot_router \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --cache-batch-size 4 \
  --train-cache-images 4096 \
  --val-cache-images 256 \
  --train-cache "${TRAIN_CACHE}" \
  --val-cache "${VAL_CACHE}" \
  --reuse-cache \
  --cache-only \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype bfloat16 \
  --train-steps "${PROBE_STEPS}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --hidden-dim 256 \
  --proposal-layers 2 \
  --matched-proposal-layers 5 \
  --slot-layers 2 \
  --num-heads 8 \
  --ff-dim 512 \
  --num-slots 4 \
  --representable-min 0.50 \
  --cluster-min 0.30 \
  --cluster-delta 0.05 \
  --cluster-temperature 0.03 \
  --seed "${SEED}" \
  --output-json "${REPORT}" \
  --save-probes "${OUTPUT_ROOT}/four_slot_vs_parameter_matched_32.pt"

echo "Parameter-matched comparison ready: ${REPORT}"
