#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
MAX_BATCHES="${MAX_BATCHES:-64}"
GRADIENT_BATCHES="${GRADIENT_BATCHES:-2}"
TRAIN_CACHE_IMAGES="${TRAIN_CACHE_IMAGES:-4096}"
VAL_CACHE_IMAGES="${VAL_CACHE_IMAGES:-256}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-32}"
REUSE_CACHE="${REUSE_CACHE:-1}"

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_assignment_10k_to25k.yaml}"
V5_ROOT="${V5_ROOT:-outputs/diagnostics/unified_lane_set_v5_1_shared_trunk_gate/seed_3407}"
TRUNK_CHECKPOINT="${TRUNK_CHECKPOINT:-${V5_ROOT}/shared_trunk/iter_0010000.pt}"
ASSIGNMENT_ROOT="${ASSIGNMENT_ROOT:-${V5_ROOT}/assignment_fork}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v5_ownership_graph_and_slot_probe}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v5_ownership_graph_and_slot_probe}"

checkpoints=(
  "${TRUNK_CHECKPOINT}"
  "${ASSIGNMENT_ROOT}/iter_0015000.pt"
  "${ASSIGNMENT_ROOT}/iter_0020000.pt"
  "${ASSIGNMENT_ROOT}/iter_0025000.pt"
)
for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing V5.1 checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing V5.1 config: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}"
GRAPH_JSON="${OUTPUT_ROOT}/ownership_graph_audit_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
SLOT_JSON="${OUTPUT_ROOT}/four_slot_probe_uniform${VAL_CACHE_IMAGES}.json"
SUMMARY_JSON="${OUTPUT_ROOT}/summary.json"

echo "===== 1/3 V5 ownership graph audit (zero optimizer steps) ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v5_ownership_graph \
  --config "${CONFIG}" \
  --checkpoints "${checkpoints[@]}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --max-batches "${MAX_BATCHES}" \
  --gradient-batches "${GRADIENT_BATCHES}" \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}" \
  --line-width 30 \
  --min-valid-rows 5 \
  --stable-iou-floor 0.50 \
  --no-lane-weight 0.10 \
  --output-json "${GRAPH_JSON}"

reuse_args=()
if [[ "${REUSE_CACHE}" == "1" ]]; then
  reuse_args+=(--reuse-cache)
fi

echo "===== 2/3 frozen 32-query versus four-slot early probe ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_v5_four_slot_router \
  --config "${CONFIG}" \
  --checkpoint "${ASSIGNMENT_ROOT}/iter_0025000.pt" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --cache-batch-size "${EVAL_BATCH_SIZE}" \
  --train-cache-images "${TRAIN_CACHE_IMAGES}" \
  --val-cache-images "${VAL_CACHE_IMAGES}" \
  --train-cache "${CACHE_ROOT}/train_${TRAIN_CACHE_IMAGES}.pt" \
  --val-cache "${CACHE_ROOT}/val_${VAL_CACHE_IMAGES}.pt" \
  "${reuse_args[@]}" \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}" \
  --train-steps "${PROBE_STEPS}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --hidden-dim 256 \
  --proposal-layers 2 \
  --slot-layers 2 \
  --num-heads 8 \
  --ff-dim 512 \
  --num-slots 4 \
  --representable-min 0.50 \
  --cluster-min 0.30 \
  --cluster-delta 0.05 \
  --cluster-temperature 0.03 \
  --output-json "${SLOT_JSON}" \
  --save-probes "${OUTPUT_ROOT}/four_slot_probe.pt"

echo "===== 3/3 decision summary ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v5_ownership_graph_and_slot_probe \
  --graph-audit "${GRAPH_JSON}" \
  --slot-probe "${SLOT_JSON}" \
  --output-json "${SUMMARY_JSON}"

echo "V5 ownership graph and early slot decision are ready: ${SUMMARY_JSON}"
