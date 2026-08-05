#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v4_long_geometry_50k_to125k/geometry/iter_0100000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-100000}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2500}"
SEEDS="${SEEDS:-3407}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_5_pointer_gate_geometry100k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_5_pointer_gate_geometry100k}"

if (( SOURCE_ITERATION != 100000 )); then
  echo "This controlled gate requires SOURCE_ITERATION=100000." >&2
  exit 1
fi
if (( TRAIN_STEPS != 10000 )); then
  echo "This controlled gate requires TRAIN_STEPS=10000." >&2
  exit 1
fi
if (( CHECKPOINT_INTERVAL != 2500 )); then
  echo "This controlled gate requires CHECKPOINT_INTERVAL=2500." >&2
  exit 1
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing 100k V4 geometry checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi

echo "V4.5 pointer gate on frozen 100k geometry"
echo "source geometry: ${SOURCE_CHECKPOINT}"
echo "geometry iteration: ${SOURCE_ITERATION} (frozen)"
echo "pointer optimizer steps: ${TRAIN_STEPS}"
echo "logical checkpoint tags: 102500, 105000, 107500, 110000"

POINTER_CONFIG="${POINTER_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_5_cluster_soft_pointer.yaml}" \
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
SOURCE_ITERATION="${SOURCE_ITERATION}" \
TRAIN_STEPS="${TRAIN_STEPS}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
SEEDS="${SEEDS}" \
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}" \
DEVICE="${DEVICE:-cuda}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
METRIC_WORKERS="${METRIC_WORKERS:-8}" \
MAX_BATCHES="${MAX_BATCHES:-64}" \
AMP_DTYPE="${AMP_DTYPE:-bfloat16}" \
RUN_TRAIN="${RUN_TRAIN:-1}" \
RUN_GRAD_AUDIT="${RUN_GRAD_AUDIT:-1}" \
RUN_EVAL="${RUN_EVAL:-1}" \
RUN_TRAJECTORY_EVAL="${RUN_TRAJECTORY_EVAL:-1}" \
MIN_FREE_GB="${MIN_FREE_GB:-1.0}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CACHE_ROOT="${CACHE_ROOT}" \
ARM_NAME="cluster_soft_pointer" \
  bash scripts/run_culane_dla34_unified_lane_set_v4_5_cluster_soft_pointer_gate_50k.sh

echo "V4.5 pointer-on-100k gate completed under ${OUTPUT_ROOT}"
