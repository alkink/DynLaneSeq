#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache/unified_lane_set_v4_5_cluster_support_preflight}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/unified_lane_set_v4_5_cluster_support_preflight.json}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCHES="${MAX_BATCHES:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REQUIRE_CACHE="${REQUIRE_CACHE:-0}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing V4 source config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Missing V4 source checkpoint: ${CKPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSON}")" "${CACHE_DIR}"

cache_flag=()
if [[ "${REQUIRE_CACHE}" == "1" ]]; then
  cache_flag+=(--require-cache)
fi

echo "V4.5 GT-cluster soft-target support preflight"
echo "No training will run. A single frozen-geometry cache pass runs only if needed."
echo "checkpoint: ${CKPT}"
echo "output: ${OUTPUT_JSON}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_5_cluster_support \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}" \
  "${cache_flag[@]}" \
  --split val \
  --max-batches "${MAX_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --sample-strategy uniform \
  --line-width 30 \
  --min-valid-rows 5 \
  --top-k 4 \
  --representable-thresholds 0.50 \
  --support-mins 0.45 0.50 \
  --quality-deltas 0.03 0.05 0.10 \
  --temperatures 0.03 0.05 0.10 \
  --output-json "${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_JSON%.json}.log"
