#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_3_pointer_stop.yaml}"
CKPT="${CKPT:-outputs/diagnostics/unified_lane_set_v4_3_pointer_stop_gate_50k/seed_3407/pointer_stop/iter_0065000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache/unified_lane_set_v4_3_pointer_stop_gate_50k/seed_3407/pointer}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/unified_lane_set_v4_3_pointer_target_alignment_uniform256.json}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing V4.3 checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "V4.3 pointer target-alignment audit (existing uniform-256 cache only)"
echo "No inference and no training will run."
echo "cache directory: ${CACHE_DIR}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_3_pointer_target_alignment \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --split val \
  --max-batches 64 \
  --eval-batch-size 4 \
  --num-workers 8 \
  --sample-strategy uniform \
  --line-width 30 \
  --min-valid-rows 5 \
  --top-k 4 \
  --iou-thresholds 0.50 0.75 \
  --output-json "${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_JSON%.json}.log"
