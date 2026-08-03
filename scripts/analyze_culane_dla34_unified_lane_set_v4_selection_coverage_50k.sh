#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache/v4_selection_coverage_50k}"
REUSE_CACHE="${REUSE_CACHE:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/unified_lane_set_v4_selection_coverage_50k_uniform256.json}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing V4 checkpoint: ${CKPT}" >&2
  exit 1
fi

iteration="$(${PYTHON} - "${CKPT}" <<'PY'
import sys
import torch

try:
    payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(sys.argv[1], map_location="cpu")
print(int(payload.get("iteration", -1)))
PY
)"
if (( iteration != 50000 )); then
  echo "Expected an exact 50k checkpoint, found iteration=${iteration}: ${CKPT}" >&2
  exit 1
fi

reuse_args=()
if [[ "${REUSE_CACHE}" == "1" ]]; then
  reuse_args=(--reuse-cache)
fi

echo "V4 selection-coverage diagnosis"
echo "checkpoint: ${CKPT}"
echo "images: up to $((EVAL_BATCH_SIZE * MAX_BATCHES)) (${SAMPLE_STRATEGY})"
echo "comparison: scalar Top-4 vs hard diversity vs MMR vs Oracle Top-4"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}" \
  --max-batches "${MAX_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --sample-strategy "${SAMPLE_STRATEGY}" \
  --stage main \
  --top-k 4 \
  --iou-thresholds 0.50 0.75 \
  --near-min-iou 0.30 \
  --line-width 30 \
  --min-valid-rows 5 \
  --row-visibility-thresh 0 \
  --nms-min-overlap-points 5 \
  --hard-diversity-distances 10 20 30 40 60 \
  --mmr-sigmas 10 20 30 \
  --mmr-penalties 0.10 0.25 0.50 0.75 \
  --output-json "${OUTPUT_JSON}" \
  "${reuse_args[@]}"

echo "Selection-coverage report: ${OUTPUT_JSON}"
