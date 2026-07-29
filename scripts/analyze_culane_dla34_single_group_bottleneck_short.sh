#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/single_group_bottleneck_dla34_uniform64}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/report.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REUSE_CACHE="${REUSE_CACHE:-0}"

for path in "$CONFIG" "$CKPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

reuse_args=()
if [[ "$REUSE_CACHE" == "1" ]]; then
  reuse_args+=(--reuse-cache)
fi

python -u -m dynlaneseq_eg.tools.analyze_single_group_bottleneck \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --cache-dir "$CACHE_DIR" \
  --sample-strategy uniform \
  --max-batches "$EVAL_MAX_BATCHES" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --stage-name main \
  --num-query-groups 4 \
  --query-group-index 0 \
  --score-thresh 0.30 \
  --quality-power 0.50 \
  --top-k 4 \
  --min-valid-rows 5 \
  --line-width 30.0 \
  --iou-thresholds 0.5 0.7 \
  --output-json "$OUTPUT_JSON" \
  "${reuse_args[@]}"
