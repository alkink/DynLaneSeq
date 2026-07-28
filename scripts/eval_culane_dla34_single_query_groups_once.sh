#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/dla34_225k_single_query_groups_val}"
SCORE_THRESH="${SCORE_THRESH:-0.30}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
METRIC_CHUNKSIZE="${METRIC_CHUNKSIZE:-16}"
AMP_DTYPE="${AMP_DTYPE:-none}"
SKIP_WRITE="${SKIP_WRITE:-0}"

for path in "$CONFIG" "$CKPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

skip_args=()
if [[ "$SKIP_WRITE" == "1" ]]; then
  skip_args+=(--skip-write)
fi

python -u -m dynlaneseq_eg.tools.evaluate_culane_query_groups \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --output-dir "$OUTPUT_DIR" \
  --num-query-groups 4 \
  --top-k "$TOP_K" \
  --score-thresh "$SCORE_THRESH" \
  --quality-power "$QUALITY_POWER" \
  --score-mode exist_quality \
  --nms-distance-thresh-px 20.0 \
  --nms-min-overlap-points 5 \
  --iou-thresholds 0.5 0.7 \
  --width 30 \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --metric-workers "$METRIC_WORKERS" \
  --metric-chunksize "$METRIC_CHUNKSIZE" \
  --amp-dtype "$AMP_DTYPE" \
  "${skip_args[@]}"
