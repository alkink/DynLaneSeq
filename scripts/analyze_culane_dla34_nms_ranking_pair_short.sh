#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
BASE_CONFIG="${BASE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_global_1to1_global_inter_finetune.yaml}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/diagnostics/dla34_225k_global_1to1_global_inter/iter_0227500.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/nms_ranking}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-4}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5 0.7}"
QUALITY_POWERS="${QUALITY_POWERS:-0.0 0.25 0.50}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:--1.0 0.30}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"

for path in "$BASE_CONFIG" "$BASE_CHECKPOINT" "$CANDIDATE_CONFIG" "$CANDIDATE_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

BASE_JSON="${OUTPUT_DIR}/base_dla34_225k_oracle_topk.json"
CANDIDATE_JSON="${OUTPUT_DIR}/global_1to1_global_inter_227500_oracle_topk.json"
SUMMARY_JSON="${OUTPUT_DIR}/paired_nms_ranking_summary.json"

run_arm() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local output_json="$4"

  echo
  echo "=== ${name}: collect once, analyze many ==="
  python -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --dataset-root "$DATA_ROOT" \
    --split val \
    --device "$DEVICE" \
    --cache-dir "$CACHE_DIR" \
    --reuse-cache \
    --max-batches "$MAX_BATCHES" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --sample-strategy "$SAMPLE_STRATEGY" \
    --top-k-values "$TOP_K" \
    --iou-thresholds ${IOU_THRESHOLDS} \
    --quality-powers ${QUALITY_POWERS} \
    --score-thresholds ${SCORE_THRESHOLDS} \
    --nms-distance-thresh-px "$NMS_DISTANCE_THRESH_PX" \
    --nms-min-overlap-points "$NMS_MIN_OVERLAP_POINTS" \
    --output-json "$output_json"
}

run_arm "historical DLA-34 baseline" \
  "$BASE_CONFIG" \
  "$BASE_CHECKPOINT" \
  "$BASE_JSON"

run_arm "global 1-to-1 + global inter-query" \
  "$CANDIDATE_CONFIG" \
  "$CANDIDATE_CHECKPOINT" \
  "$CANDIDATE_JSON"

python -u -m dynlaneseq_eg.tools.summarize_nms_ranking_pair \
  --base-json "$BASE_JSON" \
  --candidate-json "$CANDIDATE_JSON" \
  --output-json "$SUMMARY_JSON"

echo
echo "paired summary: $SUMMARY_JSON"
