#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/nms_ranking_dla34_uniform64}"
BASE_REPORT="${BASE_REPORT:-${OUTPUT_DIR}/base_dla34_225k_oracle_topk.json}"
CANDIDATE_REPORT="${CANDIDATE_REPORT:-${OUTPUT_DIR}/global_1to1_global_inter_227500_oracle_topk.json}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/paired_group_duplication_summary.json}"
BASE_CACHE="${BASE_CACHE:-}"
CANDIDATE_CACHE="${CANDIDATE_CACHE:-}"
NUM_QUERY_BLOCKS="${NUM_QUERY_BLOCKS:-4}"
TOP_K="${TOP_K:-4}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5 0.7}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"

for report in "$BASE_REPORT" "$CANDIDATE_REPORT"; do
  if [[ ! -f "$report" ]]; then
    echo "Missing diagnostic report: $report" >&2
    echo "Set BASE_REPORT and CANDIDATE_REPORT to the two oracle_topk JSON files." >&2
    exit 1
  fi
done

cache_args=()
if [[ -n "$BASE_CACHE" ]]; then
  cache_args+=(--base-cache "$BASE_CACHE")
fi
if [[ -n "$CANDIDATE_CACHE" ]]; then
  cache_args+=(--candidate-cache "$CANDIDATE_CACHE")
fi

python -u -m dynlaneseq_eg.tools.analyze_group_duplication_pair \
  --base-report "$BASE_REPORT" \
  --candidate-report "$CANDIDATE_REPORT" \
  "${cache_args[@]}" \
  --num-query-blocks "$NUM_QUERY_BLOCKS" \
  --top-k "$TOP_K" \
  --quality-power "$QUALITY_POWER" \
  --iou-thresholds ${IOU_THRESHOLDS} \
  --nms-distance-thresh-px "$NMS_DISTANCE_THRESH_PX" \
  --nms-min-overlap-points "$NMS_MIN_OVERLAP_POINTS" \
  --output-json "$OUTPUT_JSON"
