#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cpu}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/object0p5_matcher_70k_pr_frontier}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
CACHE_ONLY="${CACHE_ONLY:-1}"
QUALITY_POWERS="${QUALITY_POWERS:-0.0 0.25 0.5}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:--1 0.025 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 0.90}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5 0.75}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_object0p5_matcher_5k.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_selective_cooldown_10k/iter_0070000.pt}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_5k/iter_0070000.pt}"
CONTROL_JSON="${OUTPUT_DIR}/control_pr_grid_uniform256.json"
CANDIDATE_JSON="${OUTPUT_DIR}/candidate_pr_grid_uniform256.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

for path in \
  "${CONTROL_CONFIG}" \
  "${CANDIDATE_CONFIG}" \
  "${CONTROL_CHECKPOINT}" \
  "${CANDIDATE_CHECKPOINT}"
do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

read -r -a quality_args <<< "${QUALITY_POWERS}"
read -r -a score_args <<< "${SCORE_THRESHOLDS}"
read -r -a iou_args <<< "${IOU_THRESHOLDS}"

cache_args=(--reuse-cache)
case "${CACHE_ONLY}" in
  1|true|TRUE|yes|YES)
    cache_args=(--cache-only)
    cache_mode="cache-only; model inference is forbidden"
    ;;
  0|false|FALSE|no|NO)
    cache_mode="reuse cache when available; otherwise run model inference"
    ;;
  *)
    echo "CACHE_ONLY must be 0/1 (or true/false), got: ${CACHE_ONLY}" >&2
    exit 1
    ;;
esac

common_args=(
  --split val
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --top-k-values 4
  --iou-thresholds "${iou_args[@]}"
  --quality-powers "${quality_args[@]}"
  --score-thresholds "${score_args[@]}"
  --line-width 30
  --nms-distance-thresh-px 20
  --nms-min-overlap-points 5
  --max-batches "${MAX_BATCHES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --sample-strategy uniform
  --cache-dir "${CACHE_DIR}"
  "${cache_args[@]}"
  --exact-postprocess
)

mkdir -p "${OUTPUT_DIR}"

echo "Matched official PR frontier: ${cache_mode}."
echo "Device: ${DEVICE}"
echo "Quality powers: ${QUALITY_POWERS}"
echo "Score thresholds: ${SCORE_THRESHOLDS}"
echo "Official IoU thresholds: ${IOU_THRESHOLDS}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONTROL_CONFIG}" \
  --checkpoint "${CONTROL_CHECKPOINT}" \
  --output-json "${CONTROL_JSON}" \
  "${common_args[@]}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CANDIDATE_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --output-json "${CANDIDATE_JSON}" \
  "${common_args[@]}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_official_pr_frontier_pair \
  --base-json "${CONTROL_JSON}" \
  --candidate-json "${CANDIDATE_JSON}" \
  --top-k 4 \
  --output-json "${SUMMARY_JSON}"

echo "Primary result: ${SUMMARY_JSON}"
