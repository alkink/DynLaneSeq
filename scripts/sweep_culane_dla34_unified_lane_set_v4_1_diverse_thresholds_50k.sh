#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cpu}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
CACHE_ONLY="${CACHE_ONLY:-1}"

SOURCE_CONFIG="${SOURCE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt}"
C_CONFIG="${C_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_1_score_c_set_shared.yaml}"
C_CHECKPOINT="${C_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v4_1_score_gate_50k/c_set_shared/iter_0053000.pt}"
D_CONFIG="${D_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_1_score_d_set_unique.yaml}"
D_CHECKPOINT="${D_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v4_1_score_gate_50k/d_set_unique/iter_0053000.pt}"

CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_1_score_gate_50k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/unified_lane_set_v4_1_diverse_threshold_grid_50k}"

SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.00 0.025 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.225 0.25 0.275 0.30 0.35 0.40}"
HARD_DISTANCES="${HARD_DISTANCES:-10 15 20 25 30}"
MMR_SIGMAS="${MMR_SIGMAS:-10 15 20 30 40}"
MMR_PENALTIES="${MMR_PENALTIES:-0.20 0.35 0.50 0.65 0.80}"

for path in \
  "${SOURCE_CONFIG}" "${SOURCE_CHECKPOINT}" \
  "${C_CONFIG}" "${C_CHECKPOINT}" \
  "${D_CONFIG}" "${D_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

read -r -a score_threshold_args <<<"${SCORE_THRESHOLDS}"
read -r -a hard_distance_args <<<"${HARD_DISTANCES}"
read -r -a mmr_sigma_args <<<"${MMR_SIGMAS}"
read -r -a mmr_penalty_args <<<"${MMR_PENALTIES}"

cache_flag=(--reuse-cache)
if [[ "${CACHE_ONLY}" == "1" ]]; then
  cache_flag=(--cache-only)
fi

mkdir -p "${OUTPUT_DIR}"

names=(source_v4 c_set_shared d_set_unique)
configs=("${SOURCE_CONFIG}" "${C_CONFIG}" "${D_CONFIG}")
checkpoints=("${SOURCE_CHECKPOINT}" "${C_CHECKPOINT}" "${D_CHECKPOINT}")

echo "V4.1 cached threshold/diversity grid"
echo "cache-only: ${CACHE_ONLY}"
echo "score thresholds: ${SCORE_THRESHOLDS}"
echo "hard distances: ${HARD_DISTANCES}"
echo "MMR sigmas: ${MMR_SIGMAS}"
echo "MMR penalties: ${MMR_PENALTIES}"

for index in "${!names[@]}"; do
  name="${names[$index]}"
  config="${configs[$index]}"
  checkpoint="${checkpoints[$index]}"
  report="${OUTPUT_DIR}/${name}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
  echo "===== GRID ${name} ====="
  "${PYTHON}" -u -m dynlaneseq_eg.tools.sweep_v4_diverse_thresholds \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_ROOT}/${name}" \
    "${cache_flag[@]}" \
    --max-batches "${MAX_BATCHES}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --sample-strategy uniform \
    --stage main \
    --top-k 4 \
    --iou-thresholds 0.50 0.75 \
    --score-thresholds "${score_threshold_args[@]}" \
    --hard-diversity-distances "${hard_distance_args[@]}" \
    --mmr-sigmas "${mmr_sigma_args[@]}" \
    --mmr-penalties "${mmr_penalty_args[@]}" \
    --near-min-iou 0.30 \
    --line-width 30 \
    --min-valid-rows 5 \
    --row-visibility-thresh 0 \
    --nms-min-overlap-points 5 \
    --output-json "${report}" \
    2>&1 | tee "${OUTPUT_DIR}/${name}.log"
done

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_diverse_threshold_grid \
  --source "${OUTPUT_DIR}/source_v4_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
  --c "${OUTPUT_DIR}/c_set_shared_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
  --d "${OUTPUT_DIR}/d_set_unique_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
  --output-json "${OUTPUT_DIR}/summary.json"

cp "${OUTPUT_DIR}/summary.json" outputs/diagnostics/v4_1_diverse_threshold_grid_summary.json

echo "Grid completed: ${OUTPUT_DIR}/summary.json"
echo "Convenience copy: outputs/diagnostics/v4_1_diverse_threshold_grid_summary.json"
