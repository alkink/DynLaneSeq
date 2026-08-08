#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v6_b_four_slot_refinement_25k_to29k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/diagnostics/unified_lane_set_v6_b_four_slot_refinement_gate_25k/seed_3407/slot_refinement/iter_0028000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v6_b_four_slot_refinement_gate_25k}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache/unified_lane_set_v6_b_refinement_ceiling_iter28k}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/audits/slot_refinement_ceiling_iter_0028000_uniform256.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
MAX_BATCHES="${MAX_BATCHES:-64}"
DELTA_BOUND_PX="${DELTA_BOUND_PX:-24.0}"
MATCH_MIN_QUALITY="${MATCH_MIN_QUALITY:-0.20}"

for required in "${CONFIG}" "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V6-B ceiling-audit artefact: ${required}" >&2
    exit 1
  fi
done

echo "V6-B slot refinement zero-training ceiling audit"
echo "checkpoint: ${CHECKPOINT}"
echo "counterfactuals: learned x, GT range, bounded x, second bounded x, exact x"

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v6_b_slot_refinement_ceiling \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}" \
  --max-batches "${MAX_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --sample-strategy uniform \
  --iou-thresholds 0.50 0.75 \
  --line-width 30 \
  --min-valid-rows 5 \
  --match-min-quality "${MATCH_MIN_QUALITY}" \
  --delta-bound-px "${DELTA_BOUND_PX}" \
  --output-json "${OUTPUT_JSON}"

echo "V6-B ceiling report: ${OUTPUT_JSON}"
