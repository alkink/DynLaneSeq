#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-12}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SEED=3407

FIELD_CONFIG="${FIELD_CONFIG:-dynlaneseq_eg/configs/culane_v22_lane_field_stage_a.yaml}"
V20_CONFIG="${V20_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_233k_to241k.yaml}"
V20_ROOT="${V20_ROOT:-outputs/diagnostics/v20_slot_owned_safe_replacement_233k}"
V20_GEOMETRY_CHECKPOINT="${V20_GEOMETRY_CHECKPOINT:-${V20_ROOT}/initialization/treatment_iter_0233000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v22_lane_field_stage_a_official}"
VAL_CACHE_MANIFEST="${VAL_CACHE_MANIFEST:-${OUTPUT_ROOT}/cache/official_val_v20_exact/manifest.json}"
VAL_LIST="${DATA_ROOT}/list/val.txt"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/official_val_exact_gt_operator_oracle.json}"

for required in \
  "${FIELD_CONFIG}" "${V20_CONFIG}" "${V20_GEOMETRY_CHECKPOINT}" \
  "${VAL_CACHE_MANIFEST}" "${VAL_LIST}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V22 exact operator artifact: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v22_exact_gt_operator_oracle \
  --field-config "${FIELD_CONFIG}" \
  --v20-config "${V20_CONFIG}" \
  --v20-geometry-checkpoint "${V20_GEOMETRY_CHECKPOINT}" \
  --v20-cache-manifest "${VAL_CACHE_MANIFEST}" \
  --dataset-root "${DATA_ROOT}" \
  --val-list "${VAL_LIST}" \
  --output-json "${OUTPUT_JSON}" \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --seed "${SEED}"

echo "Exact GT operator oracle finished on all 9,675 official val.txt rows. No filtering, deduplication, training, checkpoint/threshold selection, or test evaluation was performed."
