#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-12}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
SEED=3407
STEPS=10000
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4

FIELD_CONFIG="${FIELD_CONFIG:-dynlaneseq_eg/configs/culane_v22_lane_field_stage_a.yaml}"
V20_CONFIG="${V20_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_233k_to241k.yaml}"
V7_CHECKPOINT="${V7_CHECKPOINT:-/workspace/DynLaneSeq/outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
V20_ROOT="${V20_ROOT:-outputs/diagnostics/v20_slot_owned_safe_replacement_233k}"
V20_GEOMETRY_CHECKPOINT="${V20_GEOMETRY_CHECKPOINT:-${V20_ROOT}/initialization/treatment_iter_0233000.pt}"
V20_SCORING_CHECKPOINT="${V20_SCORING_CHECKPOINT:-${V20_ROOT}/train_treatment/iter_0241000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v22_lane_field_stage_a_official}"

# These paths are fixed by the official CULane protocol. The Python entry
# points independently verify the canonical paths and exact populations.
TRAIN_LIST="${DATA_ROOT}/list/train_gt.txt"
VAL_LIST="${DATA_ROOT}/list/val.txt"
WRONG_VAL_LIST="${OUTPUT_ROOT}/controls/official_val_cross_clip_wrong.txt"
WRONG_VAL_REPORT="${OUTPUT_ROOT}/controls/official_val_cross_clip_wrong.json"
VAL_CACHE_DIR="${OUTPUT_ROOT}/cache/official_val_v20_exact"
VAL_CACHE_MANIFEST="${VAL_CACHE_DIR}/manifest.json"
FIELD_TRAIN_DIR="${OUTPUT_ROOT}/train"
FIELD_CHECKPOINT="${FIELD_TRAIN_DIR}/lane_field_endpoint.pt"
FIELD_REPORT="${OUTPUT_ROOT}/official_val_stage_a_report.json"

mkdir -p "${OUTPUT_ROOT}/controls" "${OUTPUT_ROOT}/cache" "${FIELD_TRAIN_DIR}"

for required in \
  "${FIELD_CONFIG}" "${V20_CONFIG}" "${V7_CHECKPOINT}" \
  "${V20_GEOMETRY_CHECKPOINT}" "${V20_SCORING_CHECKPOINT}" \
  "${TRAIN_LIST}" "${VAL_LIST}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V22 official-protocol artifact: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" -c \
  'import json,sys; from dynlaneseq_eg.tools.v22_official_protocol import official_culane_list_contract as c; print(json.dumps({"train": c(sys.argv[1], split="train"), "val": c(sys.argv[1], split="val")}, indent=2, sort_keys=True))' \
  "${DATA_ROOT}" > "${OUTPUT_ROOT}/official_population_contract.json"

if [[ ! -f "${WRONG_VAL_LIST}" || ! -f "${WRONG_VAL_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.build_cross_clip_derangement \
    --input-list "${VAL_LIST}" \
    --output-list "${WRONG_VAL_LIST}" \
    --output-json "${WRONG_VAL_REPORT}" \
    --seed "${SEED}"
fi

if [[ ! -f "${VAL_CACHE_MANIFEST}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.cache_v20_slot_owned_replacement \
    --config "${V20_CONFIG}" \
    --checkpoint "${V20_GEOMETRY_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --list-path "${VAL_LIST}" \
    --output-dir "${VAL_CACHE_DIR}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --shard-size 64 \
    --seed "${SEED}"
fi

if [[ ! -f "${FIELD_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train_v22_lane_field_stage_a \
    --config "${FIELD_CONFIG}" \
    --v7-checkpoint "${V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --output-dir "${FIELD_TRAIN_DIR}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --seed "${SEED}" \
    --log-interval 50
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_v22_lane_field_stage_a \
  --field-config "${FIELD_CONFIG}" \
  --field-checkpoint "${FIELD_CHECKPOINT}" \
  --v20-config "${V20_CONFIG}" \
  --v20-geometry-checkpoint "${V20_GEOMETRY_CHECKPOINT}" \
  --v20-scoring-checkpoint "${V20_SCORING_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --domain "official_val|val|${VAL_LIST}|${VAL_CACHE_MANIFEST}|${WRONG_VAL_LIST}|${WRONG_VAL_REPORT}" \
  --output-json "${FIELD_REPORT}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}"

echo "V22 Stage-A finished on official train_gt.txt and all official val.txt rows. No filtering, deduplication, test evaluation, checkpoint selection, or Stage B was performed."
