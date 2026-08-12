#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION="${SOURCE_ITERATION:-225000}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v9_slot_owned_geometry_gate_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/v9_slot_owned_geometry_gate_225k}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${OUTPUT_ROOT}/initialization/iter_0225000.pt}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v9_slot_owned_control_225k_to227k.yaml}"
TREATMENT_CONFIG="${TREATMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v9_slot_owned_treatment_225k_to227k.yaml}"
CONTROL_MEMORY_CONFIG="${CONTROL_MEMORY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v9_slot_owned_control_memorize64.yaml}"
TREATMENT_MEMORY_CONFIG="${TREATMENT_MEMORY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v9_slot_owned_treatment_memorize64.yaml}"

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/initialization"
if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.initialize_v9_slot_owned_checkpoint \
    --config "${CONTROL_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

export PYTHON DATA_ROOT DEVICE SEED SOURCE_ITERATION OUTPUT_ROOT CACHE_ROOT
export SOURCE_CHECKPOINT="${INITIAL_CHECKPOINT}"
export CONTROL_CONFIG TREATMENT_CONFIG CONTROL_MEMORY_CONFIG TREATMENT_MEMORY_CONFIG
export SUMMARY_PREFIX=v9

bash scripts/run_culane_dla34_v8_1_geometry_router_state_gate_225k_to227k.sh
