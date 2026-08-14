#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-1}"
CACHE_SHARD_SIZE="${CACHE_SHARD_SIZE:-64}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-64}"
RUN_CACHE="${RUN_CACHE:-1}"

TREATMENT_CONFIG="${TREATMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_233k_to241k.yaml}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_control_233k_to241k.yaml}"
V20_ROOT="${V20_ROOT:-outputs/diagnostics/v20_slot_owned_safe_replacement_233k}"
LIST_ROOT="${LIST_ROOT:-${V20_ROOT}/lists}"
TREATMENT_INITIAL="${TREATMENT_INITIAL:-${V20_ROOT}/initialization/treatment_iter_0233000.pt}"
TREATMENT_ENDPOINT="${TREATMENT_ENDPOINT:-${V20_ROOT}/train_treatment/iter_0241000.pt}"
CONTROL_ENDPOINT="${CONTROL_ENDPOINT:-${V20_ROOT}/train_control/iter_0241000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v20_decision_sufficiency_autopsy_241k}"

TRAIN_LIST="${LIST_ROOT}/train_clip640_image8192.txt"
SAMECLIP_LIST="${LIST_ROOT}/same_clip_unseen_image256.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"

TRAIN_CACHE="${V20_ROOT}/cache/train8192_exact_raster/manifest.json"
SAMECLIP_CACHE_DIR="${OUTPUT_ROOT}/cache/sameclip256_exact_raster"
HELDOUT_CACHE_DIR="${OUTPUT_ROOT}/cache/heldout256_exact_raster"
VAL_CACHE_DIR="${OUTPUT_ROOT}/cache/validation256_exact_raster"
SAMECLIP_CACHE="${SAMECLIP_CACHE_DIR}/manifest.json"
HELDOUT_CACHE="${HELDOUT_CACHE_DIR}/manifest.json"
VAL_CACHE="${VAL_CACHE_DIR}/manifest.json"

mkdir -p "${OUTPUT_ROOT}/cache" "${OUTPUT_ROOT}/reports"

for required in \
  "${TREATMENT_CONFIG}" "${CONTROL_CONFIG}" \
  "${TREATMENT_INITIAL}" "${TREATMENT_ENDPOINT}" "${CONTROL_ENDPOINT}" \
  "${TRAIN_LIST}" "${SAMECLIP_LIST}" "${HELDOUT_LIST}" "${VAL_LIST}" \
  "${TRAIN_CACHE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V20 autopsy artifact: ${required}" >&2
    exit 1
  fi
done

cache_domain() {
  local split="$1"
  local list_path="$2"
  local output_dir="$3"
  if [[ -f "${output_dir}/manifest.json" ]]; then
    return
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.cache_v20_slot_owned_replacement \
    --config "${TREATMENT_CONFIG}" \
    --checkpoint "${TREATMENT_INITIAL}" \
    --dataset-root "${DATA_ROOT}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --batch-size "${CACHE_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --shard-size "${CACHE_SHARD_SIZE}" \
    --seed "${SEED}"
}

if [[ "${RUN_CACHE}" == "1" ]]; then
  cache_domain train "${SAMECLIP_LIST}" "${SAMECLIP_CACHE_DIR}"
  cache_domain train "${HELDOUT_LIST}" "${HELDOUT_CACHE_DIR}"
  cache_domain val "${VAL_LIST}" "${VAL_CACHE_DIR}"
fi

for manifest in "${SAMECLIP_CACHE}" "${HELDOUT_CACHE}" "${VAL_CACHE}"; do
  if [[ ! -f "${manifest}" ]]; then
    echo "Missing V20 autopsy cache: ${manifest}" >&2
    exit 1
  fi
done

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v20_decision_sufficiency \
  --treatment-config "${TREATMENT_CONFIG}" \
  --control-config "${CONTROL_CONFIG}" \
  --treatment-checkpoint "${TREATMENT_ENDPOINT}" \
  --control-checkpoint "${CONTROL_ENDPOINT}" \
  --domain "train8192=${TRAIN_CACHE}" \
  --domain "sameclip256=${SAMECLIP_CACHE}" \
  --domain "heldout256=${HELDOUT_CACHE}" \
  --domain "validation256=${VAL_CACHE}" \
  --device "${DEVICE}" \
  --batch-size "${AUDIT_BATCH_SIZE}" \
  --output-json "${OUTPUT_ROOT}/reports/v20_decision_sufficiency_autopsy.json" \
  --output-markdown "${OUTPUT_ROOT}/reports/V20_DECISION_SUFFICIENCY_AUTOPSY.md"

echo "V20 decision-sufficiency autopsy complete. No training, threshold selection, full validation or test was started."
