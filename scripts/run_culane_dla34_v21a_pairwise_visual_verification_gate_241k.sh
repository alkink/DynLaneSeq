#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED=3407
STEPS=4000
BATCH_SIZE=64
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-1}"
CACHE_SHARD_SIZE="${CACHE_SHARD_SIZE:-256}"
RUN_CACHE="${RUN_CACHE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_233k_to241k.yaml}"
V20_ROOT="${V20_ROOT:-outputs/diagnostics/v20_slot_owned_safe_replacement_233k}"
V20_AUDIT_ROOT="${V20_AUDIT_ROOT:-outputs/diagnostics/v20_decision_sufficiency_autopsy_241k}"
V20_INITIAL="${V20_INITIAL:-${V20_ROOT}/initialization/treatment_iter_0233000.pt}"
V20_ENDPOINT="${V20_ENDPOINT:-${V20_ROOT}/train_treatment/iter_0241000.pt}"
LIST_ROOT="${LIST_ROOT:-${V20_ROOT}/lists}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v21a_pairwise_visual_verification_241k}"

TRAIN_LIST="${LIST_ROOT}/train_clip640_image8192.txt"
SAMECLIP_LIST="${LIST_ROOT}/same_clip_unseen_image256.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"

TRAIN_V20_CACHE="${V20_ROOT}/cache/train8192_exact_raster/manifest.json"
SAMECLIP_V20_CACHE="${V20_AUDIT_ROOT}/cache/sameclip256_exact_raster/manifest.json"
HELDOUT_V20_CACHE="${V20_AUDIT_ROOT}/cache/heldout256_exact_raster/manifest.json"
VAL_V20_CACHE="${V20_AUDIT_ROOT}/cache/validation256_exact_raster/manifest.json"

mkdir -p "${OUTPUT_ROOT}/lists" "${OUTPUT_ROOT}/cache" "${OUTPUT_ROOT}/train"

for required in \
  "${CONFIG}" "${V20_INITIAL}" "${V20_ENDPOINT}" \
  "${TRAIN_LIST}" "${SAMECLIP_LIST}" "${HELDOUT_LIST}" "${VAL_LIST}" \
  "${TRAIN_V20_CACHE}" "${SAMECLIP_V20_CACHE}" \
  "${HELDOUT_V20_CACHE}" "${VAL_V20_CACHE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V21A artifact: ${required}" >&2
    exit 1
  fi
done

prepare_wrong_list() {
  local name="$1"
  local source_list="$2"
  local wrong_list="${OUTPUT_ROOT}/lists/${name}_cross_clip_wrong.txt"
  local report="${OUTPUT_ROOT}/lists/${name}_cross_clip_wrong.json"
  if [[ ! -f "${wrong_list}" || ! -f "${report}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.build_cross_clip_derangement \
      --input-list "${source_list}" \
      --output-list "${wrong_list}" \
      --output-json "${report}" \
      --seed "${SEED}"
  fi
}

prepare_wrong_list train8192 "${TRAIN_LIST}"
prepare_wrong_list sameclip256 "${SAMECLIP_LIST}"
prepare_wrong_list heldout256 "${HELDOUT_LIST}"
prepare_wrong_list validation256 "${VAL_LIST}"

cache_domain() {
  local name="$1"
  local split="$2"
  local source_list="$3"
  local v20_cache="$4"
  local output_dir="${OUTPUT_ROOT}/cache/${name}"
  if [[ -f "${output_dir}/manifest.json" ]]; then
    return
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.cache_v21a_pairwise_visual_verification \
    --config "${CONFIG}" \
    --geometry-checkpoint "${V20_INITIAL}" \
    --scoring-checkpoint "${V20_ENDPOINT}" \
    --v20-cache-manifest "${v20_cache}" \
    --dataset-root "${DATA_ROOT}" \
    --split "${split}" \
    --list-path "${source_list}" \
    --wrong-list-path "${OUTPUT_ROOT}/lists/${name}_cross_clip_wrong.txt" \
    --wrong-list-report "${OUTPUT_ROOT}/lists/${name}_cross_clip_wrong.json" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --batch-size "${CACHE_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --shard-size "${CACHE_SHARD_SIZE}" \
    --curve-samples 24 \
    --seed "${SEED}"
}

if [[ "${RUN_CACHE}" == "1" ]]; then
  cache_domain train8192 train "${TRAIN_LIST}" "${TRAIN_V20_CACHE}"
  cache_domain sameclip256 train "${SAMECLIP_LIST}" "${SAMECLIP_V20_CACHE}"
  cache_domain heldout256 train "${HELDOUT_LIST}" "${HELDOUT_V20_CACHE}"
  cache_domain validation256 val "${VAL_LIST}" "${VAL_V20_CACHE}"
fi

for name in train8192 sameclip256 heldout256 validation256; do
  if [[ ! -f "${OUTPUT_ROOT}/cache/${name}/manifest.json" ]]; then
    echo "Missing V21A visual cache: ${name}" >&2
    exit 1
  fi
done

if [[ "${RUN_TRAIN}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train_v21a_cached_visual_verifier \
    --train-cache "${OUTPUT_ROOT}/cache/train8192/manifest.json" \
    --eval-domain "train8192=${OUTPUT_ROOT}/cache/train8192/manifest.json" \
    --eval-domain "sameclip256=${OUTPUT_ROOT}/cache/sameclip256/manifest.json" \
    --eval-domain "heldout256=${OUTPUT_ROOT}/cache/heldout256/manifest.json" \
    --eval-domain "validation256=${OUTPUT_ROOT}/cache/validation256/manifest.json" \
    --output-dir "${OUTPUT_ROOT}/train" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}"
fi

echo "V21A diagnostic gate complete. No deployment, full validation, test, checkpoint selection or threshold search was started."
