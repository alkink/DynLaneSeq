#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_6_1_stable_pointer.yaml}"
DEFAULT_STABLE_CKPT="outputs/diagnostics/unified_lane_set_v4_6_1_stable_pointer_gate_105k/seed_3407/stable_low_lr_cosine/iter_0110000.pt"
DEFAULT_SOURCE_CKPT="outputs/diagnostics/unified_lane_set_v4_5_pointer_gate_geometry100k/seed_3407/cluster_soft_pointer/iter_0105000.pt"
if [[ -z "${CKPT:-}" ]]; then
  if [[ -f "${DEFAULT_STABLE_CKPT}" ]]; then
    CKPT="${DEFAULT_STABLE_CKPT}"
  else
    CKPT="${DEFAULT_SOURCE_CKPT}"
  fi
fi

AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
DEVELOPMENT_IMAGES="${DEVELOPMENT_IMAGES:-4096}"
HOLDOUT_IMAGES="${HOLDOUT_IMAGES:-1024}"
EARLY_STOP_IMAGES="${EARLY_STOP_IMAGES:-512}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEEDS="${SEEDS:-3407 5419 7823}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
REUSE_CACHE="${REUSE_CACHE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_pointer_large_representation}"
CHECKPOINT_TAG="$(basename "${CKPT}" .pt)"
TOTAL_IMAGES=$(( DEVELOPMENT_IMAGES + HOLDOUT_IMAGES ))
CACHE_PATH="${CACHE_PATH:-outputs/diagnostic_cache/unified_lane_set_v4_pointer_large_representation/${CHECKPOINT_TAG}_train_uniform${TOTAL_IMAGES}.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/large_representation_summary.json}"

for required in "${CONFIG}" "${CKPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V4 large representation artefact: ${required}" >&2
    exit 1
  fi
done
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing CULane dataset root: ${DATA_ROOT}" >&2
  exit 1
fi
if (( DEVELOPMENT_IMAGES <= EARLY_STOP_IMAGES )); then
  echo "DEVELOPMENT_IMAGES must exceed EARLY_STOP_IMAGES." >&2
  exit 1
fi
if (( HOLDOUT_IMAGES <= 0 )); then
  echo "HOLDOUT_IMAGES must be positive." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${CACHE_PATH}")"

reuse_args=()
if [[ "${REUSE_CACHE}" == "1" && -f "${CACHE_PATH}" ]]; then
  reuse_args+=(--reuse-cache)
fi

read -r -a seed_args <<< "${SEEDS}"

echo "V4 large image-disjoint representative-quality probe"
echo "checkpoint: ${CKPT}"
echo "development: ${DEVELOPMENT_IMAGES} (fit $(( DEVELOPMENT_IMAGES - EARLY_STOP_IMAGES )) + early-stop ${EARLY_STOP_IMAGES})"
echo "untouched holdout: ${HOLDOUT_IMAGES}"
echo "probe seeds: ${SEEDS}"
echo "arms: raw/hidden x linear/nonlinear"
echo "geometry: frozen; augmentation: disabled; official test: unused"

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_pointer_large_representation \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split train \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  --development-images "${DEVELOPMENT_IMAGES}" \
  --holdout-images "${HOLDOUT_IMAGES}" \
  --early-stop-images "${EARLY_STOP_IMAGES}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --sample-strategy uniform \
  --cache-path "${CACHE_PATH}" \
  "${reuse_args[@]}" \
  --seeds "${seed_args[@]}" \
  --batch-size "${PROBE_BATCH_SIZE}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --linear-lr 0.003 \
  --nonlinear-lr 0.001 \
  --weight-decay 0.001 \
  --nonlinear-hidden 128 \
  --dropout 0.10 \
  --representable-min 0.50 \
  --cluster-support-min 0.20 \
  --pair-support-delta 0.20 \
  --pair-min-quality-gap 0.02 \
  --quality-aux-weight 0.05 \
  --split-seed 1907 \
  --output-json "${OUTPUT_JSON}"

echo "Large representation audit complete: ${OUTPUT_JSON}"
