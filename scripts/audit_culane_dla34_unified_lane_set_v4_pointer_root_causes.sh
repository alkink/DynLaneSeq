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
SAMPLE_COUNT="${SAMPLE_COUNT:-256}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MEMORIZE_IMAGES="${MEMORIZE_IMAGES:-64}"
MEMORIZE_STEPS="${MEMORIZE_STEPS:-3000}"
MEMORIZE_BATCH_SIZE="${MEMORIZE_BATCH_SIZE:-16}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-32}"
PARALLEL_STEPS="${PARALLEL_STEPS:-3000}"
PARALLEL_BATCH_SIZE="${PARALLEL_BATCH_SIZE:-32}"
SEED="${SEED:-3407}"
REUSE_CACHE="${REUSE_CACHE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_pointer_root_cause_audits}"
CHECKPOINT_TAG="$(basename "${CKPT}" .pt)"
CACHE_PATH="${CACHE_PATH:-outputs/diagnostic_cache/unified_lane_set_v4_pointer_root_cause_audits/${CHECKPOINT_TAG}_frozen_train_uniform${SAMPLE_COUNT}.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/root_cause_summary.json}"

for required in "${CONFIG}" "${CKPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V4 pointer audit artefact: ${required}" >&2
    exit 1
  fi
done
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing CULane dataset root: ${DATA_ROOT}" >&2
  exit 1
fi
if (( SAMPLE_COUNT < MEMORIZE_IMAGES )); then
  echo "SAMPLE_COUNT must be at least MEMORIZE_IMAGES." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${CACHE_PATH}")"

reuse_args=()
if [[ "${REUSE_CACHE}" == "1" && -f "${CACHE_PATH}" ]]; then
  reuse_args+=(--reuse-cache)
fi

echo "V4 pointer root-cause audit suite"
echo "checkpoint: ${CKPT}"
echo "fixed no-augmentation train images: ${SAMPLE_COUNT}"
echo "audit 1: ${MEMORIZE_IMAGES} images, fixed teacher, ${MEMORIZE_STEPS} selector steps"
echo "audit 2: frozen hidden quality probes, ${PROBE_STEPS} steps"
echo "audit 3: parallel Hungarian selector, ${PARALLEL_STEPS} steps"

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_pointer_root_causes \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split train \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  --sample-count "${SAMPLE_COUNT}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --sample-strategy uniform \
  --cache-path "${CACHE_PATH}" \
  "${reuse_args[@]}" \
  --memorize-images "${MEMORIZE_IMAGES}" \
  --memorize-steps "${MEMORIZE_STEPS}" \
  --memorize-batch-size "${MEMORIZE_BATCH_SIZE}" \
  --probe-steps "${PROBE_STEPS}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --parallel-steps "${PARALLEL_STEPS}" \
  --parallel-batch-size "${PARALLEL_BATCH_SIZE}" \
  --holdout-stride 4 \
  --representable-min 0.20 \
  --max-selections 4 \
  --seed "${SEED}" \
  --output-json "${OUTPUT_JSON}"

echo "Root-cause audit complete: ${OUTPUT_JSON}"
