#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_6_1_stable_pointer.yaml}"
CKPT="${CKPT:-outputs/diagnostics/unified_lane_set_v4_6_1_stable_pointer_gate_105k/seed_3407/stable_low_lr_cosine/iter_0110000.pt}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
DEVELOPMENT_IMAGES="${DEVELOPMENT_IMAGES:-4096}"
HOLDOUT_IMAGES="${HOLDOUT_IMAGES:-1024}"
EARLY_STOP_IMAGES="${EARLY_STOP_IMAGES:-512}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ROW_SAMPLES="${ROW_SAMPLES:-24}"
CHANNEL_PROJECTION="${CHANNEL_PROJECTION:-64}"
SEEDS="${SEEDS:-3407 5419 7823}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-64}"
ROW_BATCH_SIZE="${ROW_BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
REUSE_ROW_CACHE="${REUSE_ROW_CACHE:-1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-2}"
TOTAL_IMAGES=$(( DEVELOPMENT_IMAGES + HOLDOUT_IMAGES ))
CHECKPOINT_TAG="$(basename "${CKPT}" .pt)"
BASE_CACHE="${BASE_CACHE:-outputs/diagnostic_cache/unified_lane_set_v4_pointer_large_representation/${CHECKPOINT_TAG}_train_uniform${TOTAL_IMAGES}.pt}"
ROW_CACHE="${ROW_CACHE:-outputs/diagnostic_cache/unified_lane_set_v4_pointer_row_sequence_observability/${CHECKPOINT_TAG}_train_uniform${TOTAL_IMAGES}_rows${ROW_SAMPLES}_proj${CHANNEL_PROJECTION}.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/row_sequence_observability_summary.json}"

for required in "${CONFIG}" "${CKPT}" "${BASE_CACHE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V4 row observability artefact: ${required}" >&2
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

mkdir -p "$(dirname "${ROW_CACHE}")" "$(dirname "${OUTPUT_JSON}")"
if [[ ! -f "${ROW_CACHE}" ]]; then
  free_kib="$(df -Pk "$(dirname "${ROW_CACHE}")" | awk 'NR==2 {print $4}')"
  required_kib=$(( MIN_FREE_GIB * 1024 * 1024 ))
  if (( free_kib < required_kib )); then
    free_gib="$(${PYTHON} -c "print(round(${free_kib}/1024/1024, 2))")"
    echo "Refusing row cache: only ${free_gib} GiB free; need ${MIN_FREE_GIB} GiB." >&2
    exit 1
  fi
fi

reuse_args=()
if [[ "${REUSE_ROW_CACHE}" == "1" && -f "${ROW_CACHE}" ]]; then
  reuse_args+=(--reuse-row-cache)
fi
read -r -a seed_args <<< "${SEEDS}"

echo "V4 pooled-vs-row representative observability audit"
echo "checkpoint: ${CKPT}"
echo "pooled base cache: ${BASE_CACHE}"
echo "row cache: ${ROW_CACHE}"
echo "development: ${DEVELOPMENT_IMAGES} (fit $(( DEVELOPMENT_IMAGES - EARLY_STOP_IMAGES )) + early-stop ${EARLY_STOP_IMAGES})"
echo "untouched holdout: ${HOLDOUT_IMAGES}"
echo "row view: ${ROW_SAMPLES} rows, ${CHANNEL_PROJECTION}+${CHANNEL_PROJECTION} projected state/evidence channels"
echo "arms: raw residual; hidden linear/nonlinear residual; row transformer residual"
echo "all scores retain the frozen checkpoint unary as an additive prior"

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_pointer_row_sequence_observability \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split train \
  --device "${DEVICE}" \
  --amp-dtype "${AMP_DTYPE}" \
  --base-cache "${BASE_CACHE}" \
  --row-cache "${ROW_CACHE}" \
  "${reuse_args[@]}" \
  --development-images "${DEVELOPMENT_IMAGES}" \
  --holdout-images "${HOLDOUT_IMAGES}" \
  --early-stop-images "${EARLY_STOP_IMAGES}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --row-samples "${ROW_SAMPLES}" \
  --channel-projection "${CHANNEL_PROJECTION}" \
  --projection-seed 8849 \
  --seeds "${seed_args[@]}" \
  --candidate-batch-size "${CANDIDATE_BATCH_SIZE}" \
  --row-batch-size "${ROW_BATCH_SIZE}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --candidate-lr 0.001 \
  --row-lr 0.0003 \
  --weight-decay 0.001 \
  --candidate-hidden 128 \
  --row-hidden 128 \
  --row-layers 2 \
  --row-heads 4 \
  --dropout 0.10 \
  --representable-min 0.50 \
  --cluster-quality-min 0.0001 \
  --target-temperature 0.05 \
  --pair-min-quality-gap 0.02 \
  --pairwise-weight 0.50 \
  --residual-l2-weight 0.0001 \
  --split-seed 1907 \
  --output-json "${OUTPUT_JSON}"

echo "Row observability audit complete: ${OUTPUT_JSON}"
