#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml}"
HEALTHY_CHECKPOINT="${HEALTHY_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k/iter_0025000.pt}"
FAILED_CHECKPOINT="${FAILED_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k/iter_0050000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
TOP_K="${TOP_K:-4}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.20 0.30}"
EXTENDED="${EXTENDED:-0}"
INTEGRITY_ONLY="${INTEGRITY_ONLY:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/unified_lane_set_v3_module_swaps_uniform64.json}"

for checkpoint in "${HEALTHY_CHECKPOINT}" "${FAILED_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

variants=(
  control_healthy
  control_failed
  healthy_encoder_failed_decoder
  failed_encoder_healthy_decoder
  healthy_row_readout
  failed_row_readout_on_healthy
  healthy_row_norm
  failed_row_norm_on_healthy
  healthy_row_x
  failed_row_x_on_healthy
  healthy_lane_state_core
  failed_lane_state_core_on_healthy
  healthy_fpn_bn_buffers
  failed_fpn_bn_buffers_on_healthy
)
if [[ "${EXTENDED}" == "1" ]]; then
  variants+=(
    healthy_full_fpn
    failed_full_fpn_on_healthy
    healthy_row_reference_path
    failed_row_reference_path_on_healthy
  )
fi
read -r -a score_threshold_values <<< "${SCORE_THRESHOLDS}"
integrity_args=()
if [[ "${INTEGRITY_ONLY}" == "1" ]]; then
  integrity_args=(--integrity-only)
fi

echo "Unified lane-set V3 zero-training module-swap audit"
echo "healthy: ${HEALTHY_CHECKPOINT}"
echo "failed:  ${FAILED_CHECKPOINT}"
echo "sample:  $((EVAL_BATCH_SIZE * MAX_BATCHES)) validation images"
echo "variants: ${variants[*]}"
echo "output:   ${OUTPUT_JSON}"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.analyze_unified_lane_set_module_swaps \
  --config "${CONFIG}" \
  --healthy-checkpoint "${HEALTHY_CHECKPOINT}" \
  --failed-checkpoint "${FAILED_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --sample-strategy uniform \
  --amp-dtype "${AMP_DTYPE}" \
  --top-k "${TOP_K}" \
  --score-thresholds "${score_threshold_values[@]}" \
  --iou-thresholds 0.50 0.75 \
  --line-width 30 \
  --min-valid-rows 5 \
  --row-visibility-thresh 0 \
  --variants "${variants[@]}" \
  "${integrity_args[@]}" \
  --output-json "${OUTPUT_JSON}"
