#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/venv/clrernet/bin/python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-12}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
ROOT="${ROOT:-outputs/diagnostics/v31_selection_bridge_exact_pair_35k_to50k}"
REPLAY_ROOT="${ROOT}/refiner_2x2_replay"

CONTROL_CONFIG="dynlaneseq_eg/configs/culane_v30_field_only_35k_to50k.yaml"
TREATMENT_CONFIG="dynlaneseq_eg/configs/culane_v31_selection_bridge_35k_to50k.yaml"
CONTROL_CHECKPOINT="${ROOT}/train/field_only/iter_0050000.pt"
TREATMENT_CHECKPOINT="${ROOT}/train/selection_bridge/iter_0050000.pt"
CONTROL_METRICS="${ROOT}/reports/field_only_full_val/metrics.json"
TREATMENT_METRICS="${ROOT}/reports/selection_bridge_full_val/metrics.json"
REFINER_PREFIX="structured_query_head.set_selection_head.slot_refinement"

CONTROL_WITH_TREATMENT="${REPLAY_ROOT}/control_selection_treatment_refiner"
TREATMENT_WITH_CONTROL="${REPLAY_ROOT}/treatment_selection_control_refiner"

for required in \
  "${CONTROL_CHECKPOINT}" \
  "${TREATMENT_CHECKPOINT}" \
  "${CONTROL_METRICS}" \
  "${TREATMENT_METRICS}" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing V31 refiner-replay artifact: ${required}" >&2
    exit 1
  fi
done

evaluate_hybrid() {
  local config="$1"
  local checkpoint="$2"
  local refiner_checkpoint="$3"
  local output_dir="$4"
  if [[ -s "${output_dir}/metrics.json" ]]; then
    return
  fi
  mkdir -p "${output_dir}"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --module-override-checkpoint "${refiner_checkpoint}" \
    --module-override-prefix "${REFINER_PREFIX}" \
    --split val \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --score-mode four_slot \
    --score-thresh 0.0 \
    --quality-score-power 0.0 \
    --top-k 4 \
    --nms-distance-thresh-px 0.0 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --iou-thresholds 0.5 0.75 \
    --pred-dir "${output_dir}/predictions" \
    --output-txt "${output_dir}/metrics.txt" \
    --output-json "${output_dir}/metrics.json" \
    --no-pretrained-init \
    --amp-dtype none \
    2>&1 | tee "${output_dir}/eval.log"
}

# A and D already exist from the exact-paired pipeline. Only the two crossed
# policies require inference. The base checkpoint owns all upstream state and
# selection; the donor contributes only exact refiner parameters.
evaluate_hybrid \
  "${CONTROL_CONFIG}" "${CONTROL_CHECKPOINT}" "${TREATMENT_CHECKPOINT}" \
  "${CONTROL_WITH_TREATMENT}"
evaluate_hybrid \
  "${TREATMENT_CONFIG}" "${TREATMENT_CHECKPOINT}" "${CONTROL_CHECKPOINT}" \
  "${TREATMENT_WITH_CONTROL}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v31_refiner_2x2 \
  --control-selection-control-refiner "${CONTROL_METRICS}" \
  --control-selection-treatment-refiner "${CONTROL_WITH_TREATMENT}/metrics.json" \
  --treatment-selection-control-refiner "${TREATMENT_WITH_CONTROL}/metrics.json" \
  --treatment-selection-treatment-refiner "${TREATMENT_METRICS}" \
  --output-json "${REPLAY_ROOT}/refiner_2x2_summary.json" \
  2>&1 | tee "${REPLAY_ROOT}/refiner_2x2_summary.log"

echo "V31 50K refiner 2x2 replay complete. Test split remained closed."
