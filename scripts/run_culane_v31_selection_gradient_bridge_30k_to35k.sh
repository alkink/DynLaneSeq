#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION=30000
ENDPOINT_ITERATION=35000
TRAINING_STEPS=5000
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_v30_joint_slot_field_35k_route_residual_off.yaml}"
TREATMENT_CONFIG="${TREATMENT_CONFIG:-dynlaneseq_eg/configs/culane_v31_field_only_selection_gradient_bridge_30k_to35k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/source_checkpoints/v7_resume_safe_iter_0030000.pt}"
CONTROL_ROOT="${CONTROL_ROOT:-outputs/diagnostics/v30_field_only_exact_pair_30k_to35k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v31_selection_gradient_bridge_exact_pair_30k_to35k}"

CONTROL_ENDPOINT="${CONTROL_ROOT}/train/field_only/iter_0035000.pt"
CONTROL_COVERAGE="${CONTROL_ROOT}/reports/field_only_uniform256.json"
CONTROL_VAL_DIR="${CONTROL_ROOT}/reports/field_only_full_val"
CONTROL_METRICS="${CONTROL_VAL_DIR}/metrics.json"
V7_VAL_DIR="${CONTROL_ROOT}/reports/control_full_val"
V7_METRICS="${V7_VAL_DIR}/metrics.json"

TREATMENT_DIR="${OUTPUT_ROOT}/train/selection_bridge"
TREATMENT_ENDPOINT="${TREATMENT_DIR}/iter_0035000.pt"
PAIR_CONTRACT="${OUTPUT_ROOT}/audits/exact_pair_contract.json"
GRADIENT_CONTRACT="${OUTPUT_ROOT}/audits/gradient_bridge_contract.json"
TREATMENT_COVERAGE="${OUTPUT_ROOT}/reports/selection_bridge_uniform256.json"
TREATMENT_VAL_DIR="${OUTPUT_ROOT}/reports/selection_bridge_full_val"
TREATMENT_METRICS="${TREATMENT_VAL_DIR}/metrics.json"
PAIRED_FIELD="${OUTPUT_ROOT}/paired_field_only_vs_selection_bridge.json"
PAIRED_V7="${OUTPUT_ROOT}/paired_v7_vs_selection_bridge.json"
TRANSITION_FIELD="${OUTPUT_ROOT}/paired_field_only_vs_selection_bridge_transitions.json"
SUMMARY="${OUTPUT_ROOT}/exact_pair_summary.json"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${CONTROL_CONFIG}" \
  "${TREATMENT_CONFIG}" \
  "${SOURCE_CHECKPOINT}" \
  "${CONTROL_ENDPOINT}" \
  "${CONTROL_COVERAGE}" \
  "${CONTROL_METRICS}" \
  "${V7_METRICS}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V31 artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_CHECKPOINT is not iteration 30000." >&2
  exit 1
fi
if (( $(checkpoint_iteration "${CONTROL_ENDPOINT}") != ENDPOINT_ITERATION )); then
  echo "Field-only control is not the global-35K endpoint." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V31 exact pair must preserve V7 effective batch 16." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports" "${TREATMENT_DIR}"

if [[ ! -s "${PAIR_CONTRACT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v31_exact_pair_contract \
    --control-config "${CONTROL_CONFIG}" \
    --treatment-config "${TREATMENT_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --start-iteration "${SOURCE_ITERATION}" \
    --optimizer-steps "${TRAINING_STEPS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --output-json "${PAIR_CONTRACT}" \
    2>&1 | tee "${OUTPUT_ROOT}/audits/exact_pair_contract.log"
fi

if [[ ! -s "${GRADIENT_CONTRACT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_joint_slot_field_contract \
    --source-config "${CONTROL_CONFIG}" \
    --treatment-config "${TREATMENT_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --start-iteration "${SOURCE_ITERATION}" \
    --field-only \
    --selection-gradient-bridge \
    --output-json "${GRADIENT_CONTRACT}" \
    2>&1 | tee "${OUTPUT_ROOT}/audits/gradient_bridge_contract.log"
fi

"${PYTHON}" - "${PAIR_CONTRACT}" "${GRADIENT_CONTRACT}" <<'PY'
import json, sys
for path in sys.argv[1:]:
    if json.load(open(path)).get("passed") is not True:
        raise SystemExit(f"V31 preflight failed: {path}")
PY

TRAIN_LOG="${TREATMENT_DIR}/train.log"
if [[ ! -f "${TREATMENT_ENDPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
    --config "${TREATMENT_CONFIG}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --output-dir "${TREATMENT_DIR}" \
    --resume "${SOURCE_CHECKPOINT}" \
    --resume-remap-optimizer-groups \
    --max-iters "${TRAINING_STEPS}" \
    --checkpoint-interval "${TRAINING_STEPS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --seg-aux-amp-dtype bfloat16 \
    --compile-model true \
    --resume-safe-data true \
    2>&1 | tee -a "${TRAIN_LOG}"
fi
if [[ ! -f "${TREATMENT_ENDPOINT}" ]] || \
   (( $(checkpoint_iteration "${TREATMENT_ENDPOINT}") != ENDPOINT_ITERATION )); then
  echo "V31 global-35K endpoint is missing or invalid." >&2
  exit 1
fi
if ! grep -Eq "rng_state_restored_after_optimizer_remap.*True" "${TRAIN_LOG}"; then
  echo "V31 did not prove checkpoint RNG restoration." >&2
  exit 1
fi

if [[ ! -s "${TREATMENT_COVERAGE}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${TREATMENT_CONFIG}" \
    --checkpoint "${TREATMENT_ENDPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --list-path "${DATA_ROOT}/list/val.txt" \
    --device "${DEVICE}" \
    --cache-dir "${OUTPUT_ROOT}/cache/selection_bridge_uniform256" \
    --max-batches 64 \
    --eval-batch-size 4 \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --sample-strategy uniform \
    --top-k 4 \
    --iou-thresholds 0.50 0.75 \
    --output-json "${TREATMENT_COVERAGE}"
fi

if [[ ! -s "${TREATMENT_METRICS}" ]]; then
  mkdir -p "${TREATMENT_VAL_DIR}"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${TREATMENT_CONFIG}" \
    --checkpoint "${TREATMENT_ENDPOINT}" \
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
    --pred-dir "${TREATMENT_VAL_DIR}/predictions" \
    --output-txt "${TREATMENT_VAL_DIR}/metrics.txt" \
    --output-json "${TREATMENT_METRICS}" \
    --no-pretrained-init \
    --amp-dtype none \
    2>&1 | tee "${TREATMENT_VAL_DIR}/eval.log"
fi

paired_audit() {
  local name="$1"
  local source_dir="$2"
  local source_metrics="$3"
  local output="$4"
  if [[ ! -s "${output}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_full_validation_paired_effect \
      --experiment-name "${name}" \
      --source-pred-dir "${source_dir}/predictions" \
      --candidate-pred-dir "${TREATMENT_VAL_DIR}/predictions" \
      --dataset-root "${DATA_ROOT}" \
      --list-path "${DATA_ROOT}/list/val.txt" \
      --uniform-report "${TREATMENT_COVERAGE}" \
      --source-metrics "${source_metrics}" \
      --candidate-metrics "${TREATMENT_METRICS}" \
      --iou-thresholds 0.5 0.75 \
      --workers "${METRIC_WORKERS}" \
      --bootstrap-samples 10000 \
      --bootstrap-seed "${SEED}" \
      --output-json "${output}"
  fi
}

paired_audit \
  "V31 exact-paired field-only vs selection-gradient bridge" \
  "${CONTROL_VAL_DIR}" "${CONTROL_METRICS}" "${PAIRED_FIELD}"
paired_audit \
  "V31 exact-paired V7 vs selection-gradient bridge" \
  "${V7_VAL_DIR}" "${V7_METRICS}" "${PAIRED_V7}"

if [[ ! -s "${TRANSITION_FIELD}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_paired_gt_transitions \
    --experiment-name "V31 field-only vs selection-gradient bridge transitions" \
    --source-pred-dir "${CONTROL_VAL_DIR}/predictions" \
    --candidate-pred-dir "${TREATMENT_VAL_DIR}/predictions" \
    --dataset-root "${DATA_ROOT}" \
    --list-path "${DATA_ROOT}/list/val.txt" \
    --iou-thresholds 0.5 0.75 \
    --workers "${METRIC_WORKERS}" \
    --output-json "${TRANSITION_FIELD}"
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v30_exact_pair \
  --control-metrics "${CONTROL_METRICS}" \
  --treatment-metrics "${TREATMENT_METRICS}" \
  --control-coverage "${CONTROL_COVERAGE}" \
  --treatment-coverage "${TREATMENT_COVERAGE}" \
  --paired-audit "${PAIRED_FIELD}" \
  --transition-audit "${TRANSITION_FIELD}" \
  --output-json "${SUMMARY}"

echo "V31 selection-gradient bridge exact-pair gate complete. Test remained closed."
