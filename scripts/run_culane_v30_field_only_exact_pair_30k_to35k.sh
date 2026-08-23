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

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
TREATMENT_CONFIG="${TREATMENT_CONFIG:-dynlaneseq_eg/configs/culane_v30_joint_slot_field_35k_route_residual_off.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0030000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v30_field_only_exact_pair_30k_to35k}"

CONTROL_DIR="${OUTPUT_ROOT}/train/control"
TREATMENT_DIR="${OUTPUT_ROOT}/train/field_only"
CONTROL_ENDPOINT="${CONTROL_DIR}/iter_0035000.pt"
TREATMENT_ENDPOINT="${TREATMENT_DIR}/iter_0035000.pt"
PAIR_CONTRACT="${OUTPUT_ROOT}/audits/exact_pair_contract.json"
FIELD_CONTRACT="${OUTPUT_ROOT}/audits/field_zero_step_contract.json"
CONTROL_COVERAGE="${OUTPUT_ROOT}/reports/control_uniform256.json"
TREATMENT_COVERAGE="${OUTPUT_ROOT}/reports/field_only_uniform256.json"
CONTROL_VAL_DIR="${OUTPUT_ROOT}/reports/control_full_val"
TREATMENT_VAL_DIR="${OUTPUT_ROOT}/reports/field_only_full_val"
CONTROL_METRICS="${CONTROL_VAL_DIR}/metrics.json"
TREATMENT_METRICS="${TREATMENT_VAL_DIR}/metrics.json"
PAIRED_AUDIT="${OUTPUT_ROOT}/paired_control_vs_field_only.json"
TRANSITION_AUDIT="${OUTPUT_ROOT}/paired_gt_transitions.json"
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
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing exact-pair artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_CHECKPOINT is not iteration 30000." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Exact pair must preserve V7 effective batch 16." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/reports" \
  "${CONTROL_DIR}" \
  "${TREATMENT_DIR}"

if [[ ! -s "${PAIR_CONTRACT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_exact_pair_contract \
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

if [[ ! -s "${FIELD_CONTRACT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_joint_slot_field_contract \
    --source-config "${CONTROL_CONFIG}" \
    --treatment-config "${TREATMENT_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --start-iteration "${SOURCE_ITERATION}" \
    --field-only \
    --output-json "${FIELD_CONTRACT}" \
    2>&1 | tee "${OUTPUT_ROOT}/audits/field_zero_step_contract.log"
fi

"${PYTHON}" - "${PAIR_CONTRACT}" "${FIELD_CONTRACT}" <<'PY'
import json, sys
for path in sys.argv[1:]:
    if json.load(open(path)).get("passed") is not True:
        raise SystemExit(f"Exact-pair preflight failed: {path}")
PY

train_arm() {
  local label="$1"
  local config="$2"
  local output_dir="$3"
  local endpoint="$4"
  local log="${output_dir}/train.log"
  if [[ ! -f "${endpoint}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
      --config "${config}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --output-dir "${output_dir}" \
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
      2>&1 | tee -a "${log}"
  fi
  if [[ ! -f "${endpoint}" ]] || \
     (( $(checkpoint_iteration "${endpoint}") != ENDPOINT_ITERATION )); then
    echo "${label} global-35K endpoint is missing or invalid." >&2
    exit 1
  fi
  if ! grep -Eq "rng_state_restored_after_optimizer_remap.*True" "${log}"; then
    echo "${label} did not prove checkpoint RNG restoration." >&2
    exit 1
  fi
}

# Sequential execution fits the single RTX 5070 Ti.  Both arms restore the
# exact same 30K RNG and consume the same iteration-addressed augmentation
# stream, so execution order cannot alter the pairing contract.
train_arm "control" "${CONTROL_CONFIG}" "${CONTROL_DIR}" "${CONTROL_ENDPOINT}"
train_arm "field-only" "${TREATMENT_CONFIG}" "${TREATMENT_DIR}" "${TREATMENT_ENDPOINT}"

coverage_arm() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local output_json="$4"
  if [[ ! -s "${output_json}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --list-path "${DATA_ROOT}/list/val.txt" \
      --device "${DEVICE}" \
      --cache-dir "${OUTPUT_ROOT}/cache/${label}_uniform256" \
      --max-batches 64 \
      --eval-batch-size 4 \
      --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" \
      --sample-strategy uniform \
      --top-k 4 \
      --iou-thresholds 0.50 0.75 \
      --output-json "${output_json}"
  fi
}

evaluate_arm() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local output_dir="$4"
  local metrics="${output_dir}/metrics.json"
  if [[ ! -s "${metrics}" ]]; then
    mkdir -p "${output_dir}"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
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
      --output-json "${metrics}" \
      --no-pretrained-init \
      --amp-dtype none \
      2>&1 | tee "${output_dir}/eval.log"
  fi
}

coverage_arm "control" "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}" "${CONTROL_COVERAGE}"
coverage_arm "field_only" "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}" "${TREATMENT_COVERAGE}"
evaluate_arm "control" "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}" "${CONTROL_VAL_DIR}"
evaluate_arm "field_only" "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}" "${TREATMENT_VAL_DIR}"

if [[ ! -s "${PAIRED_AUDIT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_full_validation_paired_effect \
    --experiment-name "V30 field-only exact-paired 30K-to-35K audit" \
    --source-pred-dir "${CONTROL_VAL_DIR}/predictions" \
    --candidate-pred-dir "${TREATMENT_VAL_DIR}/predictions" \
    --dataset-root "${DATA_ROOT}" \
    --list-path "${DATA_ROOT}/list/val.txt" \
    --uniform-report "${TREATMENT_COVERAGE}" \
    --source-metrics "${CONTROL_METRICS}" \
    --candidate-metrics "${TREATMENT_METRICS}" \
    --iou-thresholds 0.5 0.75 \
    --workers "${METRIC_WORKERS}" \
    --bootstrap-samples 10000 \
    --bootstrap-seed 3407 \
    --output-json "${PAIRED_AUDIT}" \
    2>&1 | tee "${OUTPUT_ROOT}/paired_control_vs_field_only.log"
fi

if [[ ! -s "${TRANSITION_AUDIT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_paired_gt_transitions \
    --experiment-name "V30 field-only exact-paired GT transitions" \
    --source-pred-dir "${CONTROL_VAL_DIR}/predictions" \
    --candidate-pred-dir "${TREATMENT_VAL_DIR}/predictions" \
    --dataset-root "${DATA_ROOT}" \
    --list-path "${DATA_ROOT}/list/val.txt" \
    --iou-thresholds 0.5 0.75 \
    --workers "${METRIC_WORKERS}" \
    --output-json "${TRANSITION_AUDIT}" \
    2>&1 | tee "${OUTPUT_ROOT}/paired_gt_transitions.log"
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v30_exact_pair \
  --control-metrics "${CONTROL_METRICS}" \
  --treatment-metrics "${TREATMENT_METRICS}" \
  --control-coverage "${CONTROL_COVERAGE}" \
  --treatment-coverage "${TREATMENT_COVERAGE}" \
  --paired-audit "${PAIRED_AUDIT}" \
  --transition-audit "${TRANSITION_AUDIT}" \
  --output-json "${SUMMARY}"

echo "V30 exact-paired 30K-to-35K gate complete. Test remained closed."
