#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/venv/clrernet/bin/python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION=35000
ENDPOINT_ITERATION=50000
TRAINING_STEPS=15000
BATCH_SIZE=4
GRAD_ACCUM=4
NUM_WORKERS="${NUM_WORKERS:-12}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"

CONTROL_CONFIG="dynlaneseq_eg/configs/culane_v30_field_only_35k_to50k.yaml"
TREATMENT_CONFIG="dynlaneseq_eg/configs/culane_v31_selection_bridge_35k_to50k.yaml"
CONTROL_SOURCE="outputs/diagnostics/v30_field_only_exact_pair_30k_to35k/train/field_only/iter_0035000.pt"
TREATMENT_SOURCE="outputs/diagnostics/v31_selection_gradient_bridge_exact_pair_30k_to35k/train/selection_bridge/iter_0035000.pt"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v31_selection_bridge_exact_pair_35k_to50k}"

CONTROL_DIR="${OUTPUT_ROOT}/train/field_only"
TREATMENT_DIR="${OUTPUT_ROOT}/train/selection_bridge"
CONTROL_ENDPOINT="${CONTROL_DIR}/iter_0050000.pt"
TREATMENT_ENDPOINT="${TREATMENT_DIR}/iter_0050000.pt"
PAIR_CONTRACT="${OUTPUT_ROOT}/audits/exact_pair_contract.json"
CONTROL_COVERAGE="${OUTPUT_ROOT}/reports/field_only_uniform256.json"
TREATMENT_COVERAGE="${OUTPUT_ROOT}/reports/selection_bridge_uniform256.json"
CONTROL_VAL_DIR="${OUTPUT_ROOT}/reports/field_only_full_val"
TREATMENT_VAL_DIR="${OUTPUT_ROOT}/reports/selection_bridge_full_val"
CONTROL_METRICS="${CONTROL_VAL_DIR}/metrics.json"
TREATMENT_METRICS="${TREATMENT_VAL_DIR}/metrics.json"
PAIRED_AUDIT="${OUTPUT_ROOT}/paired_field_only_vs_selection_bridge.json"
TRANSITION_AUDIT="${OUTPUT_ROOT}/paired_field_only_vs_selection_bridge_transitions.json"
SUMMARY="${OUTPUT_ROOT}/exact_pair_summary.json"
AUTOPSY_ROOT="${OUTPUT_ROOT}/stage_autopsy"

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
  "${CONTROL_SOURCE}" \
  "${TREATMENT_SOURCE}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V31 35K-to-50K artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${CONTROL_SOURCE}") != SOURCE_ITERATION )); then
  echo "Control source is not global iteration 35K." >&2
  exit 1
fi
if (( $(checkpoint_iteration "${TREATMENT_SOURCE}") != SOURCE_ITERATION )); then
  echo "Treatment source is not global iteration 35K." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports" \
  "${CONTROL_DIR}" "${TREATMENT_DIR}" "${AUTOPSY_ROOT}"

if [[ ! -s "${PAIR_CONTRACT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v31_continuation_pair_contract \
    --control-config "${CONTROL_CONFIG}" \
    --treatment-config "${TREATMENT_CONFIG}" \
    --control-checkpoint "${CONTROL_SOURCE}" \
    --treatment-checkpoint "${TREATMENT_SOURCE}" \
    --rng-reference-checkpoint "${CONTROL_SOURCE}" \
    --dataset-root "${DATA_ROOT}" \
    --start-iteration "${SOURCE_ITERATION}" \
    --optimizer-steps "${TRAINING_STEPS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --output-json "${PAIR_CONTRACT}" \
    2>&1 | tee "${OUTPUT_ROOT}/audits/exact_pair_contract.log"
fi
"${PYTHON}" - "${PAIR_CONTRACT}" <<'PY'
import json, sys
if json.load(open(sys.argv[1])).get("passed") is not True:
    raise SystemExit("35K-to-50K exact-pair preflight failed")
PY

train_arm() {
  local label="$1"
  local config="$2"
  local source="$3"
  local output_dir="$4"
  local endpoint="$5"
  local log="${output_dir}/train.log"
  if [[ ! -f "${endpoint}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
      --config "${config}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --output-dir "${output_dir}" \
      --resume "${source}" \
      --resume-rng-from "${CONTROL_SOURCE}" \
      --max-iters "${TRAINING_STEPS}" \
      --checkpoint-interval 5000 \
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
    echo "${label} global-50K endpoint is missing or invalid." >&2
    exit 1
  fi
  if ! grep -Eq "resume_rng_override.*restored.*True" "${log}"; then
    echo "${label} did not prove common RNG restoration." >&2
    exit 1
  fi
}

# Sequential arms share one GPU. Both restore the V30 35K RNG reference and
# consume the identical iteration-addressed sample/augmentation stream.
train_arm "field-only" "${CONTROL_CONFIG}" "${CONTROL_SOURCE}" \
  "${CONTROL_DIR}" "${CONTROL_ENDPOINT}"
train_arm "selection-bridge" "${TREATMENT_CONFIG}" "${TREATMENT_SOURCE}" \
  "${TREATMENT_DIR}" "${TREATMENT_ENDPOINT}"

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
  local config="$1"
  local checkpoint="$2"
  local output_dir="$3"
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

coverage_arm "field_only" "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}" "${CONTROL_COVERAGE}"
coverage_arm "selection_bridge" "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}" "${TREATMENT_COVERAGE}"
evaluate_arm "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}" "${CONTROL_VAL_DIR}"
evaluate_arm "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}" "${TREATMENT_VAL_DIR}"

if [[ ! -s "${PAIRED_AUDIT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_full_validation_paired_effect \
    --experiment-name "V31 exact-paired 35K-to-50K field-only vs selection bridge" \
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
    --bootstrap-seed "${SEED}" \
    --output-json "${PAIRED_AUDIT}"
fi

if [[ ! -s "${TRANSITION_AUDIT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_paired_gt_transitions \
    --experiment-name "V31 35K-to-50K field-only vs selection bridge transitions" \
    --source-pred-dir "${CONTROL_VAL_DIR}/predictions" \
    --candidate-pred-dir "${TREATMENT_VAL_DIR}/predictions" \
    --dataset-root "${DATA_ROOT}" \
    --list-path "${DATA_ROOT}/list/val.txt" \
    --iou-thresholds 0.5 0.75 \
    --workers "${METRIC_WORKERS}" \
    --output-json "${TRANSITION_AUDIT}"
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v30_exact_pair \
  --control-metrics "${CONTROL_METRICS}" \
  --treatment-metrics "${TREATMENT_METRICS}" \
  --control-coverage "${CONTROL_COVERAGE}" \
  --treatment-coverage "${TREATMENT_COVERAGE}" \
  --paired-audit "${PAIRED_AUDIT}" \
  --transition-audit "${TRANSITION_AUDIT}" \
  --output-json "${SUMMARY}"

full_coverage_arm() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local cache_dir="${AUTOPSY_ROOT}/cache/${label}"
  local report="${AUTOPSY_ROOT}/${label}_full_stage_coverage.json"
  mkdir -p "${cache_dir}"
  if [[ ! -s "${report}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --list-path "${DATA_ROOT}/list/val.txt" \
      --device "${DEVICE}" \
      --cache-dir "${cache_dir}" \
      --max-batches 0 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" \
      --amp-dtype none \
      --sample-strategy sequential \
      --top-k 4 \
      --iou-thresholds 0.50 0.75 \
      --hard-diversity-distances 20 \
      --mmr-penalties 0.75 \
      --mmr-sigmas 20 \
      --output-json "${report}" \
      2>&1 | tee "${report%.json}.log"
  fi
}

full_coverage_arm "field_only" "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}"
full_coverage_arm "selection_bridge" "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}"

mapfile -t CONTROL_CACHES < <(find "${AUTOPSY_ROOT}/cache/field_only" -type f -name '*.pt' | sort)
mapfile -t TREATMENT_CACHES < <(find "${AUTOPSY_ROOT}/cache/selection_bridge" -type f -name '*.pt' | sort)
if (( ${#CONTROL_CACHES[@]} != 1 || ${#TREATMENT_CACHES[@]} != 1 )); then
  echo "Expected exactly one complete full-stage cache per arm." >&2
  exit 1
fi
FACTORIAL_REPORT="${AUTOPSY_ROOT}/field_only_vs_selection_bridge_factorial.json"
if [[ ! -s "${FACTORIAL_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_activity_geometry_factorial \
    --source-cache "${CONTROL_CACHES[0]}" \
    --candidate-cache "${TREATMENT_CACHES[0]}" \
    --iou-thresholds 0.50 0.75 \
    --output-json "${FACTORIAL_REPORT}" \
    2>&1 | tee "${FACTORIAL_REPORT%.json}.log"
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v30_stage_autopsy \
  --control-coverage "${AUTOPSY_ROOT}/field_only_full_stage_coverage.json" \
  --treatment-coverage "${AUTOPSY_ROOT}/selection_bridge_full_stage_coverage.json" \
  --factorial-audit "${FACTORIAL_REPORT}" \
  --exact-pair-summary "${SUMMARY}" \
  --output-json "${AUTOPSY_ROOT}/stage_autopsy_summary.json" \
  2>&1 | tee "${AUTOPSY_ROOT}/stage_autopsy_summary.log"

echo "V31 exact-paired 35K-to-50K pipeline complete. Test split remained closed."
