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

V7_CONFIG="dynlaneseq_eg/configs/culane_v7_exact_control_35k_to50k.yaml"
PULSE_CONFIG="dynlaneseq_eg/configs/culane_v32_auxiliary_pulse_consolidation_35k_to50k.yaml"
V7_SOURCE="outputs/diagnostics/v30_field_only_exact_pair_30k_to35k/train/control/iter_0035000.pt"
FIELD_SOURCE="outputs/diagnostics/v30_field_only_exact_pair_30k_to35k/train/field_only/iter_0035000.pt"
BRIDGE_SOURCE="outputs/diagnostics/v31_selection_gradient_bridge_exact_pair_30k_to35k/train/selection_bridge/iter_0035000.pt"
V7_ROOT="outputs/diagnostics/v30_field_only_vs_v7_exact_35k_to50k"
V7_ENDPOINT="${V7_ROOT}/train/v7_control/iter_0050000.pt"
V7_VAL_DIR="${V7_ROOT}/reports/v7_exact_full_val"
V7_METRICS="${V7_VAL_DIR}/metrics.json"
V7_FULL_COVERAGE="${V7_ROOT}/field_decay_autopsy/full_stage/v7_exact_full_stage_coverage.json"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v32_auxiliary_pulse_exact_35k_to50k}"
FIELD_DIR="${OUTPUT_ROOT}/train/field_pulse"
BRIDGE_DIR="${OUTPUT_ROOT}/train/bridge_pulse"
FIELD_ENDPOINT="${FIELD_DIR}/iter_0050000.pt"
BRIDGE_ENDPOINT="${BRIDGE_DIR}/iter_0050000.pt"
CONTRACT="${OUTPUT_ROOT}/audits/exact_pulse_contract.json"
FIELD_VAL_DIR="${OUTPUT_ROOT}/reports/field_pulse_full_val"
BRIDGE_VAL_DIR="${OUTPUT_ROOT}/reports/bridge_pulse_full_val"
FIELD_METRICS="${FIELD_VAL_DIR}/metrics.json"
BRIDGE_METRICS="${BRIDGE_VAL_DIR}/metrics.json"
FIELD_UNIFORM="${OUTPUT_ROOT}/reports/field_pulse_uniform256.json"
BRIDGE_UNIFORM="${OUTPUT_ROOT}/reports/bridge_pulse_uniform256.json"
FIELD_FULL_COVERAGE="${OUTPUT_ROOT}/stage_autopsy/field_pulse_full_stage_coverage.json"
BRIDGE_FULL_COVERAGE="${OUTPUT_ROOT}/stage_autopsy/bridge_pulse_full_stage_coverage.json"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V7_CONFIG}" "${PULSE_CONFIG}" "${V7_SOURCE}" "${FIELD_SOURCE}" \
  "${BRIDGE_SOURCE}" "${V7_ENDPOINT}" "${V7_METRICS}" \
  "${V7_FULL_COVERAGE}" "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing V32 pulse artifact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/stage_autopsy" "${OUTPUT_ROOT}/trajectory" \
  "${FIELD_DIR}" "${BRIDGE_DIR}"

if [[ ! -s "${CONTRACT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v32_pulse_contract \
    --v7-config "${V7_CONFIG}" \
    --pulse-config "${PULSE_CONFIG}" \
    --v7-source "${V7_SOURCE}" \
    --v7-endpoint "${V7_ENDPOINT}" \
    --field-source "${FIELD_SOURCE}" \
    --bridge-source "${BRIDGE_SOURCE}" \
    --rng-reference-checkpoint "${FIELD_SOURCE}" \
    --dataset-root "${DATA_ROOT}" \
    --start-iteration "${SOURCE_ITERATION}" \
    --end-iteration "${ENDPOINT_ITERATION}" \
    --optimizer-steps "${TRAINING_STEPS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --output-json "${CONTRACT}" \
    2>&1 | tee "${OUTPUT_ROOT}/audits/exact_pulse_contract.log"
fi
"${PYTHON}" - "${CONTRACT}" <<'PY'
import json, sys
if json.load(open(sys.argv[1])).get("passed") is not True:
    raise SystemExit("V32 pulse preflight failed")
PY

train_arm() {
  local label="$1"
  local source="$2"
  local output_dir="$3"
  local endpoint="$4"
  local log="${output_dir}/train.log"
  if [[ ! -f "${endpoint}" ]]; then
    local resume_checkpoint="${source}"
    local resume_iteration="${SOURCE_ITERATION}"
    local candidate
    while IFS= read -r candidate; do
      local candidate_iteration
      candidate_iteration="$(checkpoint_iteration "${candidate}")"
      if (( candidate_iteration > resume_iteration && candidate_iteration < ENDPOINT_ITERATION )); then
        resume_checkpoint="${candidate}"
        resume_iteration="${candidate_iteration}"
      fi
    done < <(find "${output_dir}" -maxdepth 1 -type f -name 'iter_*.pt' | sort)

    local remaining_steps=$(( ENDPOINT_ITERATION - resume_iteration ))
    local rng_args=()
    if (( resume_iteration == SOURCE_ITERATION )); then
      rng_args=(--resume-rng-from "${FIELD_SOURCE}")
    fi
    echo "${label}: global ${resume_iteration} -> ${ENDPOINT_ITERATION}; auxiliary losses OFF" | tee -a "${log}"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
      --config "${PULSE_CONFIG}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --output-dir "${output_dir}" \
      --resume "${resume_checkpoint}" \
      "${rng_args[@]}" \
      --max-iters "${remaining_steps}" \
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
    echo "${label} did not prove restoration of the common RNG reference." >&2
    exit 1
  fi
}

# One 5090 is available, so arms run sequentially.  Both restore the same V30
# 35K RNG reference and consume the same iteration-addressed augmentation stream.
train_arm "field-pulse" "${FIELD_SOURCE}" "${FIELD_DIR}" "${FIELD_ENDPOINT}"
train_arm "field+bridge-pulse" "${BRIDGE_SOURCE}" "${BRIDGE_DIR}" "${BRIDGE_ENDPOINT}"

uniform_coverage() {
  local label="$1"
  local checkpoint="$2"
  local output_json="$3"
  if [[ ! -s "${output_json}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${PULSE_CONFIG}" \
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

# 40K/45K are trajectory diagnostics only.  The decision endpoint remains 50K.
for iteration in 40000 45000; do
  checkpoint_name="$(printf 'iter_%07d.pt' "${iteration}")"
  uniform_coverage \
    "field_pulse_${iteration}" \
    "${FIELD_DIR}/${checkpoint_name}" \
    "${OUTPUT_ROOT}/trajectory/field_pulse_${iteration}_uniform256.json"
  uniform_coverage \
    "bridge_pulse_${iteration}" \
    "${BRIDGE_DIR}/${checkpoint_name}" \
    "${OUTPUT_ROOT}/trajectory/bridge_pulse_${iteration}_uniform256.json"
done
uniform_coverage "field_pulse_50000" "${FIELD_ENDPOINT}" "${FIELD_UNIFORM}"
uniform_coverage "bridge_pulse_50000" "${BRIDGE_ENDPOINT}" "${BRIDGE_UNIFORM}"

evaluate_arm() {
  local checkpoint="$1"
  local output_dir="$2"
  local metrics="${output_dir}/metrics.json"
  if [[ ! -s "${metrics}" ]]; then
    mkdir -p "${output_dir}"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
      --config "${PULSE_CONFIG}" \
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

evaluate_arm "${FIELD_ENDPOINT}" "${FIELD_VAL_DIR}"
evaluate_arm "${BRIDGE_ENDPOINT}" "${BRIDGE_VAL_DIR}"

full_coverage() {
  local label="$1"
  local checkpoint="$2"
  local output_json="$3"
  if [[ ! -s "${output_json}" ]]; then
    mkdir -p "${OUTPUT_ROOT}/stage_autopsy/cache/${label}"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${PULSE_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --list-path "${DATA_ROOT}/list/val.txt" \
      --device "${DEVICE}" \
      --cache-dir "${OUTPUT_ROOT}/stage_autopsy/cache/${label}" \
      --max-batches 0 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" \
      --amp-dtype none \
      --sample-strategy sequential \
      --top-k 4 \
      --iou-thresholds 0.50 0.75 \
      --output-json "${output_json}" \
      2>&1 | tee "${output_json%.json}.log"
  fi
}

full_coverage "field_pulse" "${FIELD_ENDPOINT}" "${FIELD_FULL_COVERAGE}"
full_coverage "bridge_pulse" "${BRIDGE_ENDPOINT}" "${BRIDGE_FULL_COVERAGE}"

paired_audit() {
  local name="$1"
  local source_pred="$2"
  local candidate_pred="$3"
  local source_metrics="$4"
  local candidate_metrics="$5"
  local uniform_report="$6"
  local output_json="$7"
  if [[ ! -s "${output_json}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_full_validation_paired_effect \
      --experiment-name "${name}" \
      --source-pred-dir "${source_pred}" \
      --candidate-pred-dir "${candidate_pred}" \
      --dataset-root "${DATA_ROOT}" \
      --list-path "${DATA_ROOT}/list/val.txt" \
      --uniform-report "${uniform_report}" \
      --source-metrics "${source_metrics}" \
      --candidate-metrics "${candidate_metrics}" \
      --iou-thresholds 0.5 0.75 \
      --workers "${METRIC_WORKERS}" \
      --bootstrap-samples 10000 \
      --bootstrap-seed "${SEED}" \
      --output-json "${output_json}"
  fi
}

transition_audit() {
  local name="$1"
  local source_pred="$2"
  local candidate_pred="$3"
  local output_json="$4"
  if [[ ! -s "${output_json}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_paired_gt_transitions \
      --experiment-name "${name}" \
      --source-pred-dir "${source_pred}" \
      --candidate-pred-dir "${candidate_pred}" \
      --dataset-root "${DATA_ROOT}" \
      --list-path "${DATA_ROOT}/list/val.txt" \
      --iou-thresholds 0.5 0.75 \
      --workers "${METRIC_WORKERS}" \
      --output-json "${output_json}"
  fi
}

V7_FIELD_PAIRED="${OUTPUT_ROOT}/paired_v7_vs_field_pulse.json"
V7_BRIDGE_PAIRED="${OUTPUT_ROOT}/paired_v7_vs_bridge_pulse.json"
FIELD_BRIDGE_PAIRED="${OUTPUT_ROOT}/paired_field_pulse_vs_bridge_pulse.json"
V7_FIELD_TRANSITIONS="${OUTPUT_ROOT}/transitions_v7_vs_field_pulse.json"
V7_BRIDGE_TRANSITIONS="${OUTPUT_ROOT}/transitions_v7_vs_bridge_pulse.json"
FIELD_BRIDGE_TRANSITIONS="${OUTPUT_ROOT}/transitions_field_pulse_vs_bridge_pulse.json"

paired_audit "V32 exact V7 versus Field Pulse at 50K" \
  "${V7_VAL_DIR}/predictions" "${FIELD_VAL_DIR}/predictions" \
  "${V7_METRICS}" "${FIELD_METRICS}" "${FIELD_UNIFORM}" "${V7_FIELD_PAIRED}"
paired_audit "V32 exact V7 versus Field+Bridge Pulse at 50K" \
  "${V7_VAL_DIR}/predictions" "${BRIDGE_VAL_DIR}/predictions" \
  "${V7_METRICS}" "${BRIDGE_METRICS}" "${BRIDGE_UNIFORM}" "${V7_BRIDGE_PAIRED}"
paired_audit "V32 Field Pulse versus Field+Bridge Pulse at 50K" \
  "${FIELD_VAL_DIR}/predictions" "${BRIDGE_VAL_DIR}/predictions" \
  "${FIELD_METRICS}" "${BRIDGE_METRICS}" "${BRIDGE_UNIFORM}" "${FIELD_BRIDGE_PAIRED}"

transition_audit "V32 exact V7 versus Field Pulse transitions" \
  "${V7_VAL_DIR}/predictions" "${FIELD_VAL_DIR}/predictions" "${V7_FIELD_TRANSITIONS}"
transition_audit "V32 exact V7 versus Field+Bridge Pulse transitions" \
  "${V7_VAL_DIR}/predictions" "${BRIDGE_VAL_DIR}/predictions" "${V7_BRIDGE_TRANSITIONS}"
transition_audit "V32 Field Pulse versus Field+Bridge Pulse transitions" \
  "${FIELD_VAL_DIR}/predictions" "${BRIDGE_VAL_DIR}/predictions" "${FIELD_BRIDGE_TRANSITIONS}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v32_pulse \
  --v7-metrics "${V7_METRICS}" \
  --field-pulse-metrics "${FIELD_METRICS}" \
  --bridge-pulse-metrics "${BRIDGE_METRICS}" \
  --v7-coverage "${V7_FULL_COVERAGE}" \
  --field-pulse-coverage "${FIELD_FULL_COVERAGE}" \
  --bridge-pulse-coverage "${BRIDGE_FULL_COVERAGE}" \
  --v7-vs-field-paired "${V7_FIELD_PAIRED}" \
  --v7-vs-bridge-paired "${V7_BRIDGE_PAIRED}" \
  --field-vs-bridge-paired "${FIELD_BRIDGE_PAIRED}" \
  --v7-vs-field-transitions "${V7_FIELD_TRANSITIONS}" \
  --v7-vs-bridge-transitions "${V7_BRIDGE_TRANSITIONS}" \
  --field-vs-bridge-transitions "${FIELD_BRIDGE_TRANSITIONS}" \
  --pair-contract "${CONTRACT}" \
  --output-json "${OUTPUT_ROOT}/pulse_summary.json" \
  2>&1 | tee "${OUTPUT_ROOT}/pulse_summary.log"

echo "V32 auxiliary-pulse consolidation pipeline complete. Test split remained closed."
