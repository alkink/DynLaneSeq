#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION="${SOURCE_ITERATION:-225000}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_FIXED64="${RUN_FIXED64:-1}"
RUN_GENERALIZATION="${RUN_GENERALIZATION:-1}"
EVAL_TRAJECTORY="${EVAL_TRAJECTORY:-0}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-v8_1}"
ALLOW_FAILED_FIXED64_EXPLORATORY="${ALLOW_FAILED_FIXED64_EXPLORATORY:-0}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v8_1_geometry_router_state_control_225k_to227k.yaml}"
TREATMENT_CONFIG="${TREATMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v8_1_geometry_router_state_treatment_225k_to227k.yaml}"
CONTROL_MEMORY_CONFIG="${CONTROL_MEMORY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v8_1_geometry_router_state_control_memorize64.yaml}"
TREATMENT_MEMORY_CONFIG="${TREATMENT_MEMORY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v8_1_geometry_router_state_treatment_memorize64.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v8_1_geometry_router_state_gate_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/v8_1_geometry_router_state_gate_225k}"

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
  "${CONTROL_MEMORY_CONFIG}" \
  "${TREATMENT_MEMORY_CONFIG}" \
  "${SOURCE_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V8.1 causal-gate artefact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi
if (( TRAIN_STEPS != 2000 || CHECKPOINT_INTERVAL != 500 )); then
  echo "This causal gate is predeclared as 2000 steps with 500-step checkpoints." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Generalization arms require effective batch size 16." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports" "${CACHE_ROOT}"
POPULATION_REPORT="${OUTPUT_ROOT}/audits/population_contract_32b.json"
if [[ "${RUN_PREFLIGHT}" == "1" && ! -f "${POPULATION_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_1_geometry_router_state_contract \
    --control-config "${CONTROL_CONFIG}" \
    --treatment-config "${TREATMENT_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --num-workers 0 \
    --batches 32 \
    --start-iteration "${SOURCE_ITERATION}" \
    --output-json "${POPULATION_REPORT}"
fi
"${PYTHON}" - "${POPULATION_REPORT}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print({"v8_1_population_contract_passed": report.get("passed", False)})
if report.get("passed") is not True:
    raise SystemExit("V8.1 population contract failed; training remains closed")
PY

FIXED_LIST="${OUTPUT_ROOT}/train_uniform64.txt"
"${PYTHON}" - "${DATA_ROOT}/list/train_gt.txt" "${FIXED_LIST}" <<'PY'
import sys
from pathlib import Path
source = Path(sys.argv[1])
destination = Path(sys.argv[2])
lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
count = min(64, len(lines))
if count != 64:
    raise SystemExit(f"expected at least 64 training images, found {count}")
indices = [round(index * (len(lines) - 1) / 63) for index in range(64)]
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text("\n".join(lines[index] for index in indices) + "\n", encoding="utf-8")
print({"fixed_list": str(destination), "images": 64, "first": indices[0], "last": indices[-1]})
PY

train_arm() {
  local name="$1"
  local config="$2"
  local run_dir="$3"
  local grad_accum="$4"
  local train_list="$5"
  local end_iteration=$((SOURCE_ITERATION + TRAIN_STEPS))
  local end_tag
  end_tag="$(printf '%07d' "${end_iteration}")"
  local final_checkpoint="${run_dir}/iter_${end_tag}.pt"
  mkdir -p "${run_dir}"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "${name}: reusing ${final_checkpoint}"
    return
  fi

  local latest_checkpoint=""
  local latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${run_dir}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    local iteration
    iteration="$(checkpoint_iteration "${candidate}")"
    if (( iteration > latest_iteration && iteration < end_iteration )); then
      latest_checkpoint="${candidate}"
      latest_iteration="${iteration}"
    fi
  done

  local args=(
    --config "${config}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --output-dir "${run_dir}"
    --checkpoint-base "${SOURCE_CHECKPOINT}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${grad_accum}"
    --num-workers "${NUM_WORKERS}"
    --seg-aux-amp-dtype "${AMP_DTYPE}"
    --compile-model false
    --resume-safe-data true
  )
  if [[ -n "${train_list}" ]]; then
    args+=(--train-list "${train_list}")
  fi
  if [[ -n "${latest_checkpoint}" ]]; then
    args+=(
      --resume "${latest_checkpoint}"
      --max-iters "$((end_iteration - latest_iteration))"
    )
  else
    args+=(
      --init-from "${SOURCE_CHECKPOINT}"
      --init-iteration "${SOURCE_ITERATION}"
      --max-iters "${TRAIN_STEPS}"
    )
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${args[@]}" \
    2>&1 | tee -a "${run_dir}/train.log"
}

route_report() {
  local config="$1"
  local checkpoint="$2"
  local report="$3"
  local split="$4"
  local list_path="$5"
  local max_images="$6"
  local sample_strategy="$7"
  [[ -f "${report}" ]] && return
  local args=(
    --config "${config}"
    --checkpoint "${checkpoint}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --split "${split}"
    --sample-strategy "${sample_strategy}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --metric-workers "${METRIC_WORKERS}"
    --max-images "${max_images}"
    --iou-thresholds 0.50 0.75
    --output-json "${report}"
  )
  if [[ -n "${list_path}" ]]; then
    args+=(--list-path "${list_path}")
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_route_support_reference_policies "${args[@]}"
}

coverage_report() {
  local config="$1"
  local checkpoint="$2"
  local report="$3"
  local cache="$4"
  local split="$5"
  local list_path="$6"
  local max_batches="$7"
  local sample_strategy="$8"
  [[ -f "${report}" ]] && return
  local args=(
    --config "${config}"
    --checkpoint "${checkpoint}"
    --dataset-root "${DATA_ROOT}"
    --split "${split}"
    --device "${DEVICE}"
    --cache-dir "${cache}"
    --max-batches "${max_batches}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --metric-workers "${METRIC_WORKERS}"
    --amp-dtype none
    --sample-strategy "${sample_strategy}"
    --stage main
    --top-k 4
    --iou-thresholds 0.50 0.75
    --near-min-iou 0.30
    --line-width 30
    --min-valid-rows 5
    --hard-diversity-distances 20
    --mmr-sigmas 20
    --mmr-penalties 0.50
    --output-json "${report}"
  )
  if [[ -n "${list_path}" ]]; then
    args+=(--list-path "${list_path}")
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage "${args[@]}"
}

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"

if [[ "${RUN_FIXED64}" == "1" ]]; then
  MEMORY_CONTROL_DIR="${OUTPUT_ROOT}/memorize64/control"
  MEMORY_TREATMENT_DIR="${OUTPUT_ROOT}/memorize64/treatment"
  train_arm fixed64_control "${CONTROL_MEMORY_CONFIG}" "${MEMORY_CONTROL_DIR}" 1 "${FIXED_LIST}"
  train_arm fixed64_treatment "${TREATMENT_MEMORY_CONFIG}" "${MEMORY_TREATMENT_DIR}" 1 "${FIXED_LIST}"

  FIXED_SOURCE_ROUTE="${OUTPUT_ROOT}/reports/fixed64_source_route.json"
  FIXED_SOURCE_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_source_coverage.json"
  FIXED_CONTROL_ROUTE="${OUTPUT_ROOT}/reports/fixed64_control_route.json"
  FIXED_CONTROL_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_control_coverage.json"
  FIXED_TREATMENT_ROUTE="${OUTPUT_ROOT}/reports/fixed64_treatment_route.json"
  FIXED_TREATMENT_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_treatment_coverage.json"
  route_report "${CONTROL_MEMORY_CONFIG}" "${SOURCE_CHECKPOINT}" "${FIXED_SOURCE_ROUTE}" train "${FIXED_LIST}" 0 sequential
  coverage_report "${CONTROL_MEMORY_CONFIG}" "${SOURCE_CHECKPOINT}" "${FIXED_SOURCE_COVERAGE}" "${CACHE_ROOT}/fixed64_source" train "${FIXED_LIST}" 0 sequential
  route_report "${CONTROL_MEMORY_CONFIG}" "${MEMORY_CONTROL_DIR}/iter_${END_TAG}.pt" "${FIXED_CONTROL_ROUTE}" train "${FIXED_LIST}" 0 sequential
  coverage_report "${CONTROL_MEMORY_CONFIG}" "${MEMORY_CONTROL_DIR}/iter_${END_TAG}.pt" "${FIXED_CONTROL_COVERAGE}" "${CACHE_ROOT}/fixed64_control" train "${FIXED_LIST}" 0 sequential
  route_report "${TREATMENT_MEMORY_CONFIG}" "${MEMORY_TREATMENT_DIR}/iter_${END_TAG}.pt" "${FIXED_TREATMENT_ROUTE}" train "${FIXED_LIST}" 0 sequential
  coverage_report "${TREATMENT_MEMORY_CONFIG}" "${MEMORY_TREATMENT_DIR}/iter_${END_TAG}.pt" "${FIXED_TREATMENT_COVERAGE}" "${CACHE_ROOT}/fixed64_treatment" train "${FIXED_LIST}" 0 sequential

  set +e
  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v8_1_geometry_router_state_gate \
    --mode fixed64 \
    --population-contract "${POPULATION_REPORT}" \
    --source-route "${FIXED_SOURCE_ROUTE}" \
    --source-coverage "${FIXED_SOURCE_COVERAGE}" \
    --control-route "${FIXED_CONTROL_ROUTE}" \
    --control-coverage "${FIXED_CONTROL_COVERAGE}" \
    --treatment-route "${FIXED_TREATMENT_ROUTE}" \
    --treatment-coverage "${FIXED_TREATMENT_COVERAGE}" \
    --iteration "${END_ITERATION}" \
    --output-json "${OUTPUT_ROOT}/${SUMMARY_PREFIX}_fixed64_summary.json"
  fixed64_status=$?
  set -e
  if (( fixed64_status != 0 )) && [[ "${ALLOW_FAILED_FIXED64_EXPLORATORY}" != "1" ]]; then
    exit "${fixed64_status}"
  fi
elif [[ ! -f "${OUTPUT_ROOT}/${SUMMARY_PREFIX}_fixed64_summary.json" ]]; then
  echo "RUN_FIXED64=0 requires an existing fixed-64 summary." >&2
  exit 1
else
  set +e
  "${PYTHON}" - "${OUTPUT_ROOT}/${SUMMARY_PREFIX}_fixed64_summary.json" <<'PY'
import json
import sys
from pathlib import Path
if json.loads(Path(sys.argv[1]).read_text()).get("passed") is not True:
    raise SystemExit("existing fixed-64 gate failed")
PY
  fixed64_status=$?
  set -e
  if (( fixed64_status != 0 )) && [[ "${ALLOW_FAILED_FIXED64_EXPLORATORY}" != "1" ]]; then
    exit "${fixed64_status}"
  fi
fi

if [[ "${ALLOW_FAILED_FIXED64_EXPLORATORY}" == "1" ]]; then
  echo "EXPLORATORY ONLY: continuing past the recorded fixed-64 FAIL."
  echo "This does not authorize long training or alter the fixed-64 verdict."
fi

if [[ "${RUN_GENERALIZATION}" != "1" ]]; then
  echo "V8.1 fixed-64 gate complete; paired generalization remains paused."
  exit 0
fi

CONTROL_DIR="${OUTPUT_ROOT}/generalization/control"
TREATMENT_DIR="${OUTPUT_ROOT}/generalization/treatment"
train_arm generalization_control "${CONTROL_CONFIG}" "${CONTROL_DIR}" "${GRAD_ACCUM}" ""
train_arm generalization_treatment "${TREATMENT_CONFIG}" "${TREATMENT_DIR}" "${GRAD_ACCUM}" ""

VAL_MAX_IMAGES=256
VAL_MAX_BATCHES=$(((VAL_MAX_IMAGES + EVAL_BATCH_SIZE - 1) / EVAL_BATCH_SIZE))
VAL_SOURCE_ROUTE="${OUTPUT_ROOT}/reports/val_source_route_uniform256.json"
VAL_SOURCE_COVERAGE="${OUTPUT_ROOT}/reports/val_source_coverage_uniform256.json"
route_report "${CONTROL_CONFIG}" "${SOURCE_CHECKPOINT}" "${VAL_SOURCE_ROUTE}" val "" "${VAL_MAX_IMAGES}" uniform
coverage_report "${CONTROL_CONFIG}" "${SOURCE_CHECKPOINT}" "${VAL_SOURCE_COVERAGE}" "${CACHE_ROOT}/val_source" val "" "${VAL_MAX_BATCHES}" uniform

start_iteration="${END_ITERATION}"
if [[ "${EVAL_TRAJECTORY}" == "1" ]]; then
  start_iteration=$((SOURCE_ITERATION + CHECKPOINT_INTERVAL))
fi
for ((iteration=start_iteration; iteration<=END_ITERATION; iteration+=CHECKPOINT_INTERVAL)); do
  tag="$(printf '%07d' "${iteration}")"
  control_checkpoint="${CONTROL_DIR}/iter_${tag}.pt"
  treatment_checkpoint="${TREATMENT_DIR}/iter_${tag}.pt"
  control_route="${OUTPUT_ROOT}/reports/val_control_iter_${tag}_route_uniform256.json"
  treatment_route="${OUTPUT_ROOT}/reports/val_treatment_iter_${tag}_route_uniform256.json"
  control_coverage="${OUTPUT_ROOT}/reports/val_control_iter_${tag}_coverage_uniform256.json"
  treatment_coverage="${OUTPUT_ROOT}/reports/val_treatment_iter_${tag}_coverage_uniform256.json"
  route_report "${CONTROL_CONFIG}" "${control_checkpoint}" "${control_route}" val "" "${VAL_MAX_IMAGES}" uniform
  route_report "${TREATMENT_CONFIG}" "${treatment_checkpoint}" "${treatment_route}" val "" "${VAL_MAX_IMAGES}" uniform
  coverage_report "${CONTROL_CONFIG}" "${control_checkpoint}" "${control_coverage}" "${CACHE_ROOT}/val_control_${tag}" val "" "${VAL_MAX_BATCHES}" uniform
  coverage_report "${TREATMENT_CONFIG}" "${treatment_checkpoint}" "${treatment_coverage}" "${CACHE_ROOT}/val_treatment_${tag}" val "" "${VAL_MAX_BATCHES}" uniform

  summary="${OUTPUT_ROOT}/${SUMMARY_PREFIX}_generalization_iter_${tag}_summary.json"
  if (( iteration == END_ITERATION )); then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v8_1_geometry_router_state_gate \
      --mode generalization \
      --population-contract "${POPULATION_REPORT}" \
      --source-route "${VAL_SOURCE_ROUTE}" \
      --source-coverage "${VAL_SOURCE_COVERAGE}" \
      --control-route "${control_route}" \
      --control-coverage "${control_coverage}" \
      --treatment-route "${treatment_route}" \
      --treatment-coverage "${treatment_coverage}" \
      --iteration "${iteration}" \
      --output-json "${summary}"
  else
    # Intermediate endpoints are descriptive only and cannot select the arm.
    set +e
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v8_1_geometry_router_state_gate \
      --mode generalization \
      --population-contract "${POPULATION_REPORT}" \
      --source-route "${VAL_SOURCE_ROUTE}" \
      --source-coverage "${VAL_SOURCE_COVERAGE}" \
      --control-route "${control_route}" \
      --control-coverage "${control_coverage}" \
      --treatment-route "${treatment_route}" \
      --treatment-coverage "${treatment_coverage}" \
      --iteration "${iteration}" \
      --output-json "${summary}"
    set -e
  fi
done

echo "V8.1 paired 2k gate passed its predeclared uniform-256 endpoint."
echo "Long training is still closed until paired full-validation confirms the result."
