#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION="${SOURCE_ITERATION:-225000}"
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_FIXED64="${RUN_FIXED64:-1}"
RUN_GENERALIZATION="${RUN_GENERALIZATION:-0}"
EVAL_TRAJECTORY="${EVAL_TRAJECTORY:-0}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v10_legacy_control_225k_to228k.yaml}"
V10_CONFIG="${V10_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v10_global_visual_geometry_225k_to228k.yaml}"
V10_MEMORY_CONFIG="${V10_MEMORY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v10_global_visual_geometry_memorize64.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v10_global_visual_geometry_gate_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/v10_global_visual_geometry_gate_225k}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${OUTPUT_ROOT}/initialization/iter_0225000.pt}"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V7_CONFIG}" \
  "${CONTROL_CONFIG}" \
  "${V10_CONFIG}" \
  "${V10_MEMORY_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V10 gate artefact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi
if (( TRAIN_STEPS != 3000 || CHECKPOINT_INTERVAL != 500 )); then
  echo "V10 is predeclared as a 3k gate with 500-step checkpoints." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V10 generalization requires effective batch size 16." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/initialization" \
  "${OUTPUT_ROOT}/reports" "${CACHE_ROOT}"
if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.initialize_v10_global_visual_checkpoint \
    --config "${V10_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

CONTRACT_REPORT="${OUTPUT_ROOT}/audits/v10_zero_step_contract.json"
if [[ "${RUN_PREFLIGHT}" == "1" && ! -f "${CONTRACT_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v10_global_visual_geometry_contract \
    --config "${V10_CONFIG}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --num-workers 0 \
    --batches 16 \
    --start-iteration "${SOURCE_ITERATION}" \
    --output-json "${CONTRACT_REPORT}"
fi
"${PYTHON}" - "${CONTRACT_REPORT}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print({"v10_zero_step_contract_passed": report.get("passed", False)})
if report.get("passed") is not True:
    raise SystemExit("V10 zero-step contract failed; training remains closed")
PY

FIXED_LIST="${OUTPUT_ROOT}/train_uniform64.txt"
"${PYTHON}" - "${DATA_ROOT}/list/train_gt.txt" "${FIXED_LIST}" <<'PY'
import sys
from pathlib import Path
source = Path(sys.argv[1])
destination = Path(sys.argv[2])
lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(lines) < 64:
    raise SystemExit(f"expected at least 64 training images, found {len(lines)}")
indices = [round(index * (len(lines) - 1) / 63) for index in range(64)]
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text("\n".join(lines[index] for index in indices) + "\n", encoding="utf-8")
print({"fixed_list": str(destination), "images": 64, "first": indices[0], "last": indices[-1]})
PY

train_arm() {
  local name="$1"
  local config="$2"
  local base_checkpoint="$3"
  local run_dir="$4"
  local grad_accum="$5"
  local train_list="$6"
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
    --checkpoint-base "${base_checkpoint}"
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
      --init-from "${base_checkpoint}"
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
  local strategy="$7"
  [[ -f "${report}" ]] && return
  local args=(
    --config "${config}"
    --checkpoint "${checkpoint}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --split "${split}"
    --sample-strategy "${strategy}"
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
  local strategy="$8"
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
    --sample-strategy "${strategy}"
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
  MEMORY_DIR="${OUTPUT_ROOT}/memorize64/v10"
  train_arm fixed64_v10 "${V10_MEMORY_CONFIG}" "${INITIAL_CHECKPOINT}" \
    "${MEMORY_DIR}" 1 "${FIXED_LIST}"

  SOURCE_ROUTE="${OUTPUT_ROOT}/reports/fixed64_source_v7_route.json"
  SOURCE_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_source_v7_coverage.json"
  INIT_ROUTE="${OUTPUT_ROOT}/reports/fixed64_v10_init_route.json"
  INIT_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_v10_init_coverage.json"
  END_ROUTE="${OUTPUT_ROOT}/reports/fixed64_v10_end_route.json"
  END_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_v10_end_coverage.json"
  route_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" "${SOURCE_ROUTE}" train "${FIXED_LIST}" 0 sequential
  coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" "${SOURCE_COVERAGE}" "${CACHE_ROOT}/fixed64_source" train "${FIXED_LIST}" 0 sequential
  route_report "${V10_MEMORY_CONFIG}" "${INITIAL_CHECKPOINT}" "${INIT_ROUTE}" train "${FIXED_LIST}" 0 sequential
  coverage_report "${V10_MEMORY_CONFIG}" "${INITIAL_CHECKPOINT}" "${INIT_COVERAGE}" "${CACHE_ROOT}/fixed64_init" train "${FIXED_LIST}" 0 sequential
  route_report "${V10_MEMORY_CONFIG}" "${MEMORY_DIR}/iter_${END_TAG}.pt" "${END_ROUTE}" train "${FIXED_LIST}" 0 sequential
  coverage_report "${V10_MEMORY_CONFIG}" "${MEMORY_DIR}/iter_${END_TAG}.pt" "${END_COVERAGE}" "${CACHE_ROOT}/fixed64_end" train "${FIXED_LIST}" 0 sequential

  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v10_global_visual_geometry_gate \
    --mode fixed64 \
    --contract "${CONTRACT_REPORT}" \
    --source-route "${SOURCE_ROUTE}" \
    --source-coverage "${SOURCE_COVERAGE}" \
    --init-route "${INIT_ROUTE}" \
    --init-coverage "${INIT_COVERAGE}" \
    --control-route "${SOURCE_ROUTE}" \
    --control-coverage "${SOURCE_COVERAGE}" \
    --treatment-route "${END_ROUTE}" \
    --treatment-coverage "${END_COVERAGE}" \
    --iteration "${END_ITERATION}" \
    --output-json "${OUTPUT_ROOT}/v10_fixed64_summary.json"
elif [[ ! -f "${OUTPUT_ROOT}/v10_fixed64_summary.json" ]]; then
  echo "RUN_FIXED64=0 requires an existing V10 fixed-64 summary." >&2
  exit 1
else
  "${PYTHON}" - "${OUTPUT_ROOT}/v10_fixed64_summary.json" <<'PY'
import json
import sys
from pathlib import Path
if json.loads(Path(sys.argv[1]).read_text()).get("passed") is not True:
    raise SystemExit("existing V10 fixed-64 gate failed")
PY
fi

if [[ "${RUN_GENERALIZATION}" != "1" ]]; then
  echo "V10 fixed-64 gate complete. Generalization remains paused."
  echo "If and only if v10_fixed64_summary.json passed, rerun with RUN_FIXED64=0 RUN_GENERALIZATION=1."
  exit 0
fi

CONTROL_DIR="${OUTPUT_ROOT}/generalization/legacy_control"
TREATMENT_DIR="${OUTPUT_ROOT}/generalization/v10"
train_arm generalization_control "${CONTROL_CONFIG}" "${SOURCE_V7_CHECKPOINT}" "${CONTROL_DIR}" "${GRAD_ACCUM}" ""
train_arm generalization_v10 "${V10_CONFIG}" "${INITIAL_CHECKPOINT}" "${TREATMENT_DIR}" "${GRAD_ACCUM}" ""

VAL_MAX_IMAGES=256
VAL_MAX_BATCHES=$(((VAL_MAX_IMAGES + EVAL_BATCH_SIZE - 1) / EVAL_BATCH_SIZE))
SOURCE_ROUTE="${OUTPUT_ROOT}/reports/val_source_v7_route_uniform256.json"
SOURCE_COVERAGE="${OUTPUT_ROOT}/reports/val_source_v7_coverage_uniform256.json"
INIT_ROUTE="${OUTPUT_ROOT}/reports/val_v10_init_route_uniform256.json"
INIT_COVERAGE="${OUTPUT_ROOT}/reports/val_v10_init_coverage_uniform256.json"
route_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" "${SOURCE_ROUTE}" val "" "${VAL_MAX_IMAGES}" uniform
coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" "${SOURCE_COVERAGE}" "${CACHE_ROOT}/val_source" val "" "${VAL_MAX_BATCHES}" uniform
route_report "${V10_CONFIG}" "${INITIAL_CHECKPOINT}" "${INIT_ROUTE}" val "" "${VAL_MAX_IMAGES}" uniform
coverage_report "${V10_CONFIG}" "${INITIAL_CHECKPOINT}" "${INIT_COVERAGE}" "${CACHE_ROOT}/val_init" val "" "${VAL_MAX_BATCHES}" uniform

start_iteration="${END_ITERATION}"
if [[ "${EVAL_TRAJECTORY}" == "1" ]]; then
  start_iteration=$((SOURCE_ITERATION + CHECKPOINT_INTERVAL))
fi
for ((iteration=start_iteration; iteration<=END_ITERATION; iteration+=CHECKPOINT_INTERVAL)); do
  tag="$(printf '%07d' "${iteration}")"
  control_checkpoint="${CONTROL_DIR}/iter_${tag}.pt"
  treatment_checkpoint="${TREATMENT_DIR}/iter_${tag}.pt"
  control_route="${OUTPUT_ROOT}/reports/val_control_iter_${tag}_route_uniform256.json"
  control_coverage="${OUTPUT_ROOT}/reports/val_control_iter_${tag}_coverage_uniform256.json"
  treatment_route="${OUTPUT_ROOT}/reports/val_v10_iter_${tag}_route_uniform256.json"
  treatment_coverage="${OUTPUT_ROOT}/reports/val_v10_iter_${tag}_coverage_uniform256.json"
  route_report "${CONTROL_CONFIG}" "${control_checkpoint}" "${control_route}" val "" "${VAL_MAX_IMAGES}" uniform
  coverage_report "${CONTROL_CONFIG}" "${control_checkpoint}" "${control_coverage}" "${CACHE_ROOT}/val_control_${tag}" val "" "${VAL_MAX_BATCHES}" uniform
  route_report "${V10_CONFIG}" "${treatment_checkpoint}" "${treatment_route}" val "" "${VAL_MAX_IMAGES}" uniform
  coverage_report "${V10_CONFIG}" "${treatment_checkpoint}" "${treatment_coverage}" "${CACHE_ROOT}/val_v10_${tag}" val "" "${VAL_MAX_BATCHES}" uniform

  summary="${OUTPUT_ROOT}/v10_generalization_iter_${tag}_summary.json"
  if (( iteration == END_ITERATION )); then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v10_global_visual_geometry_gate \
      --mode generalization \
      --contract "${CONTRACT_REPORT}" \
      --source-route "${SOURCE_ROUTE}" \
      --source-coverage "${SOURCE_COVERAGE}" \
      --init-route "${INIT_ROUTE}" \
      --init-coverage "${INIT_COVERAGE}" \
      --control-route "${control_route}" \
      --control-coverage "${control_coverage}" \
      --treatment-route "${treatment_route}" \
      --treatment-coverage "${treatment_coverage}" \
      --iteration "${iteration}" \
      --output-json "${summary}"
  else
    set +e
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v10_global_visual_geometry_gate \
      --mode generalization \
      --contract "${CONTRACT_REPORT}" \
      --source-route "${SOURCE_ROUTE}" \
      --source-coverage "${SOURCE_COVERAGE}" \
      --init-route "${INIT_ROUTE}" \
      --init-coverage "${INIT_COVERAGE}" \
      --control-route "${control_route}" \
      --control-coverage "${control_coverage}" \
      --treatment-route "${treatment_route}" \
      --treatment-coverage "${treatment_coverage}" \
      --iteration "${iteration}" \
      --output-json "${summary}"
    set -e
  fi
done

echo "V10 paired 3k gate passed. Long training remains closed pending paired full validation."
