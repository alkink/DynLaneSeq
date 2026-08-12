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

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
V11_CONFIG="${V11_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v11_unified_slot_row_225k_to228k.yaml}"
V11_MEMORY_CONFIG="${V11_MEMORY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v11_unified_slot_row_memorize64.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v11_unified_slot_row_gate_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/v11_unified_slot_row_gate_225k}"
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
  "${V11_CONFIG}" \
  "${V11_MEMORY_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V11 gate artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi
if (( TRAIN_STEPS != 3000 || CHECKPOINT_INTERVAL != 500 )); then
  echo "V11 is predeclared as a 3k gate with 500-step checkpoints." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V11 generalization requires effective batch size 16." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/initialization" \
  "${OUTPUT_ROOT}/reports" \
  "${CACHE_ROOT}"

if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.initialize_v11_unified_slot_row_checkpoint \
    --config "${V11_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

"${PYTHON}" - \
  "${V7_CONFIG}" \
  "${V11_CONFIG}" \
  "${V11_MEMORY_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${INITIAL_CHECKPOINT}" \
  "${OUTPUT_ROOT}/audits/provenance.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from dynlaneseq_eg.config import load_config


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


v7_config, v11_config, memory_config, source, initialization, output = sys.argv[1:]
resolved = load_config(v11_config)
resolved_bytes = json.dumps(
    resolved,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
report = {
    "git_commit": subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True
    ).strip(),
    "git_branch": subprocess.check_output(
        ("git", "branch", "--show-current"), text=True
    ).strip(),
    "files": {
        str(Path(path)): {"sha256": digest(path)}
        for path in (
            v7_config,
            v11_config,
            memory_config,
            source,
            initialization,
        )
    },
    "resolved_v11_config_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
    "trainable_parameter_prefixes": resolved["training"][
        "trainable_parameter_prefixes"
    ],
    "checkpoint_model_prefixes": resolved["training"][
        "checkpoint_model_prefixes"
    ],
    "optimizer": resolved["optimizer"],
    "scheduler": resolved["scheduler"],
    "loss": resolved["loss"],
    "test_set_used": False,
}
destination = Path(output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print({
    "v11_git_commit": report["git_commit"],
    "resolved_config_sha256": report["resolved_v11_config_sha256"],
    "provenance": str(destination),
})
PY

CONTRACT_REPORT="${OUTPUT_ROOT}/audits/v11_initialization_contract.json"
if [[ "${RUN_PREFLIGHT}" == "1" && ! -f "${CONTRACT_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v11_unified_slot_row_contract \
    --config "${V11_CONFIG}" \
    --source-config "${V7_CONFIG}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
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
print({"v11_initialization_contract_passed": report.get("passed", False)})
if report.get("passed") is not True:
    raise SystemExit("V11 initialization contract failed; training remains closed")
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
    --checkpoint-base "${INITIAL_CHECKPOINT}"
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
      --init-from "${INITIAL_CHECKPOINT}"
      --init-iteration "${SOURCE_ITERATION}"
      --max-iters "${TRAIN_STEPS}"
    )
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${args[@]}" \
    2>&1 | tee -a "${run_dir}/train.log"
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

state_report() {
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
    --max-images "${max_images}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --output-json "${report}"
  )
  if [[ -n "${list_path}" ]]; then
    args+=(--list-path "${list_path}")
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v11_unified_checkpoint_state "${args[@]}"
}

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"

if [[ "${RUN_FIXED64}" == "1" ]]; then
  FIXED_DIR="${OUTPUT_ROOT}/memorize64/v11"
  train_arm fixed64_v11 "${V11_MEMORY_CONFIG}" "${FIXED_DIR}" 1 "${FIXED_LIST}"
  SOURCE_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_source_v7_coverage.json"
  INIT_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_v11_init_coverage.json"
  END_COVERAGE="${OUTPUT_ROOT}/reports/fixed64_v11_end_coverage.json"
  INIT_STATE="${OUTPUT_ROOT}/reports/fixed64_v11_init_state.json"
  END_STATE="${OUTPUT_ROOT}/reports/fixed64_v11_end_state.json"
  coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" \
    "${SOURCE_COVERAGE}" "${CACHE_ROOT}/fixed64_source" train "${FIXED_LIST}" 0 sequential
  coverage_report "${V11_MEMORY_CONFIG}" "${INITIAL_CHECKPOINT}" \
    "${INIT_COVERAGE}" "${CACHE_ROOT}/fixed64_init" train "${FIXED_LIST}" 0 sequential
  coverage_report "${V11_MEMORY_CONFIG}" "${FIXED_DIR}/iter_${END_TAG}.pt" \
    "${END_COVERAGE}" "${CACHE_ROOT}/fixed64_end" train "${FIXED_LIST}" 0 sequential
  state_report "${V11_MEMORY_CONFIG}" "${INITIAL_CHECKPOINT}" \
    "${INIT_STATE}" train "${FIXED_LIST}" 0 sequential
  state_report "${V11_MEMORY_CONFIG}" "${FIXED_DIR}/iter_${END_TAG}.pt" \
    "${END_STATE}" train "${FIXED_LIST}" 0 sequential
  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v11_unified_slot_row_gate \
    --mode fixed64 \
    --contract "${CONTRACT_REPORT}" \
    --source-coverage "${SOURCE_COVERAGE}" \
    --init-coverage "${INIT_COVERAGE}" \
    --treatment-coverage "${END_COVERAGE}" \
    --init-state "${INIT_STATE}" \
    --treatment-state "${END_STATE}" \
    --iteration "${END_ITERATION}" \
    --output-json "${OUTPUT_ROOT}/v11_fixed64_summary.json"
fi

"${PYTHON}" - "${OUTPUT_ROOT}/v11_fixed64_summary.json" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print({"v11_fixed64_passed": report.get("passed", False), "next_action": report.get("next_action")})
if report.get("passed") is not True:
    raise SystemExit("V11 fixed-64 gate failed; generalization remains closed")
PY

if [[ "${RUN_GENERALIZATION}" != "1" ]]; then
  echo "V11 fixed-64 gate complete. Generalization remains paused."
  echo "Only if v11_fixed64_summary.json passed, rerun with RUN_FIXED64=0 RUN_GENERALIZATION=1."
  exit 0
fi

GENERALIZATION_DIR="${OUTPUT_ROOT}/generalization/v11"
train_arm generalization_v11 "${V11_CONFIG}" "${GENERALIZATION_DIR}" "${GRAD_ACCUM}" ""
VAL_MAX_IMAGES=256
VAL_MAX_BATCHES=$(((VAL_MAX_IMAGES + EVAL_BATCH_SIZE - 1) / EVAL_BATCH_SIZE))
SOURCE_COVERAGE="${OUTPUT_ROOT}/reports/val_source_v7_coverage_uniform256.json"
INIT_COVERAGE="${OUTPUT_ROOT}/reports/val_v11_init_coverage_uniform256.json"
END_COVERAGE="${OUTPUT_ROOT}/reports/val_v11_end_coverage_uniform256.json"
INIT_STATE="${OUTPUT_ROOT}/reports/val_v11_init_state_uniform256.json"
END_STATE="${OUTPUT_ROOT}/reports/val_v11_end_state_uniform256.json"
coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" \
  "${SOURCE_COVERAGE}" "${CACHE_ROOT}/val_source" val "" "${VAL_MAX_BATCHES}" uniform
coverage_report "${V11_CONFIG}" "${INITIAL_CHECKPOINT}" \
  "${INIT_COVERAGE}" "${CACHE_ROOT}/val_init" val "" "${VAL_MAX_BATCHES}" uniform
coverage_report "${V11_CONFIG}" "${GENERALIZATION_DIR}/iter_${END_TAG}.pt" \
  "${END_COVERAGE}" "${CACHE_ROOT}/val_end" val "" "${VAL_MAX_BATCHES}" uniform
state_report "${V11_CONFIG}" "${INITIAL_CHECKPOINT}" \
  "${INIT_STATE}" val "" "${VAL_MAX_IMAGES}" uniform
state_report "${V11_CONFIG}" "${GENERALIZATION_DIR}/iter_${END_TAG}.pt" \
  "${END_STATE}" val "" "${VAL_MAX_IMAGES}" uniform
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v11_unified_slot_row_gate \
  --mode generalization \
  --contract "${CONTRACT_REPORT}" \
  --source-coverage "${SOURCE_COVERAGE}" \
  --init-coverage "${INIT_COVERAGE}" \
  --treatment-coverage "${END_COVERAGE}" \
  --init-state "${INIT_STATE}" \
  --treatment-state "${END_STATE}" \
  --iteration "${END_ITERATION}" \
  --output-json "${OUTPUT_ROOT}/v11_generalization_iter_${END_TAG}_summary.json"

echo "V11 generalization gate complete. Long training is still disabled."
