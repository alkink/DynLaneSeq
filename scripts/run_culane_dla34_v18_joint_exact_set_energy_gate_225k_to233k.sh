#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION=225000
ENDPOINT_ITERATION=233000
TRAINING_STEPS=8000
CHECKPOINT_INTERVAL=1000
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# V18 official replay holds source+endpoint models and two cross-clip feature
# batches at once.  Two is deliberate even when ordinary evaluation uses 8.
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
TREATMENT_CONFIG="${TREATMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v18_joint_exact_set_energy_225k_to233k.yaml}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v18_joint_exact_set_energy_control_225k_to233k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v18_joint_exact_set_energy_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/v18_joint_exact_set_energy_225k}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
INITIAL_CHECKPOINT="${OUTPUT_ROOT}/initialization/iter_0225000.pt"
TREATMENT_DIR="${OUTPUT_ROOT}/train/treatment"
CONTROL_DIR="${OUTPUT_ROOT}/train/control"
TREATMENT_ENDPOINT="${TREATMENT_DIR}/iter_0233000.pt"
CONTROL_ENDPOINT="${CONTROL_DIR}/iter_0233000.pt"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V7_CONFIG}" \
  "${TREATMENT_CONFIG}" \
  "${CONTROL_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V18 artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration 225000." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V18 requires effective batch 16." >&2
  exit 1
fi
if (( EVAL_BATCH_SIZE < 1 || EVAL_BATCH_SIZE > 4 )); then
  echo "V18 causal replay eval batch must be in [1,4]." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/initialization" \
  "${TREATMENT_DIR}" \
  "${CONTROL_DIR}" \
  "${LIST_ROOT}" \
  "${CACHE_ROOT}"

# CULane has 705 path-level clips under this exact clip key; 704 have the
# 13-frame capacity required by the balanced 12/13-image train allocation.
# Reserve 64 completely disjoint clips and use the remaining 640 for training.
"${PYTHON}" -u -m dynlaneseq_eg.tools.build_v11_bridge_lists \
  --train-list "${DATA_ROOT}/list/train_gt.txt" \
  --val-list "${DATA_ROOT}/list/val.txt" \
  --output-dir "${LIST_ROOT}" \
  --seed "${SEED}" \
  --train-clips 640 \
  --train-images 8192 \
  --balanced-train-remainder \
  --seen-images 256 \
  --same-clip-unseen-images 256 \
  --heldout-clips 64 \
  --heldout-images 256 \
  --val-images 256 \
  --optimizer-steps "${TRAINING_STEPS}" \
  --effective-batch-size "$((BATCH_SIZE * GRAD_ACCUM))" \
  --experiment-name "V18 joint exact set-energy paired fixed gate" \
  --output-json "${OUTPUT_ROOT}/audits/list_protocol.json"

TRAIN_LIST="${LIST_ROOT}/train_clip640_image8192.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"
HELDOUT_WRONG_LIST="${LIST_ROOT}/heldout_cross_clip_wrong_image256.txt"
VAL_WRONG_LIST="${LIST_ROOT}/val_cross_clip_wrong_image256.txt"
HELDOUT_CROSSCLIP="${OUTPUT_ROOT}/audits/heldout_cross_clip_derangement.json"
VAL_CROSSCLIP="${OUTPUT_ROOT}/audits/val_cross_clip_derangement.json"

"${PYTHON}" -u -m dynlaneseq_eg.tools.build_cross_clip_derangement \
  --input-list "${HELDOUT_LIST}" \
  --output-list "${HELDOUT_WRONG_LIST}" \
  --output-json "${HELDOUT_CROSSCLIP}" \
  --seed "${SEED}"
"${PYTHON}" -u -m dynlaneseq_eg.tools.build_cross_clip_derangement \
  --input-list "${VAL_LIST}" \
  --output-list "${VAL_WRONG_LIST}" \
  --output-json "${VAL_CROSSCLIP}" \
  --seed "${SEED}"

if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.initialize_v18_joint_exact_set_energy_checkpoint \
    --config "${TREATMENT_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

GATE0="${OUTPUT_ROOT}/audits/zero_step_contract.json"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v18_joint_exact_set_energy_contract \
    --config "${TREATMENT_CONFIG}" \
    --control-config "${CONTROL_CONFIG}" \
    --source-config "${V7_CONFIG}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --num-workers 0 \
    --start-iteration "${SOURCE_ITERATION}" \
    --output-json "${GATE0}"
fi
if [[ ! -f "${GATE0}" ]]; then
  echo "V18 Gate 0 report is missing." >&2
  exit 1
fi
GATE0_PASSED="$(${PYTHON} - "${GATE0}" <<'PY'
import json, sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text()).get("passed") is True else "0")
PY
)"
if [[ "${GATE0_PASSED}" != "1" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}/v18_completion.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "experiment": "V18 joint exact set-energy",
    "gate0_passed": False,
    "training_executed": False,
    "fixed_endpoint_gate_passed": False,
    "full_validation_executed": False,
    "long_training_authorized": False,
    "test_set_used": False,
    "decision": "v18_complete_gate0_fail_stop_for_review",
}, indent=2, sort_keys=True) + "\n")
PY
  echo "V18 Gate 0 FAIL: no optimizer step is authorized."
  exit 0
fi

train_arm() {
  local name="$1"
  local config="$2"
  local directory="$3"
  local endpoint="$4"
  if [[ -f "${endpoint}" ]]; then
    if (( $(checkpoint_iteration "${endpoint}") != ENDPOINT_ITERATION )); then
      echo "Existing V18 ${name} endpoint has wrong iteration." >&2
      exit 1
    fi
    echo "Reusing fixed V18 ${name} endpoint ${endpoint}"
    return
  fi
  local latest=""
  local latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${directory}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    local candidate_iteration
    candidate_iteration="$(checkpoint_iteration "${candidate}")"
    if (( candidate_iteration > latest_iteration && candidate_iteration < ENDPOINT_ITERATION )); then
      latest="${candidate}"
      latest_iteration="${candidate_iteration}"
    fi
  done
  local command=(
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train
    --config "${config}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --output-dir "${directory}"
    --checkpoint-base "${INITIAL_CHECKPOINT}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --num-workers "${NUM_WORKERS}"
    --seg-aux-amp-dtype "${AMP_DTYPE}"
    # PyTorch 2.11/RTX 5090 steady-state A/B: 19.5 img/s compiled versus
    # 13.1 img/s eager after the arithmetic-preserving V18 optimizations.
    # Treatment and control both use the same compiled execution path.
    --compile-model true
    --resume-safe-data true
    --train-list "${TRAIN_LIST}"
  )
  if [[ -n "${latest}" ]]; then
    command+=(--resume "${latest}" --max-iters "$((ENDPOINT_ITERATION - latest_iteration))")
  else
    command+=(--init-from "${INITIAL_CHECKPOINT}" --init-iteration "${SOURCE_ITERATION}" --max-iters "${TRAINING_STEPS}")
  fi
  "${command[@]}" 2>&1 | tee -a "${directory}/train.log"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_arm treatment "${TREATMENT_CONFIG}" "${TREATMENT_DIR}" "${TREATMENT_ENDPOINT}"
  train_arm control "${CONTROL_CONFIG}" "${CONTROL_DIR}" "${CONTROL_ENDPOINT}"
fi
if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "V18 training stage complete; evaluation remains paused."
  exit 0
fi
for endpoint in "${TREATMENT_ENDPOINT}" "${CONTROL_ENDPOINT}"; do
  if [[ ! -f "${endpoint}" ]] || (( $(checkpoint_iteration "${endpoint}") != ENDPOINT_ITERATION )); then
    echo "Missing or invalid fixed V18 endpoint: ${endpoint}" >&2
    exit 1
  fi
done

official_report() {
  local arm="$1"
  local config="$2"
  local checkpoint="$3"
  local split="$4"
  local list_path="$5"
  local wrong_list="$6"
  local crossclip="$7"
  local output="$8"
  [[ -f "${output}" ]] && return
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v18_joint_exact_set_energy_official \
    --config "${config}" \
    --source-config "${V7_CONFIG}" \
    --checkpoint "${checkpoint}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --wrong-list-path "${wrong_list}" \
    --cross-clip-report "${crossclip}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --output-json "${output}"
}

for arm in treatment control; do
  if [[ "${arm}" == "treatment" ]]; then
    arm_config="${TREATMENT_CONFIG}"
    arm_checkpoint="${TREATMENT_ENDPOINT}"
  else
    arm_config="${CONTROL_CONFIG}"
    arm_checkpoint="${CONTROL_ENDPOINT}"
  fi
  official_report "${arm}" "${arm_config}" "${arm_checkpoint}" train \
    "${HELDOUT_LIST}" "${HELDOUT_WRONG_LIST}" "${HELDOUT_CROSSCLIP}" \
    "${OUTPUT_ROOT}/reports/heldout_${arm}_official.json"
  official_report "${arm}" "${arm_config}" "${arm_checkpoint}" val \
    "${VAL_LIST}" "${VAL_WRONG_LIST}" "${VAL_CROSSCLIP}" \
    "${OUTPUT_ROOT}/reports/validation_${arm}_official.json"
done

coverage_report() {
  local config="$1"
  local checkpoint="$2"
  local split="$3"
  local list_path="$4"
  local name="$5"
  local output="${OUTPUT_ROOT}/reports/${name}_coverage.json"
  [[ -f "${output}" ]] && return
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_ROOT}/${name}" \
    --max-batches 0 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --sample-strategy sequential \
    --top-k 4 \
    --iou-thresholds 0.50 0.75 \
    --output-json "${output}"
}

coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" train "${HELDOUT_LIST}" heldout_source
coverage_report "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}" train "${HELDOUT_LIST}" heldout_treatment
coverage_report "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}" train "${HELDOUT_LIST}" heldout_control
coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" val "${VAL_LIST}" val_source
coverage_report "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}" val "${VAL_LIST}" val_treatment
coverage_report "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}" val "${VAL_LIST}" val_control

SUMMARY="${OUTPUT_ROOT}/v18_fixed_gate_summary.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v18_joint_exact_set_energy_gate \
  --contract "${GATE0}" \
  --heldout-treatment "${OUTPUT_ROOT}/reports/heldout_treatment_official.json" \
  --heldout-control "${OUTPUT_ROOT}/reports/heldout_control_official.json" \
  --validation-treatment "${OUTPUT_ROOT}/reports/validation_treatment_official.json" \
  --validation-control "${OUTPUT_ROOT}/reports/validation_control_official.json" \
  --heldout-source-coverage "${OUTPUT_ROOT}/reports/heldout_source_coverage.json" \
  --heldout-treatment-coverage "${OUTPUT_ROOT}/reports/heldout_treatment_coverage.json" \
  --heldout-control-coverage "${OUTPUT_ROOT}/reports/heldout_control_coverage.json" \
  --val-source-coverage "${OUTPUT_ROOT}/reports/val_source_coverage.json" \
  --val-treatment-coverage "${OUTPUT_ROOT}/reports/val_treatment_coverage.json" \
  --val-control-coverage "${OUTPUT_ROOT}/reports/val_control_coverage.json" \
  --output-json "${SUMMARY}"

"${PYTHON}" - "${OUTPUT_ROOT}" "${TREATMENT_CONFIG}" "${CONTROL_CONFIG}" "${SOURCE_V7_CHECKPOINT}" "${INITIAL_CHECKPOINT}" "${TREATMENT_ENDPOINT}" "${CONTROL_ENDPOINT}" "${SUMMARY}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
from dynlaneseq_eg.config import load_config

root = Path(sys.argv[1])
fixed = [Path(value) for value in sys.argv[2:]]
fixed.extend(Path(value) for value in (
    "docs/experiments/V18_JOINT_EXACT_SET_ENERGY_CORRECTED_CONTRACT_2026-08-14.md",
    "dynlaneseq_eg/modeling/v18_joint_exact_set_energy.py",
    "dynlaneseq_eg/modeling/four_slot_selection.py",
    "dynlaneseq_eg/losses/loss_s0.py",
    "dynlaneseq_eg/engine/train_one_epoch.py",
    "dynlaneseq_eg/tools/audit_v18_joint_exact_set_energy_contract.py",
    "dynlaneseq_eg/tools/audit_v18_joint_exact_set_energy_official.py",
    "dynlaneseq_eg/tools/summarize_v18_joint_exact_set_energy_gate.py",
    "scripts/run_culane_dla34_v18_joint_exact_set_energy_gate_225k_to233k.sh",
))
for directory in (root / "lists", root / "audits", root / "reports"):
    if directory.is_dir():
        fixed.extend(path for path in directory.rglob("*") if path.is_file())
paths = list(dict.fromkeys(path.resolve() for path in fixed if path.is_file()))
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
manifest = {
    "experiment": "V18 paired fixed 8000-step endpoint",
    "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
    "git_branch": subprocess.check_output(("git", "branch", "--show-current"), text=True).strip(),
    "files": {str(path): {"sha256": digest(path)} for path in paths},
    "resolved_treatment_config_sha256": hashlib.sha256(json.dumps(load_config(sys.argv[2]), sort_keys=True, default=str).encode()).hexdigest(),
    "resolved_control_config_sha256": hashlib.sha256(json.dumps(load_config(sys.argv[3]), sort_keys=True, default=str).encode()).hexdigest(),
    "checkpoint_selection_performed": False,
    "threshold_search_performed": False,
    "nms_search_performed": False,
    "full_validation_executed": False,
    "test_set_used": False,
}
(root / "provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
summary = json.loads(Path(sys.argv[8]).read_text())
(root / "v18_completion.json").write_text(json.dumps({
    "experiment": "V18 joint exact set-energy",
    "gate0_passed": True,
    "paired_training_executed": True,
    "fixed_endpoint_gate_passed": summary.get("passed") is True,
    "full_validation_executed": False,
    "long_training_authorized": False,
    "test_set_used": False,
    "decision": summary.get("decision"),
}, indent=2, sort_keys=True) + "\n")
PY

echo "V18 is complete. Stop for user/Sol review; no V19, full validation, long training, threshold search or test was started."
