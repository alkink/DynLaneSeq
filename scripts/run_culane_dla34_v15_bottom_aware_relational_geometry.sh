#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION=225000
ENDPOINT_ITERATION=228000
TRAINING_STEPS=3000
CHECKPOINT_INTERVAL=500
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
V15_CONFIG="${V15_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v15_bottom_aware_relational_geometry_225k_to228k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v15_bottom_aware_relational_geometry_225k}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
INITIAL_CHECKPOINT="${OUTPUT_ROOT}/initialization/iter_0225000.pt"
TRAIN_DIR="${OUTPUT_ROOT}/train"
ENDPOINT="${TRAIN_DIR}/iter_0228000.pt"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V7_CONFIG}" \
  "${V15_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V15 artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration 225000." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V15 requires effective batch 16." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/cache" \
  "${OUTPUT_ROOT}/initialization" \
  "${TRAIN_DIR}" \
  "${LIST_ROOT}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.build_v11_bridge_lists \
  --train-list "${DATA_ROOT}/list/train_gt.txt" \
  --val-list "${DATA_ROOT}/list/val.txt" \
  --output-dir "${LIST_ROOT}" \
  --seed "${SEED}" \
  --train-clips 512 \
  --train-images 4096 \
  --seen-images 256 \
  --same-clip-unseen-images 256 \
  --heldout-clips 64 \
  --heldout-images 256 \
  --val-images 256 \
  --optimizer-steps "${TRAINING_STEPS}" \
  --effective-batch-size "$((BATCH_SIZE * GRAD_ACCUM))" \
  --experiment-name "V15 bottom-aware relational clip-disjoint contract" \
  --output-json "${OUTPUT_ROOT}/audits/list_protocol.json"

TRAIN_LIST="${LIST_ROOT}/train_clip512_image4096.txt"
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
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.initialize_v15_bottom_aware_relational_checkpoint \
    --config "${V15_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

GATE0="${OUTPUT_ROOT}/audits/zero_step_contract.json"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v15_bottom_aware_relational_contract \
    --config "${V15_CONFIG}" \
    --source-config "${V7_CONFIG}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --num-workers 0 \
    --start-iteration "${SOURCE_ITERATION}" \
    --cross-clip-report "${HELDOUT_CROSSCLIP}" \
    --cross-clip-report "${VAL_CROSSCLIP}" \
    --output-json "${GATE0}"
fi
if [[ ! -f "${GATE0}" ]]; then
  echo "V15 Gate 0 report is missing." >&2
  exit 1
fi
GATE0_PASSED="$(${PYTHON} - "${GATE0}" <<'PY'
import json, sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text()).get("passed") is True else "0")
PY
)"
if [[ "${GATE0_PASSED}" != "1" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}/v15_completion.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "experiment": "V15 bottom-aware relational slot geometry",
    "gate0_passed": False,
    "training_executed": False,
    "fixed_endpoint_gate_passed": False,
    "full_validation_executed": False,
    "long_training_authorized": False,
    "test_set_used": False,
    "decision": "v15_complete_gate0_fail_stop_for_review",
}, indent=2, sort_keys=True) + "\n")
PY
  echo "V15 Gate 0 FAIL: no optimizer step is authorized. V15 is complete."
  exit 0
fi

train_endpoint() {
  if [[ -f "${ENDPOINT}" ]]; then
    if (( $(checkpoint_iteration "${ENDPOINT}") != ENDPOINT_ITERATION )); then
      echo "Existing V15 endpoint has the wrong iteration." >&2
      exit 1
    fi
    echo "Reusing fixed V15 endpoint ${ENDPOINT}"
    return
  fi
  local latest=""
  local latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${TRAIN_DIR}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    local iteration
    iteration="$(checkpoint_iteration "${candidate}")"
    if (( iteration > latest_iteration && iteration < ENDPOINT_ITERATION )); then
      latest="${candidate}"
      latest_iteration="${iteration}"
    fi
  done
  local command=(
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train
    --config "${V15_CONFIG}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --output-dir "${TRAIN_DIR}"
    --checkpoint-base "${INITIAL_CHECKPOINT}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --num-workers "${NUM_WORKERS}"
    --seg-aux-amp-dtype "${AMP_DTYPE}"
    --compile-model false
    --resume-safe-data true
    --train-list "${TRAIN_LIST}"
  )
  if [[ -n "${latest}" ]]; then
    command+=(
      --resume "${latest}"
      --max-iters "$((ENDPOINT_ITERATION - latest_iteration))"
    )
  else
    command+=(
      --init-from "${INITIAL_CHECKPOINT}"
      --init-iteration "${SOURCE_ITERATION}"
      --max-iters "${TRAINING_STEPS}"
    )
  fi
  "${command[@]}" 2>&1 | tee -a "${TRAIN_DIR}/train.log"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_endpoint
fi
if [[ ! -f "${ENDPOINT}" ]]; then
  echo "Fixed V15 endpoint is missing: ${ENDPOINT}" >&2
  exit 1
fi
if (( $(checkpoint_iteration "${ENDPOINT}") != ENDPOINT_ITERATION )); then
  echo "V15 endpoint is not iteration 228000." >&2
  exit 1
fi

GRADIENT_REPORT="${OUTPUT_ROOT}/audits/endpoint_gradient_population.json"
if [[ ! -f "${GRADIENT_REPORT}" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v15_endpoint_gradient_population \
    --config "${V15_CONFIG}" \
    --checkpoint "${ENDPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --list-path "${TRAIN_LIST}" \
    --device "${DEVICE}" \
    --batches 32 \
    --batch-size 4 \
    --num-workers "${NUM_WORKERS}" \
    --output-json "${GRADIENT_REPORT}"
fi

official_report() {
  local split="$1"
  local list_path="$2"
  local wrong_list="$3"
  local crossclip="$4"
  local output="$5"
  [[ -f "${output}" ]] && return
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v15_bottom_aware_relational_official \
    --config "${V15_CONFIG}" \
    --checkpoint "${ENDPOINT}" \
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

HELDOUT_OFFICIAL="${OUTPUT_ROOT}/reports/heldout_official.json"
VAL_OFFICIAL="${OUTPUT_ROOT}/reports/validation_official.json"
official_report train "${HELDOUT_LIST}" "${HELDOUT_WRONG_LIST}" \
  "${HELDOUT_CROSSCLIP}" "${HELDOUT_OFFICIAL}"
official_report val "${VAL_LIST}" "${VAL_WRONG_LIST}" \
  "${VAL_CROSSCLIP}" "${VAL_OFFICIAL}"

coverage_report() {
  local config="$1"
  local checkpoint="$2"
  local split="$3"
  local list_path="$4"
  local name="$5"
  local output="${OUTPUT_ROOT}/reports/${name}_coverage.json"
  if [[ ! -f "${output}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split "${split}" \
      --list-path "${list_path}" \
      --device "${DEVICE}" \
      --cache-dir "${OUTPUT_ROOT}/cache/${name}" \
      --max-batches 0 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" \
      --sample-strategy sequential \
      --top-k 4 \
      --iou-thresholds 0.50 0.75 \
      --output-json "${output}"
  fi
}

coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" train \
  "${HELDOUT_LIST}" heldout_source
coverage_report "${V15_CONFIG}" "${ENDPOINT}" train \
  "${HELDOUT_LIST}" heldout_treatment
coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" val \
  "${VAL_LIST}" val_source
coverage_report "${V15_CONFIG}" "${ENDPOINT}" val \
  "${VAL_LIST}" val_treatment
HELDOUT_SOURCE_COVERAGE="${OUTPUT_ROOT}/reports/heldout_source_coverage.json"
HELDOUT_TREATMENT_COVERAGE="${OUTPUT_ROOT}/reports/heldout_treatment_coverage.json"
VAL_SOURCE_COVERAGE="${OUTPUT_ROOT}/reports/val_source_coverage.json"
VAL_TREATMENT_COVERAGE="${OUTPUT_ROOT}/reports/val_treatment_coverage.json"

SUMMARY="${OUTPUT_ROOT}/v15_fixed_gate_summary.json"
"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.summarize_v15_bottom_aware_relational_gate \
  --contract "${GATE0}" \
  --gradient-population "${GRADIENT_REPORT}" \
  --heldout "${HELDOUT_OFFICIAL}" \
  --validation "${VAL_OFFICIAL}" \
  --heldout-source-coverage "${HELDOUT_SOURCE_COVERAGE}" \
  --heldout-treatment-coverage "${HELDOUT_TREATMENT_COVERAGE}" \
  --val-source-coverage "${VAL_SOURCE_COVERAGE}" \
  --val-treatment-coverage "${VAL_TREATMENT_COVERAGE}" \
  --output-json "${SUMMARY}"

"${PYTHON}" - \
  "${OUTPUT_ROOT}" "${V7_CONFIG}" "${V15_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" "${INITIAL_CHECKPOINT}" "${ENDPOINT}" \
  "${GATE0}" "${GRADIENT_REPORT}" "${HELDOUT_OFFICIAL}" \
  "${VAL_OFFICIAL}" "${SUMMARY}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
from dynlaneseq_eg.config import load_config

root = Path(sys.argv[1])
paths = [Path(value) for value in sys.argv[2:]]
paths.extend(
    Path(value)
    for value in (
        "docs/V15_BOTTOM_AWARE_RELATIONAL_SLOT_GEOMETRY_CONTRACT_2026-08-13.md",
        "dynlaneseq_eg/modeling/four_slot_selection.py",
        "dynlaneseq_eg/losses/loss_s0.py",
        "dynlaneseq_eg/tools/audit_v15_bottom_aware_relational_contract.py",
        "dynlaneseq_eg/tools/audit_v15_bottom_aware_relational_official.py",
        "dynlaneseq_eg/tools/audit_v15_endpoint_gradient_population.py",
        "dynlaneseq_eg/tools/summarize_v15_bottom_aware_relational_gate.py",
        "scripts/run_culane_dla34_v15_bottom_aware_relational_geometry.sh",
    )
)
for directory in (root / "lists", root / "audits", root / "reports"):
    if directory.is_dir():
        paths.extend(path for path in directory.rglob("*") if path.is_file())
paths = list(dict.fromkeys(path.resolve() for path in paths if path.is_file()))
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
resolved = load_config(sys.argv[3])
manifest = {
    "experiment": "V15 fixed 3000-step endpoint",
    "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
    "git_branch": subprocess.check_output(("git", "branch", "--show-current"), text=True).strip(),
    "files": {str(path): {"sha256": digest(path)} for path in paths},
    "resolved_config_sha256": hashlib.sha256(
        json.dumps(resolved, sort_keys=True, default=str).encode()
    ).hexdigest(),
    "fixed_endpoint": str(Path(sys.argv[6]).resolve()),
    "checkpoint_selection_performed": False,
    "threshold_search_performed": False,
    "nms_search_performed": False,
    "full_validation_executed": False,
    "test_set_used": False,
}
(root / "provenance.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
summary = json.loads(Path(sys.argv[11]).read_text())
(root / "v15_completion.json").write_text(json.dumps({
    "experiment": "V15 bottom-aware relational slot geometry",
    "gate0_passed": True,
    "training_executed": True,
    "fixed_endpoint_gate_passed": summary.get("passed") is True,
    "full_validation_executed": False,
    "long_training_authorized": False,
    "test_set_used": False,
    "decision": summary.get("decision"),
}, indent=2, sort_keys=True) + "\n")
PY

echo "V15 is complete. No full validation, long training, V16, threshold search, or test run was started."
