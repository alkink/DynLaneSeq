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
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
V12_CONFIG="${V12_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v12_visual_first_association_stage_a_225k_to227k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v12_visual_first_stage_a_225k}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${OUTPUT_ROOT}/initialization/iter_0225000.pt}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
PROTOCOL_REPORT="${OUTPUT_ROOT}/audits/bridge_list_protocol.json"
CONTRACT_REPORT="${OUTPUT_ROOT}/audits/v12_zero_step_contract.json"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V7_CONFIG}" \
  "${V12_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V12 Stage-A artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi
if (( TRAIN_STEPS != 2000 || CHECKPOINT_INTERVAL != 500 )); then
  echo "V12 Stage A is predeclared as 2k steps with 500-step checkpoints." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V12 Stage A requires effective batch size 16." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/initialization" \
  "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/train" \
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
  --output-json "${PROTOCOL_REPORT}"

"${PYTHON}" - "${PROTOCOL_REPORT}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print({"bridge_list_protocol_passed": report.get("passed", False)})
if report.get("passed") is not True or report.get("test_set_used") is not False:
    raise SystemExit("V12 bridge list/leakage contract failed")
PY

TRAIN_LIST="${LIST_ROOT}/train_clip512_image4096.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"

if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.initialize_v12_visual_first_checkpoint \
    --config "${V12_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v12_visual_first_contract \
    --config "${V12_CONFIG}" \
    --source-config "${V7_CONFIG}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --num-workers 0 \
    --start-iteration "${SOURCE_ITERATION}" \
    --output-json "${CONTRACT_REPORT}"
fi
"${PYTHON}" - "${CONTRACT_REPORT}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print({"v12_zero_step_contract_passed": report.get("passed", False)})
if report.get("passed") is not True:
    raise SystemExit("V12 zero-step contract failed; training closed")
PY

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
TRAIN_DIR="${OUTPUT_ROOT}/train/v12"
END_CHECKPOINT="${TRAIN_DIR}/iter_${END_TAG}.pt"

train_arm() {
  mkdir -p "${TRAIN_DIR}"
  if [[ -f "${END_CHECKPOINT}" ]]; then
    echo "V12 Stage A: reusing ${END_CHECKPOINT}"
    return
  fi
  local latest_checkpoint=""
  local latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${TRAIN_DIR}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    local iteration
    iteration="$(checkpoint_iteration "${candidate}")"
    if (( iteration > latest_iteration && iteration < END_ITERATION )); then
      latest_checkpoint="${candidate}"
      latest_iteration="${iteration}"
    fi
  done
  local args=(
    --config "${V12_CONFIG}"
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
  if [[ -n "${latest_checkpoint}" ]]; then
    args+=(
      --resume "${latest_checkpoint}"
      --max-iters "$((END_ITERATION - latest_iteration))"
    )
  else
    args+=(
      --init-from "${INITIAL_CHECKPOINT}"
      --init-iteration "${SOURCE_ITERATION}"
      --max-iters "${TRAIN_STEPS}"
    )
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${args[@]}" \
    2>&1 | tee -a "${TRAIN_DIR}/train.log"
}

state_report() {
  local checkpoint="$1"
  local report="$2"
  local split="$3"
  local list_path="$4"
  [[ -f "${report}" ]] && return
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v12_visual_first_state \
    --config "${V12_CONFIG}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --sample-strategy sequential \
    --max-images 0 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --output-json "${report}"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_arm
fi
if [[ ! -f "${END_CHECKPOINT}" ]]; then
  echo "Missing fixed V12 endpoint: ${END_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  state_report "${INITIAL_CHECKPOINT}" \
    "${OUTPUT_ROOT}/reports/heldout_init_state.json" train "${HELDOUT_LIST}"
  state_report "${END_CHECKPOINT}" \
    "${OUTPUT_ROOT}/reports/heldout_end_state.json" train "${HELDOUT_LIST}"
  state_report "${INITIAL_CHECKPOINT}" \
    "${OUTPUT_ROOT}/reports/val_init_state.json" val "${VAL_LIST}"
  state_report "${END_CHECKPOINT}" \
    "${OUTPUT_ROOT}/reports/val_end_state.json" val "${VAL_LIST}"

  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v12_visual_first_gate \
    --contract "${CONTRACT_REPORT}" \
    --heldout-init "${OUTPUT_ROOT}/reports/heldout_init_state.json" \
    --heldout-end "${OUTPUT_ROOT}/reports/heldout_end_state.json" \
    --val-init "${OUTPUT_ROOT}/reports/val_init_state.json" \
    --val-end "${OUTPUT_ROOT}/reports/val_end_state.json" \
    --output-json "${OUTPUT_ROOT}/v12_stage_a_summary.json"
fi

"${PYTHON}" - \
  "${V7_CONFIG}" \
  "${V12_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${INITIAL_CHECKPOINT}" \
  "${END_CHECKPOINT}" \
  "${PROTOCOL_REPORT}" \
  "${OUTPUT_ROOT}/provenance.json" <<'PY'
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


v7, v12, source, initialization, endpoint, protocol_path, output = sys.argv[1:]
resolved = load_config(v12)
resolved_bytes = json.dumps(
    resolved, sort_keys=True, separators=(",", ":")
).encode("utf-8")
paths = (
    v7,
    v12,
    source,
    initialization,
    endpoint,
    "dynlaneseq_eg/modeling/four_slot_selection.py",
    "dynlaneseq_eg/losses/loss_s0.py",
    "dynlaneseq_eg/tools/audit_v12_visual_first_contract.py",
    "dynlaneseq_eg/tools/audit_v12_visual_first_state.py",
    "dynlaneseq_eg/tools/summarize_v12_visual_first_gate.py",
    "scripts/run_culane_dla34_v12_visual_first_stage_a_225k_to227k.sh",
)
report = {
    "git_commit": subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True
    ).strip(),
    "git_branch": subprocess.check_output(
        ("git", "branch", "--show-current"), text=True
    ).strip(),
    "files": {str(Path(path)): {"sha256": digest(path)} for path in paths},
    "resolved_v12_config_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
    "list_protocol": json.loads(Path(protocol_path).read_text(encoding="utf-8")),
    "deployment_mode": "exact_v7",
    "legacy_route_logits_used_by_v12": False,
    "optimizer": resolved["optimizer"],
    "scheduler": resolved["scheduler"],
    "loss": resolved["loss"],
    "test_set_used": False,
}
destination = Path(output)
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print({"provenance": str(destination), "git_commit": report["git_commit"]})
PY

echo "V12 Stage-A artifacts: ${OUTPUT_ROOT}"
