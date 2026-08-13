#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION="${SOURCE_ITERATION:-227000}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

V12_CONFIG="${V12_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v12_visual_first_association_stage_a_225k_to227k.yaml}"
V13_CONFIG="${V13_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v13_visual_precision_geometry_227k_to229k.yaml}"
SOURCE_V12_CHECKPOINT="${SOURCE_V12_CHECKPOINT:-outputs/diagnostics/v12_visual_first_stage_a_225k/train/v12/iter_0227000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v13_visual_precision_geometry_227k}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${OUTPUT_ROOT}/initialization/iter_0227000.pt}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
PROTOCOL_REPORT="${OUTPUT_ROOT}/audits/bridge_list_protocol.json"
CONTRACT_REPORT="${OUTPUT_ROOT}/audits/v13_zero_step_contract.json"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V12_CONFIG}" \
  "${V13_CONFIG}" \
  "${SOURCE_V12_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V13 bridge artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V12_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V12_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi
if (( TRAIN_STEPS != 2000 || CHECKPOINT_INTERVAL != 500 )); then
  echo "V13 bridge is predeclared as 2k steps with 500-step checkpoints." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V13 bridge requires effective batch size 16." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/initialization" \
  "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/train" \
  "${LIST_ROOT}"

# Rebuild the exact deterministic bridge protocol rather than borrowing a
# mutable list path from another experiment.  The protocol keeps training,
# held-out clips and validation disjoint; the test set is never opened.
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
    raise SystemExit("V13 bridge list/leakage contract failed")
PY

TRAIN_LIST="${LIST_ROOT}/train_clip512_image4096.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"

if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.initialize_v13_visual_precision_checkpoint \
    --config "${V13_CONFIG}" \
    --source-checkpoint "${SOURCE_V12_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v13_visual_precision_contract \
    --config "${V13_CONFIG}" \
    --source-config "${V12_CONFIG}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --source-checkpoint "${SOURCE_V12_CHECKPOINT}" \
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
print({"v13_zero_step_contract_passed": report.get("passed", False)})
if report.get("passed") is not True:
    raise SystemExit("V13 zero-step contract failed; training closed")
PY

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
TRAIN_DIR="${OUTPUT_ROOT}/train/v13"
END_CHECKPOINT="${TRAIN_DIR}/iter_${END_TAG}.pt"

train_arm() {
  mkdir -p "${TRAIN_DIR}"
  if [[ -f "${END_CHECKPOINT}" ]]; then
    echo "V13 bridge: reusing ${END_CHECKPOINT}"
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
    --config "${V13_CONFIG}"
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

official_report() {
  local report="$1"
  local split="$2"
  local list_path="$3"
  [[ -f "${report}" ]] && return
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v13_visual_precision_official \
    --config "${V13_CONFIG}" \
    --checkpoint "${END_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --sample-strategy sequential \
    --max-images 0 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --iou-thresholds 0.50 0.75 \
    --line-width 30 \
    --min-valid-rows 5 \
    --output-json "${report}"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_arm
fi
if [[ ! -f "${END_CHECKPOINT}" ]]; then
  echo "Missing fixed V13 endpoint: ${END_CHECKPOINT}" >&2
  exit 1
fi

HELDOUT_REPORT="${OUTPUT_ROOT}/reports/heldout_v13_official.json"
VAL_REPORT="${OUTPUT_ROOT}/reports/validation_v13_official.json"
if [[ "${RUN_EVAL}" == "1" ]]; then
  official_report "${HELDOUT_REPORT}" train "${HELDOUT_LIST}"
  official_report "${VAL_REPORT}" val "${VAL_LIST}"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v13_visual_precision_gate \
    --contract "${CONTRACT_REPORT}" \
    --heldout "${HELDOUT_REPORT}" \
    --validation "${VAL_REPORT}" \
    --output-json "${OUTPUT_ROOT}/v13_bridge_summary.json"
fi

"${PYTHON}" - \
  "${V12_CONFIG}" \
  "${V13_CONFIG}" \
  "${SOURCE_V12_CHECKPOINT}" \
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


v12, v13, source, initialization, endpoint, protocol_path, output = sys.argv[1:]
resolved = load_config(v13)
resolved_bytes = json.dumps(
    resolved, sort_keys=True, separators=(",", ":")
).encode("utf-8")
paths = (
    v12,
    v13,
    source,
    initialization,
    endpoint,
    "dynlaneseq_eg/modeling/four_slot_selection.py",
    "dynlaneseq_eg/modeling/structured_queries.py",
    "dynlaneseq_eg/losses/loss_s0.py",
    "dynlaneseq_eg/factory.py",
    "dynlaneseq_eg/tools/initialize_v13_visual_precision_checkpoint.py",
    "dynlaneseq_eg/tools/audit_v13_visual_precision_contract.py",
    "dynlaneseq_eg/tools/audit_v13_visual_precision_official.py",
    "dynlaneseq_eg/tools/summarize_v13_visual_precision_gate.py",
    "docs/diagnostics/V13_VISUAL_PRECISION_GEOMETRY_GATE.md",
    "scripts/run_culane_dla34_v13_visual_precision_227k_to229k.sh",
)
report = {
    "git_commit": subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True
    ).strip(),
    "git_branch": subprocess.check_output(
        ("git", "branch", "--show-current"), text=True
    ).strip(),
    "files": {str(Path(path)): {"sha256": digest(path)} for path in paths},
    "resolved_v13_config_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
    "list_protocol": json.loads(Path(protocol_path).read_text(encoding="utf-8")),
    "source_iteration": 227000,
    "endpoint_iteration": 229000,
    "hard_proposal_id_produces_final_geometry": False,
    "activity_and_score_source": "exact_v7",
    "full_validation_predeclared": False,
    "long_training_authorized": False,
    "optimizer": resolved["optimizer"],
    "scheduler": resolved["scheduler"],
    "loss": resolved["loss"],
    "test_set_used": False,
}
destination = Path(output)
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print({"provenance": str(destination), "git_commit": report["git_commit"]})
PY

echo "V13 bridge artifacts: ${OUTPUT_ROOT}"
