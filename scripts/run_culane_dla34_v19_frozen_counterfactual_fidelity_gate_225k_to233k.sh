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
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
V19_CONFIG="${V19_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v19_frozen_counterfactual_fidelity_225k_to233k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v19_frozen_counterfactual_fidelity_225k}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
INITIAL_CHECKPOINT="${OUTPUT_ROOT}/initialization/iter_0225000.pt"
TRAIN_DIR="${OUTPUT_ROOT}/train"
ENDPOINT="${TRAIN_DIR}/iter_0233000.pt"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V7_CONFIG}" \
  "${V19_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V19 artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration 225000." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V19 requires effective batch 16." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/initialization" \
  "${TRAIN_DIR}" \
  "${LIST_ROOT}"

# Reproduce the exact 8192-image/640-clip bridge population used for V18.
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
  --experiment-name "V19 frozen counterfactual proposal fidelity fixed gate" \
  --output-json "${OUTPUT_ROOT}/audits/list_protocol.json"

TRAIN_LIST="${LIST_ROOT}/train_clip640_image8192.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"

if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.initialize_v19_counterfactual_fidelity_checkpoint \
    --config "${V19_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

GATE0="${OUTPUT_ROOT}/audits/zero_step_contract.json"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_contract \
    --config "${V19_CONFIG}" \
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
  echo "V19 Gate 0 report is missing." >&2
  exit 1
fi
GATE0_PASSED="$(${PYTHON} - "${GATE0}" <<'PY'
import json, sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text()).get("passed") is True else "0")
PY
)"
if [[ "${GATE0_PASSED}" != "1" ]]; then
  echo "V19 Gate 0 FAIL: no optimizer step is authorized."
  exit 0
fi

if [[ "${RUN_TRAIN}" == "1" && ! -f "${ENDPOINT}" ]]; then
  latest=""
  latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${TRAIN_DIR}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    candidate_iteration="$(checkpoint_iteration "${candidate}")"
    if (( candidate_iteration > latest_iteration && candidate_iteration < ENDPOINT_ITERATION )); then
      latest="${candidate}"
      latest_iteration="${candidate_iteration}"
    fi
  done
  command=(
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train
    --config "${V19_CONFIG}"
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
    --compile-model "${COMPILE_MODEL}"
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
fi

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "V19 training stage complete; evaluation remains paused."
  exit 0
fi
if [[ ! -f "${ENDPOINT}" ]] || (( $(checkpoint_iteration "${ENDPOINT}") != ENDPOINT_ITERATION )); then
  echo "Missing or invalid fixed V19 endpoint: ${ENDPOINT}" >&2
  exit 1
fi

official_report() {
  local split="$1"
  local list_path="$2"
  local output="$3"
  [[ -f "${output}" ]] && return
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_official \
    --config "${V19_CONFIG}" \
    --source-config "${V7_CONFIG}" \
    --checkpoint "${ENDPOINT}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --output-json "${output}"
}

official_report train "${HELDOUT_LIST}" "${OUTPUT_ROOT}/reports/heldout_official.json"
official_report val "${VAL_LIST}" "${OUTPUT_ROOT}/reports/validation_official.json"

SUMMARY="${OUTPUT_ROOT}/v19_fixed_gate_summary.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v19_counterfactual_fidelity_gate \
  --contract "${GATE0}" \
  --heldout "${OUTPUT_ROOT}/reports/heldout_official.json" \
  --validation "${OUTPUT_ROOT}/reports/validation_official.json" \
  --output-json "${SUMMARY}"

"${PYTHON}" - "${OUTPUT_ROOT}" "${V19_CONFIG}" "${SOURCE_V7_CHECKPOINT}" "${INITIAL_CHECKPOINT}" "${ENDPOINT}" "${SUMMARY}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
from dynlaneseq_eg.config import load_config

root = Path(sys.argv[1])
fixed = [Path(value) for value in sys.argv[2:]]
fixed.extend(Path(value) for value in (
    "dynlaneseq_eg/modeling/v19_counterfactual_fidelity.py",
    "dynlaneseq_eg/modeling/four_slot_selection.py",
    "dynlaneseq_eg/losses/loss_s0.py",
    "dynlaneseq_eg/tools/audit_v19_counterfactual_fidelity_contract.py",
    "dynlaneseq_eg/tools/audit_v19_counterfactual_fidelity_official.py",
    "dynlaneseq_eg/tools/summarize_v19_counterfactual_fidelity_gate.py",
    "scripts/run_culane_dla34_v19_frozen_counterfactual_fidelity_gate_225k_to233k.sh",
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
summary = json.loads(Path(sys.argv[6]).read_text())
manifest = {
    "experiment": "V19 frozen counterfactual proposal fidelity",
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "artifacts": [
        {"path": str(path), "sha256": digest(path)} for path in paths
    ],
    "resolved_config": load_config(sys.argv[2]),
    "fixed_endpoint_gate_passed": bool(summary.get("passed")),
    "decision": summary.get("decision"),
    "long_training_authorized": False,
    "full_validation_executed": False,
    "test_set_used": False,
}
(root / "v19_completion.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

echo "V19 fixed gate complete. Stop here for joint planning."
