#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION=225000
STAGE_A_STEPS=2000
CHECKPOINT_INTERVAL=500
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TRAJECTORY="${RUN_TRAJECTORY:-1}"
RUN_STAGE_B="${RUN_STAGE_B:-1}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
V14_A_CONFIG="${V14_A_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v14_corrected_visual_first_stage_a_225k_to227k.yaml}"
V14_B_CONFIG="${V14_B_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v14_corrected_visual_first_stage_b_227k_to229k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v14_corrected_visual_first_stage_ab_225k}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
STAGE_A_INITIAL="${OUTPUT_ROOT}/stage_a/initialization/iter_0225000.pt"
STAGE_A_TRAIN_DIR="${OUTPUT_ROOT}/stage_a/train"
STAGE_A_ENDPOINT="${STAGE_A_TRAIN_DIR}/iter_0227000.pt"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V7_CONFIG}" \
  "${V14_A_CONFIG}" \
  "${V14_B_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V14 artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration 225000." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V14 Stage A requires effective batch 16." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/reports/stage_a_trajectory" \
  "${OUTPUT_ROOT}/stage_a/initialization" \
  "${STAGE_A_TRAIN_DIR}" \
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
  --optimizer-steps "${STAGE_A_STEPS}" \
  --effective-batch-size "$((BATCH_SIZE * GRAD_ACCUM))" \
  --experiment-name "V14 corrected visual-first clip-disjoint list contract" \
  --output-json "${OUTPUT_ROOT}/audits/bridge_list_protocol.json"

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

if [[ ! -f "${STAGE_A_INITIAL}" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.initialize_v14_corrected_visual_first_checkpoint \
    --config "${V14_A_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${STAGE_A_INITIAL}" \
    --output-json "${OUTPUT_ROOT}/audits/stage_a_initialization.json"
fi

STAGE_A_CONTRACT="${OUTPUT_ROOT}/audits/stage_a_zero_step_contract.json"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v14_corrected_visual_first_contract \
    --config "${V14_A_CONFIG}" \
    --source-config "${V7_CONFIG}" \
    --checkpoint "${STAGE_A_INITIAL}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --num-workers 0 \
    --start-iteration "${SOURCE_ITERATION}" \
    --cross-clip-report "${HELDOUT_CROSSCLIP}" \
    --cross-clip-report "${VAL_CROSSCLIP}" \
    --output-json "${STAGE_A_CONTRACT}"
fi
GATE0_PASSED="$(${PYTHON} - "${STAGE_A_CONTRACT}" <<'PY'
import json,sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if report.get("passed") is True else "0")
PY
)"
if [[ "${GATE0_PASSED}" != "1" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}" "${V7_CONFIG}" "${V14_A_CONFIG}" \
    "${V14_B_CONFIG}" "${SOURCE_V7_CHECKPOINT}" "${STAGE_A_INITIAL}" \
    "${STAGE_A_CONTRACT}" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
items={
    "v7_config":Path(sys.argv[2]), "stage_a_config":Path(sys.argv[3]),
    "stage_b_config":Path(sys.argv[4]), "source_v7_checkpoint":Path(sys.argv[5]),
    "stage_a_initialization":Path(sys.argv[6]), "stage_a_gate0":Path(sys.argv[7]),
}
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()
(root/"provenance.json").write_text(json.dumps({
    "git_commit":subprocess.check_output(("git","rev-parse","HEAD"),text=True).strip(),
    "artifacts":{k:{"path":str(v.resolve()),"sha256":sha(v)} for k,v in items.items()},
    "checkpoint_selection_performed":False,"test_set_used":False,
},indent=2,sort_keys=True)+"\n")
(root/"v14_completion.json").write_text(json.dumps({
    "experiment":"V14 corrected visual-first Stage A/B",
    "stage_a_gate0_passed":False,"stage_a_training_executed":False,
    "stage_b_executed":False,"long_training_authorized":False,
    "test_set_used":False,"decision":"v14_complete_gate0_fail_stop_for_sol",
},indent=2,sort_keys=True)+"\n")
PY
  echo "V14 Stage-A Gate 0 FAIL: no optimizer step or later stage is authorized."
  exit 0
fi

train_stage_a() {
  if [[ -f "${STAGE_A_ENDPOINT}" ]]; then
    echo "Reusing fixed V14 Stage-A endpoint ${STAGE_A_ENDPOINT}"
    return
  fi
  local latest=""
  local latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${STAGE_A_TRAIN_DIR}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    local iteration
    iteration="$(checkpoint_iteration "${candidate}")"
    if (( iteration > latest_iteration && iteration < 227000 )); then
      latest="${candidate}"
      latest_iteration="${iteration}"
    fi
  done
  local args=(
    --config "${V14_A_CONFIG}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --output-dir "${STAGE_A_TRAIN_DIR}"
    --checkpoint-base "${STAGE_A_INITIAL}"
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
    args+=(--resume "${latest}" --max-iters "$((227000 - latest_iteration))")
  else
    args+=(
      --init-from "${STAGE_A_INITIAL}"
      --init-iteration "${SOURCE_ITERATION}"
      --max-iters "${STAGE_A_STEPS}"
    )
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${args[@]}" \
    2>&1 | tee -a "${STAGE_A_TRAIN_DIR}/train.log"
}

state_report() {
  local checkpoint="$1"
  local split="$2"
  local list_path="$3"
  local wrong_list="$4"
  local crossclip="$5"
  local report="$6"
  [[ -f "${report}" ]] && return
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v14_corrected_visual_first_state \
    --config "${V14_A_CONFIG}" \
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
    --output-json "${report}"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_stage_a
fi
if [[ ! -f "${STAGE_A_ENDPOINT}" ]]; then
  echo "Missing fixed V14 Stage-A endpoint: ${STAGE_A_ENDPOINT}" >&2
  exit 1
fi

HELDOUT_END="${OUTPUT_ROOT}/reports/stage_a_heldout_endpoint.json"
VAL_END="${OUTPUT_ROOT}/reports/stage_a_validation_endpoint.json"
state_report "${STAGE_A_ENDPOINT}" train "${HELDOUT_LIST}" \
  "${HELDOUT_WRONG_LIST}" "${HELDOUT_CROSSCLIP}" "${HELDOUT_END}"
state_report "${STAGE_A_ENDPOINT}" val "${VAL_LIST}" \
  "${VAL_WRONG_LIST}" "${VAL_CROSSCLIP}" "${VAL_END}"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.summarize_v14_corrected_visual_first_stage_a \
  --contract "${STAGE_A_CONTRACT}" \
  --heldout-end "${HELDOUT_END}" \
  --val-end "${VAL_END}" \
  --output-json "${OUTPUT_ROOT}/v14_stage_a_summary.json"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.audit_v14_stage_a_gradient_population \
  --config "${V14_A_CONFIG}" \
  --checkpoint "${STAGE_A_ENDPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --list-path "${TRAIN_LIST}" \
  --device "${DEVICE}" \
  --batches 32 \
  --batch-size 4 \
  --num-workers "${NUM_WORKERS}" \
  --output-json "${OUTPUT_ROOT}/audits/stage_a_endpoint_gradient_population.json"

if [[ "${RUN_TRAJECTORY}" == "1" ]]; then
  for iteration in 225000 225500 226000 226500 227000; do
    if (( iteration == 225000 )); then
      checkpoint="${STAGE_A_INITIAL}"
    else
      checkpoint="${STAGE_A_TRAIN_DIR}/iter_$(printf '%07d' "${iteration}").pt"
    fi
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing fixed trajectory checkpoint ${checkpoint}" >&2
      exit 1
    fi
    state_report "${checkpoint}" train "${HELDOUT_LIST}" \
      "${HELDOUT_WRONG_LIST}" "${HELDOUT_CROSSCLIP}" \
      "${OUTPUT_ROOT}/reports/stage_a_trajectory/heldout_$(printf '%07d' "${iteration}").json"
    state_report "${checkpoint}" val "${VAL_LIST}" \
      "${VAL_WRONG_LIST}" "${VAL_CROSSCLIP}" \
      "${OUTPUT_ROOT}/reports/stage_a_trajectory/validation_$(printf '%07d' "${iteration}").json"
  done
  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v14_stage_a_trajectory \
    --report-root "${OUTPUT_ROOT}/reports/stage_a_trajectory" \
    --iterations 225000 225500 226000 226500 227000 \
    --output-json "${OUTPUT_ROOT}/stage_a_trajectory_summary.json"
fi

"${PYTHON}" - \
  "${V7_CONFIG}" "${V14_A_CONFIG}" "${V14_B_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" "${STAGE_A_INITIAL}" "${STAGE_A_ENDPOINT}" \
  "${OUTPUT_ROOT}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
from dynlaneseq_eg.config import load_config

v7, a, b, source, initial, endpoint, root = sys.argv[1:]
root = Path(root)
paths = [
    v7, a, b, source, initial, endpoint,
    "dynlaneseq_eg/modeling/four_slot_selection.py",
    "dynlaneseq_eg/losses/loss_s0.py",
    "dynlaneseq_eg/tools/build_cross_clip_derangement.py",
    "dynlaneseq_eg/tools/build_v11_bridge_lists.py",
    "dynlaneseq_eg/tools/audit_v14_corrected_visual_first_contract.py",
    "dynlaneseq_eg/tools/audit_v14_corrected_visual_first_state.py",
    "dynlaneseq_eg/tools/audit_v14_stage_a_gradient_population.py",
    "dynlaneseq_eg/tools/summarize_v14_corrected_visual_first_stage_a.py",
    "scripts/run_culane_dla34_v14_corrected_visual_first_stage_ab.sh",
]
paths.extend(
    str(path)
    for path in sorted(Path("dynlaneseq_eg").rglob("*v14*"))
    if path.is_file() and path.suffix in {".py", ".yaml", ".md"}
)
paths.extend(
    str(path)
    for directory in (root/"lists", root/"audits", root/"reports")
    if directory.is_dir()
    for path in sorted(directory.rglob("*"))
    if path.is_file() and path.suffix in {".json", ".txt", ".yaml", ".md"}
)
paths = list(dict.fromkeys(paths))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()
resolved = {"stage_a": load_config(a), "stage_b": load_config(b)}
(root/"resolved_configs.json").write_text(json.dumps(resolved, indent=2, sort_keys=True)+"\n")
manifest = {
    "git_commit": subprocess.check_output(("git","rev-parse","HEAD"), text=True).strip(),
    "git_branch": subprocess.check_output(("git","branch","--show-current"), text=True).strip(),
    "files": {str(Path(p)): {"sha256": digest(p)} for p in paths},
    "resolved_config_sha256": {
        key: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",",":")).encode()).hexdigest()
        for key,value in resolved.items()
    },
    "fixed_stage_a_endpoint": str(Path(endpoint).resolve()),
    "checkpoint_selection_performed": False,
    "test_set_used": False,
}
(root/"provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

STAGE_A_PASSED="$("${PYTHON}" - "${OUTPUT_ROOT}/v14_stage_a_summary.json" <<'PY'
import json,sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text()).get("passed") is True else "0")
PY
)"

if [[ "${STAGE_A_PASSED}" != "1" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}/v14_completion.json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
path.write_text(json.dumps({
    "experiment":"V14 corrected visual-first Stage A/B",
    "stage_a_passed":False,
    "stage_b_executed":False,
    "full_validation_executed":False,
    "long_training_authorized":False,
    "test_set_used":False,
    "decision":"v14_complete_stage_a_fail_stop_for_sol",
}, indent=2, sort_keys=True)+"\n")
PY
  echo "V14 Stage A FAIL: Stage B and long training are closed."
  echo "V14 is complete; stop and package these artifacts for Sol Pro."
  exit 0
fi
if [[ "${RUN_STAGE_B}" != "1" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}/v14_completion.json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
path.write_text(json.dumps({
    "experiment":"V14 corrected visual-first Stage A/B",
    "stage_a_passed":True,
    "stage_b_executed":False,
    "full_validation_executed":False,
    "long_training_authorized":False,
    "test_set_used":False,
    "decision":"v14_stage_a_pass_manual_stage_b_closed",
}, indent=2, sort_keys=True)+"\n")
PY
  echo "V14 Stage A PASS, but RUN_STAGE_B=0. No long training is authorized."
  exit 0
fi

# Stage B is intentionally a second gate inside the same V14 version.  Its
# runner is invoked only by the immutable Stage-A summary above.
"${PYTHON}" -u -m dynlaneseq_eg.tools.run_v14_stage_b \
  --stage-a-config "${V14_A_CONFIG}" \
  --stage-b-config "${V14_B_CONFIG}" \
  --v7-config "${V7_CONFIG}" \
  --stage-a-checkpoint "${STAGE_A_ENDPOINT}" \
  --source-v7-checkpoint "${SOURCE_V7_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --train-list "${TRAIN_LIST}" \
  --heldout-list "${HELDOUT_LIST}" \
  --heldout-wrong-list "${HELDOUT_WRONG_LIST}" \
  --heldout-cross-clip-report "${HELDOUT_CROSSCLIP}" \
  --val-list "${VAL_LIST}" \
  --val-wrong-list "${VAL_WRONG_LIST}" \
  --val-cross-clip-report "${VAL_CROSSCLIP}" \
  --output-root "${OUTPUT_ROOT}/stage_b" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --num-workers "${NUM_WORKERS}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --metric-workers "${METRIC_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}"

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1])
source=root/"stage_b"/"v14_completion.json"
if not source.is_file():
    raise SystemExit("Stage-B returned without a completion contract")
payload=json.loads(source.read_text())
(root/"v14_completion.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True)+"\n"
)
PY

echo "V14 Stage A/B completed. STOP: no V15 or long continuation is authorized."
