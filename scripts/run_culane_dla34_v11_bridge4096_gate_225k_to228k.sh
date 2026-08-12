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
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
V11_CONFIG="${V11_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v11_bridge4096_225k_to228k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v11_bridge4096_gate_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/v11_bridge4096_gate_225k}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${OUTPUT_ROOT}/initialization/iter_0225000.pt}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
PROTOCOL_REPORT="${OUTPUT_ROOT}/audits/bridge_list_protocol.json"
CONTRACT_REPORT="${OUTPUT_ROOT}/audits/v11_initialization_contract.json"

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
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V11 bridge artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi
if (( TRAIN_STEPS != 3000 || CHECKPOINT_INTERVAL != 500 )); then
  echo "V11 bridge is predeclared as 3k steps with 500-step checkpoints." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V11 bridge requires effective batch size 16." >&2
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/initialization" \
  "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/train" \
  "${LIST_ROOT}" \
  "${CACHE_ROOT}"

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
    raise SystemExit("V11 bridge list/leakage contract failed")
PY

TRAIN_LIST="${LIST_ROOT}/train_clip512_image4096.txt"
SEEN_LIST="${LIST_ROOT}/seen_train_image256.txt"
SAME_CLIP_LIST="${LIST_ROOT}/same_clip_unseen_image256.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"

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
  "${SOURCE_V7_CHECKPOINT}" \
  "${INITIAL_CHECKPOINT}" \
  "${PROTOCOL_REPORT}" \
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


v7_config, v11_config, source, initialization, protocol_path, output = sys.argv[1:]
resolved = load_config(v11_config)
resolved_bytes = json.dumps(
    resolved,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
code_paths = (
    "dynlaneseq_eg/tools/build_v11_bridge_lists.py",
    "dynlaneseq_eg/tools/summarize_v11_bridge_gate.py",
    "scripts/run_culane_dla34_v11_bridge4096_gate_225k_to228k.sh",
)
report = {
    "git_commit": subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True
    ).strip(),
    "git_branch": subprocess.check_output(
        ("git", "branch", "--show-current"), text=True
    ).strip(),
    "files": {
        str(Path(path)): {"sha256": digest(path)}
        for path in (v7_config, v11_config, source, initialization, *code_paths)
    },
    "resolved_v11_config_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
    "list_protocol": protocol,
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

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
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
    raise SystemExit("V11 initialization contract failed; bridge training closed")
PY

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
TRAIN_DIR="${OUTPUT_ROOT}/train/v11"
END_CHECKPOINT="${TRAIN_DIR}/iter_${END_TAG}.pt"

train_arm() {
  mkdir -p "${TRAIN_DIR}"
  if [[ -f "${END_CHECKPOINT}" ]]; then
    echo "V11 bridge: reusing ${END_CHECKPOINT}"
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
    --config "${V11_CONFIG}"
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

coverage_report() {
  local config="$1"
  local checkpoint="$2"
  local report="$3"
  local cache="$4"
  local split="$5"
  local list_path="$6"
  [[ -f "${report}" ]] && return
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --device "${DEVICE}" \
    --cache-dir "${cache}" \
    --max-batches 0 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --amp-dtype none \
    --sample-strategy sequential \
    --stage main \
    --top-k 4 \
    --iou-thresholds 0.50 0.75 \
    --near-min-iou 0.30 \
    --line-width 30 \
    --min-valid-rows 5 \
    --hard-diversity-distances 20 \
    --mmr-sigmas 20 \
    --mmr-penalties 0.50 \
    --output-json "${report}"
}

state_report() {
  local config="$1"
  local checkpoint="$2"
  local report="$3"
  local split="$4"
  local list_path="$5"
  [[ -f "${report}" ]] && return
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v11_unified_checkpoint_state \
    --config "${config}" \
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

evaluate_domain() {
  local name="$1"
  local split="$2"
  local list_path="$3"
  local prefix="${OUTPUT_ROOT}/reports/${name}"
  coverage_report "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" \
    "${prefix}_source_v7_coverage.json" "${CACHE_ROOT}/${name}_source" \
    "${split}" "${list_path}"
  coverage_report "${V11_CONFIG}" "${INITIAL_CHECKPOINT}" \
    "${prefix}_v11_init_coverage.json" "${CACHE_ROOT}/${name}_init" \
    "${split}" "${list_path}"
  coverage_report "${V11_CONFIG}" "${END_CHECKPOINT}" \
    "${prefix}_v11_end_coverage.json" "${CACHE_ROOT}/${name}_end" \
    "${split}" "${list_path}"
  state_report "${V11_CONFIG}" "${INITIAL_CHECKPOINT}" \
    "${prefix}_v11_init_state.json" "${split}" "${list_path}"
  state_report "${V11_CONFIG}" "${END_CHECKPOINT}" \
    "${prefix}_v11_end_state.json" "${split}" "${list_path}"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_arm
fi

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "V11 bridge training stage complete; evaluation remains paused."
  exit 0
fi
if [[ ! -f "${END_CHECKPOINT}" ]]; then
  echo "Missing fixed endpoint for evaluation: ${END_CHECKPOINT}" >&2
  exit 1
fi

evaluate_domain seen_train train "${SEEN_LIST}"
evaluate_domain same_clip_unseen train "${SAME_CLIP_LIST}"
evaluate_domain heldout_clip train "${HELDOUT_LIST}"
evaluate_domain val val "${VAL_LIST}"

MANIFEST="${OUTPUT_ROOT}/audits/bridge_summary_manifest.json"
"${PYTHON}" - \
  "${PROTOCOL_REPORT}" \
  "${CONTRACT_REPORT}" \
  "${OUTPUT_ROOT}" \
  "${END_ITERATION}" \
  "${SEEN_LIST}" \
  "${SAME_CLIP_LIST}" \
  "${HELDOUT_LIST}" \
  "${VAL_LIST}" \
  "${MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

protocol, contract, root, iteration, seen, same, heldout, val, output = sys.argv[1:]
root = Path(root).resolve()


def domain(name, split, list_path):
    prefix = root / "reports" / name
    return {
        "split": split,
        "list_path": str(Path(list_path).resolve()),
        "source_coverage": str(prefix) + "_source_v7_coverage.json",
        "init_coverage": str(prefix) + "_v11_init_coverage.json",
        "end_coverage": str(prefix) + "_v11_end_coverage.json",
        "init_state": str(prefix) + "_v11_init_state.json",
        "end_state": str(prefix) + "_v11_end_state.json",
    }


manifest = {
    "protocol": str(Path(protocol).resolve()),
    "contract": str(Path(contract).resolve()),
    "iteration": int(iteration),
    "test_set_used": False,
    "domains": {
        "seen_train": domain("seen_train", "train", seen),
        "same_clip_unseen": domain("same_clip_unseen", "train", same),
        "heldout_clip": domain("heldout_clip", "train", heldout),
        "val": domain("val", "val", val),
    },
}
destination = Path(output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

SUMMARY="${OUTPUT_ROOT}/v11_bridge4096_summary.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v11_bridge_gate \
  --manifest "${MANIFEST}" \
  --output-json "${SUMMARY}"

"${PYTHON}" - "${SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print({
    "v11_bridge_passed": report.get("passed", False),
    "diagnosis": report.get("diagnosis"),
    "long_training_authorized": report.get("long_training_authorized"),
    "next_action": report.get("next_action"),
})
PY

echo "V11 bridge gate complete. Test split and long training remain disabled."
