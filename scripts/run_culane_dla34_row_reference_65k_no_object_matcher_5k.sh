#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_no_object_matcher_5k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/dla34_rowref_from65k_no_object_matcher_5k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
AUTO_RESUME="${AUTO_RESUME:-1}"
EXPECTED_LAMBDA_OBJ="${EXPECTED_LAMBDA_OBJ:-0.0}"
TARGET_ITERATION=70000
FINAL_CHECKPOINT="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"

if [[ -z "${SOURCE_CHECKPOINT:-}" ]]; then
  for candidate in \
    outputs/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt \
    outputs/diagnostics/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt
  do
    if [[ -f "${candidate}" ]]; then
      SOURCE_CHECKPOINT="${candidate}"
      break
    fi
  done
fi

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Refusing unmatched effective batch size: $((BATCH_SIZE * GRAD_ACCUM)); expected 16." >&2
  exit 1
fi
if [[ -z "${SOURCE_CHECKPOINT:-}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing exact 65k source checkpoint. Set SOURCE_CHECKPOINT explicitly." >&2
  exit 1
fi
if [[ -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Matcher-weight diagnostic already reached 70k: ${FINAL_CHECKPOINT}"
  exit 0
fi

checkpoint_audit="$(
  "${PYTHON}" -c '
import math
import sys
import torch

path = sys.argv[1]
payload = torch.load(path, map_location="cpu")
iteration = int(payload.get("iteration", -1))
groups = {
    str(group.get("name", "")): float(group["lr"])
    for group in payload.get("optimizer", {}).get("param_groups", [])
}
if iteration != 65000:
    raise SystemExit(f"expected exact iteration 65000, got {iteration} in {path}")
expected = {
    "backbone_decay": 1e-6,
    "backbone_no_decay": 1e-6,
    "evidence_decay": 1e-5,
    "evidence_no_decay": 1e-5,
    "model_decay": 1e-5,
    "model_no_decay": 1e-5,
}
missing = [name for name in expected if name not in groups]
wrong = [
    (name, groups.get(name), value)
    for name, value in expected.items()
    if name in groups
    and not math.isclose(groups[name], value, rel_tol=2e-3, abs_tol=1e-12)
]
if missing or wrong:
    raise SystemExit(
        "optimizer LR audit failed: "
        + repr({"missing": missing, "wrong": wrong, "groups": groups})
    )
print(", ".join(f"{name}={groups[name]:.3g}" for name in expected))
' "${SOURCE_CHECKPOINT}"
)"

RUN_ITERS=5000
START_ITERATION=65000
RUN_CHECKPOINT="${SOURCE_CHECKPOINT}"
RESUME_DESCRIPTION="65k source with optimizer-group remapping"
TRAIN_RESUME_ARGS=(
  --resume "${SOURCE_CHECKPOINT}"
  --resume-remap-optimizer-groups
)

LATEST_PARTIAL=""
if [[ -d "${OUT_DIR}" ]]; then
  LATEST_PARTIAL="$(
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' -print 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
fi
if [[ -n "${LATEST_PARTIAL}" ]]; then
  if [[ "${AUTO_RESUME}" != "1" ]]; then
    echo "Partial candidate run exists: ${LATEST_PARTIAL}" >&2
    echo "Set AUTO_RESUME=1 or use a new empty OUT_DIR." >&2
    exit 1
  fi
  START_ITERATION="$(
    "${PYTHON}" -c '
import math
import sys
import torch

path = sys.argv[1]
expected_lambda_obj = float(sys.argv[2])
payload = torch.load(path, map_location="cpu")
iteration = int(payload.get("iteration", -1))
if not 65000 < iteration < 70000:
    raise SystemExit(
        f"partial checkpoint iteration must be between 65000 and 70000, "
        f"got {iteration} in {path}"
    )
cfg = payload.get("cfg")
if not isinstance(cfg, dict):
    raise SystemExit("partial checkpoint is missing its expanded config")
matcher = cfg.get("matcher", {})
actual_lambda_obj = float(matcher.get("lambda_obj", float("nan")))
if not math.isclose(
    actual_lambda_obj, expected_lambda_obj, rel_tol=0.0, abs_tol=1e-12
):
    raise SystemExit(
        "refusing non-candidate partial checkpoint: "
        f"matcher.lambda_obj={matcher.get('lambda_obj')}, "
        f"expected {expected_lambda_obj}"
    )
if "optimizer" not in payload or "scheduler" not in payload:
    raise SystemExit(
        "partial checkpoint must contain optimizer and scheduler state"
    )
print(iteration)
' "${LATEST_PARTIAL}" "${EXPECTED_LAMBDA_OBJ}"
  )"
  RUN_ITERS="$((TARGET_ITERATION - START_ITERATION))"
  RUN_CHECKPOINT="${LATEST_PARTIAL}"
  RESUME_DESCRIPTION="candidate partial with optimizer and scheduler restoration"
  TRAIN_RESUME_ARGS=(--resume "${LATEST_PARTIAL}")
fi

config_audit="$(
  "${PYTHON}" -c '
import math
import sys
from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
expected_lambda_obj = float(sys.argv[2])
matcher = cfg["matcher"]
if not math.isclose(
    float(matcher["lambda_obj"]),
    expected_lambda_obj,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        f"expected matcher.lambda_obj={expected_lambda_obj}, "
        f"got {matcher['"'"'lambda_obj'"'"']}"
    )
expected = {
    "lambda_point": 5.0,
    "lambda_range": 1.0,
    "lambda_line_iou": 1.0,
    "line_iou_radius": 15.0,
}
wrong = {
    key: matcher.get(key)
    for key, value in expected.items()
    if float(matcher.get(key, float("nan"))) != value
}
if wrong:
    raise SystemExit(f"geometry matcher audit failed: {wrong}")
if cfg["model"]["structured_query"].get("set_selection", {}).get("enabled", False):
    raise SystemExit("separate set-selection head must be disabled")
if float(cfg["loss"].get("w_set_selection", 0.0)) != 0.0:
    raise SystemExit("set-selection loss must be disabled")
print(
    f"lambda_obj={expected_lambda_obj:g}, lambda_point=5, lambda_range=1, "
    "lambda_line_iou=1, one matcher per supervised output"
)
' "${CONFIG}" "${EXPECTED_LAMBDA_OBJ}"
)"

echo "Matcher-weight causal diagnostic: 65000 -> ${TARGET_ITERATION}"
echo "Source checkpoint: ${SOURCE_CHECKPOINT}"
echo "Stored LR audit: ${checkpoint_audit}"
echo "Matcher audit: ${config_audit}"
echo "Architecture/new parameters: unchanged/none"
echo "Run checkpoint: ${RUN_CHECKPOINT}"
echo "Resume mode: ${RESUME_DESCRIPTION}"
echo "Remaining optimizer steps: ${RUN_ITERS}"
echo "Old AdamW moments: preserved"
echo "Old parameter LRs/schedule: exact first 5k of the matched cooldown control"
echo "Output: ${OUT_DIR}"
echo "Batch/accum/effective: ${BATCH_SIZE}/${GRAD_ACCUM}/$((BATCH_SIZE * GRAD_ACCUM))"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: audits passed; training was not started."
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --max-iters "${RUN_ITERS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}" \
  "${TRAIN_RESUME_ARGS[@]}"
