#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_coherent_lane_state_25k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_coherent_lane_state_25k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
TARGET_ITERS="${TARGET_ITERS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
AUTO_RESUME="${AUTO_RESUME:-1}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

if (( TARGET_ITERS != 25000 )); then
  echo "This controlled gate is predeclared at exactly 25,000 iterations." >&2
  exit 1
fi
if (( BATCH_SIZE != 4 || GRAD_ACCUM != 4 )); then
  echo "The matched gate requires BATCH_SIZE=4 and GRAD_ACCUM=4." >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" <<'PY'
import sys

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_matcher

cfg = load_config(sys.argv[1])
structured = cfg["model"]["structured_query"]
row_reference = structured["row_reference"]
lane_state = structured["lane_state"]
matcher = build_matcher(cfg).cfg
loss = cfg["loss"]
post = cfg["postprocess"]

checks = {
    "32 deployable primary queries": int(structured["num_instances"]) == 32,
    "no train-only auxiliary queries": not structured.get(
        "training_auxiliary_group_sizes"
    ),
    "one interaction group": int(structured["num_groups"]) == 1,
    "persistent lane state": bool(lane_state["enabled"]),
    "row reference enabled": bool(row_reference["enabled"]),
    "reference detached between layers": bool(
        row_reference["detach_between_layers"]
    ),
    "separate selector disabled": not bool(
        structured.get("set_selection", {}).get("enabled", False)
    ),
    "strict Hungarian assignment": str(matcher.assignment) == "hungarian",
    "bounded object matcher cost": str(matcher.object_cost_type)
    == "neg_probability",
    "matcher object weight 0.5": float(matcher.lambda_obj) == 0.5,
    "independent per-layer matching": not bool(
        cfg["matcher"]["reuse_final_assignment_for_intermediate"]
    ),
    "direct existence supervision": float(loss["w_exist"]) == 2.0,
    "quality supervision disabled": float(loss["w_quality"]) == 0.0,
    "set selector supervision disabled": float(loss["w_set_selection"]) == 0.0,
    "direct existence deployment score": str(post["score_mode"]) == "exist",
    "NMS-free deployment": float(post["lane_nms_distance_thresh_px"]) == 0.0,
    "direct Top-4 deployment": int(post["top_k"]) == 4,
    "full scheduler horizon": int(cfg["scheduler"]["total_iters"]) == 278000,
    "matched warmup": int(cfg["scheduler"]["warmup_iters"]) == 1000,
    "matched seed": int(cfg["training"]["seed"]) == 3407,
}
failed = [name for name, passed in checks.items() if not passed]
for name, passed in checks.items():
    print(f"[{'OK' if passed else 'FAIL'}] {name}")
if failed:
    raise SystemExit("coherent lane-state contract failed: " + ", ".join(failed))
print("coherent lane-state 25k contract audit passed")
PY

if [[ "${AUDIT_ONLY}" == "1" ]]; then
  exit 0
fi

DATA_ROOT="${DATA_ROOT}" \
DEVICE="${DEVICE}" \
CONFIG="${CONFIG}" \
OUT_DIR="${OUT_DIR}" \
TARGET_ITERS="${TARGET_ITERS}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
AUTO_RESUME="${AUTO_RESUME}" \
  bash scripts/run_culane_dla34_row_reference_full_278k.sh

checkpoint="${OUT_DIR}/iter_0025000.pt"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Training ended without the declared 25k checkpoint: ${checkpoint}" >&2
  exit 1
fi
echo "Coherent lane-state gate checkpoint ready: ${checkpoint}"
