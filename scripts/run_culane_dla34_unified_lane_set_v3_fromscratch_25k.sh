#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
TARGET_ITERS="${TARGET_ITERS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
AUTO_RESUME="${AUTO_RESUME:-1}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

if (( TARGET_ITERS != 25000 )); then
  echo "This architecture gate is fixed at 25,000 iterations." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Expected effective batch size 16, got $((BATCH_SIZE * GRAD_ACCUM))." >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" <<'PY'
import sys

from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
structured = cfg["model"]["structured_query"]
state = structured["lane_state"]
semantic = state["semantic_context"]
loss = cfg["loss"]
post = cfg["postprocess"]
checks = {
    "one 32-query deployable set": structured["num_instances"] == 32
    and structured["num_groups"] == 1
    and not structured.get("training_auxiliary_group_sizes"),
    "causal bidirectional lane state": state["enabled"]
    and state["mode"] == "causal_set",
    "one scalar foreground score": state["single_logit_score"],
    "P2 geometry plus P4/P5 semantics": cfg["model"]["multi_scale_evidence"]["scales"]
    == ["p4", "p5"]
    and semantic["enabled"]
    and semantic["scales"] == ["p4", "p5"],
    "strict global Hungarian": cfg["matcher"]["assignment"] == "hungarian"
    and cfg["matcher"]["num_groups"] == 1,
    "bounded matcher score": cfg["matcher"]["object_cost_type"]
    == "neg_probability"
    and cfg["matcher"]["lambda_obj"] == 0.5,
    "direct unique foreground supervision": loss["w_exist"] == 2.0
    and loss["w_quality"] == 0.0
    and loss["w_set_selection"] == 0.0,
    "set count and ranking constraints": loss["w_cardinality"] > 0.0
    and loss["w_score_margin"] > 0.0,
    "NMS-free exact score deployment": post["score_mode"] == "exist"
    and post["quality_score_power"] == 0.0
    and post["lane_nms_distance_thresh_px"] == 0.0
    and post["top_k"] == 4,
    "full learning-rate horizon": cfg["scheduler"]["total_iters"] == 278000,
}
for name, passed in checks.items():
    print(f"[{'OK' if passed else 'FAIL'}] {name}")
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("unified lane-set contract failed: " + ", ".join(failed))
print("unified lane-set v3 contract audit passed")
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
  echo "Training ended without ${checkpoint}." >&2
  exit 1
fi
echo "Unified lane-set checkpoint ready: ${checkpoint}"
