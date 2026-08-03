#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
TARGET_ITERS="${TARGET_ITERS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
AUTO_RESUME="${AUTO_RESUME:-1}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

if (( TARGET_ITERS < 1 || TARGET_ITERS > 278000 )); then
  echo "TARGET_ITERS must be in [1, 278000], got ${TARGET_ITERS}." >&2
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
rowref = structured["row_reference"]
state = structured["lane_state"]
matcher = cfg["matcher"]
loss = cfg["loss"]
post = cfg["postprocess"]
optimizer = cfg["optimizer"]
groups = {entry["name"]: entry for entry in optimizer["parameter_groups"]}
offsets = tuple(float(value) for value in rowref["delta_offsets_px"])
evidence_offsets = tuple(float(value) for value in rowref["offsets_px"])
checks = {
    "one deployable 32-query set": structured["num_instances"] == 32
    and structured["num_groups"] == 1
    and not structured.get("training_auxiliary_group_sizes"),
    "image-grounded row reference": rowref["enabled"]
    and rowref["detach_between_layers"],
    "bounded local coordinate action": rowref["prediction_mode"] == "bounded_delta"
    and len(offsets) == 33
    and offsets[0] == -96.0
    and offsets[-1] == 96.0
    and abs(sum(offsets)) < 1e-9,
    "bounded local evidence profile": len(evidence_offsets) == 17
    and evidence_offsets[0] == -96.0
    and evidence_offsets[-1] == 96.0
    and abs(sum(evidence_offsets)) < 1e-9,
    "score gradient isolated from geometry": state["detach_score_geometry"],
    "geometry-only Hungarian": matcher["assignment"] == "hungarian"
    and matcher["num_groups"] == 1
    and matcher["lambda_obj"] == 0.0,
    "stable final assignment reused": matcher[
        "reuse_final_assignment_for_intermediate"
    ],
    "intermediate geometry-only supervision": loss["w_intermediate_exist"] == 0.0,
    "final IoU-aware single score": state["single_logit_score"]
    and loss["exist_target_mode"] == "iou_aware"
    and loss["exist_quality_floor"] == 0.5
    and loss["w_quality"] == 0.0
    and loss["w_set_selection"] == 0.0,
    "no count or binary ranking shortcut": loss["w_cardinality"] == 0.0
    and loss["w_score_margin"] == 0.0,
    "separate low-LR recurrent geometry": groups["lane_state"]["lr"] == 5e-5
    and groups["coordinate"]["lr"] == 5e-5
    and groups["lane_identity"]["lr"] == 5e-5
    and optimizer["evidence_lr"] == 1e-4,
    "278k scheduler and training contract": cfg["scheduler"]["total_iters"] == 278000
    and cfg["training"]["max_iters"] == 278000
    and cfg["training"]["checkpoint_interval"] == 5000,
    "single-score NMS-free deployment": post["score_mode"] == "exist"
    and post["quality_score_power"] == 0.0
    and post["lane_nms_distance_thresh_px"] == 0.0
    and post["top_k"] == 4,
}
for name, passed in checks.items():
    print(f"[{'OK' if passed else 'FAIL'}] {name}")
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("unified lane-set V4 contract failed: " + ", ".join(failed))
print("unified lane-set V4 bounded-delta 278k contract audit passed")
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

checkpoint="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERS}").pt"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Training ended without ${checkpoint}." >&2
  exit 1
fi
echo "Unified lane-set V4 checkpoint ready: ${checkpoint}"
