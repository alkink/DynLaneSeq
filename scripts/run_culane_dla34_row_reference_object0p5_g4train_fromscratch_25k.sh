#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_g4train_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_g4train_g1infer_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_slots32_g4train_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
TARGET_ITERS="${TARGET_ITERS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
AUTO_RESUME="${AUTO_RESUME:-1}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

if (( TARGET_ITERS != 25000 )); then
  echo "This causal run is predeclared at exactly 25000 iterations." >&2
  echo "Got TARGET_ITERS=${TARGET_ITERS}; refusing a changed horizon." >&2
  exit 1
fi
if (( BATCH_SIZE != 4 || GRAD_ACCUM != 4 )); then
  echo "This matched run requires BATCH_SIZE=4 and GRAD_ACCUM=4." >&2
  exit 1
fi
for path in "${CONTROL_CONFIG}" "${CONFIG}" "${DEPLOY_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required config: ${path}" >&2
    exit 1
  fi
done

# Refuse accidental multi-variable experiments.  Once the intended group
# contract and inference-only view are normalized away, the candidate must be
# byte-for-byte equivalent at the resolved-config level to the object0.5
# row-reference control.
"${PYTHON}" - "${CONTROL_CONFIG}" "${CONFIG}" "${DEPLOY_CONFIG}" <<'PY'
from copy import deepcopy
import sys

from dynlaneseq_eg.config import load_config


control = load_config(sys.argv[1])
candidate = load_config(sys.argv[2])
deploy = load_config(sys.argv[3])


def strip_bookkeeping(cfg):
    cfg = deepcopy(cfg)
    cfg.pop("_config_path", None)
    cfg.pop("output_dir", None)
    return cfg


expected = strip_bookkeeping(control)
normalized_candidate = strip_bookkeeping(candidate)
normalized_candidate["model"]["structured_query"]["num_groups"] = 1
normalized_candidate["matcher"]["assignment"] = "hungarian"
normalized_candidate["matcher"]["num_groups"] = 1
if normalized_candidate != expected:
    raise SystemExit(
        "g4 candidate changes more than num_groups/assignment; refusing train"
    )

normalized_deploy = strip_bookkeeping(deploy)
normalized_train = strip_bookkeeping(candidate)
normalized_deploy["model"]["structured_query"].pop("inference_group_index", None)
normalized_deploy["postprocess"] = deepcopy(normalized_train["postprocess"])
if normalized_deploy != normalized_train:
    raise SystemExit(
        "deployment config changes train-time state beyond inference group/postprocess"
    )

structured = candidate["model"]["structured_query"]
matcher = candidate["matcher"]
if not bool(structured["row_reference"]["enabled"]):
    raise SystemExit("row-reference decoder is not enabled")
if int(structured["num_groups"]) != 4:
    raise SystemExit("candidate model.num_groups is not 4")
if matcher["assignment"] != "grouped_one_to_many" or int(matcher["num_groups"]) != 4:
    raise SystemExit("candidate matcher is not four-group one-to-many")
if float(matcher["lambda_obj"]) != 0.5:
    raise SystemExit("candidate matcher.lambda_obj is not 0.5")
if int(deploy["model"]["structured_query"]["inference_group_index"]) != 0:
    raise SystemExit("deployment group was not predeclared as group zero")
if deploy["postprocess"]["score_mode"] != "quality":
    raise SystemExit("deployment score_mode is not quality")
if float(deploy["postprocess"]["lane_nms_distance_thresh_px"]) != 0.0:
    raise SystemExit("deployment lane NMS is not disabled")
if int(candidate["scheduler"]["total_iters"]) != 278000:
    raise SystemExit("candidate does not retain the full 278k cosine horizon")
if int(candidate["training"]["seed"]) != 3407:
    raise SystemExit("candidate seed is not 3407")

print("parity audit passed: only 1 global group -> 4 isolated training groups")
print("deployment audit passed: fixed group 0, quality-only score, lane NMS off")
print("schedule audit passed: 25k observation on the unchanged 278k horizon")
PY

if [[ "${AUDIT_ONLY}" == "1" ]]; then
  echo "AUDIT_ONLY=1; configuration contract verified without starting training."
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
  echo "Training finished without expected checkpoint: ${checkpoint}" >&2
  exit 1
fi
echo "Matched train-many checkpoint ready: ${checkpoint}"
