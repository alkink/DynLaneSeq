#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_aux3x8_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_slots32_aux3x8_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
TARGET_ITERS="${TARGET_ITERS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
AUTO_RESUME="${AUTO_RESUME:-1}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

if (( TARGET_ITERS != 25000 )); then
  echo "This causal gate is fixed at 25000 iterations; got ${TARGET_ITERS}." >&2
  exit 1
fi
if (( BATCH_SIZE != 4 || GRAD_ACCUM != 4 )); then
  echo "Matched training requires BATCH_SIZE=4 and GRAD_ACCUM=4." >&2
  exit 1
fi
for path in "${CONTROL_CONFIG}" "${CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required config: ${path}" >&2
    exit 1
  fi
done

"${PYTHON}" - "${CONTROL_CONFIG}" "${CONFIG}" <<'PY'
from copy import deepcopy
import sys

from dynlaneseq_eg.config import load_config


control = load_config(sys.argv[1])
candidate = load_config(sys.argv[2])


def normalized(cfg):
    cfg = deepcopy(cfg)
    cfg.pop("_config_path", None)
    cfg.pop("output_dir", None)
    cfg["model"]["structured_query"].pop(
        "training_auxiliary_group_sizes",
        None,
    )
    cfg["loss"].pop("lambda_training_auxiliary", None)
    return cfg


if normalized(control) != normalized(candidate):
    raise SystemExit(
        "candidate changes more than train-only query groups and their loss; "
        "refusing unmatched training"
    )
structured = candidate["model"]["structured_query"]
if int(structured["num_instances"]) != 32:
    raise SystemExit("deployable primary set is not 32 queries")
if tuple(structured["training_auxiliary_group_sizes"]) != (8, 8, 8):
    raise SystemExit("candidate does not use the predeclared 3x8 auxiliary groups")
if int(structured["num_groups"]) != 1:
    raise SystemExit("primary group is not global")
matcher = candidate["matcher"]
if matcher["assignment"] != "hungarian" or int(matcher["num_groups"]) != 1:
    raise SystemExit("primary matcher is not global one-to-one")
if float(matcher["lambda_obj"]) != 0.5:
    raise SystemExit("matcher.lambda_obj is not the matched 0.5 setting")
if float(candidate["loss"]["lambda_training_auxiliary"]) != 0.5:
    raise SystemExit("training auxiliary loss weight is not 0.5")
if not bool(structured["row_reference"]["enabled"]):
    raise SystemExit("row-reference decoder is not enabled")
if not bool(structured["intermediate_supervision"]):
    raise SystemExit("deep supervision is not enabled")
if int(candidate["scheduler"]["total_iters"]) != 278000:
    raise SystemExit("candidate does not retain the full 278k cosine horizon")
if int(candidate["training"]["seed"]) != 3407:
    raise SystemExit("candidate seed is not 3407")

print("parity audit passed: main 32-query one-to-one control is unchanged")
print("hybrid audit passed: train-only auxiliary groups=(8,8,8), weight=0.5")
print("deployment contract: only the primary 32 queries are emitted")
PY

if [[ "${AUDIT_ONLY}" == "1" ]]; then
  echo "AUDIT_ONLY=1; configuration contract verified without training."
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
echo "Hybrid 25k checkpoint ready: ${checkpoint}"

