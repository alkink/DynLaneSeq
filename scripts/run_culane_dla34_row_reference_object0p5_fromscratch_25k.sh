#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep.yaml}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
TARGET_ITERS="${TARGET_ITERS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
AUTO_RESUME="${AUTO_RESUME:-1}"

if (( TARGET_ITERS < 1 || TARGET_ITERS > 278000 )); then
  echo "TARGET_ITERS must be in [1, 278000], got ${TARGET_ITERS}" >&2
  exit 1
fi
if (( BATCH_SIZE != 4 || GRAD_ACCUM != 4 )); then
  echo "This causal gate requires BATCH_SIZE=4 and GRAD_ACCUM=4." >&2
  echo "Changing the microbatch would break parity with the 25k control." >&2
  exit 1
fi

# Refuse accidental multi-variable experiments.  Only output/checkpoint
# bookkeeping and matcher.lambda_obj may differ from the resolved control.
"${PYTHON}" - "${CONTROL_CONFIG}" "${CONFIG}" <<'PY'
from copy import deepcopy
import sys

from dynlaneseq_eg.config import load_config

control = load_config(sys.argv[1])
candidate = load_config(sys.argv[2])

if float(control["matcher"]["lambda_obj"]) != 2.0:
    raise SystemExit("control matcher.lambda_obj is not 2.0")
if float(candidate["matcher"]["lambda_obj"]) != 0.5:
    raise SystemExit("candidate matcher.lambda_obj is not 0.5")

def normalized(cfg):
    cfg = deepcopy(cfg)
    cfg.pop("_config_path", None)
    cfg.pop("output_dir", None)
    cfg["matcher"]["lambda_obj"] = 2.0
    cfg["training"]["checkpoint_interval"] = 25000
    return cfg

if normalized(control) != normalized(candidate):
    raise SystemExit(
        "candidate is not a single-variable matcher intervention; refusing train"
    )

if int(candidate["scheduler"]["total_iters"]) != 278000:
    raise SystemExit("candidate scheduler horizon is not the full 278k protocol")
if int(candidate["scheduler"]["warmup_iters"]) != 1000:
    raise SystemExit("candidate warmup is not the matched 1000-iteration warmup")
if int(candidate["training"]["seed"]) != 3407:
    raise SystemExit("candidate seed is not the matched seed 3407")

print(
    "parity audit passed: only matcher.lambda_obj 2.0 -> 0.5 "
    "(plus output/checkpoint bookkeeping)"
)
print("scheduler audit passed: cosine horizon=278000, warmup=1000")
PY

DATA_ROOT="${DATA_ROOT}" \
DEVICE="${DEVICE}" \
CONFIG="${CONFIG}" \
OUT_DIR="${OUT_DIR}" \
TARGET_ITERS="${TARGET_ITERS}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
AUTO_RESUME="${AUTO_RESUME}" \
  bash scripts/run_culane_dla34_row_reference_full_278k.sh

final_checkpoint="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERS}").pt"
if [[ ! -f "${final_checkpoint}" ]]; then
  echo "Training finished without expected checkpoint: ${final_checkpoint}" >&2
  exit 1
fi
echo "Matched from-scratch candidate ready: ${final_checkpoint}"
