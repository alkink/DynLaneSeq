#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k/iter_0025000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v3_rescue_25k_to30k}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AUDIT_MAX_BATCHES="${AUDIT_MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SOURCE_ITERATION=25000
TARGET_ITERATION=30000

CONTROL_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_control_from25k_5k.yaml"
CONTRACT_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_contract_rescue_from25k_5k.yaml"
SCALE_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_scale_rescue_from25k_5k.yaml"

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing 25k source checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Expected effective batch size 16, got $((BATCH_SIZE * GRAD_ACCUM))." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON}" - "${SOURCE_CHECKPOINT}" \
  "${CONTROL_CONFIG}" "${CONTRACT_CONFIG}" "${SCALE_CONFIG}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config

checkpoint_path = Path(sys.argv[1]).resolve()
config_paths = [Path(value) for value in sys.argv[2:]]
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
iteration = int(payload.get("iteration", -1))
if iteration != 25000:
    raise SystemExit(f"Expected a 25k source checkpoint, got iteration={iteration}")
for key in ("model", "optimizer", "scheduler", "cfg"):
    if key not in payload:
        raise SystemExit(f"Source checkpoint misses required {key!r} state")
source = payload["cfg"]
configs = {path.stem: load_config(path) for path in config_paths}
for name, cfg in configs.items():
    if cfg["model"] != source["model"]:
        raise SystemExit(f"{name}: model contract differs from the source checkpoint")
    if cfg["scheduler"] != source["scheduler"]:
        raise SystemExit(f"{name}: scheduler horizon differs from the source checkpoint")
    if int(cfg["training"]["seed"]) != int(source["training"]["seed"]):
        raise SystemExit(f"{name}: seed differs from the source checkpoint")
    if int(cfg["training"]["max_iters"]) != 5000:
        raise SystemExit(f"{name}: gate must contain exactly 5000 optimizer steps")
    if int(cfg["training"]["checkpoint_interval"]) != 1000:
        raise SystemExit(f"{name}: gate must save every 1000 optimizer steps")

control = configs[config_paths[0].stem]
contract = configs[config_paths[1].stem]
scale = configs[config_paths[2].stem]
for section in ("matcher", "loss", "optimizer"):
    if control[section] != source[section]:
        raise SystemExit(f"control arm changed source {section}")
if contract["optimizer"] != source["optimizer"]:
    raise SystemExit("contract arm must not change the optimizer")
if float(contract["matcher"]["lambda_obj"]) != 0.0:
    raise SystemExit("contract arm must use geometry-only Hungarian cost")
if bool(contract["matcher"]["reuse_final_assignment_for_intermediate"]):
    raise SystemExit("contract arm must retain independent geometry assignments")
expected_loss = {
    "w_intermediate_exist": 0.0,
    "w_cardinality": 0.0,
    "w_score_margin": 0.0,
}
for key, expected in expected_loss.items():
    if float(contract["loss"][key]) != expected:
        raise SystemExit(f"contract arm has unexpected {key}")
if scale["matcher"] != source["matcher"] or scale["loss"] != source["loss"]:
    raise SystemExit("scale arm must preserve the complete source objective")
scale_optimizer = dict(scale["optimizer"])
parameter_groups = scale_optimizer.pop("parameter_groups", None)
if scale_optimizer != source["optimizer"] or len(parameter_groups or ()) != 2:
    raise SystemExit("scale arm changed more than its two optimizer subgroups")

digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
scheduler = payload["scheduler"]
report = {
    "source_checkpoint": str(checkpoint_path),
    "source_sha256": digest,
    "source_iteration": iteration,
    "model_tensors": len(payload["model"]),
    "optimizer_groups": len(payload["optimizer"].get("param_groups", [])),
    "scheduler_last_epoch": scheduler.get("last_epoch"),
    "scheduler_step_count": scheduler.get("_step_count"),
    "arms": {
        "control": "exact objective and optimizer replay",
        "contract": (
            "lambda_obj=0, no intermediate foreground, no cardinality/margin"
        ),
        "scale": (
            "unchanged objective; lane-state and row-readout base LR 5e-5"
        ),
    },
}
print(json.dumps(report, indent=2))
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight passed; training was not started."
  exit 0
fi

run_arm() {
  local name="$1"
  local config="$2"
  local output_dir="${OUTPUT_ROOT}/${name}"
  local target_checkpoint="${output_dir}/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"
  mkdir -p "${output_dir}"

  if [[ -f "${target_checkpoint}" ]]; then
    echo "[${name}] target already exists: ${target_checkpoint}"
    return
  fi

  local resume_checkpoint="${SOURCE_CHECKPOINT}"
  local start_iteration="${SOURCE_ITERATION}"
  local latest
  latest="$(find "${output_dir}" -maxdepth 1 -type f -name 'iter_*.pt' | sort -V | tail -n 1)"
  if [[ -n "${latest}" ]]; then
    resume_checkpoint="${latest}"
    start_iteration="$(${PYTHON} - "${latest}" <<'PY'
import sys
import torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("iteration", -1)))
PY
)"
  fi
  if (( start_iteration < SOURCE_ITERATION || start_iteration >= TARGET_ITERATION )); then
    echo "[${name}] invalid resume iteration ${start_iteration}: ${resume_checkpoint}" >&2
    exit 1
  fi
  local remaining="$((TARGET_ITERATION - start_iteration))"
  local remap_args=()
  if [[ "${name}" == "scale" && "${start_iteration}" == "${SOURCE_ITERATION}" ]]; then
    remap_args=(--resume-remap-optimizer-groups)
  fi

  echo "===== ${name}: ${start_iteration} -> ${TARGET_ITERATION} ====="
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
    --config "${config}" \
    --device "${DEVICE}" \
    --dataset-root "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --resume "${resume_checkpoint}" \
    --max-iters "${remaining}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --seg-aux-amp-dtype bfloat16 \
    "${remap_args[@]}" \
    2>&1 | tee "${output_dir}/train_from_$(printf '%07d' "${start_iteration}").log"
}

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
run_arm control "${CONTROL_CONFIG}"
run_arm contract "${CONTRACT_CONFIG}"
run_arm scale "${SCALE_CONFIG}"

audit_checkpoint() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local report="${OUTPUT_ROOT}/${name}_uniform64.json"
  DATA_ROOT="${DATA_ROOT}" \
  CONFIG="${config}" \
  CKPT="${checkpoint}" \
  DEVICE="${DEVICE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  MAX_BATCHES="${AUDIT_MAX_BATCHES}" \
  GRADIENT_IMAGES=0 \
  AMP_DTYPE="${AMP_DTYPE}" \
  SCORE_THRESHOLDS="0.20 0.30" \
  TOP_K=4 \
  OUTPUT_JSON="${report}" \
    bash scripts/audit_culane_dla34_unified_lane_set_v3_short.sh \
    2>&1 | tee "${OUTPUT_ROOT}/${name}_uniform64.log"
}

audit_checkpoint \
  source \
  "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml" \
  "${SOURCE_CHECKPOINT}"
audit_checkpoint control "${CONTROL_CONFIG}" \
  "${OUTPUT_ROOT}/control/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"
audit_checkpoint contract "${CONTRACT_CONFIG}" \
  "${OUTPUT_ROOT}/contract/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"
audit_checkpoint scale "${SCALE_CONFIG}" \
  "${OUTPUT_ROOT}/scale/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.summarize_unified_lane_set_v3_rescue_gate \
  --source-audit "${OUTPUT_ROOT}/source_uniform64.json" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --arm control \
    "${OUTPUT_ROOT}/control_uniform64.json" \
    "${OUTPUT_ROOT}/control/iter_$(printf '%07d' "${TARGET_ITERATION}").pt" \
  --arm contract \
    "${OUTPUT_ROOT}/contract_uniform64.json" \
    "${OUTPUT_ROOT}/contract/iter_$(printf '%07d' "${TARGET_ITERATION}").pt" \
  --arm scale \
    "${OUTPUT_ROOT}/scale_uniform64.json" \
    "${OUTPUT_ROOT}/scale/iter_$(printf '%07d' "${TARGET_ITERATION}").pt" \
  --output-json "${OUTPUT_ROOT}/summary.json"

echo "V3 matched 25k->30k rescue gate completed."
echo "summary: ${OUTPUT_ROOT}/summary.json"
