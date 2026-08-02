#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
TARGET_ITERS="${TARGET_ITERS:-75000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
RUN_AUDIT="${RUN_AUDIT:-1}"
AUDIT_ITERS="${AUDIT_ITERS:-25000 50000 75000}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-outputs/diagnostics/unified_lane_set_v3_25k_to75k}"
AUDIT_MAX_BATCHES="${AUDIT_MAX_BATCHES:-16}"
AUDIT_GRADIENT_IMAGES="${AUDIT_GRADIENT_IMAGES:-2}"
CONTRACT_ONLY="${CONTRACT_ONLY:-0}"

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Expected effective batch size 16, got $((BATCH_SIZE * GRAD_ACCUM))." >&2
  exit 1
fi
if (( TARGET_ITERS < 25000 || TARGET_ITERS > 278000 )); then
  echo "TARGET_ITERS must be in [25000, 278000]." >&2
  exit 1
fi
if (( TARGET_ITERS % 5000 != 0 )); then
  echo "TARGET_ITERS must be divisible by the 5k checkpoint interval." >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
structured = cfg["model"]["structured_query"]
state = structured["lane_state"]
checks = {
    "one deployable 32-query set": structured["num_instances"] == 32
    and structured["num_groups"] == 1
    and not structured.get("training_auxiliary_group_sizes"),
    "causal lane-row ownership": state.get("enabled")
    and state.get("mode") == "causal_set",
    "one foreground score": state.get("single_logit_score") is True,
    "P2 geometry plus score-only P4/P5": cfg["model"]["multi_scale_evidence"]["scales"]
    == ["p4", "p5"]
    and state["semantic_context"]["scales"] == ["p4", "p5"],
    "strict global one-to-one assignment": cfg["matcher"]["assignment"] == "hungarian"
    and cfg["matcher"]["num_groups"] == 1,
    "bounded matcher score": cfg["matcher"]["object_cost_type"] == "neg_probability"
    and float(cfg["matcher"]["lambda_obj"]) == 0.5,
    "single-score supervision": float(cfg["loss"]["w_exist"]) == 2.0
    and float(cfg["loss"]["w_quality"]) == 0.0
    and float(cfg["loss"]["w_set_selection"]) == 0.0,
    "NMS-free deployment": float(cfg["postprocess"]["lane_nms_distance_thresh_px"]) == 0.0
    and cfg["postprocess"]["score_mode"] == "exist"
    and float(cfg["postprocess"]["quality_score_power"]) == 0.0,
    "278k scheduler horizon": int(cfg["scheduler"]["total_iters"]) == 278000,
    "matched seed": int(cfg["training"]["seed"]) == 3407,
}
for name, passed in checks.items():
    print(f"[{'OK' if passed else 'FAIL'}] {name}")
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("V3 continuation contract failed: " + ", ".join(failed))
PY

if [[ "${CONTRACT_ONLY}" == "1" ]]; then
  exit 0
fi

resume_checkpoint="$(${PYTHON} - "${OUT_DIR}" "${TARGET_ITERS}" <<'PY'
import sys
from pathlib import Path

directory = Path(sys.argv[1])
target = int(sys.argv[2])
candidates = []
for path in directory.glob("iter_*.pt") if directory.is_dir() else ():
    try:
        iteration = int(path.stem.split("_")[-1])
    except ValueError:
        continue
    if iteration <= target:
        candidates.append((iteration, path))
if not candidates:
    raise SystemExit(f"No iter_*.pt checkpoint at or below {target} in {directory}")
print(max(candidates)[1])
PY
)"

"${PYTHON}" - "${CONFIG}" "${resume_checkpoint}" "${TARGET_ITERS}" <<'PY'
import json
import sys
import torch
from dynlaneseq_eg.config import load_config

config_path, checkpoint_path, target_raw = sys.argv[1:]
target = int(target_raw)
current = load_config(config_path)
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
iteration = int(payload.get("iteration", -1))
if iteration < 25000:
    raise SystemExit(f"Refusing pre-gate checkpoint at iteration {iteration}")
if iteration > target:
    raise SystemExit(f"Checkpoint iteration {iteration} exceeds target {target}")
for key in ("model", "optimizer", "scheduler"):
    if key not in payload:
        raise SystemExit(f"Checkpoint is missing required {key!r} state")
source = payload.get("cfg")
if not isinstance(source, dict) or not source:
    raise SystemExit("Checkpoint does not contain its expanded training config")

def signature(cfg):
    return {
        "model": cfg.get("model"),
        "matcher": cfg.get("matcher"),
        "loss": cfg.get("loss"),
        "optimizer": cfg.get("optimizer"),
        "scheduler": cfg.get("scheduler"),
        "seed": cfg.get("training", {}).get("seed"),
    }

if signature(source) != signature(current):
    raise SystemExit(
        "Checkpoint/config architecture or training-objective mismatch; "
        "refusing a non-identical continuation"
    )
scheduler = payload["scheduler"]
print(json.dumps({
    "resume_checkpoint": checkpoint_path,
    "resume_iteration": iteration,
    "target_iteration": target,
    "optimizer_groups": len(payload["optimizer"].get("param_groups", [])),
    "scheduler_last_epoch": scheduler.get("last_epoch"),
    "scheduler_step_count": scheduler.get("_step_count"),
    "note": (
        "Model, optimizer moments, and scheduler phase are restored exactly. "
        "The legacy checkpoint format did not save DataLoader/RNG iterator "
        "state, so augmentation order is deterministic but not bitwise-identical "
        "to an uninterrupted process."
    ),
}, indent=2))
PY

if [[ -f "${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERS}").pt" ]]; then
  echo "Training target already exists; skipping optimization."
else
  DATA_ROOT="${DATA_ROOT}" \
  DEVICE="${DEVICE}" \
  CONFIG="${CONFIG}" \
  OUT_DIR="${OUT_DIR}" \
  TARGET_ITERS="${TARGET_ITERS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM="${GRAD_ACCUM}" \
  SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE}" \
  AUTO_RESUME=1 \
    bash scripts/run_culane_dla34_row_reference_full_278k.sh
fi

if [[ "${RUN_AUDIT}" != "1" ]]; then
  echo "RUN_AUDIT=${RUN_AUDIT}; frozen trajectory audit skipped."
  exit 0
fi

mkdir -p "${AUDIT_OUTPUT_DIR}"
audit_reports=()
for iteration in ${AUDIT_ITERS}; do
  if (( iteration > TARGET_ITERS )); then
    continue
  fi
  checkpoint="${OUT_DIR}/iter_$(printf '%07d' "${iteration}").pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Audit checkpoint absent, skipping: ${checkpoint}" >&2
    continue
  fi
  report="${AUDIT_OUTPUT_DIR}/contract_iter_$(printf '%07d' "${iteration}")_uniform64.json"
  DATA_ROOT="${DATA_ROOT}" \
  CONFIG="${CONFIG}" \
  CKPT="${checkpoint}" \
  DEVICE="${DEVICE}" \
  EVAL_BATCH_SIZE=4 \
  NUM_WORKERS=8 \
  MAX_BATCHES="${AUDIT_MAX_BATCHES}" \
  GRADIENT_IMAGES="${AUDIT_GRADIENT_IMAGES}" \
  AMP_DTYPE=bfloat16 \
  SCORE_THRESHOLDS="0.20 0.30" \
  TOP_K=4 \
  OUTPUT_JSON="${report}" \
    bash scripts/audit_culane_dla34_unified_lane_set_v3_short.sh
  audit_reports+=("${report}")
done

if (( ${#audit_reports[@]} > 0 )); then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.summarize_unified_lane_set_contract_trajectory \
    --inputs "${audit_reports[@]}" \
    --output-json "${AUDIT_OUTPUT_DIR}/trajectory_summary.json" \
    --output-csv "${AUDIT_OUTPUT_DIR}/trajectory_summary.csv"
fi

echo "V3 continuation and frozen trajectory audit completed."
