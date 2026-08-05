#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
GEOMETRY_CONFIG="${GEOMETRY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_long_geometry.yaml}"
BASE_GEOMETRY_CONFIG="${BASE_GEOMETRY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml}"
SOURCE_GEOMETRY_CHECKPOINT="${SOURCE_GEOMETRY_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt}"
SOURCE_GEOMETRY_ITERATION="${SOURCE_GEOMETRY_ITERATION:-50000}"
GEOMETRY_TARGET_ITERATION="${GEOMETRY_TARGET_ITERATION:-125000}"
GEOMETRY_CHECKPOINT_INTERVAL="${GEOMETRY_CHECKPOINT_INTERVAL:-25000}"
GEOMETRY_OUTPUT_DIR="${GEOMETRY_OUTPUT_DIR:-outputs/diagnostics/unified_lane_set_v4_long_geometry_50k_to125k/geometry}"
GEOMETRY_REPORT_DIR="${GEOMETRY_REPORT_DIR:-outputs/diagnostics/unified_lane_set_v4_long_geometry_50k_to125k/geometry_reports}"
POINTER_CONFIG="${POINTER_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_5_cluster_soft_pointer.yaml}"
BASELINE_POINTER_CHECKPOINT="${BASELINE_POINTER_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v4_5_cluster_soft_pointer_gate_50k/seed_3407/cluster_soft_pointer/iter_0060000.pt}"
BASELINE_POINTER_CONFIG="${BASELINE_POINTER_CONFIG:-${POINTER_CONFIG}}"
POINTER_TRAIN_STEPS="${POINTER_TRAIN_STEPS:-10000}"
POINTER_CHECKPOINT_INTERVAL="${POINTER_CHECKPOINT_INTERVAL:-2500}"
POINTER_SEEDS="${POINTER_SEEDS:-3407}"
POINTER_OUTPUT_ROOT="${POINTER_OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_5_pointer_gate_geometry125k}"
POINTER_CACHE_ROOT="${POINTER_CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_5_pointer_gate_geometry125k}"
FINAL_REPORT_DIR="${FINAL_REPORT_DIR:-outputs/diagnostics/unified_lane_set_v4_long_geometry_50k_to125k}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_GEOMETRY="${RUN_GEOMETRY:-1}"
RUN_GEOMETRY_AUDIT="${RUN_GEOMETRY_AUDIT:-1}"
RUN_POINTER="${RUN_POINTER:-1}"
RUN_POINTER_GRAD_AUDIT="${RUN_POINTER_GRAD_AUDIT:-1}"
RUN_POINTER_EVAL="${RUN_POINTER_EVAL:-1}"
RUN_POINTER_TRAJECTORY_EVAL="${RUN_POINTER_TRAJECTORY_EVAL:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-2.00}"
CONTRACT_ONLY="${CONTRACT_ONLY:-0}"

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Expected effective batch 16, got $((BATCH_SIZE * GRAD_ACCUM))" >&2
  exit 1
fi
if (( SOURCE_GEOMETRY_ITERATION != 50000 )); then
  echo "This controlled trajectory requires the exact 50k V4 source." >&2
  exit 1
fi
if (( GEOMETRY_TARGET_ITERATION != 125000 )); then
  echo "This predeclared gate requires GEOMETRY_TARGET_ITERATION=125000." >&2
  exit 1
fi
if (( (GEOMETRY_TARGET_ITERATION - SOURCE_GEOMETRY_ITERATION) % GEOMETRY_CHECKPOINT_INTERVAL != 0 )); then
  echo "Geometry span must be divisible by GEOMETRY_CHECKPOINT_INTERVAL." >&2
  exit 1
fi
if (( POINTER_TRAIN_STEPS < 1 || POINTER_TRAIN_STEPS % POINTER_CHECKPOINT_INTERVAL != 0 )); then
  echo "POINTER_TRAIN_STEPS must be positive and divisible by POINTER_CHECKPOINT_INTERVAL." >&2
  exit 1
fi

for required in "${GEOMETRY_CONFIG}" "${BASE_GEOMETRY_CONFIG}" "${SOURCE_GEOMETRY_CHECKPOINT}" "${POINTER_CONFIG}" "${BASELINE_POINTER_CONFIG}" "${BASELINE_POINTER_CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing long-geometry artefact: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" - "${BASE_GEOMETRY_CONFIG}" "${GEOMETRY_CONFIG}" "${SOURCE_GEOMETRY_CHECKPOINT}" "${SOURCE_GEOMETRY_ITERATION}" "${GEOMETRY_CHECKPOINT_INTERVAL}" <<'PY'
import json
import sys
import torch
from dynlaneseq_eg.config import load_config

base_path, long_path, checkpoint_path, source_raw, interval_raw = sys.argv[1:]
base = load_config(base_path)
long = load_config(long_path)
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
source_iteration = int(source_raw)
interval = int(interval_raw)

def signature(cfg):
    return {
        "model": cfg.get("model"),
        "matcher": cfg.get("matcher"),
        "loss": cfg.get("loss"),
        "optimizer": cfg.get("optimizer"),
        "scheduler": cfg.get("scheduler"),
        "seed": cfg.get("training", {}).get("seed"),
    }

checks = {
    "long_config_same_objective": signature(base) == signature(long),
    "source_iteration_exact": int(payload.get("iteration", -1)) == source_iteration,
    "source_full_model": payload.get("model_state_mode", "full") == "full",
    "source_has_optimizer": "optimizer" in payload,
    "source_has_scheduler": "scheduler" in payload,
    "cosine_horizon_278k": int(long["scheduler"]["total_iters"]) == 278000,
    "long_max_iters_278k": int(long["training"]["max_iters"]) == 278000,
    "checkpoint_interval_matches": int(long["training"]["checkpoint_interval"]) == interval,
    "optimizer_state_saved": bool(long["training"].get("checkpoint_include_optimizer", False)),
    "no_last_alias": not bool(long["training"].get("save_last_alias", True)),
}
source_cfg = payload.get("cfg")
checks["source_checkpoint_same_objective"] = isinstance(source_cfg, dict) and signature(source_cfg) == signature(base)
print(json.dumps(checks, indent=2))
if not all(checks.values()):
    raise SystemExit("long-geometry resume contract failed")
PY

if [[ "${CONTRACT_ONLY}" == "1" ]]; then
  echo "Long-geometry and pointer contract audit passed."
  exit 0
fi

mkdir -p "${GEOMETRY_OUTPUT_DIR}" "${GEOMETRY_REPORT_DIR}" "${FINAL_REPORT_DIR}"
if [[ "${RUN_GEOMETRY}" == "1" ]]; then
  "${PYTHON}" - "${GEOMETRY_OUTPUT_DIR}" "${MIN_FREE_GB}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(path).free
minimum = float(sys.argv[2]) * 1024 ** 3
print({"checkpoint_filesystem_free_gib": round(free / 1024 ** 3, 2)})
if free < minimum:
    raise SystemExit(
        f"only {free / 1024 ** 3:.2f} GiB free; long geometry + pointer needs "
        f"at least {minimum / 1024 ** 3:.2f} GiB"
    )
PY

  resume_checkpoint="${SOURCE_GEOMETRY_CHECKPOINT}"
  mapfile -t existing < <(find "${GEOMETRY_OUTPUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' | sort -V)
  if (( ${#existing[@]} > 0 )); then
    resume_checkpoint="${existing[${#existing[@]}-1]}"
  fi
  start_iteration="$("${PYTHON}" - "${resume_checkpoint}" <<'PY'
import sys
import torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
for key in ("model", "optimizer", "scheduler"):
    if key not in payload:
        raise SystemExit(f"resume checkpoint is missing {key}: {sys.argv[1]}")
print(int(payload.get("iteration", -1)))
PY
)"
  if (( start_iteration < SOURCE_GEOMETRY_ITERATION || start_iteration > GEOMETRY_TARGET_ITERATION )); then
    echo "Invalid geometry resume iteration ${start_iteration}: ${resume_checkpoint}" >&2
    exit 1
  fi
  if (( start_iteration < GEOMETRY_TARGET_ITERATION )); then
    remaining=$((GEOMETRY_TARGET_ITERATION - start_iteration))
    echo "Exact V4 geometry resume: ${start_iteration} -> ${GEOMETRY_TARGET_ITERATION}"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
      --config "${GEOMETRY_CONFIG}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --output-dir "${GEOMETRY_OUTPUT_DIR}" \
      --max-iters "${remaining}" \
      --checkpoint-interval "${GEOMETRY_CHECKPOINT_INTERVAL}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --seg-aux-amp-dtype "${AMP_DTYPE}" \
      --resume "${resume_checkpoint}" \
      2>&1 | tee -a "${GEOMETRY_OUTPUT_DIR}/train.log"
  else
    echo "[SKIP] geometry target already exists: ${resume_checkpoint}"
  fi
fi

mature_tag="$(printf '%07d' "${GEOMETRY_TARGET_ITERATION}")"
mature_geometry_checkpoint="${GEOMETRY_OUTPUT_DIR}/iter_${mature_tag}.pt"
if [[ ! -f "${mature_geometry_checkpoint}" ]]; then
  echo "Missing mature geometry checkpoint: ${mature_geometry_checkpoint}" >&2
  exit 1
fi

geometry_summary="${GEOMETRY_REPORT_DIR}/trajectory_summary.json"
if [[ "${RUN_GEOMETRY_AUDIT}" == "1" ]]; then
  geometry_reports=()
  geometry_checkpoints=("${SOURCE_GEOMETRY_CHECKPOINT}")
  for ((iteration=SOURCE_GEOMETRY_ITERATION + GEOMETRY_CHECKPOINT_INTERVAL; iteration<=GEOMETRY_TARGET_ITERATION; iteration+=GEOMETRY_CHECKPOINT_INTERVAL)); do
    geometry_checkpoints+=("${GEOMETRY_OUTPUT_DIR}/iter_$(printf '%07d' "${iteration}").pt")
  done
  for checkpoint in "${geometry_checkpoints[@]}"; do
    iteration="$("${PYTHON}" - "${checkpoint}" <<'PY'
import sys
import torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("iteration", -1)))
PY
)"
    report="${GEOMETRY_REPORT_DIR}/contract_iter_$(printf '%07d' "${iteration}")_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
    DATA_ROOT="${DATA_ROOT}" \
    CONFIG="${BASE_GEOMETRY_CONFIG}" \
    CKPT="${checkpoint}" \
    DEVICE="${DEVICE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    MAX_BATCHES="${MAX_BATCHES}" \
    GRADIENT_IMAGES=0 \
    AMP_DTYPE="${AMP_DTYPE}" \
    SCORE_THRESHOLDS="0.20 0.30" \
    OUTPUT_JSON="${report}" \
      bash scripts/audit_culane_dla34_unified_lane_set_v4_short.sh
    geometry_reports+=("${report}")
  done
  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_unified_lane_set_contract_trajectory \
    --inputs "${geometry_reports[@]}" \
    --output-json "${geometry_summary}" \
    --output-csv "${GEOMETRY_REPORT_DIR}/trajectory_summary.csv"
elif [[ ! -f "${geometry_summary}" ]]; then
  echo "Missing geometry summary while RUN_GEOMETRY_AUDIT=0: ${geometry_summary}" >&2
  exit 1
fi

if [[ "${RUN_POINTER}" == "1" || "${RUN_POINTER_EVAL}" == "1" ]]; then
  POINTER_CONFIG="${POINTER_CONFIG}" \
  SOURCE_CHECKPOINT="${mature_geometry_checkpoint}" \
  SOURCE_ITERATION="${GEOMETRY_TARGET_ITERATION}" \
  TRAIN_STEPS="${POINTER_TRAIN_STEPS}" \
  CHECKPOINT_INTERVAL="${POINTER_CHECKPOINT_INTERVAL}" \
  SEEDS="${POINTER_SEEDS}" \
  DATA_ROOT="${DATA_ROOT}" \
  DEVICE="${DEVICE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM="${GRAD_ACCUM}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  METRIC_WORKERS="${METRIC_WORKERS}" \
  MAX_BATCHES="${MAX_BATCHES}" \
  AMP_DTYPE="${AMP_DTYPE}" \
  RUN_TRAIN="${RUN_POINTER}" \
  RUN_GRAD_AUDIT="${RUN_POINTER_GRAD_AUDIT}" \
  RUN_EVAL="${RUN_POINTER_EVAL}" \
  RUN_TRAJECTORY_EVAL="${RUN_POINTER_TRAJECTORY_EVAL}" \
  OUTPUT_ROOT="${POINTER_OUTPUT_ROOT}" \
  CACHE_ROOT="${POINTER_CACHE_ROOT}" \
    bash scripts/run_culane_dla34_unified_lane_set_v4_5_cluster_soft_pointer_gate_50k.sh
fi

if [[ "${RUN_POINTER_EVAL}" != "1" || "${RUN_POINTER_TRAJECTORY_EVAL}" != "1" ]]; then
  echo "Long geometry completed. Pointer comparison summary skipped."
  exit 0
fi

sample_count=$((EVAL_BATCH_SIZE * MAX_BATCHES))
baseline_report="${FINAL_REPORT_DIR}/baseline_v4_5_geometry50k_uniform${sample_count}.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
  --config "${BASELINE_POINTER_CONFIG}" \
  --checkpoint "${BASELINE_POINTER_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --cache-dir "${POINTER_CACHE_ROOT}/baseline_geometry50k" \
  --max-batches "${MAX_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --metric-workers "${METRIC_WORKERS}" \
  --sample-strategy uniform \
  --stage main \
  --top-k 4 \
  --iou-thresholds 0.50 0.75 \
  --near-min-iou 0.30 \
  --line-width 30 \
  --min-valid-rows 5 \
  --hard-diversity-distances 20 \
  --mmr-sigmas 20 \
  --mmr-penalties 0.50 \
  --output-json "${baseline_report}"

for seed in ${POINTER_SEEDS}; do
  mature_pointer_summary="${POINTER_OUTPUT_ROOT}/seed_${seed}/reports/v4_5_trajectory_summary.json"
  if [[ ! -f "${mature_pointer_summary}" ]]; then
    echo "Missing mature pointer summary: ${mature_pointer_summary}" >&2
    exit 1
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_long_geometry_pointer_gate \
    --baseline-pointer-report "${baseline_report}" \
    --geometry-trajectory "${geometry_summary}" \
    --mature-pointer-summary "${mature_pointer_summary}" \
    --seed "${seed}" \
    --baseline-pointer-checkpoint "${BASELINE_POINTER_CHECKPOINT}" \
    --mature-geometry-checkpoint "${mature_geometry_checkpoint}" \
    --git-commit "$(git rev-parse HEAD)" \
    --output-json "${FINAL_REPORT_DIR}/seed_${seed}_long_geometry_pointer_summary.json"
done

echo "V4 50k -> ${GEOMETRY_TARGET_ITERATION} geometry and fresh V4.5 pointer gate completed."
