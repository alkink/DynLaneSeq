#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_gate_0k_to10k.yaml}"
SEED="${SEED:-3407}"
TARGET_ITERATION="${TARGET_ITERATION:-10000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v7_joint_four_slot_gate}"
RUN_DIR="${OUTPUT_ROOT}/seed_${SEED}/joint_0k_to10k"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v7_joint_four_slot_gate/seed_${SEED}}"
PREFLIGHT_REPORT="${RUN_DIR}/contract_preflight_step0.json"
FINAL_GRADIENT_REPORT="${RUN_DIR}/gradient_contract_iter_0010000.json"
SUMMARY="${RUN_DIR}/v7_joint_four_slot_0k_to10k_summary.json"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing V7 joint gate config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing CULane root: ${DATA_ROOT}" >&2
  exit 1
fi
if (( TARGET_ITERATION != 10000 )); then
  echo "This causal gate is fixed to iteration 0 -> 10000." >&2
  exit 1
fi
if (( CHECKPOINT_INTERVAL != 5000 )); then
  echo "This causal gate requires checkpoints at 5k and 10k." >&2
  exit 1
fi
if (( BATCH_SIZE != 4 || GRAD_ACCUM != 4 )); then
  echo "This joint gate requires physical batch 4 and accumulation 4." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" ]]; then
  echo "This joint gate requires AMP_DTYPE=bfloat16." >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
selection = cfg["model"]["structured_query"]["set_selection"]
loss = cfg["loss"]
training = cfg["training"]
checks = {
    "hard_slot_assignment": loss["four_slot_assignment_mode"] == "hard_min",
    "all_gt_target": loss["four_slot_target_mode"] == "all_gt",
    "factorized_routing": selection["four_slot_factorized_routing"] is True,
    "hard_reference": (
        selection["four_slot_refinement_reference_mode"] == "hard_st"
    ),
    "structured_unique_backward": (
        selection["four_slot_refinement_structured_unique_routing"] is True
    ),
    "detached_slot_geometry_trunk": (
        selection["four_slot_refinement_detach_slot_states"] is True
    ),
    "all_slot_geometry": loss["four_slot_geometry_match_all_slots"] is True,
    "from_iteration_zero": int(training["max_iters"]) == 10000,
    "full_model_trainable": (
        "trainable_parameter_prefixes" not in training
        and "frozen_detector_eval" not in training
    ),
    "effective_batch_16": (
        int(training["batch_size"])
        * int(training["gradient_accumulation_steps"])
        == 16
    ),
    "long_cosine_horizon": int(cfg["scheduler"]["total_iters"]) == 278000,
    "optimizer_state_saved": training["checkpoint_include_optimizer"] is True,
}
print({"v7_joint_0k_to10k_contract": checks})
if not all(checks.values()):
    raise SystemExit("invalid V7 joint 0-to-10k contract")
PY

mkdir -p "${RUN_DIR}" "${CACHE_ROOT}"

if [[ ! -f "${PREFLIGHT_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v7_joint_four_slot_contract \
    --config "${CONFIG}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --num-workers "${NUM_WORKERS}" \
    --amp-dtype none \
    --output-json "${PREFLIGHT_REPORT}"
fi

"${PYTHON}" - "${RUN_DIR}" "${MIN_FREE_GB}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
free = shutil.disk_usage(path).free / 1024 ** 3
minimum = float(sys.argv[2])
print({"checkpoint_filesystem_free_gib": round(free, 2)})
if free < minimum:
    raise SystemExit(
        f"V7 joint gate requires at least {minimum:.2f} GiB free; "
        f"found {free:.2f} GiB"
    )
PY

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

latest_checkpoint=""
latest_iteration=0
for candidate in "${RUN_DIR}"/iter_*.pt; do
  [[ -f "${candidate}" ]] || continue
  iteration="$(checkpoint_iteration "${candidate}")"
  if (( iteration > latest_iteration && iteration <= TARGET_ITERATION )); then
    latest_checkpoint="${candidate}"
    latest_iteration="${iteration}"
  fi
done

if (( latest_iteration < TARGET_ITERATION )); then
  remaining=$((TARGET_ITERATION - latest_iteration))
  train_args=(
    --config "${CONFIG}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --output-dir "${RUN_DIR}"
    --max-iters "${remaining}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --num-workers "${NUM_WORKERS}"
    --seg-aux-amp-dtype "${AMP_DTYPE}"
    --compile-model false
  )
  if [[ -n "${latest_checkpoint}" ]]; then
    echo "Exact-resume joint V7: ${latest_iteration} -> ${TARGET_ITERATION}"
    train_args+=(--resume "${latest_checkpoint}")
  else
    echo "Starting joint V7 from iteration zero: 0 -> ${TARGET_ITERATION}"
  fi
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${train_args[@]}" \
    2>&1 | tee -a "${RUN_DIR}/train.log"
fi

FINAL_CHECKPOINT="${RUN_DIR}/iter_0010000.pt"
if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  echo "V7 joint gate did not produce ${FINAL_CHECKPOINT}." >&2
  exit 1
fi

summary_args=()
for iteration in 5000 10000; do
  tag="$(printf '%07d' "${iteration}")"
  checkpoint="${RUN_DIR}/iter_${tag}.pt"
  report="${RUN_DIR}/joint_iter_${tag}_uniform256.json"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing V7 joint trajectory checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  if [[ ! -f "${report}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${CONFIG}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_ROOT}/joint_${tag}" \
      --max-batches 64 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" \
      --amp-dtype "${AMP_DTYPE}" \
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
      --output-json "${report}"
  fi
  summary_args+=(--report "${iteration}=${report}")
done

if [[ ! -f "${FINAL_GRADIENT_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v7_joint_four_slot_contract \
    --config "${CONFIG}" \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --num-workers "${NUM_WORKERS}" \
    --amp-dtype none \
    --output-json "${FINAL_GRADIENT_REPORT}"
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v7_joint_four_slot_10k_gate \
  "${summary_args[@]}" \
  --preflight-contract "${PREFLIGHT_REPORT}" \
  --final-gradient-contract "${FINAL_GRADIENT_REPORT}" \
  --output-json "${SUMMARY}"

echo "V7 joint 0-to-10k gate complete: ${SUMMARY}"
