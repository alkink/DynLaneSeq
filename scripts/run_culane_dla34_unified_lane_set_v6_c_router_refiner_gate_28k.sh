#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v6_c_router_refiner_cotrain_28k_to32k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v6_b_four_slot_refinement_gate_25k/seed_3407/slot_refinement/iter_0028000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-28000}"
TRAIN_STEPS="${TRAIN_STEPS:-4000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
SEEDS="${SEEDS:-3407}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v6_c_router_refiner_cotrain_gate_28k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v6_c_router_refiner_cotrain_gate_28k}"
RUN_CONTRACT_AUDIT="${RUN_CONTRACT_AUDIT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_FULL_VAL="${RUN_FULL_VAL:-0}"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
payload = _torch_load(sys.argv[1])
print(int(payload.get("iteration", -1)))
PY
}

for required in "${CONFIG}" "${SOURCE_CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V6-C artefact: ${required}" >&2
    exit 1
  fi
done
if (( TRAIN_STEPS < 1 || CHECKPOINT_INTERVAL < 1 )); then
  echo "TRAIN_STEPS and CHECKPOINT_INTERVAL must be positive" >&2
  exit 1
fi
if (( TRAIN_STEPS % CHECKPOINT_INTERVAL != 0 )); then
  echo "TRAIN_STEPS must be divisible by CHECKPOINT_INTERVAL" >&2
  exit 1
fi
actual_source_iteration="$(checkpoint_iteration "${SOURCE_CHECKPOINT}")"
if (( actual_source_iteration != SOURCE_ITERATION )); then
  echo "Expected V6-B source iteration ${SOURCE_ITERATION}, found ${actual_source_iteration}" >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_model

cfg = load_config(sys.argv[1])
cfg["model"]["pretrained_backbone"] = False
cfg["model"]["require_pretrained_backbone"] = False
selection = cfg["model"]["structured_query"]["set_selection"]
loss = cfg["loss"]
training = cfg["training"]
model = build_model(cfg)
refiner = model.structured_query_head.set_selection_head.slot_refinement
checks = {
    "four_slot_interaction": selection.get("candidate_interaction") == "four_slot",
    "refinement_enabled": bool(selection.get("four_slot_refinement_enabled")),
    "straight_through_routing": bool(
        selection.get("four_slot_refinement_straight_through_routing")
    ),
    "slot_state_attached": not bool(
        selection.get("four_slot_refinement_detach_slot_states", True)
    ),
    "selection_and_geometry_losses": float(loss.get("w_four_slot_selection", 0.0)) == 0.25
        and float(loss.get("w_four_slot_geometry", 0.0)) == 1.0,
    "only_selector_trainable": training.get("trainable_parameter_prefixes")
        == ["structured_query_head.set_selection_head"],
    "compact_selector_delta": training.get("checkpoint_model_prefixes")
        == ["structured_query_head.set_selection_head"],
    "frozen_detector_eval": bool(training.get("frozen_detector_eval")),
    "refiner_loaded": refiner is not None,
    "slot_score_mode": cfg["postprocess"].get("score_mode") == "four_slot",
    "no_threshold_or_nms": float(cfg["postprocess"].get("score_thresh", -1.0)) == 0.0
        and float(cfg["postprocess"].get("lane_nms_distance_thresh_px", -1.0)) == 0.0,
}
print(checks)
if not all(checks.values()):
    raise SystemExit("V6-C config contract failed")
PY

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports"
contract_report="${OUTPUT_ROOT}/audits/router_refiner_gradient_contract.json"
if [[ "${RUN_CONTRACT_AUDIT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v6_c_router_refiner_contract \
    --config "${CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --expected-source-iteration "${SOURCE_ITERATION}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --amp-dtype "${AMP_DTYPE}" \
    --output-json "${contract_report}"
elif [[ ! -f "${contract_report}" ]]; then
  echo "Missing V6-C contract report while RUN_CONTRACT_AUDIT=0" >&2
  exit 1
fi

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
echo "V6-C straight-through router/refiner co-training gate"
echo "source: ${SOURCE_CHECKPOINT}"
echo "logical iterations: ${SOURCE_ITERATION} -> ${END_ITERATION}"
echo "trainable subtree: four-slot router + bounded refiner; proposal detector frozen"

source_report="${OUTPUT_ROOT}/reports/source_iter_${SOURCE_ITERATION}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
if [[ "${RUN_EVAL}" == "1" && ! -f "${source_report}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_ROOT}/source" \
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
    --output-json "${source_report}"
fi

summary_args=()
for seed in ${SEEDS}; do
  run_dir="${OUTPUT_ROOT}/seed_${seed}/router_refiner"
  mkdir -p "${run_dir}"
  final_checkpoint="${run_dir}/iter_${END_TAG}.pt"
  if [[ "${RUN_TRAIN}" == "1" && ! -f "${final_checkpoint}" ]]; then
    latest_checkpoint=""
    latest_iteration=-1
    for candidate in "${run_dir}"/iter_*.pt; do
      [[ -f "${candidate}" ]] || continue
      iteration="$(checkpoint_iteration "${candidate}")"
      if (( iteration > SOURCE_ITERATION && iteration < END_ITERATION && iteration > latest_iteration )); then
        latest_checkpoint="${candidate}"
        latest_iteration="${iteration}"
      fi
    done
    train_args=(
      --config "${CONFIG}"
      --dataset-root "${DATA_ROOT}"
      --device "${DEVICE}"
      --output-dir "${run_dir}"
      --checkpoint-interval "${CHECKPOINT_INTERVAL}"
      --seed "${seed}"
      --batch-size "${BATCH_SIZE}"
      --grad-accum "${GRAD_ACCUM}"
      --num-workers "${NUM_WORKERS}"
      --seg-aux-amp-dtype "${AMP_DTYPE}"
      --checkpoint-base "${SOURCE_CHECKPOINT}"
    )
    if [[ -n "${latest_checkpoint}" ]]; then
      remaining_steps=$((END_ITERATION - latest_iteration))
      echo "Resuming V6-C ${latest_iteration} -> ${END_ITERATION}"
      train_args+=(--resume "${latest_checkpoint}" --max-iters "${remaining_steps}")
    else
      train_args+=(
        --init-from "${SOURCE_CHECKPOINT}"
        --init-iteration "${SOURCE_ITERATION}"
        --max-iters "${TRAIN_STEPS}"
      )
    fi
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${train_args[@]}" \
      2>&1 | tee -a "${run_dir}/train.log"
  fi
  if [[ ! -f "${final_checkpoint}" ]]; then
    echo "Missing V6-C final checkpoint: ${final_checkpoint}" >&2
    exit 1
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    for ((iteration=SOURCE_ITERATION + CHECKPOINT_INTERVAL; iteration<=END_ITERATION; iteration+=CHECKPOINT_INTERVAL)); do
      tag="$(printf '%07d' "${iteration}")"
      checkpoint="${run_dir}/iter_${tag}.pt"
      report="${OUTPUT_ROOT}/reports/seed_${seed}_iter_${tag}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
      if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing V6-C trajectory checkpoint: ${checkpoint}" >&2
        exit 1
      fi
      if [[ ! -f "${report}" ]]; then
        "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
          --config "${CONFIG}" \
          --checkpoint "${checkpoint}" \
          --dataset-root "${DATA_ROOT}" \
          --split val \
          --device "${DEVICE}" \
          --cache-dir "${CACHE_ROOT}/seed_${seed}/iter_${tag}" \
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
          --output-json "${report}"
      fi
      summary_args+=(--report "${iteration}=${report}")
    done
  fi
done

summary_path="${OUTPUT_ROOT}/v6_c_summary.json"
if [[ "${RUN_EVAL}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v6_c_router_refiner_gate \
    --source-report "${source_report}" \
    "${summary_args[@]}" \
    --output-json "${summary_path}"
fi

if [[ "${RUN_FULL_VAL}" == "1" ]]; then
  best_spec="$("${PYTHON}" - "${summary_path}" <<'PY'
import json
import re
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
if not summary["passed"]:
    raise SystemExit("V6-C uniform gate did not pass; full validation blocked")
row = summary["best"]
match = re.search(r"seed_(\d+)_iter_", Path(row["report"]).name)
if match is None:
    raise SystemExit("cannot recover seed from V6-C report")
print(f"{match.group(1)}:{int(row['iteration'])}")
PY
)"
  best_seed="${best_spec%%:*}"
  best_iteration="${best_spec##*:}"
  best_tag="$(printf '%07d' "${best_iteration}")"
  best_checkpoint="${OUTPUT_ROOT}/seed_${best_seed}/router_refiner/iter_${best_tag}.pt"
  full_dir="$(dirname "${best_checkpoint}")/val_eval_iter_${best_tag}_four_slot_refined_fp32"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${CONFIG}" \
    --checkpoint "${best_checkpoint}" \
    --split val \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --score-mode four_slot \
    --score-thresh 0.0 \
    --quality-score-power 0.0 \
    --top-k 4 \
    --nms-distance-thresh-px 0.0 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --iou-thresholds 0.5 0.75 \
    --pred-dir "${full_dir}" \
    --output-txt "${full_dir}/metrics.txt" \
    --output-json "${full_dir}/metrics.json" \
    --no-pretrained-init \
    --amp-dtype none
fi

echo "V6-C summary: ${summary_path}"
