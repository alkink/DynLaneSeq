#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CONFIG="${SOURCE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_assignment_10k_to25k.yaml}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v6_a_four_slot_selector_25k_to29k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v5_1_shared_trunk_gate/seed_3407/assignment_fork/iter_0025000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-25000}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v6_a_four_slot_gate_25k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v6_a_four_slot_gate_25k}"
RUN_CONTRACT_AUDIT="${RUN_CONTRACT_AUDIT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_FULL_VAL="${RUN_FULL_VAL:-0}"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import torch
try:
    payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(sys.argv[1], map_location="cpu")
print(int(payload.get("iteration", -1)))
PY
}

for required in "${SOURCE_CONFIG}" "${CONFIG}" "${SOURCE_CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V6-A artefact: ${required}" >&2
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
  echo "Expected V5.1 source iteration ${SOURCE_ITERATION}, found ${actual_source_iteration}" >&2
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
head = model.structured_query_head.set_selection_head
checks = {
    "four_slot_interaction": selection.get("candidate_interaction") == "four_slot",
    "four_slots": int(selection.get("four_slot_num_slots", -1)) == 4,
    "slot_only_loss": float(loss.get("w_four_slot_selection", 0.0)) == 1.0,
    "detector_losses_zero": all(
        float(loss.get(name, 0.0)) == 0.0
        for name in (
            "w_exist", "w_point", "w_range", "w_line_iou", "w_set_selection",
            "w_pointer_selection", "lambda_intermediate"
        )
    ),
    "only_slot_trainable": training.get("trainable_parameter_prefixes")
        == ["structured_query_head.set_selection_head"],
    "compact_slot_delta": training.get("checkpoint_model_prefixes")
        == ["structured_query_head.set_selection_head"],
    "frozen_detector_eval": bool(training.get("frozen_detector_eval")),
    "no_nms": float(cfg["postprocess"].get("lane_nms_distance_thresh_px", -1.0)) == 0.0,
    "slot_score_mode": cfg["postprocess"].get("score_mode") == "four_slot",
    "probe_parameter_parity": sum(p.numel() for p in head.parameters()) == 2_980_711,
}
print(checks)
if not all(checks.values()):
    raise SystemExit("V6-A config contract failed")
PY

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports"
contract_report="${OUTPUT_ROOT}/audits/four_slot_gradient_contract.json"
if [[ "${RUN_CONTRACT_AUDIT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v6_four_slot_contract \
    --config "${CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --expected-source-iteration "${SOURCE_ITERATION}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --num-workers "${NUM_WORKERS}" \
    --amp-dtype "${AMP_DTYPE}" \
    --output-json "${contract_report}"
elif [[ ! -f "${contract_report}" ]]; then
  echo "Missing V6-A contract report while RUN_CONTRACT_AUDIT=0" >&2
  exit 1
fi

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
echo "V6-A frozen-proposal four-slot gate"
echo "source: ${SOURCE_CHECKPOINT}"
echo "logical iterations: ${SOURCE_ITERATION} -> ${END_ITERATION}"
echo "trainable parameters: four-slot decoder only (2,980,711)"

source_report="${OUTPUT_ROOT}/reports/source_v5_1_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
if [[ "${RUN_EVAL}" == "1" && ! -f "${source_report}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${SOURCE_CONFIG}" \
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
  run_dir="${OUTPUT_ROOT}/seed_${seed}/four_slot"
  mkdir -p "${run_dir}"
  final_checkpoint="${run_dir}/iter_${END_TAG}.pt"
  if [[ "${RUN_TRAIN}" == "1" && ! -f "${final_checkpoint}" ]]; then
    latest_checkpoint=""
    for candidate in "${run_dir}"/iter_*.pt; do
      [[ -f "${candidate}" ]] || continue
      iteration="$(checkpoint_iteration "${candidate}")"
      if (( iteration > SOURCE_ITERATION && iteration < END_ITERATION )); then
        if [[ -z "${latest_checkpoint}" ]] || (( iteration > $(checkpoint_iteration "${latest_checkpoint}") )); then
          latest_checkpoint="${candidate}"
        fi
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
      --seg-aux-amp-dtype "${AMP_DTYPE}"
      --checkpoint-base "${SOURCE_CHECKPOINT}"
    )
    if [[ -n "${latest_checkpoint}" ]]; then
      current_iteration="$(checkpoint_iteration "${latest_checkpoint}")"
      remaining_steps=$((END_ITERATION - current_iteration))
      echo "Resuming V6-A selector ${current_iteration} -> ${END_ITERATION}"
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
    echo "Missing V6-A final checkpoint: ${final_checkpoint}" >&2
    exit 1
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    for ((iteration=SOURCE_ITERATION + CHECKPOINT_INTERVAL; iteration<=END_ITERATION; iteration+=CHECKPOINT_INTERVAL)); do
      tag="$(printf '%07d' "${iteration}")"
      checkpoint="${run_dir}/iter_${tag}.pt"
      report="${OUTPUT_ROOT}/reports/seed_${seed}_iter_${tag}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
      if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing V6-A trajectory checkpoint: ${checkpoint}" >&2
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

summary_path="${OUTPUT_ROOT}/v6_a_summary.json"
if [[ "${RUN_EVAL}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v6_a_four_slot_gate \
    --source-report "${source_report}" \
    "${summary_args[@]}" \
    --output-json "${summary_path}"
fi

if [[ "${RUN_FULL_VAL}" == "1" ]]; then
  best_checkpoint="$("${PYTHON}" - "${summary_path}" "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
if not summary["passed"]:
    raise SystemExit("V6-A uniform gate did not pass; full validation blocked")
best = summary["best"]
iteration = int(best["iteration"])
report_path = str(best["report"])
seed_part = next(part for part in Path(report_path).stem.split("_") if part.isdigit())
print(Path(sys.argv[2]) / f"seed_{seed_part}" / "four_slot" / f"iter_{iteration:07d}.pt")
PY
)"
  full_dir="$(dirname "${best_checkpoint}")/val_eval_$(basename "${best_checkpoint%.pt}")_four_slot_fp32"
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

echo "V6-A summary: ${summary_path}"
