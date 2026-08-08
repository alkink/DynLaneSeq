#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
SOURCE_ITERATION="${SOURCE_ITERATION:-26000}"
TARGET_ITERATION="${TARGET_ITERATION:-27000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v7_hard_slot_assignment_gate}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v7_hard_slot_assignment_gate}"

CONFIG=dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_hard_reference_memorize64.yaml
INIT_CHECKPOINT="${OUTPUT_ROOT}/shared_slot_init_iter_0025000.pt"
TRAIN_LIST="${OUTPUT_ROOT}/train_uniform64.txt"
TRAIN_DIR="${OUTPUT_ROOT}/memorize64/hard"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${TRAIN_DIR}/iter_0026000.pt}"
SUMMARY="${OUTPUT_ROOT}/v7_hard_reference_memorization_26k_to27k_summary.json"

if (( SOURCE_ITERATION != 26000 || TARGET_ITERATION != 27000 )); then
  echo "This causal continuation is fixed to iteration 26000 -> 27000." >&2
  exit 1
fi
if (( CHECKPOINT_INTERVAL != 500 )); then
  echo "This trajectory gate requires CHECKPOINT_INTERVAL=500." >&2
  exit 1
fi
if (( BATCH_SIZE != 4 )); then
  echo "This exact fixed-64 continuation requires BATCH_SIZE=4." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" ]]; then
  echo "This exact continuation requires AMP_DTYPE=bfloat16." >&2
  exit 1
fi
for required in \
  "${CONFIG}" \
  "${INIT_CHECKPOINT}" \
  "${SOURCE_CHECKPOINT}" \
  "${TRAIN_LIST}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing hard-reference continuation artefact: ${required}" >&2
    exit 1
  fi
done

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

if (( $(checkpoint_iteration "${SOURCE_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi
if (( $(wc -l < "${TRAIN_LIST}") != 64 )); then
  echo "Expected the original 64-image memorization list: ${TRAIN_LIST}" >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
selection = cfg["model"]["structured_query"]["set_selection"]
loss = cfg["loss"]
checks = {
    "hard_reference": selection["four_slot_refinement_reference_mode"] == "hard_st",
    "hard_slot_assignment": loss["four_slot_assignment_mode"] == "hard_min",
    "all_gt": loss["four_slot_target_mode"] == "all_gt",
    "frozen_detector": cfg["training"]["frozen_detector_eval"] is True,
    "fixed_batch": int(cfg["training"]["batch_size"]) == 4,
    "no_accumulation": int(cfg["training"]["gradient_accumulation_steps"]) == 1,
}
print({"hard_reference_continuation_contract": checks})
if not all(checks.values()):
    raise SystemExit("invalid hard-reference continuation contract")
PY

mkdir -p "${TRAIN_DIR}" "${CACHE_ROOT}"
FINAL_TAG="$(printf '%07d' "${TARGET_ITERATION}")"
FINAL_CHECKPOINT="${TRAIN_DIR}/iter_${FINAL_TAG}.pt"
if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
    --config "${CONFIG}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --output-dir "${TRAIN_DIR}" \
    --checkpoint-base "${INIT_CHECKPOINT}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum 1 \
    --num-workers "${NUM_WORKERS}" \
    --seg-aux-amp-dtype "${AMP_DTYPE}" \
    --compile-model false \
    --train-list "${TRAIN_LIST}" \
    --resume "${SOURCE_CHECKPOINT}" \
    --max-iters "$((TARGET_ITERATION - SOURCE_ITERATION))" \
    2>&1 | tee -a "${TRAIN_DIR}/train.log"
fi
if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Missing final hard-reference checkpoint: ${FINAL_CHECKPOINT}" >&2
  exit 1
fi

evaluate_report() {
  local iteration="$1"
  local tag
  tag="$(printf '%07d' "${iteration}")"
  local checkpoint="${TRAIN_DIR}/iter_${tag}.pt"
  local report="${OUTPUT_ROOT}/memorize64_hard_iter_${tag}.json"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing trajectory checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  if [[ ! -f "${report}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${CONFIG}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split train \
      --list-path "${TRAIN_LIST}" \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_ROOT}/memorize_hard_${tag}" \
      --max-batches 0 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" \
      --sample-strategy sequential \
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
}

evaluate_report 26000
evaluate_report 26500
evaluate_report 27000

"${PYTHON}" - \
  "${OUTPUT_ROOT}/memorize64_hard_iter_0026000.json" \
  "${OUTPUT_ROOT}/memorize64_hard_iter_0026500.json" \
  "${OUTPUT_ROOT}/memorize64_hard_iter_0027000.json" \
  "${SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

iterations = (26000, 26500, 27000)

def extract(path, iteration):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    refined = report["methods"]["four_slot_refined"]
    unrefined = report["methods"]["four_slot_global_unique"]
    slot = report["four_slot_diagnostics"]
    return {
        "iteration": iteration,
        "f1_050": refined["0.50"]["f1"],
        "f1_075": refined["0.75"]["f1"],
        "precision_050": refined["0.50"]["precision"],
        "recall_050": refined["0.50"]["recall"],
        "tp_050": refined["0.50"]["tp"],
        "fp_050": refined["0.50"]["fp"],
        "fn_050": refined["0.50"]["fn"],
        "mean_selected": refined["0.50"]["mean_selected_per_image"],
        "unrefined_f1_050": unrefined["0.50"]["f1"],
        "unrefined_f1_075": unrefined["0.75"]["f1"],
        "refinement_gain_050": (
            refined["0.50"]["f1"] - unrefined["0.50"]["f1"]
        ),
        "refinement_gain_075": (
            refined["0.75"]["f1"] - unrefined["0.75"]["f1"]
        ),
        "cardinality_exact": slot["cardinality"]["exact_fraction"],
        "cardinality_under": slot["cardinality"]["under_fraction"],
        "cardinality_over": slot["cardinality"]["over_fraction"],
        "semantic_duplicate": slot["semantic_duplicate_cluster_fraction"],
        "close_pair_fraction_20px": refined["0.50"][
            "selected_curve_diversity"
        ]["close_pair_fraction_below_20px"],
        "repair_fraction": slot["global_assignment_repair_fraction"],
        "route_entropy": slot["mean_route_entropy"],
        "oracle_recall_050": report["capacity"]["0.50"][
            "all_candidate_oracle"
        ]["recall"],
        "oracle_recall_075": report["capacity"]["0.75"][
            "all_candidate_oracle"
        ]["recall"],
    }

trajectory = [
    extract(path, iteration)
    for path, iteration in zip(sys.argv[1:4], iterations)
]
final = trajectory[-1]
thresholds = {
    "f1_050_min": 0.90,
    "cardinality_exact_min": 0.90,
    "semantic_duplicate_max": 0.02,
    "close_pair_fraction_20px_max": 0.02,
}
gate = {
    "f1_050": final["f1_050"] >= thresholds["f1_050_min"],
    "cardinality_exact": (
        final["cardinality_exact"] >= thresholds["cardinality_exact_min"]
    ),
    "semantic_duplicate": (
        final["semantic_duplicate"] <= thresholds["semantic_duplicate_max"]
    ),
    "close_pair_fraction_20px": (
        final["close_pair_fraction_20px"]
        <= thresholds["close_pair_fraction_20px_max"]
    ),
    "oracle_unchanged": all(
        row["oracle_recall_050"] == trajectory[0]["oracle_recall_050"]
        and row["oracle_recall_075"] == trajectory[0]["oracle_recall_075"]
        for row in trajectory
    ),
}
payload = {
    "experiment": "V7 hard-reference hard-ownership fixed-64 continuation",
    "iterations": list(iterations),
    "thresholds": thresholds,
    "trajectory": trajectory,
    "gate": gate,
    "pass": all(gate.values()),
    "best_iteration_050": max(
        trajectory,
        key=lambda row: row["f1_050"],
    )["iteration"],
}
Path(sys.argv[4]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["pass"]:
    raise SystemExit("hard-reference 2k memorization gate failed")
PY

echo "Hard-reference 2k memorization gate passed: ${SUMMARY}"
