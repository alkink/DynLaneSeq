#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION=30000
ENDPOINT_ITERATION=35000
TRAINING_STEPS=5000
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"

SOURCE_CONFIG="${SOURCE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
TREATMENT_CONFIG="${TREATMENT_CONFIG:-dynlaneseq_eg/configs/culane_v30_joint_slot_field_30k_to35k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0030000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v30_joint_slot_field_30k_to35k}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-V30 joint slot field, historical V7-35K comparison}"
CONTRACT_MODE="${CONTRACT_MODE:-full}"
TRAIN_DIR="${OUTPUT_ROOT}/train/joint_field"
ENDPOINT="${TRAIN_DIR}/iter_0035000.pt"
CONTRACT="${OUTPUT_ROOT}/audits/zero_step_contract.json"
COVERAGE="${OUTPUT_ROOT}/reports/iter_0035000_uniform256.json"
FULL_VAL_DIR="${OUTPUT_ROOT}/reports/iter_0035000_full_val_four_slot_fp32"
FULL_VAL="${FULL_VAL_DIR}/metrics.json"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${SOURCE_CONFIG}" \
  "${TREATMENT_CONFIG}" \
  "${SOURCE_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V30 artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_CHECKPOINT is not iteration 30000." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V30 preserves V7 effective batch 16." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports" "${TRAIN_DIR}"

case "${CONTRACT_MODE}" in
  full)
    CONTRACT_EXTRA_ARGS=()
    ;;
  field_only)
    CONTRACT_EXTRA_ARGS=(--field-only)
    ;;
  *)
    echo "CONTRACT_MODE must be 'full' or 'field_only'." >&2
    exit 1
    ;;
esac

if [[ ! -s "${CONTRACT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_joint_slot_field_contract \
    --source-config "${SOURCE_CONFIG}" \
    --treatment-config "${TREATMENT_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --start-iteration "${SOURCE_ITERATION}" \
    "${CONTRACT_EXTRA_ARGS[@]}" \
    --output-json "${CONTRACT}" \
    2>&1 | tee "${OUTPUT_ROOT}/audits/zero_step_contract.log"
fi
"${PYTHON}" - "${CONTRACT}" <<'PY'
import json, sys
if json.load(open(sys.argv[1])).get("passed") is not True:
    raise SystemExit("V30 zero-step contract did not pass")
PY

if [[ ! -f "${ENDPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
    --config "${TREATMENT_CONFIG}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --output-dir "${TRAIN_DIR}" \
    --resume "${SOURCE_CHECKPOINT}" \
    --resume-remap-optimizer-groups \
    --max-iters "${TRAINING_STEPS}" \
    --checkpoint-interval "${TRAINING_STEPS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --seg-aux-amp-dtype bfloat16 \
    --compile-model true \
    --resume-safe-data true \
    2>&1 | tee -a "${TRAIN_DIR}/train.log"
fi
if [[ ! -f "${ENDPOINT}" ]] || (( $(checkpoint_iteration "${ENDPOINT}") != ENDPOINT_ITERATION )); then
  echo "V30 global-35K endpoint is missing or invalid." >&2
  exit 1
fi

if [[ ! -s "${COVERAGE}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${TREATMENT_CONFIG}" \
    --checkpoint "${ENDPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --list-path "${DATA_ROOT}/list/val.txt" \
    --device "${DEVICE}" \
    --cache-dir "${OUTPUT_ROOT}/cache/iter_0035000_uniform256" \
    --max-batches 64 \
    --eval-batch-size 4 \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --sample-strategy uniform \
    --top-k 4 \
    --iou-thresholds 0.50 0.75 \
    --output-json "${COVERAGE}"
fi

# Full official validation uses the exact optimized V7 protocol: refined four
# slots, threshold 0, top-4, no NMS, FP32/TF32 inference.  The test split is
# intentionally never opened by this experiment.
if [[ ! -s "${FULL_VAL}" ]]; then
  mkdir -p "${FULL_VAL_DIR}"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${TREATMENT_CONFIG}" \
    --checkpoint "${ENDPOINT}" \
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
    --pred-dir "${FULL_VAL_DIR}/predictions" \
    --output-txt "${FULL_VAL_DIR}/metrics.txt" \
    --output-json "${FULL_VAL}" \
    --no-pretrained-init \
    --amp-dtype none \
    2>&1 | tee "${FULL_VAL_DIR}/eval.log"
fi

"${PYTHON}" - "${COVERAGE}" "${FULL_VAL}" "${OUTPUT_ROOT}/v30_summary.json" "${EXPERIMENT_LABEL}" <<'PY'
import json, sys
coverage = json.load(open(sys.argv[1]))
metrics = json.load(open(sys.argv[2]))
results = metrics.get("results", metrics)
def value(threshold, key="F1"):
    row = results.get(str(threshold), results.get(f"{threshold:.2f}", {}))
    if not isinstance(row, dict):
        return None
    return row.get(key, row.get(key.lower()))
oracle_050 = coverage.get("capacity", {}).get("0.50", {}).get("all_candidate_oracle", {})
oracle_075 = coverage.get("capacity", {}).get("0.75", {}).get("all_candidate_oracle", {})
summary = {
    "experiment": sys.argv[4],
    "endpoint_iteration": 35000,
    "historical_v7_35k": {
        "f1_050": 0.7819890656674242,
        "f1_075": 0.5514773634744149,
    },
    "treatment": {"f1_050": value(0.5), "f1_075": value(0.75)},
    "all32_oracle_recall": {
        "0.50": oracle_050.get("recall"),
        "0.75": oracle_075.get("recall"),
    },
    "comparison_contract": "historical control; model/seed/schedule matched, exact legacy worker RNG unrecoverable",
    "test_split_used": False,
}
for key in ("f1_050", "f1_075"):
    left = summary["treatment"].get(key)
    right = summary["historical_v7_35k"].get(key)
    summary.setdefault("treatment_minus_historical", {})[key] = (
        None if left is None else left - right
    )
with open(sys.argv[3], "w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "V30 30K->35K treatment and validation evaluation complete. Test remained closed."
