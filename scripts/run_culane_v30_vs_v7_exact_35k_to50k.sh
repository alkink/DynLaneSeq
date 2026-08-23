#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/venv/clrernet/bin/python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION=35000
ENDPOINT_ITERATION=50000
TRAINING_STEPS=15000
BATCH_SIZE=4
GRAD_ACCUM=4
NUM_WORKERS="${NUM_WORKERS:-12}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"

V7_CONFIG="dynlaneseq_eg/configs/culane_v7_exact_control_35k_to50k.yaml"
V30_CONFIG="dynlaneseq_eg/configs/culane_v30_field_only_35k_to50k.yaml"
V7_SOURCE="outputs/diagnostics/v30_field_only_exact_pair_30k_to35k/train/control/iter_0035000.pt"
V30_SOURCE="outputs/diagnostics/v30_field_only_exact_pair_30k_to35k/train/field_only/iter_0035000.pt"
V30_ROOT="outputs/diagnostics/v31_selection_bridge_exact_pair_35k_to50k"
V30_ENDPOINT="${V30_ROOT}/train/field_only/iter_0050000.pt"
V30_METRICS="${V30_ROOT}/reports/field_only_full_val/metrics.json"
V30_COVERAGE="${V30_ROOT}/reports/field_only_uniform256.json"
HISTORICAL_V7_METRICS="${V30_ROOT}/reports/historical_v7_50k_full_val/metrics.json"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v30_field_only_vs_v7_exact_35k_to50k}"
V7_DIR="${OUTPUT_ROOT}/train/v7_control"
V7_ENDPOINT="${V7_DIR}/iter_0050000.pt"
V7_VAL_DIR="${OUTPUT_ROOT}/reports/v7_exact_full_val"
V7_METRICS="${V7_VAL_DIR}/metrics.json"
V7_COVERAGE="${OUTPUT_ROOT}/reports/v7_exact_uniform256.json"
PAIR_CONTRACT="${OUTPUT_ROOT}/audits/exact_pair_contract.json"
LINEAGE_CONTRACT="${OUTPUT_ROOT}/audits/source_lineage_contract.json"
PAIRED_AUDIT="${OUTPUT_ROOT}/paired_v7_vs_field_only.json"
TRANSITION_AUDIT="${OUTPUT_ROOT}/paired_v7_vs_field_only_transitions.json"
SUMMARY="${OUTPUT_ROOT}/exact_pair_summary.json"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V7_CONFIG}" "${V30_CONFIG}" "${V7_SOURCE}" "${V30_SOURCE}" \
  "${V30_ENDPOINT}" "${V30_METRICS}" "${V30_COVERAGE}" \
  "${DATA_ROOT}/list/train_gt.txt" "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing V30/V7 exact-pair artifact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports" "${V7_DIR}"

if [[ ! -s "${PAIR_CONTRACT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_exact_pair_contract \
    --control-config "${V7_CONFIG}" \
    --treatment-config "${V30_CONFIG}" \
    --checkpoint "${V7_SOURCE}" \
    --dataset-root "${DATA_ROOT}" \
    --start-iteration "${SOURCE_ITERATION}" \
    --optimizer-steps "${TRAINING_STEPS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --output-json "${PAIR_CONTRACT}" \
    2>&1 | tee "${OUTPUT_ROOT}/audits/exact_pair_contract.log"
fi

"${PYTHON}" - "${V7_SOURCE}" "${V30_SOURCE}" "${V30_ENDPOINT}" \
  "${PAIR_CONTRACT}" "${LINEAGE_CONTRACT}" <<'PY'
import json
import sys
from pathlib import Path
from dynlaneseq_eg.engine.checkpoint import _torch_load
from dynlaneseq_eg.tools.audit_v31_continuation_pair_contract import _nested_equal

v7_path, v30_path, endpoint_path, pair_path, output_path = map(Path, sys.argv[1:])
v7 = _torch_load(v7_path)
v30 = _torch_load(v30_path)
endpoint = _torch_load(endpoint_path)
pair = json.loads(pair_path.read_text(encoding="utf-8"))
checks = {
    "pair_stream_contract_passed": pair.get("passed") is True,
    "v7_source_is_35k": int(v7.get("iteration", -1)) == 35000,
    "v30_source_is_35k": int(v30.get("iteration", -1)) == 35000,
    "v30_endpoint_is_50k": int(endpoint.get("iteration", -1)) == 50000,
    "scheduler_state_exact_at_35k": _nested_equal(v7.get("scheduler"), v30.get("scheduler")),
    "optimizer_present_in_both": isinstance(v7.get("optimizer"), dict) and isinstance(v30.get("optimizer"), dict),
    "rng_reference_present": isinstance(v30.get("rng_state"), dict) and bool(v30.get("rng_state")),
}
report = {
    "experiment": "V7/V30 exact 35K source-lineage contract",
    "v7_source": str(v7_path),
    "v30_source_and_rng_reference": str(v30_path),
    "v30_endpoint": str(endpoint_path),
    "checks": checks,
    "passed": all(checks.values()),
    "test_split_used": False,
}
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit("V7/V30 source-lineage contract failed")
PY

if [[ ! -f "${V7_ENDPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
    --config "${V7_CONFIG}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --output-dir "${V7_DIR}" \
    --resume "${V7_SOURCE}" \
    --resume-rng-from "${V30_SOURCE}" \
    --max-iters "${TRAINING_STEPS}" \
    --checkpoint-interval 5000 \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --seg-aux-amp-dtype bfloat16 \
    --compile-model true \
    --resume-safe-data true \
    2>&1 | tee -a "${V7_DIR}/train.log"
fi
if (( $(checkpoint_iteration "${V7_ENDPOINT}") != ENDPOINT_ITERATION )); then
  echo "Exact V7 global-50K endpoint is missing or invalid." >&2
  exit 1
fi
if ! grep -Eq "resume_rng_override.*restored.*True" "${V7_DIR}/train.log"; then
  echo "Exact V7 did not prove common V30 RNG restoration." >&2
  exit 1
fi

if [[ ! -s "${V7_COVERAGE}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${V7_CONFIG}" --checkpoint "${V7_ENDPOINT}" \
    --dataset-root "${DATA_ROOT}" --split val \
    --list-path "${DATA_ROOT}/list/val.txt" --device "${DEVICE}" \
    --cache-dir "${OUTPUT_ROOT}/cache/v7_exact_uniform256" \
    --max-batches 64 --eval-batch-size 4 --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" --sample-strategy uniform \
    --top-k 4 --iou-thresholds 0.50 0.75 --output-json "${V7_COVERAGE}"
fi

if [[ ! -s "${V7_METRICS}" ]]; then
  mkdir -p "${V7_VAL_DIR}"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${V7_CONFIG}" --checkpoint "${V7_ENDPOINT}" --split val \
    --dataset-root "${DATA_ROOT}" --device "${DEVICE}" \
    --score-mode four_slot --score-thresh 0.0 --quality-score-power 0.0 \
    --top-k 4 --nms-distance-thresh-px 0.0 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" --eval-num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" --iou-thresholds 0.5 0.75 \
    --pred-dir "${V7_VAL_DIR}/predictions" \
    --output-txt "${V7_VAL_DIR}/metrics.txt" --output-json "${V7_METRICS}" \
    --no-pretrained-init --amp-dtype none \
    2>&1 | tee "${V7_VAL_DIR}/eval.log"
fi

if [[ ! -s "${PAIRED_AUDIT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_full_validation_paired_effect \
    --experiment-name "V30 Field-only versus exact-paired V7 at global 50K" \
    --source-pred-dir "${V7_VAL_DIR}/predictions" \
    --candidate-pred-dir "${V30_ROOT}/reports/field_only_full_val/predictions" \
    --dataset-root "${DATA_ROOT}" --list-path "${DATA_ROOT}/list/val.txt" \
    --uniform-report "${V30_COVERAGE}" --source-metrics "${V7_METRICS}" \
    --candidate-metrics "${V30_METRICS}" --iou-thresholds 0.5 0.75 \
    --workers "${METRIC_WORKERS}" --bootstrap-samples 10000 \
    --bootstrap-seed "${SEED}" --output-json "${PAIRED_AUDIT}"
fi

if [[ ! -s "${TRANSITION_AUDIT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_paired_gt_transitions \
    --experiment-name "V30 Field-only versus exact-paired V7 50K transitions" \
    --source-pred-dir "${V7_VAL_DIR}/predictions" \
    --candidate-pred-dir "${V30_ROOT}/reports/field_only_full_val/predictions" \
    --dataset-root "${DATA_ROOT}" --list-path "${DATA_ROOT}/list/val.txt" \
    --iou-thresholds 0.5 0.75 --workers "${METRIC_WORKERS}" \
    --output-json "${TRANSITION_AUDIT}"
fi

HISTORICAL_ARGS=()
if [[ -s "${HISTORICAL_V7_METRICS}" ]]; then
  HISTORICAL_ARGS=(--historical-v7-metrics "${HISTORICAL_V7_METRICS}")
fi
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v30_v7_exact_50k \
  --v7-metrics "${V7_METRICS}" --v30-metrics "${V30_METRICS}" \
  "${HISTORICAL_ARGS[@]}" --v7-coverage "${V7_COVERAGE}" \
  --v30-coverage "${V30_COVERAGE}" --paired-audit "${PAIRED_AUDIT}" \
  --transition-audit "${TRANSITION_AUDIT}" --pair-contract "${PAIR_CONTRACT}" \
  --output-json "${SUMMARY}" 2>&1 | tee "${OUTPUT_ROOT}/exact_pair_summary.log"

echo "V30 Field-only versus exact V7 50K pipeline complete. Test remained closed."
