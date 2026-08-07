#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v6_a_four_slot_selector_25k_to29k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v5_1_shared_trunk_gate/seed_3407/assignment_fork/iter_0025000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-25000}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v5_four_slot_parameter_matched/four_slot_vs_parameter_matched_32.pt}"
REFERENCE_REPORT="${REFERENCE_REPORT:-outputs/diagnostics/unified_lane_set_v5_four_slot_parameter_matched/four_slot_vs_parameter_matched_32_uniform256.json}"
FAILED_GATE_SUMMARY="${FAILED_GATE_SUMMARY:-outputs/diagnostics/unified_lane_set_v6_a_four_slot_gate_25k/v6_a_summary.json}"
FAILED_TRAIN_LOG="${FAILED_TRAIN_LOG:-outputs/diagnostics/unified_lane_set_v6_a_four_slot_gate_25k/seed_3407/four_slot/train.log}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v6_a_probe_mismatch}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v6_a_probe_mismatch}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SEED="${SEED:-3407}"

# The user may copy compact reports to outputs/diagnostics for inspection.
# Accept those aliases without changing the authoritative checkpoint paths.
if [[ ! -f "${REFERENCE_REPORT}" && -f outputs/diagnostics/four_slot_vs_parameter_matched_32_uniform256.json ]]; then
  REFERENCE_REPORT=outputs/diagnostics/four_slot_vs_parameter_matched_32_uniform256.json
fi
if [[ ! -f "${FAILED_GATE_SUMMARY}" && -f outputs/diagnostics/v6_a_summary.json ]]; then
  FAILED_GATE_SUMMARY=outputs/diagnostics/v6_a_summary.json
fi
if [[ ! -f "${FAILED_TRAIN_LOG}" && -f outputs/diagnostics/train.log ]]; then
  FAILED_TRAIN_LOG=outputs/diagnostics/train.log
fi

for required in \
  "${CONFIG}" \
  "${SOURCE_CHECKPOINT}" \
  "${PROBE_CHECKPOINT}" \
  "${REFERENCE_REPORT}" \
  "${FAILED_GATE_SUMMARY}" \
  "${FAILED_TRAIN_LOG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V6-A mismatch-audit artefact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}"

IMPORTED_CHECKPOINT="${OUTPUT_ROOT}/probe_imported_production_delta.pt"
IMPORT_REPORT="${OUTPUT_ROOT}/probe_import_contract.json"
PARITY_REPORT="${OUTPUT_ROOT}/probe_weights_production_uniform256.json"
TARGET_REPORT="${OUTPUT_ROOT}/target_distribution_uniform256.json"
SUMMARY_REPORT="${OUTPUT_ROOT}/v6_a_probe_mismatch_summary.json"

echo "V6-A zero-training probe/production mismatch audit"
echo "source detector: ${SOURCE_CHECKPOINT}"
echo "successful probe: ${PROBE_CHECKPOINT}"
echo "No optimizer step will be run."

"${PYTHON}" -u -m dynlaneseq_eg.tools.import_v6_a_probe_checkpoint \
  --config "${CONFIG}" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --probe-checkpoint "${PROBE_CHECKPOINT}" \
  --reference-report "${REFERENCE_REPORT}" \
  --expected-source-iteration "${SOURCE_ITERATION}" \
  --output-checkpoint "${IMPORTED_CHECKPOINT}" \
  --output-json "${IMPORT_REPORT}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
  --config "${CONFIG}" \
  --checkpoint "${IMPORTED_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_ROOT}/probe_weights_production" \
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
  --output-json "${PARITY_REPORT}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v6_a_target_distribution \
  --config "${CONFIG}" \
  --checkpoint "${SOURCE_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --max-batches "${MAX_BATCHES}" \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}" \
  --seed "${SEED}" \
  --output-json "${TARGET_REPORT}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v6_a_probe_mismatch \
  --reference-probe-report "${REFERENCE_REPORT}" \
  --probe-import-report "${IMPORT_REPORT}" \
  --production-parity-report "${PARITY_REPORT}" \
  --target-distribution-report "${TARGET_REPORT}" \
  --failed-gate-summary "${FAILED_GATE_SUMMARY}" \
  --failed-train-log "${FAILED_TRAIN_LOG}" \
  --output-json "${SUMMARY_REPORT}"

echo "V6-A mismatch audit complete: ${SUMMARY_REPORT}"
