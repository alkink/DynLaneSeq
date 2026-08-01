#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/diagnostics/dla34_rowref_unified_selection_gate_10k}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${CHECKPOINT_DIR}/iter_0010000.pt}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostics/dla34_unified_selector_frozen_10k}"
TRAIN_CACHE_IMAGES="${TRAIN_CACHE_IMAGES:-2048}"
VAL_CACHE_IMAGES="${VAL_CACHE_IMAGES:-256}"
TRAIN_CACHE="${TRAIN_CACHE:-${CACHE_DIR}/train_features_${TRAIN_CACHE_IMAGES}.pt}"
VAL_CACHE="${VAL_CACHE:-${CACHE_DIR}/val_features_${VAL_CACHE_IMAGES}.pt}"
REFERENCE_REPORT="${REFERENCE_REPORT:-outputs/diagnostics/dla34_four_slot_coverage_frozen_10k/four_slot_coverage_probe.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/dla34_hierarchical_cluster_selector_10k}"
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${REFERENCE_REPORT}" && -f outputs/diagnostics/four_slot_coverage_probe.json ]]; then
  REFERENCE_REPORT=outputs/diagnostics/four_slot_coverage_probe.json
fi

for path in "${SOURCE_CHECKPOINT}" "${TRAIN_CACHE}" "${VAL_CACHE}" "${REFERENCE_REPORT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_hierarchical_cluster_selector \
  --config "${CONFIG}" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --train-cache "${TRAIN_CACHE}" \
  --val-cache "${VAL_CACHE}" \
  --reference-report "${REFERENCE_REPORT}" \
  --device "${DEVICE}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --output-json "${OUTPUT_DIR}/hierarchical_cluster_selector.json" \
  --save-probe "${OUTPUT_DIR}/hierarchical_cluster_selector.pt"

echo "Hierarchical frozen diagnosis completed:"
echo "  ${OUTPUT_DIR}/hierarchical_cluster_selector.json"
