#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/diagnostics/dla34_rowref_unified_selection_gate_10k}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${CHECKPOINT_DIR}/iter_0010000.pt}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostics/dla34_unified_selector_frozen_10k}"
VAL_CACHE_IMAGES="${VAL_CACHE_IMAGES:-256}"
VAL_CACHE="${VAL_CACHE:-${CACHE_DIR}/val_features_${VAL_CACHE_IMAGES}.pt}"
REFERENCE_REPORT="${REFERENCE_REPORT:-outputs/diagnostics/dla34_four_slot_coverage_frozen_10k/four_slot_coverage_probe.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/dla34_cluster_representative_decomposition_10k}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${REFERENCE_REPORT}" && -f outputs/diagnostics/four_slot_coverage_probe.json ]]; then
  REFERENCE_REPORT=outputs/diagnostics/four_slot_coverage_probe.json
fi

for path in "${SOURCE_CHECKPOINT}" "${VAL_CACHE}" "${REFERENCE_REPORT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.decompose_cluster_representative_selection \
  --config "${CONFIG}" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --val-cache "${VAL_CACHE}" \
  --reference-report "${REFERENCE_REPORT}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --output-json "${OUTPUT_DIR}/cluster_representative_decomposition.json"

echo "Cluster/representative decomposition completed:"
echo "  ${OUTPUT_DIR}/cluster_representative_decomposition.json"
