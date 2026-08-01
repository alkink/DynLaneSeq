#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/diagnostics/dla34_rowref_unified_selection_gate_10k}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${CHECKPOINT_DIR}/iter_0010000.pt}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostics/dla34_unified_selector_frozen_10k}"
TRAIN_CACHE="${TRAIN_CACHE:-${CACHE_DIR}/train_features_2048.pt}"
VAL_CACHE="${VAL_CACHE:-${CACHE_DIR}/val_features_256.pt}"
REFERENCE_REPORT="${REFERENCE_REPORT:-outputs/diagnostics/dla34_hierarchical_cluster_selector_10k/hierarchical_cluster_selector.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/dla34_hard_decision_selector_10k}"
TRAIN_STEPS="${TRAIN_STEPS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${REFERENCE_REPORT}" && -f outputs/diagnostics/hierarchical_cluster_selector.json ]]; then
  REFERENCE_REPORT=outputs/diagnostics/hierarchical_cluster_selector.json
fi

for path in "${SOURCE_CHECKPOINT}" "${TRAIN_CACHE}" "${VAL_CACHE}" "${REFERENCE_REPORT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_hard_decision_selector \
  --config "${CONFIG}" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --train-cache "${TRAIN_CACHE}" \
  --val-cache "${VAL_CACHE}" \
  --reference-report "${REFERENCE_REPORT}" \
  --device "${DEVICE}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --output-json "${OUTPUT_DIR}/hard_decision_selector.json" \
  --save-probe "${OUTPUT_DIR}/hard_decision_selector.pt"

echo "Hard-decision frozen diagnosis completed:"
echo "  ${OUTPUT_DIR}/hard_decision_selector.json"
