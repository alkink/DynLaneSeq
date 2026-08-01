#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEFAULT_PROBE_DIR="outputs/diagnostics/dla34_hierarchical_cluster_selector_10k"
ROOT_PROBE_DIR="outputs/diagnostics"
PROBE_REPORT="${PROBE_REPORT:-${DEFAULT_PROBE_DIR}/hierarchical_cluster_selector.json}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-${DEFAULT_PROBE_DIR}/hierarchical_cluster_selector.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/hierarchical_cluster_generalization_audit.json}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${PROBE_REPORT}" && -f "${ROOT_PROBE_DIR}/hierarchical_cluster_selector.json" ]]; then
  PROBE_REPORT="${ROOT_PROBE_DIR}/hierarchical_cluster_selector.json"
fi
if [[ ! -f "${PROBE_CHECKPOINT}" && -f "${ROOT_PROBE_DIR}/hierarchical_cluster_selector.pt" ]]; then
  PROBE_CHECKPOINT="${ROOT_PROBE_DIR}/hierarchical_cluster_selector.pt"
fi

for path in "${PROBE_REPORT}" "${PROBE_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.audit_hierarchical_probe_generalization \
  --probe-report "${PROBE_REPORT}" \
  --probe-checkpoint "${PROBE_CHECKPOINT}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --output-json "${OUTPUT_JSON}"

echo "Frozen train-validation audit completed:"
echo "  ${OUTPUT_JSON}"
