#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
V30_ROOT="${V30_ROOT:-outputs/diagnostics/v30_field_only_exact_pair_30k_to35k}"
V31_ROOT="${V31_ROOT:-outputs/diagnostics/v31_selection_gradient_bridge_exact_pair_30k_to35k}"
AUTOPSY_ROOT="${AUTOPSY_ROOT:-${V31_ROOT}/stage_autopsy}"

CONTROL_REPORT="${V30_ROOT}/stage_autopsy/field_only_full_stage_coverage.json"
CONTROL_CACHE_DIR="${V30_ROOT}/stage_autopsy/cache/field_only"
TREATMENT_CONFIG="dynlaneseq_eg/configs/culane_v31_field_only_selection_gradient_bridge_30k_to35k.yaml"
TREATMENT_CHECKPOINT="${V31_ROOT}/train/selection_bridge/iter_0035000.pt"
TREATMENT_CACHE_DIR="${AUTOPSY_ROOT}/cache/selection_bridge"
TREATMENT_REPORT="${AUTOPSY_ROOT}/selection_bridge_full_stage_coverage.json"
FACTORIAL_REPORT="${AUTOPSY_ROOT}/field_only_vs_selection_bridge_factorial.json"
SUMMARY_REPORT="${AUTOPSY_ROOT}/stage_autopsy_summary.json"

for required in \
  "${CONTROL_REPORT}" \
  "${TREATMENT_CHECKPOINT}" \
  "${V31_ROOT}/exact_pair_summary.json" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing V31 stage-autopsy artifact: ${required}" >&2
    exit 1
  fi
done

mapfile -t CONTROL_CACHES < <(find "${CONTROL_CACHE_DIR}" -type f -name '*.pt' | sort)
if (( ${#CONTROL_CACHES[@]} != 1 )); then
  echo "Expected exactly one complete field-only control cache." >&2
  exit 1
fi

mkdir -p "${TREATMENT_CACHE_DIR}"
if [[ ! -s "${TREATMENT_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${TREATMENT_CONFIG}" \
    --checkpoint "${TREATMENT_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --list-path "${DATA_ROOT}/list/val.txt" \
    --device "${DEVICE}" \
    --cache-dir "${TREATMENT_CACHE_DIR}" \
    --max-batches 0 \
    --eval-batch-size 8 \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --amp-dtype none \
    --sample-strategy sequential \
    --top-k 4 \
    --iou-thresholds 0.50 0.75 \
    --hard-diversity-distances 20 \
    --mmr-penalties 0.75 \
    --mmr-sigmas 20 \
    --output-json "${TREATMENT_REPORT}" \
    2>&1 | tee "${TREATMENT_REPORT%.json}.log"
fi

mapfile -t TREATMENT_CACHES < <(find "${TREATMENT_CACHE_DIR}" -type f -name '*.pt' | sort)
if (( ${#TREATMENT_CACHES[@]} != 1 )); then
  echo "Expected exactly one complete selection-bridge cache." >&2
  exit 1
fi

if [[ ! -s "${FACTORIAL_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_activity_geometry_factorial \
    --source-cache "${CONTROL_CACHES[0]}" \
    --candidate-cache "${TREATMENT_CACHES[0]}" \
    --iou-thresholds 0.50 0.75 \
    --output-json "${FACTORIAL_REPORT}" \
    2>&1 | tee "${FACTORIAL_REPORT%.json}.log"
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v30_stage_autopsy \
  --control-coverage "${CONTROL_REPORT}" \
  --treatment-coverage "${TREATMENT_REPORT}" \
  --factorial-audit "${FACTORIAL_REPORT}" \
  --exact-pair-summary "${V31_ROOT}/exact_pair_summary.json" \
  --output-json "${SUMMARY_REPORT}" \
  2>&1 | tee "${SUMMARY_REPORT%.json}.log"

echo "V31 selection-gradient bridge stage autopsy complete. Test remained closed."
