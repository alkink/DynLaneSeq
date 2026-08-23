#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
EXACT_ROOT="${EXACT_ROOT:-outputs/diagnostics/v30_field_only_exact_pair_30k_to35k}"
AUTOPSY_ROOT="${AUTOPSY_ROOT:-${EXACT_ROOT}/stage_autopsy}"

CONTROL_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml"
TREATMENT_CONFIG="dynlaneseq_eg/configs/culane_v30_joint_slot_field_35k_route_residual_off.yaml"
CONTROL_CHECKPOINT="${EXACT_ROOT}/train/control/iter_0035000.pt"
TREATMENT_CHECKPOINT="${EXACT_ROOT}/train/field_only/iter_0035000.pt"
CONTROL_REPORT="${AUTOPSY_ROOT}/control_full_stage_coverage.json"
TREATMENT_REPORT="${AUTOPSY_ROOT}/field_only_full_stage_coverage.json"
FACTORIAL_REPORT="${AUTOPSY_ROOT}/activity_geometry_factorial.json"
SUMMARY_REPORT="${AUTOPSY_ROOT}/stage_autopsy_summary.json"

for required in \
  "${CONTROL_CHECKPOINT}" \
  "${TREATMENT_CHECKPOINT}" \
  "${EXACT_ROOT}/exact_pair_summary.json" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing stage-autopsy artifact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${AUTOPSY_ROOT}/cache/control" "${AUTOPSY_ROOT}/cache/field_only"

coverage_arm() {
  local config="$1"
  local checkpoint="$2"
  local cache_dir="$3"
  local report="$4"
  if [[ ! -s "${report}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --list-path "${DATA_ROOT}/list/val.txt" \
      --device "${DEVICE}" \
      --cache-dir "${cache_dir}" \
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
      --output-json "${report}" \
      2>&1 | tee "${report%.json}.log"
  fi
}

coverage_arm \
  "${CONTROL_CONFIG}" \
  "${CONTROL_CHECKPOINT}" \
  "${AUTOPSY_ROOT}/cache/control" \
  "${CONTROL_REPORT}"
coverage_arm \
  "${TREATMENT_CONFIG}" \
  "${TREATMENT_CHECKPOINT}" \
  "${AUTOPSY_ROOT}/cache/field_only" \
  "${TREATMENT_REPORT}"

mapfile -t CONTROL_CACHES < <(find "${AUTOPSY_ROOT}/cache/control" -type f -name '*.pt' | sort)
mapfile -t TREATMENT_CACHES < <(find "${AUTOPSY_ROOT}/cache/field_only" -type f -name '*.pt' | sort)
if (( ${#CONTROL_CACHES[@]} != 1 || ${#TREATMENT_CACHES[@]} != 1 )); then
  echo "Expected exactly one complete cache per arm." >&2
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
  --exact-pair-summary "${EXACT_ROOT}/exact_pair_summary.json" \
  --output-json "${SUMMARY_REPORT}" \
  2>&1 | tee "${SUMMARY_REPORT%.json}.log"

echo "V30 exact-paired stage autopsy complete. Test remained closed."
