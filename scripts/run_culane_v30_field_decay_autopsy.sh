#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/venv/clrernet/bin/python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-12}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
METRIC_WORKERS="${METRIC_WORKERS:-16}"
SEED="${SEED:-3407}"

V7_CONFIG="dynlaneseq_eg/configs/culane_v7_exact_control_35k_to50k.yaml"
V30_CONFIG="dynlaneseq_eg/configs/culane_v30_field_only_35k_to50k.yaml"
PAIR_35_ROOT="outputs/diagnostics/v30_field_only_exact_pair_30k_to35k"
PAIR_50_ROOT="outputs/diagnostics/v31_selection_bridge_exact_pair_35k_to50k"
EXACT_50_ROOT="outputs/diagnostics/v30_field_only_vs_v7_exact_35k_to50k"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXACT_50_ROOT}/field_decay_autopsy}"
TRAJECTORY_ROOT="${OUTPUT_ROOT}/trajectory"

V7_35="${PAIR_35_ROOT}/train/control/iter_0035000.pt"
V30_35="${PAIR_35_ROOT}/train/field_only/iter_0035000.pt"
V7_40="${EXACT_50_ROOT}/train/v7_control/iter_0040000.pt"
V30_40="${PAIR_50_ROOT}/train/field_only/iter_0040000.pt"
V7_45="${EXACT_50_ROOT}/train/v7_control/iter_0045000.pt"
V30_45="${PAIR_50_ROOT}/train/field_only/iter_0045000.pt"
V7_50="${EXACT_50_ROOT}/train/v7_control/iter_0050000.pt"
V30_50="${PAIR_50_ROOT}/train/field_only/iter_0050000.pt"

for required in \
  "${V7_CONFIG}" "${V30_CONFIG}" \
  "${V7_35}" "${V30_35}" "${V7_40}" "${V30_40}" \
  "${V7_45}" "${V30_45}" "${V7_50}" "${V30_50}" \
  "${DATA_ROOT}/list/val.txt" "${DATA_ROOT}/list/train_gt.txt"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing V30 field-decay artifact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${TRAJECTORY_ROOT}" "${OUTPUT_ROOT}/full_stage/cache/v7" \
  "${OUTPUT_ROOT}/gradient"

evaluate_arm() {
  local iteration="$1"
  local arm="$2"
  local config="$3"
  local checkpoint="$4"
  local directory="${TRAJECTORY_ROOT}/iter_$(printf '%07d' "${iteration}")/${arm}"
  mkdir -p "${directory}"
  if [[ ! -s "${directory}/metrics.json" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
      --config "${config}" --checkpoint "${checkpoint}" --split val \
      --dataset-root "${DATA_ROOT}" --device "${DEVICE}" \
      --score-mode four_slot --score-thresh 0.0 --quality-score-power 0.0 \
      --top-k 4 --nms-distance-thresh-px 0.0 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" --eval-num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" --iou-thresholds 0.5 0.75 \
      --pred-dir "${directory}/predictions" \
      --output-txt "${directory}/metrics.txt" \
      --output-json "${directory}/metrics.json" \
      --no-pretrained-init --amp-dtype none \
      2>&1 | tee "${directory}/eval.log"
  fi
}

coverage_arm() {
  local iteration="$1"
  local arm="$2"
  local config="$3"
  local checkpoint="$4"
  local directory="${TRAJECTORY_ROOT}/iter_$(printf '%07d' "${iteration}")"
  local report="${directory}/${arm}_uniform256.json"
  if [[ ! -s "${report}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${config}" --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" --split val \
      --list-path "${DATA_ROOT}/list/val.txt" --device "${DEVICE}" \
      --cache-dir "${directory}/cache/${arm}_uniform256" \
      --max-batches 64 --eval-batch-size 4 --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" --sample-strategy uniform \
      --top-k 4 --iou-thresholds 0.50 0.75 --output-json "${report}" \
      2>&1 | tee "${report%.json}.log"
  fi
}

paired_arm() {
  local iteration="$1"
  local directory="${TRAJECTORY_ROOT}/iter_$(printf '%07d' "${iteration}")"
  if [[ ! -s "${directory}/paired.json" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_full_validation_paired_effect \
      --experiment-name "V30 Field-only versus exact V7 at global ${iteration}" \
      --source-pred-dir "${directory}/v7/predictions" \
      --candidate-pred-dir "${directory}/v30/predictions" \
      --dataset-root "${DATA_ROOT}" --list-path "${DATA_ROOT}/list/val.txt" \
      --uniform-report "${directory}/v30_uniform256.json" \
      --source-metrics "${directory}/v7/metrics.json" \
      --candidate-metrics "${directory}/v30/metrics.json" \
      --iou-thresholds 0.5 0.75 --workers "${METRIC_WORKERS}" \
      --bootstrap-samples 10000 --bootstrap-seed "$((SEED + iteration))" \
      --output-json "${directory}/paired.json"
  fi
}

for spec in \
  "35000:v7:${V7_CONFIG}:${V7_35}" \
  "35000:v30:${V30_CONFIG}:${V30_35}" \
  "40000:v7:${V7_CONFIG}:${V7_40}" \
  "40000:v30:${V30_CONFIG}:${V30_40}" \
  "45000:v7:${V7_CONFIG}:${V7_45}" \
  "45000:v30:${V30_CONFIG}:${V30_45}"; do
  IFS=: read -r iteration arm config checkpoint <<<"${spec}"
  evaluate_arm "${iteration}" "${arm}" "${config}" "${checkpoint}"
  coverage_arm "${iteration}" "${arm}" "${config}" "${checkpoint}"
done
for iteration in 35000 40000 45000; do
  paired_arm "${iteration}"
done

# Reuse the already audited global-50K reports without rerunning inference.
ITER50="${TRAJECTORY_ROOT}/iter_0050000"
mkdir -p "${ITER50}/v7" "${ITER50}/v30"
cp -f "${EXACT_50_ROOT}/reports/v7_exact_full_val/metrics.json" "${ITER50}/v7/metrics.json"
cp -f "${PAIR_50_ROOT}/reports/field_only_full_val/metrics.json" "${ITER50}/v30/metrics.json"
cp -f "${EXACT_50_ROOT}/reports/v7_exact_uniform256.json" "${ITER50}/v7_uniform256.json"
cp -f "${PAIR_50_ROOT}/reports/field_only_uniform256.json" "${ITER50}/v30_uniform256.json"
cp -f "${EXACT_50_ROOT}/paired_v7_vs_field_only.json" "${ITER50}/paired.json"

V7_FULL_REPORT="${OUTPUT_ROOT}/full_stage/v7_exact_full_stage_coverage.json"
if [[ ! -s "${V7_FULL_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${V7_CONFIG}" --checkpoint "${V7_50}" \
    --dataset-root "${DATA_ROOT}" --split val \
    --list-path "${DATA_ROOT}/list/val.txt" --device "${DEVICE}" \
    --cache-dir "${OUTPUT_ROOT}/full_stage/cache/v7" --max-batches 0 \
    --eval-batch-size 8 --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" --amp-dtype none \
    --sample-strategy sequential --top-k 4 --iou-thresholds 0.50 0.75 \
    --hard-diversity-distances 20 --mmr-penalties 0.75 --mmr-sigmas 20 \
    --output-json "${V7_FULL_REPORT}" \
    2>&1 | tee "${V7_FULL_REPORT%.json}.log"
fi

mapfile -t V7_FULL_CACHES < <(find "${OUTPUT_ROOT}/full_stage/cache/v7" -type f -name '*.pt' | sort)
mapfile -t V30_FULL_CACHES < <(find "${PAIR_50_ROOT}/stage_autopsy/cache/field_only" -type f -name '*.pt' | sort)
if (( ${#V7_FULL_CACHES[@]} != 1 || ${#V30_FULL_CACHES[@]} != 1 )); then
  echo "Expected one exact-V7 and one V30 full-stage cache." >&2
  exit 1
fi

GEOMETRY_DRIFT="${OUTPUT_ROOT}/full_stage/v7_vs_v30_geometry_drift.json"
if [[ ! -s "${GEOMETRY_DRIFT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_field_geometry_drift \
    --source-cache "${V7_FULL_CACHES[0]}" \
    --candidate-cache "${V30_FULL_CACHES[0]}" \
    --thresholds 0.50 0.75 --output-json "${GEOMETRY_DRIFT}" \
    2>&1 | tee "${GEOMETRY_DRIFT%.json}.log"
fi

GRADIENT_35="${OUTPUT_ROOT}/gradient/field_conflict_35k.json"
GRADIENT_50="${OUTPUT_ROOT}/gradient/field_conflict_50k.json"
if [[ ! -s "${GRADIENT_35}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_field_gradient_conflict \
    --config "${V30_CONFIG}" --checkpoint "${V30_35}" \
    --dataset-root "${DATA_ROOT}" --device "${DEVICE}" \
    --batch-size 1 --num-workers 0 --max-batches 12 --seed "${SEED}" \
    --output-json "${GRADIENT_35}" \
    2>&1 | tee "${GRADIENT_35%.json}.log"
fi
if [[ ! -s "${GRADIENT_50}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v30_field_gradient_conflict \
    --config "${V30_CONFIG}" --checkpoint "${V30_50}" \
    --dataset-root "${DATA_ROOT}" --device "${DEVICE}" \
    --batch-size 1 --num-workers 0 --max-batches 12 --seed "${SEED}" \
    --output-json "${GRADIENT_50}" \
    2>&1 | tee "${GRADIENT_50%.json}.log"
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v30_field_decay_autopsy \
  --trajectory-root "${TRAJECTORY_ROOT}" \
  --geometry-drift "${GEOMETRY_DRIFT}" \
  --gradient-35k "${GRADIENT_35}" --gradient-50k "${GRADIENT_50}" \
  --output-json "${OUTPUT_ROOT}/field_decay_summary.json" \
  2>&1 | tee "${OUTPUT_ROOT}/field_decay_summary.log"

echo "V30 Field-only decay autopsy complete. Test remained closed."
