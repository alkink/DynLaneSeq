#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/diagnostics/unified_lane_set_v7_joint_four_slot_long/seed_3407/joint_four_slot}"
GATE_ROOT="${GATE_ROOT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k}"
LEGACY_CONFIG="${LEGACY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_278k.yaml}"
FIXED_CONFIG="${FIXED_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GATE_ROOT}/full_test_comparison_thr0p25_fp32}"

SCORE_THRESH="${SCORE_THRESH:-0.25}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-4}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
METRIC_CHUNKSIZE="${METRIC_CHUNKSIZE:-64}"
AMP_DTYPE="${AMP_DTYPE:-none}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing CULane root: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ "${SCORE_THRESH}" != "0.25" ]]; then
  echo "This comparison is fixed to the historical V7 score threshold 0.25." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "none" ]]; then
  echo "This comparison requires the historical FP32/TF32 inference path (AMP_DTYPE=none)." >&2
  exit 1
fi

labels=(
  source_iter_0125000
  legacy_iter_0130000
  legacy_iter_0135000
  resume_safe_iter_0130000
  resume_safe_iter_0135000
)
configs=(
  "${LEGACY_CONFIG}"
  "${LEGACY_CONFIG}"
  "${LEGACY_CONFIG}"
  "${FIXED_CONFIG}"
  "${FIXED_CONFIG}"
)
checkpoints=(
  "${SOURCE_ROOT}/iter_0125000.pt"
  "${GATE_ROOT}/seed_3407/legacy/iter_0130000.pt"
  "${GATE_ROOT}/seed_3407/legacy/iter_0135000.pt"
  "${GATE_ROOT}/seed_3407/resume_safe/iter_0130000.pt"
  "${GATE_ROOT}/seed_3407/resume_safe/iter_0135000.pt"
)

for config in "${LEGACY_CONFIG}" "${FIXED_CONFIG}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Missing V7 config: ${config}" >&2
    exit 1
  fi
done
for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing V7 checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"

for index in "${!labels[@]}"; do
  label="${labels[${index}]}"
  config="${configs[${index}]}"
  checkpoint="${checkpoints[${index}]}"
  output_dir="${OUTPUT_ROOT}/${label}"
  metrics_json="${output_dir}/metrics.json"

  if [[ -s "${metrics_json}" ]]; then
    echo "[SKIP] ${label}: ${metrics_json} already exists"
    continue
  fi

  mkdir -p "${output_dir}"
  echo "[EVAL] ${label}"
  echo "       checkpoint=${checkpoint}"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --score-mode four_slot \
    --score-thresh "${SCORE_THRESH}" \
    --quality-score-power 0.0 \
    --top-k 4 \
    --nms-distance-thresh-px 0.0 \
    --nms-min-overlap-points 5 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-num-workers "${EVAL_NUM_WORKERS}" \
    --eval-prefetch-factor "${EVAL_PREFETCH_FACTOR}" \
    --metric-workers "${METRIC_WORKERS}" \
    --metric-chunksize "${METRIC_CHUNKSIZE}" \
    --iou-thresholds 0.5 0.75 \
    --categories \
    --pred-dir "${output_dir}/predictions" \
    --output-txt "${output_dir}/metrics.txt" \
    --output-json "${metrics_json}" \
    --no-pretrained-init \
    --amp-dtype "${AMP_DTYPE}" \
    2>&1 | tee "${output_dir}/eval.log"
done

echo
echo "Completed V7 full-test comparison: ${OUTPUT_ROOT}"
for index in "${!labels[@]}"; do
  label="${labels[${index}]}"
  metrics="${OUTPUT_ROOT}/${label}/metrics.txt"
  if [[ -f "${metrics}" ]]; then
    echo
    echo "[${label}]"
    grep -E '^IoU (0\.50|0\.75):|^mean:' "${metrics}" || true
  fi
done
