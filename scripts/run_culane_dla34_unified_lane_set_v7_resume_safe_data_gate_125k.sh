#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
LEGACY_CONFIG="${LEGACY_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_278k.yaml}"
FIXED_CONFIG="${FIXED_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_joint_four_slot_long/seed_3407/joint_four_slot/iter_0125000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-125000}"
SPLIT_ITERATION="${SPLIT_ITERATION:-130000}"
END_ITERATION="${END_ITERATION:-135000}"
SEED="${SEED:-3407}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
EVAL_AMP_DTYPE="${EVAL_AMP_DTYPE:-none}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v7_resume_safe_data_gate_125k}"

alternate_source="outputs/diagnostic_cache/unified_lane_set_v7_joint_four_slot_long/seed_3407/joint_four_slot/iter_0125000.pt"
if [[ ! -f "${SOURCE_CHECKPOINT}" && -f "${alternate_source}" ]]; then
  SOURCE_CHECKPOINT="${alternate_source}"
fi

for required in "${LEGACY_CONFIG}" "${FIXED_CONFIG}" "${SOURCE_CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V7 resume-safe gate artefact: ${required}" >&2
    exit 1
  fi
done
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing CULane root: ${DATA_ROOT}" >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V7 paired gate requires effective batch 16." >&2
  exit 1
fi
if (( SOURCE_ITERATION != 125000 || SPLIT_ITERATION != 130000 || END_ITERATION != 135000 )); then
  echo "The causal gate is fixed to 125k -> 130k -> 135k." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" ]]; then
  echo "V7 paired training requires AMP_DTYPE=bfloat16." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/reports" "${CACHE_ROOT}"
PREFLIGHT="${OUTPUT_ROOT}/resume_safe_data_preflight.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v7_resume_safe_data_contract \
  --legacy-config "${LEGACY_CONFIG}" \
  --fixed-config "${FIXED_CONFIG}" \
  --checkpoint "${SOURCE_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --expected-iteration "${SOURCE_ITERATION}" \
  --output-json "${PREFLIGHT}"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

train_segment() {
  local arm="$1"
  local config="$2"
  local resume_mode="$3"
  local source="$4"
  local expected_source="$5"
  local target="$6"
  local run_dir="${OUTPUT_ROOT}/seed_${SEED}/${arm}"
  local target_tag
  target_tag="$(printf '%07d' "${target}")"
  local target_checkpoint="${run_dir}/iter_${target_tag}.pt"
  mkdir -p "${run_dir}"
  if [[ -f "${target_checkpoint}" ]]; then
    echo "[SKIP] ${arm} target exists: ${target_checkpoint}"
    return
  fi
  local actual_source
  actual_source="$(checkpoint_iteration "${source}")"
  if (( actual_source != expected_source )); then
    echo "${arm} source iteration ${actual_source}, expected ${expected_source}." >&2
    exit 1
  fi
  local steps=$((target - expected_source))
  echo "Training ${arm}: ${expected_source} -> ${target} (${steps} steps)"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
    --config "${config}" \
    --resume "${source}" \
    --resume-safe-data "${resume_mode}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --output-dir "${run_dir}" \
    --max-iters "${steps}" \
    --checkpoint-interval "${steps}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --num-workers "${NUM_WORKERS}" \
    --seg-aux-amp-dtype "${AMP_DTYPE}" \
    --compile-model false \
    2>&1 | tee -a "${run_dir}/train.log"
  if [[ ! -f "${target_checkpoint}" ]]; then
    echo "${arm} did not produce ${target_checkpoint}." >&2
    exit 1
  fi
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  # Both arms are deliberately stopped and restarted at 130k. The legacy arm
  # replays its private loader/worker RNG stream; the fixed arm addresses the
  # next micro-batch directly from iteration 130000.
  train_segment legacy "${LEGACY_CONFIG}" false \
    "${SOURCE_CHECKPOINT}" "${SOURCE_ITERATION}" "${SPLIT_ITERATION}"
  train_segment legacy "${LEGACY_CONFIG}" false \
    "${OUTPUT_ROOT}/seed_${SEED}/legacy/iter_0130000.pt" \
    "${SPLIT_ITERATION}" "${END_ITERATION}"
  train_segment resume_safe "${FIXED_CONFIG}" true \
    "${SOURCE_CHECKPOINT}" "${SOURCE_ITERATION}" "${SPLIT_ITERATION}"
  train_segment resume_safe "${FIXED_CONFIG}" true \
    "${OUTPUT_ROOT}/seed_${SEED}/resume_safe/iter_0130000.pt" \
    "${SPLIT_ITERATION}" "${END_ITERATION}"
fi

evaluate_uniform() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local report="$4"
  local cache="$5"
  if [[ -f "${report}" ]]; then
    echo "[SKIP] ${label} report exists: ${report}"
    return
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing ${label} checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --device "${DEVICE}" \
    --cache-dir "${cache}" \
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
    --amp-dtype "${EVAL_AMP_DTYPE}" \
    --output-json "${report}"
}

if [[ "${RUN_EVAL}" == "1" ]]; then
  source_report="${OUTPUT_ROOT}/reports/source_iter_0125000_uniform256.json"
  evaluate_uniform source "${LEGACY_CONFIG}" "${SOURCE_CHECKPOINT}" \
    "${source_report}" "${CACHE_ROOT}/source"

  legacy_args=()
  fixed_args=()
  for iteration in "${SPLIT_ITERATION}" "${END_ITERATION}"; do
    tag="$(printf '%07d' "${iteration}")"
    legacy_report="${OUTPUT_ROOT}/reports/legacy_iter_${tag}_uniform256.json"
    fixed_report="${OUTPUT_ROOT}/reports/resume_safe_iter_${tag}_uniform256.json"
    evaluate_uniform "legacy-${iteration}" "${LEGACY_CONFIG}" \
      "${OUTPUT_ROOT}/seed_${SEED}/legacy/iter_${tag}.pt" \
      "${legacy_report}" "${CACHE_ROOT}/legacy/iter_${tag}"
    evaluate_uniform "resume-safe-${iteration}" "${FIXED_CONFIG}" \
      "${OUTPUT_ROOT}/seed_${SEED}/resume_safe/iter_${tag}.pt" \
      "${fixed_report}" "${CACHE_ROOT}/resume_safe/iter_${tag}"
    legacy_args+=(--legacy-report "${iteration}=${legacy_report}")
    fixed_args+=(--fixed-report "${iteration}=${fixed_report}")
  done

  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v7_resume_safe_data_gate \
    --source-report "${source_report}" \
    "${legacy_args[@]}" \
    "${fixed_args[@]}" \
    --preflight-contract "${PREFLIGHT}" \
    --output-json "${OUTPUT_ROOT}/v7_resume_safe_data_summary.json"
fi

echo "V7 resume-safe paired gate: ${OUTPUT_ROOT}"
