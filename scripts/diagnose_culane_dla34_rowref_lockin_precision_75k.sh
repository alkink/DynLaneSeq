#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
LOCKIN_MAX_BATCHES="${LOCKIN_MAX_BATCHES:-16}"
PRECISION_MAX_BATCHES="${PRECISION_MAX_BATCHES:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/rowref_lockin_precision_75k}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache/rowref_lockin_precision_75k}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_selective_cooldown_10k/iter_0075000.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_object0p5_matcher_10k.yaml}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_10k/iter_0075000.pt}"

for path in \
  "${CONTROL_CONFIG}" \
  "${CONTROL_CHECKPOINT}" \
  "${CANDIDATE_CONFIG}" \
  "${CANDIDATE_CHECKPOINT}"
do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required diagnostic input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

LOCKIN_JSON="${OUTPUT_DIR}/control_lockin_uniform64.json"
CONTROL_JSON="${OUTPUT_DIR}/control_precision_uniform256.json"
CANDIDATE_JSON="${OUTPUT_DIR}/candidate_precision_uniform256.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

echo "===== Stage 1/4: inference-only reference lock-in counterfactual ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_row_reference_lockin \
  --config "${CONTROL_CONFIG}" \
  --checkpoint "${CONTROL_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${LOCKIN_MAX_BATCHES}" \
  --sample-strategy uniform \
  --amp-dtype "${AMP_DTYPE}" \
  --quality-power 0.50 \
  --score-threshold 0.30 \
  --rescue-layer 3 \
  --output-json "${LOCKIN_JSON}"

echo "===== Stage 2/4: control official-raster precision decomposition ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_row_reference_precision \
  --config "${CONTROL_CONFIG}" \
  --checkpoint "${CONTROL_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}" \
  --reuse-cache \
  --max-batches "${PRECISION_MAX_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --sample-strategy uniform \
  --score-threshold 0.30 \
  --quality-power 0.50 \
  --output-json "${CONTROL_JSON}"

echo "===== Stage 3/4: lambda_obj=0.5 official-raster precision decomposition ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_row_reference_precision \
  --config "${CANDIDATE_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}" \
  --reuse-cache \
  --max-batches "${PRECISION_MAX_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --sample-strategy uniform \
  --score-threshold 0.20 \
  --quality-power 0.25 \
  --output-json "${CANDIDATE_JSON}"

echo "===== Stage 4/4: paired summary ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_row_reference_precision_pair \
  --control-json "${CONTROL_JSON}" \
  --candidate-json "${CANDIDATE_JSON}" \
  --lockin-json "${LOCKIN_JSON}" \
  --output-json "${SUMMARY_JSON}"

echo "Diagnostic complete: ${SUMMARY_JSON}"
