#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-none}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache/rowref_lockin_precision_75k}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/rowref_assignment_score_75k/summary.json}"

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

mkdir -p "$(dirname "${OUTPUT_JSON}")" "${CACHE_DIR}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_row_reference_assignment_score_trace \
  --control-config "${CONTROL_CONFIG}" \
  --control-checkpoint "${CONTROL_CHECKPOINT}" \
  --candidate-config "${CANDIDATE_CONFIG}" \
  --candidate-checkpoint "${CANDIDATE_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}" \
  --reuse-cache \
  --max-batches "${MAX_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --sample-strategy uniform \
  --amp-dtype "${AMP_DTYPE}" \
  --control-score-threshold 0.30 \
  --control-quality-power 0.50 \
  --candidate-score-threshold 0.20 \
  --candidate-quality-power 0.25 \
  --top-k 4 \
  --iou-thresholds 0.50 0.75 \
  --output-json "${OUTPUT_JSON}"

echo "Assignment/score trace complete: ${OUTPUT_JSON}"
