#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/no_object_matcher_70k_gate}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_no_object_matcher_5k.yaml}"
CONTROL_JSON="${OUTPUT_DIR}/control_70k_uniform256.json"
CANDIDATE_JSON="${OUTPUT_DIR}/candidate_70k_uniform256.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

if [[ -z "${CONTROL_CHECKPOINT:-}" ]]; then
  for candidate in \
    outputs/diagnostics/dla34_rowref_from65k_selective_cooldown_10k/iter_0070000.pt \
    outputs/dla34_rowref_from65k_selective_cooldown_10k/iter_0070000.pt
  do
    if [[ -f "${candidate}" ]]; then
      CONTROL_CHECKPOINT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CANDIDATE_CHECKPOINT:-}" ]]; then
  CANDIDATE_CHECKPOINT="outputs/diagnostics/dla34_rowref_from65k_no_object_matcher_5k/iter_0070000.pt"
fi
if [[ -z "${CONTROL_CHECKPOINT:-}" || ! -f "${CONTROL_CHECKPOINT}" ]]; then
  echo "Missing matched 70k cooldown control. Set CONTROL_CHECKPOINT." >&2
  exit 1
fi
if [[ ! -f "${CANDIDATE_CHECKPOINT}" ]]; then
  echo "Missing 70k no-object matcher candidate: ${CANDIDATE_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
REUSE_ARGS=()
if [[ "${REUSE_CACHE:-0}" == "1" ]]; then
  REUSE_ARGS+=(--reuse-cache)
fi

common_args=(
  --split val
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --top-k-values 4
  --iou-thresholds 0.5 0.7
  --quality-powers 0.0 0.5
  --score-thresholds -1 0.15
  --line-width 30
  --nms-distance-thresh-px 20
  --nms-min-overlap-points 5
  --max-batches "${MAX_BATCHES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --sample-strategy uniform
  --cache-dir "${CACHE_DIR}"
  --exact-postprocess
)

echo "Matched official-raster matcher gate on $((MAX_BATCHES * EVAL_BATCH_SIZE)) validation images"
echo "Control: ${CONTROL_CHECKPOINT}"
echo "Candidate: ${CANDIDATE_CHECKPOINT}"
echo "Fixed ranking settings: q in {0.0,0.5}; score threshold in {-1,0.15}; Top-K=4"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONTROL_CONFIG}" \
  --checkpoint "${CONTROL_CHECKPOINT}" \
  --output-json "${CONTROL_JSON}" \
  "${common_args[@]}" \
  "${REUSE_ARGS[@]}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CANDIDATE_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --output-json "${CANDIDATE_JSON}" \
  "${common_args[@]}" \
  "${REUSE_ARGS[@]}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_nms_ranking_pair \
  --base-json "${CONTROL_JSON}" \
  --candidate-json "${CANDIDATE_JSON}" \
  --output-json "${SUMMARY_JSON}"

echo "Primary result: ${SUMMARY_JSON}"
