#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_hybrid_control_gate_10k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml}"
CONTROL_OUT="${CONTROL_OUT:-outputs/diagnostics/dla34_rowref_hybrid_control_gate_10k}"
CANDIDATE_OUT="${CANDIDATE_OUT:-outputs/diagnostics/dla34_rowref_unified_selection_gate_10k}"
REPORT_DIR="${REPORT_DIR:-outputs/diagnostics/rowref_unified_selection_gate_10k}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
OFFICIAL_IOU_WORKERS="${OFFICIAL_IOU_WORKERS:-12}"
ITERATIONS="${ITERATIONS:-2500 5000 7500 10000}"
CACHE_DIR="${CACHE_DIR:-${REPORT_DIR}/cache}"

mkdir -p "${REPORT_DIR}" "${CACHE_DIR}"
pair_args=()

for iteration in ${ITERATIONS}; do
  tag="$(printf '%07d' "${iteration}")"
  control_checkpoint="${CONTROL_OUT}/iter_${tag}.pt"
  candidate_checkpoint="${CANDIDATE_OUT}/iter_${tag}.pt"
  control_json="${REPORT_DIR}/control_iter_${tag}_uniform.json"
  candidate_json="${REPORT_DIR}/candidate_iter_${tag}_uniform.json"

  for checkpoint in "${control_checkpoint}" "${candidate_checkpoint}"; do
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing gate checkpoint: ${checkpoint}" >&2
      exit 2
    fi
  done

  if [[ ! -f "${control_json}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
      --config "${CONTROL_CONFIG}" \
      --checkpoint "${control_checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --top-k-values 4 \
      --iou-thresholds 0.5 0.75 \
      --quality-powers 0.5 \
      --score-thresholds 0.0 \
      --line-width 30.0 \
      --nms-distance-thresh-px 20.0 \
      --max-batches "${MAX_BATCHES}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --official-iou-workers "${OFFICIAL_IOU_WORKERS}" \
      --sample-strategy uniform \
      --exact-postprocess \
      --reuse-cache \
      --cache-dir "${CACHE_DIR}" \
      --output-json "${control_json}"
  else
    echo "reusing ${control_json}"
  fi

  if [[ ! -f "${candidate_json}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
      --config "${CANDIDATE_CONFIG}" \
      --checkpoint "${candidate_checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --top-k-values 4 \
      --iou-thresholds 0.5 0.75 \
      --quality-powers 0.5 \
      --score-thresholds 0.0 \
      --line-width 30.0 \
      --nms-distance-thresh-px 20.0 \
      --max-batches "${MAX_BATCHES}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --official-iou-workers "${OFFICIAL_IOU_WORKERS}" \
      --sample-strategy uniform \
      --exact-postprocess \
      --reuse-cache \
      --cache-dir "${CACHE_DIR}" \
      --output-json "${candidate_json}"
  else
    echo "reusing ${candidate_json}"
  fi

  pair_args+=(--pair "${iteration}" "${control_json}" "${candidate_json}")
done

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_unified_selection_gate \
  "${pair_args[@]}" \
  --output-json "${REPORT_DIR}/trajectory_summary.json" \
  --output-csv "${REPORT_DIR}/trajectory_summary.csv"
