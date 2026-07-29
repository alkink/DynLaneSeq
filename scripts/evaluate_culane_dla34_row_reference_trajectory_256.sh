#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_reference_gate_control_10k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_row_reference_gate_10k.yaml}"
CONTROL_OUT="${CONTROL_OUT:-outputs/diagnostics/dla34_reference_gate_control_10k}"
CANDIDATE_OUT="${CANDIDATE_OUT:-outputs/diagnostics/dla34_row_reference_gate_10k}"
RESULT_DIR="${RESULT_DIR:-outputs/diagnostics/row_reference_gate/trajectory_uniform256}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-2500 5000 7500 10000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

mkdir -p "${RESULT_DIR}"
result_files=()

for iteration in ${CHECKPOINT_ITERS}; do
  checkpoint_tag="$(printf '%07d' "${iteration}")"
  control_checkpoint="${CONTROL_OUT}/iter_${checkpoint_tag}.pt"
  candidate_checkpoint="${CANDIDATE_OUT}/iter_${checkpoint_tag}.pt"
  output_json="${RESULT_DIR}/gate_iter_${checkpoint_tag}.json"

  if [[ ! -f "${control_checkpoint}" ]]; then
    echo "Missing control checkpoint: ${control_checkpoint}" >&2
    exit 1
  fi
  if [[ ! -f "${candidate_checkpoint}" ]]; then
    echo "Missing candidate checkpoint: ${candidate_checkpoint}" >&2
    exit 1
  fi

  if [[ -f "${output_json}" && "${FORCE:-0}" != "1" ]]; then
    echo "Using existing result: ${output_json}"
  else
    "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_row_reference_gate \
      --control-config "${CONTROL_CONFIG}" \
      --control-checkpoint "${control_checkpoint}" \
      --candidate-config "${CANDIDATE_CONFIG}" \
      --candidate-checkpoint "${candidate_checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --max-batches "${MAX_BATCHES}" \
      --sample-strategy uniform \
      --line-width 30.0 \
      --top-k 4 \
      --amp-dtype "${AMP_DTYPE}" \
      --min-recovered-lanes 5 \
      --max-lost-lanes 2 \
      --min-image-specificity-points 5.0 \
      --output-json "${output_json}"
  fi
  result_files+=("${output_json}")
done

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_row_reference_gate_trajectory \
  --inputs "${result_files[@]}" \
  --output-json "${RESULT_DIR}/trajectory_summary.json" \
  --output-csv "${RESULT_DIR}/trajectory_summary.csv"
