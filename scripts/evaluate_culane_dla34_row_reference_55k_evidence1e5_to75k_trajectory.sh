#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_floor_to75k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_to75k.yaml}"
CONTROL_OUT="${CONTROL_OUT:-outputs/diagnostics/dla34_rowref_from50k_evidence_lr2e5_cooldown_5k}"
CANDIDATE_OUT="${CANDIDATE_OUT:-outputs/diagnostics/dla34_rowref_from55k_evidence1e5_to75k}"
RESULT_DIR="${RESULT_DIR:-outputs/diagnostics/row_reference_evidence_lr_probe/trajectory_uniform256}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-60000 65000 70000 75000}"
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

  for checkpoint in "${control_checkpoint}" "${candidate_checkpoint}"; do
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing trajectory checkpoint: ${checkpoint}" >&2
      exit 1
    fi
  done

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

echo "Matched 1e-5 trajectory: ${RESULT_DIR}/trajectory_summary.json"
