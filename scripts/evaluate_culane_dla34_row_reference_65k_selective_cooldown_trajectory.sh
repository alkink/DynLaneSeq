#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_to75k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml}"
CANDIDATE_OUT="${CANDIDATE_OUT:-outputs/diagnostics/dla34_rowref_from65k_selective_cooldown_10k}"
RESULT_DIR="${RESULT_DIR:-outputs/diagnostics/row_reference_selective_cooldown/trajectory_uniform256}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-70000 75000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

if [[ -z "${CONTROL_CHECKPOINT:-}" ]]; then
  for candidate in \
    outputs/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt \
    outputs/diagnostics/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt
  do
    if [[ -f "${candidate}" ]]; then
      CONTROL_CHECKPOINT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CONTROL_CHECKPOINT:-}" || ! -f "${CONTROL_CHECKPOINT}" ]]; then
  echo "Missing fixed 65k control checkpoint. Set CONTROL_CHECKPOINT." >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"
result_files=()
for iteration in ${CHECKPOINT_ITERS}; do
  checkpoint_tag="$(printf '%07d' "${iteration}")"
  candidate_checkpoint="${CANDIDATE_OUT}/iter_${checkpoint_tag}.pt"
  output_json="${RESULT_DIR}/gate_iter_${checkpoint_tag}.json"

  if [[ ! -f "${candidate_checkpoint}" ]]; then
    echo "Missing selective-cooldown checkpoint: ${candidate_checkpoint}" >&2
    exit 1
  fi

  if [[ -f "${output_json}" && "${FORCE:-0}" != "1" ]]; then
    echo "Using existing result: ${output_json}"
  else
    "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_row_reference_gate \
      --control-config "${CONTROL_CONFIG}" \
      --control-checkpoint "${CONTROL_CHECKPOINT}" \
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
  --allow-fixed-control \
  --output-json "${RESULT_DIR}/trajectory_summary.json" \
  --output-csv "${RESULT_DIR}/trajectory_summary.csv"

echo "Selective-cooldown trajectory: ${RESULT_DIR}/trajectory_summary.json"
