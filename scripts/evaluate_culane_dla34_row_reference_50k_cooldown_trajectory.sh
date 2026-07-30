#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from50k_cooldown_5k.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep/iter_0050000.pt}"
CANDIDATE_OUT="${CANDIDATE_OUT:-outputs/diagnostics/dla34_rowref_from50k_evidence_lr2e5_cooldown_5k}"
RESULT_DIR="${RESULT_DIR:-outputs/diagnostics/row_reference_cooldown/from50k_uniform64}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-52500 55000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

if [[ ! -f "${CONTROL_CHECKPOINT}" ]]; then
  echo "Missing 50k control checkpoint: ${CONTROL_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"
result_files=()
for iteration in ${CHECKPOINT_ITERS}; do
  checkpoint_tag="$(printf '%07d' "${iteration}")"
  candidate_checkpoint="${CANDIDATE_OUT}/iter_${checkpoint_tag}.pt"
  output_json="${RESULT_DIR}/gate_iter_${checkpoint_tag}.json"
  if [[ ! -f "${candidate_checkpoint}" ]]; then
    echo "Missing cooldown checkpoint: ${candidate_checkpoint}" >&2
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
  --output-json "${RESULT_DIR}/trajectory_summary.json" \
  --output-csv "${RESULT_DIR}/trajectory_summary.csv"

echo "Cooldown trajectory: ${RESULT_DIR}/trajectory_summary.json"
