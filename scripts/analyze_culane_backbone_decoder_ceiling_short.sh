#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-dataset}"
MAX_BATCHES="${MAX_BATCHES:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP_DTYPE="${AMP_DTYPE:-none}"
OUT_ROOT="${OUT_ROOT:-outputs/decoder_ceiling_short}"

run_one() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Skipping missing checkpoint: ${checkpoint}" >&2
    return
  fi
  python -u -m dynlaneseq_eg.tools.analyze_decoder_layer_progression \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-batches "${MAX_BATCHES}" \
    --amp-dtype "${AMP_DTYPE}" \
    --iou-thresholds 0.5 0.7 \
    --output-json "${OUT_ROOT}/${name}.json"
}

R34_CONFIG="${R34_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
DLA_CONFIG="${DLA_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"

run_one \
  r34_iter025k \
  "${R34_CONFIG}" \
  "${R34_25K:-outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0025000.pt}"
run_one \
  r34_iter225k \
  "${R34_CONFIG}" \
  "${R34_225K:-outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
run_one \
  dla_iter025k \
  "${DLA_CONFIG}" \
  "${DLA_25K:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0025000.pt}"
run_one \
  dla_iter225k \
  "${DLA_CONFIG}" \
  "${DLA_225K:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
