#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/alki/projects/DynLaneSeq/outputs/diagnostics}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_BATCHES="${MAX_BATCHES:-20}"
MAX_PROBE_ROWS="${MAX_PROBE_ROWS:-12000}"
PROBE_STEPS="${PROBE_STEPS:-160}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
SEED="${SEED:-3407}"

R34_CONFIG="${R34_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
R34_CKPT="${R34_CKPT:-${CHECKPOINT_ROOT}/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
DLA_CONFIG="${DLA_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
DLA_CKPT="${DLA_CKPT:-${CHECKPOINT_ROOT}/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
R34_OUTPUT="${R34_OUTPUT:-${OUTPUT_ROOT}/r34_225k_frozen_residual_probe.json}"
DLA_OUTPUT="${DLA_OUTPUT:-${OUTPUT_ROOT}/dla34_225k_frozen_residual_probe.json}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-${OUTPUT_ROOT}/r34_vs_dla34_frozen_residual_probe.json}"

for path in "${R34_CONFIG}" "${R34_CKPT}" "${DLA_CONFIG}" "${DLA_CKPT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing CULane root: ${DATA_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

run_probe() {
  local config="$1"
  local checkpoint="$2"
  local output_json="$3"
  python -u -m dynlaneseq_eg.tools.analyze_frozen_residual_offset_probe \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --data-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-batches "${MAX_BATCHES}" \
    --max-probe-rows "${MAX_PROBE_ROWS}" \
    --probe-steps "${PROBE_STEPS}" \
    --probe-batch-size "${PROBE_BATCH_SIZE}" \
    --amp-dtype "${AMP_DTYPE}" \
    --anchor-layers L2 L3 \
    --offsets-px -32 -16 -8 -4 0 4 8 16 32 \
    --group-index 0 \
    --common-channels 256 \
    --seed "${SEED}" \
    --output-json "${output_json}"
}

run_probe "${R34_CONFIG}" "${R34_CKPT}" "${R34_OUTPUT}"
run_probe "${DLA_CONFIG}" "${DLA_CKPT}" "${DLA_OUTPUT}"

python -m dynlaneseq_eg.tools.compare_frozen_residual_offset_probes \
  "${R34_OUTPUT}" \
  "${DLA_OUTPUT}" \
  --output-json "${COMBINED_OUTPUT}"
