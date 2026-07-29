#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/lanerownet_rowref_speed_matrix}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_lanerownet}"

OLD_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml"
ROWREF_CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep.yaml"

mkdir -p "${OUTPUT_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_case() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  set +e
  "${PYTHON}" -u scripts/profile_row_reference_training_step.py \
    --dataset-root "${DATA_ROOT}" \
    --warmup-steps 2 \
    --breakdown-steps 3 \
    --profiler-steps 0 \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/${name}.log"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "===== ${name}: exit ${status} ====="
  return 0
}

# Historical workload: eager FP16, batch 8 x accumulation 2, no row-reference
# decoder and no intermediate supervision.  This is the relevant same-GPU
# control for the previously observed ~31 img/s run.
run_case old_eager_b8x2 \
  --config "${OLD_CONFIG}" \
  --batch-size 8 \
  --grad-accum 2

# Current validated full objective and the measured safe 16-GiB setting.
run_case rowref_deepsup_compiled_b4x4 \
  --config "${ROWREF_CONFIG}" \
  --batch-size 4 \
  --grad-accum 4 \
  --compile-model \
  --compile-mode default

# Same architecture, objective, effective batch, and schedule; only the
# micro-batch/accumulation split changes.  If this fits, it is the highest
# priority throughput setting for the full run.
run_case rowref_deepsup_compiled_b8x2 \
  --config "${ROWREF_CONFIG}" \
  --batch-size 8 \
  --grad-accum 2 \
  --compile-model \
  --compile-mode default

# Diagnostic only: isolate the cost of auxiliary decoder-layer matching and
# losses.  This must not be used as the paper training config without a new
# accuracy experiment.
run_case rowref_no_deepsup_compiled_b4x4 \
  --config "${ROWREF_CONFIG}" \
  --batch-size 4 \
  --grad-accum 4 \
  --compile-model \
  --compile-mode default \
  --disable-intermediate-supervision

echo
echo "Logs: ${OUTPUT_DIR}"
