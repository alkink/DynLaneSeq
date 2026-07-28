#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/gt_curve_feature_probe}"
# Three hundred steps were sufficient to separate C2 from P2 on both audited
# checkpoints; doubling this only lengthens the diagnostic without changing
# the question it answers.
TRAIN_STEPS="${TRAIN_STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# Thirty-two batches of two audit the same fixed 64 validation images while
# keeping curve-profile sampling latency lower on the tested RTX 3090.
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
BACKBONES="${BACKBONES:-r34 dla34}"

mkdir -p "${OUTPUT_ROOT}"

run_probe() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"

  "${PYTHON}" -u -m dynlaneseq_eg.tools.probe_gt_curve_aligned_features \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --train-steps "${TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --eval-max-batches "${EVAL_MAX_BATCHES}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --amp-dtype "${AMP_DTYPE}" \
    --save-probes "${OUTPUT_ROOT}/${name}_probes.pt" \
    --output-json "${OUTPUT_ROOT}/${name}.json"
}

for backbone in ${BACKBONES}; do
  case "${backbone}" in
    r34)
      run_probe \
        r34_225k \
        dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml \
        "${CHECKPOINT_ROOT}/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt"
      ;;
    dla34)
      run_probe \
        dla34_225k \
        dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml \
        "${CHECKPOINT_ROOT}/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt"
      ;;
    *)
      echo "Unsupported BACKBONES entry: ${backbone}" >&2
      exit 2
      ;;
  esac
done
