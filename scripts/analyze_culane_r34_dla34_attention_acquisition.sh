#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/attention_acquisition}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_BATCHES="${MAX_BATCHES:-32}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
CORRIDOR_RADIUS_PX="${CORRIDOR_RADIUS_PX:-16}"
TARGET_LAYER="${TARGET_LAYER:-3}"
BIAS_STRENGTHS="${BIAS_STRENGTHS:-0.5 1.0 2.0 4.0}"
BIAS_MODES="${BIAS_MODES:-single cascade}"
BACKBONES="${BACKBONES:-r34 dla34}"

mkdir -p "${OUTPUT_ROOT}"

run_audit() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"

  # shellcheck disable=SC2086
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_attention_acquisition \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-batches "${MAX_BATCHES}" \
    --amp-dtype "${AMP_DTYPE}" \
    --corridor-radius-px "${CORRIDOR_RADIUS_PX}" \
    --target-layer "${TARGET_LAYER}" \
    --bias-strengths ${BIAS_STRENGTHS} \
    --bias-modes ${BIAS_MODES} \
    --output-json "${OUTPUT_ROOT}/${name}.json"
}

for backbone in ${BACKBONES}; do
  case "${backbone}" in
    r34)
      run_audit \
        r34_225k \
        dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml \
        "${CHECKPOINT_ROOT}/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt"
      ;;
    dla34)
      run_audit \
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
