#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CHECKPOINT="${CHECKPOINT:-/home/alki/projects/DynLaneSeq/outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/gt_curve_feature_probe/dla34_225k_probes.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/noisy_reference_counterfactual/dla34_225k_uniform64.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
EVAL_SHIFT_PX="${EVAL_SHIFT_PX:-32}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_noisy_reference_counterfactual \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --probe-checkpoint "${PROBE_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-max-batches "${EVAL_MAX_BATCHES}" \
  --eval-sample-strategy uniform \
  --eval-shift-px "${EVAL_SHIFT_PX}" \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}" \
  --output-json "${OUTPUT_JSON}"
