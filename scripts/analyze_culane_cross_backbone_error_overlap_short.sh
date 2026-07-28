#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_BATCHES="${MAX_BATCHES:-32}"
AMP_DTYPE="${AMP_DTYPE:-none}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/culane_cross_backbone_error_overlap_r34_dla34_225k.json}"

python -m dynlaneseq_eg.tools.analyze_cross_backbone_error_overlap \
  --r34-config dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml \
  --r34-checkpoint /home/alki/projects/DynLaneSeq/outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt \
  --dla34-config dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml \
  --dla34-checkpoint /home/alki/projects/DynLaneSeq/outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --sample-strategy "${SAMPLE_STRATEGY}" \
  --line-width 30.0 \
  --iou-thresholds 0.5 0.7 \
  --amp-dtype "${AMP_DTYPE}" \
  --output-json "${OUTPUT_JSON}"
