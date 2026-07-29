#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CKPT="${CKPT:-/home/alki/projects/DynLaneSeq/outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-16}"
OUTPUT_JSON="${OUTPUT_JSON:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/seed_conditioned_p2_identity/dla34_225k_uniform64_1000.json}"
SAVE_PROBE="${SAVE_PROBE:-/home/alki/projects/DynLaneSeq/outputs/diagnostics/seed_conditioned_p2_identity/dla34_225k_uniform64_probe_1000.pt}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.probe_seed_conditioned_p2_identity \
  --config dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml \
  --checkpoint "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --eval-max-batches "${EVAL_MAX_BATCHES}" \
  --sample-strategy uniform \
  --hidden-dim 64 \
  --x-bins 200 \
  --amp-dtype bfloat16 \
  --log-interval 100 \
  --save-probe "${SAVE_PROBE}" \
  --output-json "${OUTPUT_JSON}"
