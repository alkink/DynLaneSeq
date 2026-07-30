#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_to75k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
QUALITY_FOCAL_BETA="${QUALITY_FOCAL_BETA:-2.0}"
TARGET_MODE="${TARGET_MODE:-all_proposal}"
RANK_LOSS_WEIGHT="${RANK_LOSS_WEIGHT:-0.25}"
RANK_TARGET_MARGIN="${RANK_TARGET_MARGIN:-0.10}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
MIN_GAIN_050_POINTS="${MIN_GAIN_050_POINTS:-1.0}"
MIN_GAIN_070_POINTS="${MIN_GAIN_070_POINTS:-0.5}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/row_reference_quality_rescoring_65k_uniform256.json}"
SAVE_PROBE="${SAVE_PROBE:-outputs/diagnostics/row_reference_quality_rescoring_65k.pt}"

for path in "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

python -u -m dynlaneseq_eg.tools.probe_row_reference_quality_rescoring \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --train-steps "$TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-max-batches "$EVAL_MAX_BATCHES" \
  --num-workers "$NUM_WORKERS" \
  --sample-strategy uniform \
  --amp-dtype "$AMP_DTYPE" \
  --learning-rate "$LEARNING_RATE" \
  --quality-focal-beta "$QUALITY_FOCAL_BETA" \
  --target-mode "$TARGET_MODE" \
  --rank-loss-weight "$RANK_LOSS_WEIGHT" \
  --rank-target-margin "$RANK_TARGET_MARGIN" \
  --line-width 30 \
  --top-k 4 \
  --quality-power 0.5 \
  --seed 3407 \
  --log-interval "$LOG_INTERVAL" \
  --min-gain-050-points "$MIN_GAIN_050_POINTS" \
  --min-gain-070-points "$MIN_GAIN_070_POINTS" \
  --output-json "$OUTPUT_JSON" \
  --save-probe "$SAVE_PROBE"

echo
echo "quality rescoring report: $OUTPUT_JSON"
echo "probe weights: $SAVE_PROBE"
