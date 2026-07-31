#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_object0p5_matcher_10k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_10k/iter_0075000.pt}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/dla34_joint_score_recovery_75k_uniform256.json}"
SAVE_PROBE="${SAVE_PROBE:-outputs/diagnostics/dla34_joint_score_recovery_75k.pt}"

for path in "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

python -u -m dynlaneseq_eg.tools.probe_row_reference_joint_score_recovery \
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
  --weight-decay 0.0001 \
  --balanced-rank-weight 0.25 \
  --rank-target-margin 0.10 \
  --line-width 30 \
  --top-k 4 \
  --quality-power 0.25 \
  --iou-thresholds 0.50 0.75 \
  --seed 3407 \
  --log-interval 50 \
  --output-json "$OUTPUT_JSON" \
  --save-probe "$SAVE_PROBE"

echo
echo "joint score recovery report: $OUTPUT_JSON"
echo "probe weights: $SAVE_PROBE"
