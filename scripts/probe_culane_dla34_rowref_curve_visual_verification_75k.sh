#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_object0p5_matcher_10k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_10k/iter_0075000.pt}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_IMAGES="${TRAIN_IMAGES:-4096}"
VAL_IMAGES="${VAL_IMAGES:-256}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/dla34_rowref_object0p5_curve_visual_verification_75k_uniform256.json}"
SAVE_PROBE="${SAVE_PROBE:-outputs/diagnostics/dla34_rowref_object0p5_curve_visual_verification_75k.pt}"

for path in "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

python -u -m dynlaneseq_eg.tools.probe_curve_aligned_visual_verification \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --train-images "$TRAIN_IMAGES" \
  --val-images "$VAL_IMAGES" \
  --train-steps "$TRAIN_STEPS" \
  --num-workers "$NUM_WORKERS" \
  --amp-dtype "$AMP_DTYPE" \
  --curve-samples 20 \
  --offsets-px -48 -24 -12 0 12 24 48 \
  --visual-dim 128 \
  --hidden-dim 256 \
  --row-layers 1 \
  --row-heads 4 \
  --set-layers 2 \
  --set-heads 8 \
  --set-ff-dim 512 \
  --dropout 0.1 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --quality-focal-beta 2.0 \
  --rank-loss-weight 0.25 \
  --rank-target-margin 0.10 \
  --line-width 30 \
  --min-valid-rows 5 \
  --row-visibility-thresh 0.0 \
  --range-temperature 0.03 \
  --top-k 4 \
  --quality-power 0.25 \
  --nms-distance 20 \
  --nms-min-overlap-points 5 \
  --seed 3407 \
  --log-interval 50 \
  --min-gain-050-points 1.0 \
  --min-gain-070-points 0.5 \
  --min-visual-over-control-points 0.25 \
  --output-json "$OUTPUT_JSON" \
  --save-probe "$SAVE_PROBE"

echo
echo "curve-aligned visual-verification report: $OUTPUT_JSON"
echo "probe weights: $SAVE_PROBE"
