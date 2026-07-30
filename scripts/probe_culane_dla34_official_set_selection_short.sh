#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_to75k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/diagnostics/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
TRAIN_CACHE_IMAGES="${TRAIN_CACHE_IMAGES:-4096}"
VAL_CACHE_IMAGES="${VAL_CACHE_IMAGES:-256}"
TRAIN_CACHE="${TRAIN_CACHE:-outputs/diagnostics/cache/dla34_rowref_65k_official_selection_train4096.pt}"
VAL_CACHE="${VAL_CACHE:-outputs/diagnostics/cache/dla34_rowref_65k_official_selection_val256.pt}"
REUSE_CACHE="${REUSE_CACHE:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
CURVE_SAMPLES="${CURVE_SAMPLES:-20}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/row_reference_official_set_selection_65k_uniform256.json}"
SAVE_PROBE="${SAVE_PROBE:-outputs/diagnostics/row_reference_official_set_selection_65k.pt}"

for path in "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

reuse_args=()
if [[ "$REUSE_CACHE" == "1" ]]; then
  reuse_args+=(--reuse-cache)
fi

python -u -m dynlaneseq_eg.tools.probe_official_set_selection \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --cache-batch-size "$CACHE_BATCH_SIZE" \
  --train-cache-images "$TRAIN_CACHE_IMAGES" \
  --val-cache-images "$VAL_CACHE_IMAGES" \
  --train-cache "$TRAIN_CACHE" \
  --val-cache "$VAL_CACHE" \
  "${reuse_args[@]}" \
  --num-workers "$NUM_WORKERS" \
  --amp-dtype "$AMP_DTYPE" \
  --curve-samples "$CURVE_SAMPLES" \
  --line-width 30 \
  --min-valid-rows 5 \
  --row-visibility-thresh 0.0 \
  --train-steps "$TRAIN_STEPS" \
  --probe-batch-size "$PROBE_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay 0.0001 \
  --quality-focal-beta 2.0 \
  --rank-loss-weight 0.25 \
  --rank-target-margin 0.10 \
  --hidden-dim 256 \
  --set-layers 2 \
  --set-heads 8 \
  --set-ff-dim 512 \
  --set-dropout 0.1 \
  --top-k 4 \
  --quality-power 0.5 \
  --nms-distance 20 \
  --nms-min-overlap-points 5 \
  --seed 3407 \
  --log-interval 50 \
  --min-gain-050-points 1.0 \
  --min-gain-070-points 0.5 \
  --output-json "$OUTPUT_JSON" \
  --save-probe "$SAVE_PROBE"

echo
echo "official scalar/set selection report: $OUTPUT_JSON"
echo "probe weights: $SAVE_PROBE"
