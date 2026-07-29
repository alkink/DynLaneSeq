#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:?Set PROBE_CHECKPOINT to the pretrained lane-shared probe}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/joint_lane_shared_coherence}"
STEPS="${STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

mkdir -p "$OUT_DIR"
CANDIDATE_CHECKPOINT="$OUT_DIR/iter_$((225000 + STEPS)).pt"
CANDIDATE_PROBE="$OUT_DIR/lane_shared_probe_$((225000 + STEPS)).pt"
TRAIN_JSON="$OUT_DIR/train_summary.json"
PAIR_JSON="$OUT_DIR/base_vs_candidate_uniform64.json"
AUX_JSON="$OUT_DIR/candidate_lane_shared_probe_uniform64.json"

python -u -m dynlaneseq_eg.tools.finetune_joint_lane_shared_coherence \
  --config "$CONFIG" \
  --checkpoint "$BASE_CHECKPOINT" \
  --probe-checkpoint "$PROBE_CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --model-learning-rate 2e-5 \
  --fpn-learning-rate 1e-5 \
  --probe-learning-rate 2e-4 \
  --coherence-weight 0.25 \
  --amp-dtype "$AMP_DTYPE" \
  --output-checkpoint "$CANDIDATE_CHECKPOINT" \
  --output-probe "$CANDIDATE_PROBE" \
  --output-json "$TRAIN_JSON"

BASE_CONFIG="$CONFIG" \
BASE_CHECKPOINT="$BASE_CHECKPOINT" \
CANDIDATE_CONFIG="$CONFIG" \
CANDIDATE_CHECKPOINT="$CANDIDATE_CHECKPOINT" \
DATA_ROOT="$DATA_ROOT" \
OUTPUT_JSON="$PAIR_JSON" \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS="$NUM_WORKERS" \
MAX_BATCHES=16 \
AMP_DTYPE=none \
bash scripts/analyze_culane_dla34_assignment_ablation_short.sh

python -u -m dynlaneseq_eg.tools.probe_query_conditioned_dense_curve \
  --config "$CONFIG" \
  --checkpoint "$CANDIDATE_CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --split val \
  --train-steps 0 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size 4 \
  --eval-max-batches 16 \
  --sample-strategy uniform \
  --num-workers "$NUM_WORKERS" \
  --hidden-dim 64 \
  --evidence-width 400 \
  --state-source final \
  --conditioning-mode lane_shared \
  --explicit-coordinates \
  --amp-dtype "$AMP_DTYPE" \
  --load-probe "$CANDIDATE_PROBE" \
  --output-json "$AUX_JSON"

echo "train summary: $TRAIN_JSON"
echo "main-output comparison: $PAIR_JSON"
echo "auxiliary probe comparison: $AUX_JSON"
