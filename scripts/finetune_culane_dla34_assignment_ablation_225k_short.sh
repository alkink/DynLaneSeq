#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:?Set CONFIG to one assignment-ablation YAML file}"
RESUME="${RESUME:-outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for this causal arm}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
EXTRA_ITERS="${EXTRA_ITERS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$RESUME" ]]; then
  echo "Missing resume checkpoint: $RESUME" >&2
  exit 1
fi

START_ITER="$(python -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("iteration", 0)))' "$RESUME")"
TARGET_ITER=$((START_ITER + EXTRA_ITERS))

echo "diagnostic: DLA-34 assignment/inter-query causal continuation"
echo "config: $CONFIG"
echo "resume: $RESUME"
echo "start iteration: $START_ITER"
echo "target iteration: $TARGET_ITER"
echo "output: $OUT_DIR"

python -u -m dynlaneseq_eg.tools.train \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --dataset-root "$DATA_ROOT" \
  --resume "$RESUME" \
  --max-iters "$EXTRA_ITERS" \
  --output-dir "$OUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --seg-aux-amp-dtype "$SEG_AUX_AMP_DTYPE"
