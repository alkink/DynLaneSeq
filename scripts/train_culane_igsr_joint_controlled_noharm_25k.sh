#!/bin/bash
set -e

# Run this script from the project root:
# bash scripts/train_culane_igsr_joint_controlled_noharm_25k.sh

export PYTHONPATH="$(pwd):$PYTHONPATH"

CONFIG="dynlaneseq_eg/configs/culane_igsr_joint_controlled_noharm_25k.yaml"
# Start from S0 175k checkpoint
RESUME="outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt"

if [ ! -f "$RESUME" ]; then
    echo "Warning: RESUME checkpoint $RESUME not found. Please set correct path."
    # If the user has a different path for the S0 175k checkpoint, they should change this.
fi

echo "Starting No-Harm Joint Controlled Training from $RESUME..."

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=29500 \
    -m dynlaneseq_eg.tools.train \
    --config "$CONFIG" \
    --init-from "$RESUME"
