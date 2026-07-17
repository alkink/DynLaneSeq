#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-dynlaneseq_eg/configs/tusimple_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml}"
CKPT="${CKPT:-outputs/tusimple_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0015890.pt}"

# Shared, frozen protocol from the structured ResNet-34 TuSimple run.
# Do not tune these values separately on the unstructured test predictions.
SCORE_THRESH="${SCORE_THRESH:-0.20}"
QUALITY_POWER="${QUALITY_POWER:-0.25}"

export CONFIG CKPT SCORE_THRESH QUALITY_POWER
exec bash "$(dirname "$0")/eval_tusimple_s0_structured_query_backbone_test.sh"
