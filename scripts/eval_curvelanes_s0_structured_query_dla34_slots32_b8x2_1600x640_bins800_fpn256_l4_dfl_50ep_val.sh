#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="dynlaneseq_eg/configs/curvelanes_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml" \
  exec bash scripts/eval_curvelanes_s0_structured_query_backbone_val.sh
