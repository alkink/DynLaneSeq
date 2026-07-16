#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-dynlaneseq_eg/configs/tusimple_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml}"
CKPT="${CKPT:-outputs/tusimple_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0015890.pt}"
export CONFIG CKPT
exec bash "$(dirname "$0")/eval_tusimple_s0_structured_query_backbone_test.sh"
