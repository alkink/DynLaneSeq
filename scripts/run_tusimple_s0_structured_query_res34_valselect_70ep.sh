#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-dynlaneseq_eg/configs/tusimple_s0_structured_query_res34_valselect_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml}" \
  exec bash "$(dirname "$0")/run_tusimple_s0_structured_query_backbone_70ep.sh"
