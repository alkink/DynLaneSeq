#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-dynlaneseq_eg/configs/tusimple_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml}"
RESUME="${RESUME:-outputs/tusimple_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/last.pt}"
export CONFIG RESUME
exec bash "$(dirname "$0")/resume_tusimple_s0_structured_query_backbone_to70ep.sh"
