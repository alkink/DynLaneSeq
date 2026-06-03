#!/usr/bin/env bash
set -eo pipefail

cd /home/alki/projects/DynLaneSeq
source /home/alki/miniconda3/etc/profile.d/conda.sh
conda activate clrernet

python -u -m dynlaneseq_eg.tools.train \
  --config dynlaneseq_eg/configs/culane_s0_res34_legacy_b16_90k.yaml \
  --device cuda
