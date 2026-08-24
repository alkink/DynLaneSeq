#!/usr/bin/env bash
set -euo pipefail

project=${DYNLANESEQ_PROJECT:-/workspace/DynLaneSeq}
dataset=${CULANE_ROOT:-/workspace/CULane}
python=${DYNLANESEQ_PYTHON:-/venv/clrernet/bin/python}
checkpoint=${V34_V7_CHECKPOINT:-${project}/outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}
root=${V34_OUTPUT_DIR:-${project}/outputs/diagnostics/v34_temporal_candidate_observability}

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${checkpoint}" ]]; then
  echo "missing V7 checkpoint: ${checkpoint}" >&2
  exit 2
fi

mkdir -p "${root}"
"${python}" -m dynlaneseq_eg.tools.audit_v34_temporal_candidate_observability \
  --config dynlaneseq_eg/configs/culane_v34_temporal_observability_v7_225k.yaml \
  --checkpoint "${checkpoint}" \
  --dataset-root "${dataset}" \
  --val-list list/val.txt \
  --output-dir "${root}" \
  --device cuda \
  --sample-per-fold 512 \
  --eval-batch-size 8 \
  --num-workers 6 \
  --metric-workers 20 \
  --flow-workers 8 \
  --flow-scale 0.5 \
  --fb-threshold 3.0 \
  --min-warp-rows 20 \
  --bootstrap-reps 2000 \
  --amp-dtype bf16 \
  --reuse-cache

