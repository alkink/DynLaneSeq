#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
V34_ROOT="${V34_ROOT:-${PROJECT_ROOT}/outputs/diagnostics/v34_temporal_candidate_observability}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/diagnostics/v35_raw_rgb_temporal_observability}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

exec "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.audit_v35_raw_rgb_temporal_observability \
  --target-cache "${V34_ROOT}/target_main_official_iou.pt" \
  --union-cache "${V34_ROOT}/cache/iter_0225000_6ffdfced3d820220.pt" \
  --manifest "${V34_ROOT}/temporal_manifest.json" \
  --v34-report "${V34_ROOT}/v34_temporal_observability.json" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --row-samples 64 \
  --ribbon-samples 25 \
  --fine-radius 32 \
  --coarse-radius 160 \
  --flow-scale 0.5 \
  --fb-threshold 3.0 \
  --steps 2000 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --quality-weight 0.25 \
  --log-interval 100 \
  --bootstrap-reps 2000 \
  --seeds 3407 5741 9011 \
  --reuse-ribbon-cache \
  --reuse-training
