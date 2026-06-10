#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"

echo
echo "== S1 geometry evidence: S0 geometry -> S1 =="
bash scripts/run_debug_s1_residual_geometry_evidence_2k_init_giou.sh
if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
  bash scripts/eval_debug_s1_residual_geometry_evidence_2k_init_giou.sh
fi

echo
echo "== S2 geometry evidence: S1 geometry -> S2 =="
bash scripts/run_debug_s2_residual_geometry_evidence_2k_init_giou.sh
if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
  bash scripts/eval_debug_s2_residual_geometry_evidence_2k_init_giou.sh
fi

echo
echo "== S3 active corridor + qualitycal geometry evidence: S2 geometry -> S3 =="
bash scripts/run_debug_s3_active_corridor_qualitycal_geometry_evidence_from_s2geometry_2k.sh
if [[ "${EVAL_AFTER_TRAIN}" == "1" ]]; then
  bash scripts/eval_debug_s3_active_corridor_qualitycal_geometry_evidence_from_s2geometry_2k.sh
fi
