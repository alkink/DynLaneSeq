#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${DEVICE:=cuda}"

DEVICE="${DEVICE}" bash scripts/run_culane_s0_structured_query_res34_b16.sh
DEVICE="${DEVICE}" bash scripts/run_culane_s1_residual_structured_query_from_s0.sh
DEVICE="${DEVICE}" bash scripts/run_culane_s2_residual_structured_query_from_s1.sh
DEVICE="${DEVICE}" bash scripts/run_culane_s3_active_corridor_qualitycal_structured_query_from_s2.sh
