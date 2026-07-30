#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-52500 55000 60000 65000 70000 75000}"
export CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_floor_to75k.yaml}"

exec bash scripts/evaluate_culane_dla34_row_reference_50k_cooldown_trajectory.sh
