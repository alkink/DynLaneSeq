#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
export TAG="${TAG:-stage_chain_official_val}"
exec bash scripts/run_stage_chain_diagnostics_2k.sh
