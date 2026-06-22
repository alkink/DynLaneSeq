#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
export TAG="${TAG:-$(basename "$(dirname "${CKPT:?Set CKPT to the checkpoint}")")_$(basename "$CKPT" .pt)_official_val}"
exec bash scripts/run_candidate_diagnostics_2k.sh
