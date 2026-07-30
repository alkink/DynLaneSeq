#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_floor_to75k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/dla34_rowref_from50k_evidence_lr2e5_cooldown_5k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
START_ITERATION=55000
TARGET_ITERATION=75000
FINAL_CHECKPOINT="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Refusing unmatched effective batch size: $((BATCH_SIZE * GRAD_ACCUM)); expected 16." >&2
  exit 1
fi
if [[ -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Floor-LR extension already reached its target: ${FINAL_CHECKPOINT}"
  exit 0
fi
if [[ ! -d "${OUT_DIR}" ]]; then
  echo "Missing cooldown output directory: ${OUT_DIR}" >&2
  exit 1
fi

resume_checkpoint="$(
  find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' \
    | sort -V \
    | tail -n 1
)"
if [[ -z "${resume_checkpoint}" || ! -f "${resume_checkpoint}" ]]; then
  echo "No cooldown checkpoint found in ${OUT_DIR}." >&2
  exit 1
fi

resume_iteration="$(
  "${PYTHON}" -c \
    'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("iteration", -1)))' \
    "${resume_checkpoint}"
)"
if (( resume_iteration < START_ITERATION || resume_iteration >= TARGET_ITERATION )); then
  echo "Expected a 55k-70k cooldown checkpoint; got ${resume_checkpoint} at ${resume_iteration}." >&2
  exit 1
fi

# Loading a 55k checkpoint with a fresh cosine scheduler would raise the LR
# again.  Refuse to start unless the optimizer stored in the checkpoint has
# actually reached the predeclared floor values.
lr_audit="$(
  "${PYTHON}" -c '
import math
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
groups = {
    str(group.get("name", "")): float(group["lr"])
    for group in payload.get("optimizer", {}).get("param_groups", [])
}
expected = {
    "backbone_decay": 1e-6,
    "backbone_no_decay": 1e-6,
    "evidence_decay": 2e-6,
    "evidence_no_decay": 2e-6,
    "model_decay": 1e-5,
    "model_no_decay": 1e-5,
}
missing = [name for name in expected if name not in groups]
wrong = [
    (name, groups.get(name), value)
    for name, value in expected.items()
    if name in groups and not math.isclose(groups[name], value, rel_tol=2e-3, abs_tol=1e-12)
]
if missing or wrong:
    raise SystemExit(
        "optimizer LR audit failed: "
        + repr({"missing": missing, "wrong": wrong, "groups": groups})
    )
print(", ".join(f"{name}={groups[name]:.3g}" for name in expected))
' "${resume_checkpoint}"
)"

run_iters="$((TARGET_ITERATION - resume_iteration))"
echo "Floor-LR stability extension: ${resume_iteration} -> ${TARGET_ITERATION}"
echo "Resume with optimizer moments: ${resume_checkpoint}"
echo "Stored LR audit: ${lr_audit}"
echo "Scheduler: constant (no restart and no LR increase)"
echo "Output: ${OUT_DIR}"
echo "Batch/accum/effective: ${BATCH_SIZE}/${GRAD_ACCUM}/$((BATCH_SIZE * GRAD_ACCUM))"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --max-iters "${run_iters}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}" \
  --resume "${resume_checkpoint}"
