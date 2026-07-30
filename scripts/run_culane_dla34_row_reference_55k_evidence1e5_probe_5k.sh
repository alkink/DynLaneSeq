#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_probe_5k.yaml}"
SOURCE_OUT="${SOURCE_OUT:-outputs/diagnostics/dla34_rowref_from50k_evidence_lr2e5_cooldown_5k}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SOURCE_OUT}/iter_0055000.pt}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/dla34_rowref_from55k_evidence1e5_probe_5k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
START_ITERATION=55000
TARGET_ITERATION=60000
EVIDENCE_LR=0.00001
FINAL_CHECKPOINT="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Refusing unmatched effective batch size: $((BATCH_SIZE * GRAD_ACCUM)); expected 16." >&2
  exit 1
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing exact 55k source checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi
if [[ -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Evidence-LR probe already reached its target: ${FINAL_CHECKPOINT}"
  exit 0
fi

resume_checkpoint="${SOURCE_CHECKPOINT}"
if [[ -d "${OUT_DIR}" ]]; then
  candidate_resume="$(
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' \
      | sort -V \
      | tail -n 1
  )"
  if [[ -n "${candidate_resume}" && -f "${candidate_resume}" ]]; then
    resume_checkpoint="${candidate_resume}"
  fi
fi

checkpoint_audit="$(
  "${PYTHON}" -c '
import math
import sys
import torch

path = sys.argv[1]
payload = torch.load(path, map_location="cpu")
iteration = int(payload.get("iteration", -1))
groups = {
    str(group.get("name", "")): float(group["lr"])
    for group in payload.get("optimizer", {}).get("param_groups", [])
}
if not 55000 <= iteration < 60000:
    raise SystemExit(
        f"expected a 55k-57.5k checkpoint, got {path} at iteration {iteration}"
    )
expected = {
    "backbone_decay": 1e-6,
    "backbone_no_decay": 1e-6,
    "model_decay": 1e-5,
    "model_no_decay": 1e-5,
    "evidence_decay": 2e-6 if iteration == 55000 else 1e-5,
    "evidence_no_decay": 2e-6 if iteration == 55000 else 1e-5,
}
missing = [name for name in expected if name not in groups]
wrong = [
    (name, groups.get(name), value)
    for name, value in expected.items()
    if name in groups
    and not math.isclose(groups[name], value, rel_tol=2e-3, abs_tol=1e-12)
]
if missing or wrong:
    raise SystemExit(
        "optimizer LR audit failed: "
        + repr({"missing": missing, "wrong": wrong, "groups": groups})
    )
print(iteration)
print(", ".join(f"{name}={groups[name]:.3g}" for name in expected))
' "${resume_checkpoint}"
)"
resume_iteration="${checkpoint_audit%%$'\n'*}"
lr_audit="${checkpoint_audit#*$'\n'}"
run_iters="$((TARGET_ITERATION - resume_iteration))"

echo "Controlled evidence-LR probe: ${resume_iteration} -> ${TARGET_ITERATION}"
echo "Resume with exact AdamW moments: ${resume_checkpoint}"
echo "Stored LR audit: ${lr_audit}"
echo "Intervention: evidence_decay/evidence_no_decay -> ${EVIDENCE_LR}"
echo "Held fixed: model=1e-5, backbone=1e-6, constant scheduler"
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
  --resume "${resume_checkpoint}" \
  --resume-group-lr "evidence_decay=${EVIDENCE_LR}" \
  --resume-group-lr "evidence_no_decay=${EVIDENCE_LR}"
