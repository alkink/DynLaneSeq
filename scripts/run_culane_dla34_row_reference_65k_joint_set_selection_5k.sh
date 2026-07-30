#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_joint_set_selection_5k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/dla34_rowref_from65k_joint_set_selection_5k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
TARGET_ITERATION=70000
FINAL_CHECKPOINT="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"

if [[ -z "${SOURCE_CHECKPOINT:-}" ]]; then
  for candidate in \
    outputs/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt \
    outputs/diagnostics/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt
  do
    if [[ -f "${candidate}" ]]; then
      SOURCE_CHECKPOINT="${candidate}"
      break
    fi
  done
fi

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Refusing unmatched effective batch size: $((BATCH_SIZE * GRAD_ACCUM)); expected 16." >&2
  exit 1
fi
if [[ -z "${SOURCE_CHECKPOINT:-}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing exact 65k source checkpoint. Set SOURCE_CHECKPOINT explicitly." >&2
  exit 1
fi
if [[ -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Joint set-selection diagnostic already reached 70k: ${FINAL_CHECKPOINT}"
  exit 0
fi
if find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to mix a partial run into ${OUT_DIR}." >&2
  echo "Use a new empty OUT_DIR or resume that checkpoint explicitly." >&2
  exit 1
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
if iteration != 65000:
    raise SystemExit(f"expected exact iteration 65000, got {iteration} in {path}")
expected = {
    "backbone_decay": 1e-6,
    "backbone_no_decay": 1e-6,
    "evidence_decay": 1e-5,
    "evidence_no_decay": 1e-5,
    "model_decay": 1e-5,
    "model_no_decay": 1e-5,
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
print(", ".join(f"{name}={groups[name]:.3g}" for name in expected))
' "${SOURCE_CHECKPOINT}"
)"

echo "Joint set-selection diagnostic: 65000 -> ${TARGET_ITERATION}"
echo "Source checkpoint: ${SOURCE_CHECKPOINT}"
echo "Stored LR audit: ${checkpoint_audit}"
echo "Old AdamW moments: preserved; new selector moments: initialized empty"
echo "Old parameter LRs/schedule: matched to the 65k->70k cooldown control"
echo "New selector LR: 1e-4; zero-residual initialization preserves initial ranking"
echo "Output: ${OUT_DIR}"
echo "Batch/accum/effective: ${BATCH_SIZE}/${GRAD_ACCUM}/$((BATCH_SIZE * GRAD_ACCUM))"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: audit passed; training was not started."
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --max-iters 5000 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}" \
  --resume "${SOURCE_CHECKPOINT}" \
  --resume-remap-optimizer-groups
