#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from65k_object0p5_matcher_10k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_10k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
AUTO_RESUME="${AUTO_RESUME:-1}"
START_BOUNDARY=65000
SOURCE_ITERATION=70000
TARGET_ITERATION=75000
FINAL_CHECKPOINT="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"

if [[ -z "${SOURCE_CHECKPOINT:-}" ]]; then
  for candidate in \
    outputs/diagnostics/dla34_rowref_from65k_object0p5_matcher_5k/iter_0070000.pt \
    outputs/dla34_rowref_from65k_object0p5_matcher_5k/iter_0070000.pt
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
  echo "Missing exact 70k lambda_obj=0.5 source checkpoint." >&2
  echo "Set SOURCE_CHECKPOINT explicitly." >&2
  exit 1
fi
if [[ -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Matched lambda_obj=0.5 continuation already reached 75k: ${FINAL_CHECKPOINT}"
  exit 0
fi

config_audit="$(
  "${PYTHON}" -c '
import math
import sys
from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
matcher = cfg["matcher"]
expected_matcher = {
    "lambda_obj": 0.5,
    "lambda_point": 5.0,
    "lambda_range": 1.0,
    "lambda_line_iou": 1.0,
    "line_iou_radius": 15.0,
}
wrong_matcher = {
    key: matcher.get(key)
    for key, expected in expected_matcher.items()
    if not math.isclose(
        float(matcher.get(key, float("nan"))),
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
}
if wrong_matcher:
    raise SystemExit(f"matcher audit failed: {wrong_matcher}")
if matcher.get("object_cost_type") != "neg_probability":
    raise SystemExit(
        "expected matcher.object_cost_type=neg_probability, got "
        + repr(matcher.get("object_cost_type"))
    )
scheduler = cfg["scheduler"]
expected_scheduler = {
    "name": "cosine",
    "total_iters": 10000,
    "warmup_iters": 0,
    "min_lr_ratio": 0.2,
}
if scheduler != expected_scheduler:
    raise SystemExit(
        f"scheduler audit failed: {scheduler!r}; "
        f"expected {expected_scheduler!r}"
    )
training = cfg["training"]
if int(training["seed"]) != 3407:
    raise SystemExit(f"expected seed 3407, got {training['"'"'seed'"'"']}")
if int(training["max_iters"]) != 10000:
    raise SystemExit(
        f"expected a complete 10k matched schedule, got "
        f"{training['"'"'max_iters'"'"']}"
    )
if cfg["model"]["structured_query"].get("set_selection", {}).get("enabled", False):
    raise SystemExit("separate set-selection head must be disabled")
if float(cfg["loss"].get("w_set_selection", 0.0)) != 0.0:
    raise SystemExit("set-selection loss must be disabled")
print(
    "lambda_obj=0.5; geometry costs=5/1/1; radius=15; "
    "cosine=10k,min_ratio=0.2; seed=3407"
)
' "${CONFIG}"
)"

checkpoint_iteration() {
  "${PYTHON}" -c '
import math
import sys
import torch
from dynlaneseq_eg.config import load_config

path = sys.argv[1]
minimum = int(sys.argv[2])
maximum = int(sys.argv[3])
start_boundary = int(sys.argv[4])
target_cfg = load_config(sys.argv[5])
payload = torch.load(path, map_location="cpu")
iteration = int(payload.get("iteration", -1))
if not minimum <= iteration <= maximum:
    raise SystemExit(
        f"checkpoint iteration must be in [{minimum}, {maximum}], "
        f"got {iteration} in {path}"
    )
cfg = payload.get("cfg")
if not isinstance(cfg, dict):
    raise SystemExit("checkpoint is missing its expanded config")
for key in (
    "model",
    "loss",
    "augmentation",
    "dataloader",
    "optimizer",
    "scheduler",
):
    if cfg.get(key) != target_cfg.get(key):
        raise SystemExit(
            f"checkpoint/target config mismatch in {key}; "
            "refusing an unmatched continuation"
        )
# ``--dataset-root`` is a machine-local path override written into every
# checkpoint. It is not an experimental setting, so compare the remaining
# dataset protocol fields while deliberately ignoring only ``root``.
checkpoint_dataset = dict(cfg.get("dataset", {}))
target_dataset = dict(target_cfg.get("dataset", {}))
checkpoint_dataset.pop("root", None)
target_dataset.pop("root", None)
if checkpoint_dataset != target_dataset:
    raise SystemExit(
        "checkpoint/target config mismatch in dataset protocol; "
        "refusing an unmatched continuation"
    )
matcher = cfg.get("matcher", {})
target_matcher = target_cfg["matcher"]
if matcher != target_matcher or not math.isclose(
    float(matcher.get("lambda_obj", float("nan"))),
    0.5,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(
        "checkpoint is not the lambda_obj=0.5 candidate: "
        + repr(matcher.get("lambda_obj"))
    )
optimizer = payload.get("optimizer")
scheduler = payload.get("scheduler")
if not isinstance(optimizer, dict) or not optimizer.get("param_groups"):
    raise SystemExit("checkpoint is missing optimizer parameter groups")
if not isinstance(scheduler, dict):
    raise SystemExit("checkpoint is missing scheduler state")
expected_step = iteration - start_boundary
actual_step = int(scheduler.get("last_epoch", -10**9))
if abs(actual_step - expected_step) > 1:
    raise SystemExit(
        f"scheduler phase mismatch: iteration={iteration}, "
        f"last_epoch={actual_step}, expected approximately {expected_step}"
    )
progress = min(max(actual_step / 10000.0, 0.0), 1.0)
expected_factor = 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))
bad_lrs = []
for group in optimizer["param_groups"]:
    name = str(group.get("name", "<unnamed>"))
    lr = float(group["lr"])
    initial_lr = float(group.get("initial_lr", float("nan")))
    if not math.isfinite(initial_lr):
        bad_lrs.append((name, lr, "missing initial_lr"))
    elif initial_lr == 0.0:
        if lr != 0.0:
            bad_lrs.append((name, lr, 0.0))
    elif not math.isclose(
        lr / initial_lr,
        expected_factor,
        rel_tol=3e-3,
        abs_tol=1e-8,
    ):
        bad_lrs.append((name, lr / initial_lr, expected_factor))
if bad_lrs:
    raise SystemExit(
        "optimizer LR/scheduler phase audit failed: " + repr(bad_lrs)
    )
print(iteration)
' "$1" "$2" "$3" "${START_BOUNDARY}" "${CONFIG}"
}

source_iteration="$(checkpoint_iteration \
  "${SOURCE_CHECKPOINT}" \
  "${SOURCE_ITERATION}" \
  "${SOURCE_ITERATION}")"

RUN_CHECKPOINT="${SOURCE_CHECKPOINT}"
RUN_START="${source_iteration}"
RESUME_DESCRIPTION="exact 70k candidate; AdamW moments and scheduler phase restored"

LATEST_PARTIAL=""
if [[ -d "${OUT_DIR}" ]]; then
  LATEST_PARTIAL="$(
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' -print 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
fi
if [[ -n "${LATEST_PARTIAL}" ]]; then
  if [[ "${AUTO_RESUME}" != "1" ]]; then
    echo "Partial continuation exists: ${LATEST_PARTIAL}" >&2
    echo "Set AUTO_RESUME=1 or choose an empty OUT_DIR." >&2
    exit 1
  fi
  RUN_START="$(checkpoint_iteration \
    "${LATEST_PARTIAL}" \
    "$((SOURCE_ITERATION + 1))" \
    "$((TARGET_ITERATION - 1))")"
  RUN_CHECKPOINT="${LATEST_PARTIAL}"
  RESUME_DESCRIPTION="validated partial candidate; AdamW moments and scheduler phase restored"
fi

RUN_ITERS="$((TARGET_ITERATION - RUN_START))"

echo "Matched matcher adaptation continuation: ${RUN_START} -> ${TARGET_ITERATION}"
echo "Original 70k source: ${SOURCE_CHECKPOINT}"
echo "Run checkpoint: ${RUN_CHECKPOINT}"
echo "Resume mode: ${RESUME_DESCRIPTION}"
echo "Config audit: ${config_audit}"
echo "Remaining optimizer steps: ${RUN_ITERS}"
echo "Architecture/new parameters: unchanged/none"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Output: ${OUT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: all audits passed; training was not started."
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --max-iters "${RUN_ITERS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}" \
  --resume "${RUN_CHECKPOINT}"
