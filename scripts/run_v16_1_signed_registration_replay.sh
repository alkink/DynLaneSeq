#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
V14_ROOT="${V14_ROOT:-outputs/diagnostics/v14_corrected_visual_first_stage_ab_225k}"
V14_CONFIG="${V14_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v14_corrected_visual_first_stage_a_225k_to227k.yaml}"
V14_CHECKPOINT="${V14_CHECKPOINT:-${V14_ROOT}/stage_a/train/iter_0227000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v16_1_signed_registration_replay_227k}"

HELDOUT_LIST="${V14_ROOT}/lists/heldout_clip_image256.txt"
HELDOUT_WRONG_LIST="${V14_ROOT}/lists/heldout_cross_clip_wrong_image256.txt"
HELDOUT_CROSSCLIP="${V14_ROOT}/audits/heldout_cross_clip_derangement.json"
VAL_LIST="${V14_ROOT}/lists/val_clip_balanced_image256.txt"
VAL_WRONG_LIST="${V14_ROOT}/lists/val_cross_clip_wrong_image256.txt"
VAL_CROSSCLIP="${V14_ROOT}/audits/val_cross_clip_derangement.json"

for required in \
  "${V14_CONFIG}" \
  "${V14_CHECKPOINT}" \
  "${HELDOUT_LIST}" \
  "${HELDOUT_WRONG_LIST}" \
  "${HELDOUT_CROSSCLIP}" \
  "${VAL_LIST}" \
  "${VAL_WRONG_LIST}" \
  "${VAL_CROSSCLIP}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V16.1 artifact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/reports"

run_domain() {
  local split="$1"
  local list_path="$2"
  local wrong_list="$3"
  local crossclip="$4"
  local inspect_index="$5"
  local output_json="$6"
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v16_1_signed_registration_replay \
    --config "${V14_CONFIG}" \
    --checkpoint "${V14_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --wrong-list-path "${wrong_list}" \
    --cross-clip-report "${crossclip}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --inspect-index "${inspect_index}" \
    --output-json "${output_json}"
}

HELDOUT_REPORT="${OUTPUT_ROOT}/reports/heldout_signed_registration.json"
VALIDATION_REPORT="${OUTPUT_ROOT}/reports/validation_signed_registration.json"

run_domain \
  train "${HELDOUT_LIST}" "${HELDOUT_WRONG_LIST}" \
  "${HELDOUT_CROSSCLIP}" 151 "${HELDOUT_REPORT}"
run_domain \
  val "${VAL_LIST}" "${VAL_WRONG_LIST}" \
  "${VAL_CROSSCLIP}" 151 "${VALIDATION_REPORT}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v16_1_signed_registration \
  --heldout "${HELDOUT_REPORT}" \
  --validation "${VALIDATION_REPORT}" \
  --output-json "${OUTPUT_ROOT}/v16_1_signed_registration_summary.json"

"${PYTHON}" - \
  "${OUTPUT_ROOT}" "${V14_CONFIG}" "${V14_CHECKPOINT}" \
  "${HELDOUT_LIST}" "${VAL_LIST}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root, config, checkpoint, heldout, validation = map(Path, sys.argv[1:])
paths = {
    "v14_config": config,
    "v14_checkpoint": checkpoint,
    "heldout_list": heldout,
    "validation_list": validation,
    "contract": Path(
        "docs/experiments/"
        "V16_1_SIGNED_REGISTRATION_REPLAY_CONTRACT_2026-08-13.md"
    ),
    "registration": Path(
        "dynlaneseq_eg/evaluation/signed_lane_registration.py"
    ),
    "audit": Path(
        "dynlaneseq_eg/tools/audit_v16_1_signed_registration_replay.py"
    ),
    "summarizer": Path(
        "dynlaneseq_eg/tools/summarize_v16_1_signed_registration.py"
    ),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = {
    "git_commit": subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True
    ).strip(),
    "artifacts": {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in paths.items()
    },
    "optimizer_steps": 0,
    "checkpoint_selection_performed": False,
    "full_validation_run": False,
    "test_set_used": False,
}
(root / "provenance.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
summary = json.loads(
    (root / "v16_1_signed_registration_summary.json").read_text(
        encoding="utf-8"
    )
)
(root / "v16_1_completion.json").write_text(
    json.dumps(
        {
            "experiment": "V16.1 signed-registration training-free replay",
            "passed": bool(summary.get("passed")),
            "decision": summary.get("decision"),
            "optimizer_steps": 0,
            "training_started": False,
            "full_validation_started": False,
            "long_training_started": False,
            "test_set_used": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "V16.1 terminal summary: ${OUTPUT_ROOT}/v16_1_signed_registration_summary.json"

