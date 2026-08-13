#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v16_candidate_aligned_local_reranker_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/v16_candidate_aligned_local_reranker_225k}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"

for required in \
  "${V7_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V16 preflight artifact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/reports" "${OUTPUT_ROOT}/audits" "${LIST_ROOT}" "${CACHE_ROOT}"

PROTOCOL_REPORT="${OUTPUT_ROOT}/audits/list_protocol.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.build_v11_bridge_lists \
  --train-list "${DATA_ROOT}/list/train_gt.txt" \
  --val-list "${DATA_ROOT}/list/val.txt" \
  --output-dir "${LIST_ROOT}" \
  --seed "${SEED}" \
  --train-clips 512 \
  --train-images 4096 \
  --seen-images 256 \
  --same-clip-unseen-images 256 \
  --heldout-clips 64 \
  --heldout-images 256 \
  --val-images 256 \
  --experiment-name "V16 candidate-aligned clip-disjoint preflight" \
  --output-json "${PROTOCOL_REPORT}"

"${PYTHON}" - "${PROTOCOL_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("passed") is not True or report.get("test_set_used") is not False:
    raise SystemExit("V16 list/leakage contract failed")
PY

HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"
HELDOUT_JSON="${OUTPUT_ROOT}/reports/heldout_candidate_group_preflight.json"
HELDOUT_MD="${OUTPUT_ROOT}/reports/heldout_candidate_group_preflight.md"
VAL_JSON="${OUTPUT_ROOT}/reports/validation_candidate_group_preflight.json"
VAL_MD="${OUTPUT_ROOT}/reports/validation_candidate_group_preflight.md"

run_domain() {
  local name="$1"
  local split="$2"
  local list_path="$3"
  local output_json="$4"
  local output_md="$5"
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v16_anchor_candidate_groups \
    --config "${V7_CONFIG}" \
    --checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_ROOT}/${name}" \
    --reuse-cache \
    --sample-strategy sequential \
    --max-images 256 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --official-iou-workers "${METRIC_WORKERS}" \
    --output-json "${output_json}" \
    --output-md "${output_md}"
}

run_domain heldout train "${HELDOUT_LIST}" "${HELDOUT_JSON}" "${HELDOUT_MD}"
run_domain validation val "${VAL_LIST}" "${VAL_JSON}" "${VAL_MD}"

SUMMARY="${OUTPUT_ROOT}/v16_candidate_group_preflight_summary.json"
SUMMARY_MD="${OUTPUT_ROOT}/v16_candidate_group_preflight_summary.md"
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v16_candidate_group_preflight \
  --heldout "${HELDOUT_JSON}" \
  --validation "${VAL_JSON}" \
  --output-json "${SUMMARY}" \
  --output-md "${SUMMARY_MD}"

"${PYTHON}" - \
  "${SUMMARY}" \
  "${OUTPUT_ROOT}/v16_preflight_completion.json" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

summary_path, output_path = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
completion = {
    "experiment": "V16 candidate-aligned local reranker",
    "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
    "git_branch": subprocess.check_output(("git", "branch", "--show-current"), text=True).strip(),
    "phase_completed": "training_free_candidate_group_preflight",
    "formal_result": summary["formal_result"],
    "stage_a_authorized": summary["stage_a_authorized"],
    "long_training_authorized": False,
    "full_validation_authorized": False,
    "test_set_used": False,
    "decision": summary["decision"],
}
Path(output_path).write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
print(json.dumps(completion, indent=2))
PY

echo "V16 training-free preflight complete. No optimizer step, full validation, test, or long training was started."
