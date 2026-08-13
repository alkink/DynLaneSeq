#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/geometry_proposal_clustering_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/geometry_proposal_clustering_225k}"
MAX_IMAGES="${MAX_IMAGES:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
REUSE_CACHE="${REUSE_CACHE:-1}"

for required in \
  "${V7_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing geometry-clustering artifact: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}"

run_domain() {
  local split="$1"
  local output_json="${OUTPUT_ROOT}/${split}_uniform${MAX_IMAGES}.json"
  local output_md="${OUTPUT_ROOT}/${split}_uniform${MAX_IMAGES}.md"
  local args=(
    --config "${V7_CONFIG}"
    --checkpoint "${SOURCE_V7_CHECKPOINT}"
    --dataset-root "${DATA_ROOT}"
    --split "${split}"
    --device "${DEVICE}"
    --cache-dir "${CACHE_ROOT}/${split}"
    --stage main
    --sample-strategy uniform
    --max-images "${MAX_IMAGES}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --official-iou-workers "${METRIC_WORKERS}"
    --line-width 30
    --min-valid-rows 5
    --label-iou-threshold 0.50
    --iou-thresholds 0.50 0.75
    --top-k 4
    --output-json "${output_json}"
    --output-md "${output_md}"
  )
  if [[ "${REUSE_CACHE}" == "1" ]]; then
    args+=(--reuse-cache)
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_geometry_proposal_clustering "${args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/${split}_uniform${MAX_IMAGES}.log"
}

run_domain train
run_domain val

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_geometry_proposal_clustering \
  --calibration-json "${OUTPUT_ROOT}/train_uniform${MAX_IMAGES}.json" \
  --validation-json "${OUTPUT_ROOT}/val_uniform${MAX_IMAGES}.json" \
  --output-json "${OUTPUT_ROOT}/geometry_clustering_decision.json" \
  --output-md "${OUTPUT_ROOT}/geometry_clustering_decision.md" \
  2>&1 | tee "${OUTPUT_ROOT}/geometry_clustering_decision.log"

"${PYTHON}" - "${OUTPUT_ROOT}" "${MAX_IMAGES}" "${V7_CONFIG}" "${SOURCE_V7_CHECKPOINT}" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path

root=Path(sys.argv[1])
count=str(sys.argv[2])
config=Path(sys.argv[3])
checkpoint=Path(sys.argv[4])
paths=[
    root/f"train_uniform{count}.json",
    root/f"val_uniform{count}.json",
    root/"geometry_clustering_decision.json",
]
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()
payload={
    "git_commit":subprocess.check_output(("git","rev-parse","HEAD"),text=True).strip(),
    "config":{"path":str(config.resolve()),"sha256":sha(config)},
    "source_checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
    "artifacts":{path.name:{"path":str(path.resolve()),"sha256":sha(path)} for path in paths},
    "training_performed":False,
    "optimizer_steps":0,
    "backward_performed":False,
    "checkpoint_selection_performed":False,
    "threshold_search_performed":False,
    "test_set_used":False,
    "new_model_version_started":False,
}
(root/"provenance.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
decision=json.loads((root/"geometry_clustering_decision.json").read_text(encoding="utf-8"))
completion={
    "formal_result":decision["formal_result"],
    "passed":bool(decision["passed"]),
    "decision":decision["decision"],
    "training_performed":False,
    "optimizer_steps":0,
    "full_validation_started":False,
    "long_training_started":False,
    "new_model_version_started":False,
    "test_set_used":False,
    "required_next_action":"stop_for_user_review",
}
(root/"completion.json").write_text(json.dumps(completion,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY

echo "Geometry proposal clustering audit complete. Stop for user review; no training was run."
