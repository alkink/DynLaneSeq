#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"

BASE_CONFIG="${BASE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep/iter_0025000.pt}"
CANDIDATE_ALL_CONFIG="${CANDIDATE_ALL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_g4train_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
CANDIDATE_GROUP_CONFIG="${CANDIDATE_GROUP_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_g4train_g1infer_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_g4train_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep/iter_0025000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/rowref_obj0p5_g4train_g1infer_25k_gate}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"

for path in \
  "${BASE_CONFIG}" \
  "${BASE_CHECKPOINT}" \
  "${CANDIDATE_ALL_CONFIG}" \
  "${CANDIDATE_GROUP_CONFIG}" \
  "${CANDIDATE_CHECKPOINT}"
do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required gate input: ${path}" >&2
    exit 1
  fi
done

"${PYTHON}" - "${BASE_CHECKPOINT}" "${CANDIDATE_CHECKPOINT}" <<'PY'
import sys
import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


base = load(sys.argv[1])
candidate = load(sys.argv[2])
for name, payload in (("base", base), ("candidate", candidate)):
    iteration = int(payload.get("iteration", -1))
    if iteration != 25000:
        raise SystemExit(f"{name} checkpoint iteration={iteration}, expected 25000")
    matcher = payload.get("cfg", {}).get("matcher", {})
    if float(matcher.get("lambda_obj", float("nan"))) != 0.5:
        raise SystemExit(f"{name} checkpoint lambda_obj is not 0.5")

base_matcher = base["cfg"]["matcher"]
candidate_matcher = candidate["cfg"]["matcher"]
if base_matcher.get("assignment") != "hungarian" or int(base_matcher.get("num_groups", -1)) != 1:
    raise SystemExit("base checkpoint is not the global one-to-one control")
if candidate_matcher.get("assignment") != "grouped_one_to_many" or int(candidate_matcher.get("num_groups", -1)) != 4:
    raise SystemExit("candidate checkpoint is not four-group one-to-many")
print("checkpoint audit passed: paired 25k, lambda_obj=0.5, g1 vs g4")
PY

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"
BASE_JSON="${OUTPUT_DIR}/base_global1to1_25k_uniform.json"
CANDIDATE_ALL_JSON="${OUTPUT_DIR}/candidate_g4_all32_25k_uniform.json"
CANDIDATE_GROUP_JSON="${OUTPUT_DIR}/candidate_g4_group0_25k_uniform.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

common_args=(
  --split val
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --top-k-values 4
  --iou-thresholds 0.5 0.75
  --quality-powers 0.0 0.25 0.50
  --score-thresholds 0.05 0.10 0.15 0.20 0.25 0.30
  --line-width 30
  --nms-min-overlap-points 5
  --max-batches "${MAX_BATCHES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --sample-strategy uniform
  --cache-dir "${CACHE_DIR}"
  --reuse-cache
  --exact-postprocess
)

echo "===== 1/4 global one-to-one row-reference control ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${BASE_CONFIG}" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --nms-distance-thresh-px 20.0 \
  --output-json "${BASE_JSON}" \
  "${common_args[@]}"

echo "===== 2/4 four-group candidate, all 32 train-time proposals ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CANDIDATE_ALL_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --nms-distance-thresh-px 20.0 \
  --output-json "${CANDIDATE_ALL_JSON}" \
  "${common_args[@]}"

echo "===== 3/4 predeclared group zero, quality-only, no lane NMS ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CANDIDATE_GROUP_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --nms-distance-thresh-px 0.0 \
  --output-json "${CANDIDATE_GROUP_JSON}" \
  "${common_args[@]}"

echo "===== 4/4 paired causal summary ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_train_many_infer_one_gate \
  --base-json "${BASE_JSON}" \
  --candidate-all-json "${CANDIDATE_ALL_JSON}" \
  --candidate-group-json "${CANDIDATE_GROUP_JSON}" \
  --output-json "${SUMMARY_JSON}"

echo "Primary result: ${SUMMARY_JSON}"

