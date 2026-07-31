#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep/iter_0025000.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep/iter_0025000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/rowref_object0p5_fromscratch_25k_gate}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"

for path in \
  "${CONTROL_CONFIG}" \
  "${CONTROL_CHECKPOINT}" \
  "${CANDIDATE_CONFIG}" \
  "${CANDIDATE_CHECKPOINT}"
do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required gate input: ${path}" >&2
    exit 1
  fi
done

# A filename is not enough for a causal comparison.  Verify the embedded
# iteration and matcher settings before spending time on inference.
"${PYTHON}" - \
  "${CONTROL_CHECKPOINT}" \
  "${CANDIDATE_CHECKPOINT}" <<'PY'
import sys

import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


control = load(sys.argv[1])
candidate = load(sys.argv[2])
for name, payload, expected_lambda in (
    ("control", control, 2.0),
    ("candidate", candidate, 0.5),
):
    iteration = int(payload.get("iteration", -1))
    if iteration != 25000:
        raise SystemExit(f"{name} checkpoint iteration is {iteration}, expected 25000")
    matcher = payload.get("cfg", {}).get("matcher", {})
    actual_lambda = float(matcher.get("lambda_obj", float("nan")))
    if actual_lambda != expected_lambda:
        raise SystemExit(
            f"{name} checkpoint matcher.lambda_obj={actual_lambda}, "
            f"expected {expected_lambda}"
        )
print("checkpoint audit passed: matched iteration=25000, lambda_obj=2.0 vs 0.5")
PY

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

CONTROL_JSON="${OUTPUT_DIR}/control_25k_uniform256.json"
CANDIDATE_JSON="${OUTPUT_DIR}/candidate_obj0p5_25k_uniform256.json"
RANKING_SUMMARY="${OUTPUT_DIR}/paired_threshold_free_ranking.json"
ASSIGNMENT_REPORT="${OUTPUT_DIR}/paired_query_ownership.json"
GATE_SUMMARY="${OUTPUT_DIR}/summary.json"

common_oracle_args=(
  --split val
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --top-k-values 4
  --iou-thresholds 0.5 0.75
  --quality-powers 0.0 0.25 0.50
  --score-thresholds -1.0
  --line-width 30
  --nms-distance-thresh-px 20
  --nms-min-overlap-points 5
  --max-batches "${MAX_BATCHES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --sample-strategy uniform
  --cache-dir "${CACHE_DIR}"
  --reuse-cache
  --exact-postprocess
)

echo "===== 1/4 exact official-raster control cache ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONTROL_CONFIG}" \
  --checkpoint "${CONTROL_CHECKPOINT}" \
  --output-json "${CONTROL_JSON}" \
  "${common_oracle_args[@]}"

echo "===== 2/4 exact official-raster candidate cache ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CANDIDATE_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --output-json "${CANDIDATE_JSON}" \
  "${common_oracle_args[@]}"

echo "===== 3/4 threshold-free paired ranking/NMS summary ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_nms_ranking_pair \
  --base-json "${CONTROL_JSON}" \
  --candidate-json "${CANDIDATE_JSON}" \
  --output-json "${RANKING_SUMMARY}"

echo "===== 4/4 query ownership and final causal gate ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_cross_backbone_error_overlap \
  --r34-config "${CONTROL_CONFIG}" \
  --r34-checkpoint "${CONTROL_CHECKPOINT}" \
  --dla34-config "${CANDIDATE_CONFIG}" \
  --dla34-checkpoint "${CANDIDATE_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --sample-strategy uniform \
  --line-width 30 \
  --iou-thresholds 0.5 0.75 \
  --amp-dtype none \
  --output-json "${ASSIGNMENT_REPORT}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_fromscratch_matcher_gate \
  --ranking-summary "${RANKING_SUMMARY}" \
  --assignment-report "${ASSIGNMENT_REPORT}" \
  --output-json "${GATE_SUMMARY}"

echo "Primary result: ${GATE_SUMMARY}"
