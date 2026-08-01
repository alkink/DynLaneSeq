#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep/iter_0025000.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_aux3x8_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_aux3x8_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep/iter_0025000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/rowref_obj0p5_primary32_aux3x8_25k_gate}"
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

"${PYTHON}" - "${CONTROL_CHECKPOINT}" "${CANDIDATE_CHECKPOINT}" <<'PY'
import sys
import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


control = load(sys.argv[1])
candidate = load(sys.argv[2])
for name, payload in (("control", control), ("candidate", candidate)):
    if int(payload.get("iteration", -1)) != 25000:
        raise SystemExit(f"{name} checkpoint is not iteration 25000")
    matcher = payload.get("cfg", {}).get("matcher", {})
    if matcher.get("assignment") != "hungarian":
        raise SystemExit(f"{name} primary matcher is not Hungarian")
    if int(matcher.get("num_groups", -1)) != 1:
        raise SystemExit(f"{name} primary matcher is not one global group")
    if float(matcher.get("lambda_obj", float("nan"))) != 0.5:
        raise SystemExit(f"{name} matcher.lambda_obj is not 0.5")

candidate_cfg = candidate["cfg"]
sizes = candidate_cfg["model"]["structured_query"].get(
    "training_auxiliary_group_sizes"
)
if tuple(sizes or ()) != (8, 8, 8):
    raise SystemExit("candidate checkpoint is not primary32+aux3x8")
if float(candidate_cfg["loss"].get("lambda_training_auxiliary", 0.0)) != 0.5:
    raise SystemExit("candidate checkpoint auxiliary weight is not 0.5")
print("checkpoint audit passed: paired 25k global primary, candidate aux3x8")
PY

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"
CONTROL_JSON="${OUTPUT_DIR}/control_primary32_25k_uniform256.json"
CANDIDATE_JSON="${OUTPUT_DIR}/candidate_primary32_aux3x8_25k_uniform256.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

common_args=(
  --split val
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --top-k-values 4
  --iou-thresholds 0.5 0.75
  --quality-powers 0.25
  --score-thresholds 0.05 0.10 0.15 0.20 0.25 0.30
  --line-width 30
  --nms-distance-thresh-px 20.0
  --nms-min-overlap-points 5
  --max-batches "${MAX_BATCHES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --sample-strategy uniform
  --cache-dir "${CACHE_DIR}"
  --reuse-cache
  --exact-postprocess
)

echo "===== 1/3 matched global one-to-one control ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONTROL_CONFIG}" \
  --checkpoint "${CONTROL_CHECKPOINT}" \
  --output-json "${CONTROL_JSON}" \
  "${common_args[@]}"

echo "===== 2/3 hybrid training, deployable primary 32 only ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CANDIDATE_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --output-json "${CANDIDATE_JSON}" \
  "${common_args[@]}"

echo "===== 3/3 paired causal gate ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_hybrid_primary_auxiliary_gate \
  --control-json "${CONTROL_JSON}" \
  --candidate-json "${CANDIDATE_JSON}" \
  --output-json "${SUMMARY_JSON}"

echo "Primary result: ${SUMMARY_JSON}"

