#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep/iter_0025000.pt}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_coherent_lane_state_25k.yaml}"
CANDIDATE_DIR="${CANDIDATE_DIR:-outputs/culane_s0_structured_query_dla34_coherent_lane_state_25k}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-${CANDIDATE_DIR}/iter_0025000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/coherent_lane_state_25k_gate}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"

candidate_trajectory=()
for iteration in 5000 10000 15000 20000 25000; do
  candidate_trajectory+=(
    "${CANDIDATE_DIR}/iter_$(printf '%07d' "${iteration}").pt"
  )
done

for path in \
  "${CONTROL_CONFIG}" \
  "${CONTROL_CHECKPOINT}" \
  "${CANDIDATE_CONFIG}" \
  "${candidate_trajectory[@]}"
do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required gate input: ${path}" >&2
    exit 1
  fi
done

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
for name, payload in (("control", control), ("candidate", candidate)):
    iteration = int(payload.get("iteration", -1))
    if iteration != 25000:
        raise SystemExit(f"{name} iteration={iteration}, expected 25000")
    matcher = payload.get("cfg", {}).get("matcher", {})
    if float(matcher.get("lambda_obj", float("nan"))) != 0.5:
        raise SystemExit(f"{name} matcher.lambda_obj is not 0.5")

candidate_cfg = candidate.get("cfg", {})
structured = candidate_cfg.get("model", {}).get("structured_query", {})
loss = candidate_cfg.get("loss", {})
post = candidate_cfg.get("postprocess", {})
checks = {
    "persistent lane state": bool(
        structured.get("lane_state", {}).get("enabled", False)
    ),
    "detached iterative reference": bool(
        structured.get("row_reference", {}).get(
            "detach_between_layers", False
        )
    ),
    "no selector": not bool(
        structured.get("set_selection", {}).get("enabled", False)
    ),
    "direct existence loss": float(loss.get("w_exist", 0.0)) == 2.0,
    "quality loss off": float(loss.get("w_quality", -1.0)) == 0.0,
    "direct existence score": str(post.get("score_mode")) == "exist",
    "NMS disabled": float(post.get("lane_nms_distance_thresh_px", -1.0))
    == 0.0,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("candidate checkpoint contract failed: " + ", ".join(failed))
print("checkpoint audit passed: matched 25k/lambda_obj=0.5 and coherent contract")
PY

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

CONTROL_JSON="${OUTPUT_DIR}/control_object0p5_25k_uniform256.json"
CANDIDATE_JSON="${OUTPUT_DIR}/candidate_coherent_25k_uniform256.json"
RANKING_JSON="${OUTPUT_DIR}/paired_threshold_free_ranking.json"
CONTROL_CALIBRATION_JSON="${OUTPUT_DIR}/control_nmsfree_calibration.json"
CANDIDATE_CALIBRATION_JSON="${OUTPUT_DIR}/candidate_nmsfree_calibration.json"
OWNERSHIP_JSON="${OUTPUT_DIR}/candidate_ownership_5k_to25k_uniform256.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

oracle_args=(
  --split val
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --top-k-values 4
  --iou-thresholds 0.5 0.75
  --quality-powers 0.0
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

echo "===== 1/7 matched control capacity/ranking ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONTROL_CONFIG}" \
  --checkpoint "${CONTROL_CHECKPOINT}" \
  --output-json "${CONTROL_JSON}" \
  "${oracle_args[@]}"

echo "===== 2/7 coherent candidate capacity/ranking ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CANDIDATE_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --output-json "${CANDIDATE_JSON}" \
  "${oracle_args[@]}"

echo "===== 3/7 exact paired ranking and NMS-dependency summary ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_nms_ranking_pair \
  --base-json "${CONTROL_JSON}" \
  --candidate-json "${CANDIDATE_JSON}" \
  --output-json "${RANKING_JSON}"

calibration_args=(
  --split val
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --top-k-values 4
  --iou-thresholds 0.5 0.75
  --quality-powers 0.0
  --score-thresholds 0.02 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50
  --line-width 30
  --nms-distance-thresh-px 0
  --nms-min-overlap-points 5
  --max-batches "${MAX_BATCHES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --sample-strategy uniform
  --cache-dir "${CACHE_DIR}"
  --cache-only
  --exact-postprocess
)

echo "===== 4/7 cached NMS-free control calibration ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONTROL_CONFIG}" \
  --checkpoint "${CONTROL_CHECKPOINT}" \
  --output-json "${CONTROL_CALIBRATION_JSON}" \
  "${calibration_args[@]}"

echo "===== 5/7 cached NMS-free candidate calibration ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CANDIDATE_CONFIG}" \
  --checkpoint "${CANDIDATE_CHECKPOINT}" \
  --output-json "${CANDIDATE_CALIBRATION_JSON}" \
  "${calibration_args[@]}"

echo "===== 6/7 same-query ownership trajectory ====="
"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.analyze_unified_selector_ownership_stability \
  --config "${CANDIDATE_CONFIG}" \
  --checkpoints "${candidate_trajectory[@]}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --max-batches "${MAX_BATCHES}" \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype "${AMP_DTYPE}" \
  --line-width 30 \
  --top-k 4 \
  --score-mode exist \
  --quality-power 0.0 \
  --stable-iou-floor 0.30 \
  --near-tie-margin 0.02 \
  --min-owner-retention 0.75 \
  --output-json "${OWNERSHIP_JSON}"

echo "===== 7/7 predeclared causal gate ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_coherent_lane_state_gate \
  --ranking-summary "${RANKING_JSON}" \
  --ownership-report "${OWNERSHIP_JSON}" \
  --control-calibration "${CONTROL_CALIBRATION_JSON}" \
  --candidate-calibration "${CANDIDATE_CALIBRATION_JSON}" \
  --output-json "${SUMMARY_JSON}"

echo "Primary result: ${SUMMARY_JSON}"
