#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k/iter_0025000.pt}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/unified_lane_set_v3_25k}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

common=(
  --config "${CONFIG}"
  --checkpoint "${CKPT}"
  --split val
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --top-k-values 4
  --iou-thresholds 0.5 0.75
  --quality-powers 0.0
  --score-thresholds 0.02 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60
  --line-width 30
  --nms-min-overlap-points 5
  --max-batches "${MAX_BATCHES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --sample-strategy uniform
  --cache-dir "${CACHE_DIR}"
  --exact-postprocess
)

echo "===== one inference pass + NMS-free operating points ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  "${common[@]}" \
  --nms-distance-thresh-px 0 \
  --reuse-cache \
  --output-json "${OUTPUT_DIR}/nmsfree_uniform256.json"

echo "===== cached NMS=20 counterfactual ====="
"${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  "${common[@]}" \
  --nms-distance-thresh-px 20 \
  --cache-only \
  --output-json "${OUTPUT_DIR}/nms20_uniform256.json"

"${PYTHON}" - \
  "${OUTPUT_DIR}/nmsfree_uniform256.json" \
  "${OUTPUT_DIR}/nms20_uniform256.json" \
  "${OUTPUT_DIR}/summary.json" <<'PY'
import json
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def points(report):
    result = []
    for row in report.get("rows", []):
        if (
            row.get("strategy") == "model_topk_nms"
            and int(row.get("top_k", 0)) == 4
            and float(row.get("quality_power", -1.0)) == 0.0
            and row.get("score_threshold") is not None
            and float(row.get("iou_threshold", -1.0)) == 0.5
        ):
            metric = row.get("official_iou_0.5")
            if metric is not None:
                result.append(
                    {
                        "score_threshold": float(row["score_threshold"]),
                        "precision": float(metric["precision"]),
                        "recall": float(metric["recall"]),
                        "f1": float(metric["f1"]),
                    }
                )
    if not result:
        raise SystemExit("No exact IoU@0.50 operating points found")
    return sorted(result, key=lambda item: item["score_threshold"])


nmsfree = points(load(sys.argv[1]))
nms20 = points(load(sys.argv[2]))
best_free = max(nmsfree, key=lambda item: (item["f1"], item["precision"]))
best_nms = max(nms20, key=lambda item: (item["f1"], item["precision"]))
same_threshold_nms = next(
    item
    for item in nms20
    if item["score_threshold"] == best_free["score_threshold"]
)
summary = {
    "diagnostic_only": True,
    "selection_rule": "best NMS-free F1@0.50 on the fixed uniform-256 validation sample",
    "best_nmsfree": best_free,
    "nms20_at_same_threshold": same_threshold_nms,
    "nms_gain_at_frozen_threshold_points": 100.0
    * (same_threshold_nms["f1"] - best_free["f1"]),
    "best_nms20": best_nms,
    "nmsfree_operating_points": nmsfree,
    "nms20_operating_points": nms20,
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"output_json: {sys.argv[3]}")
PY
