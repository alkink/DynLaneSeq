#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Joint validation selection over checkpoint and post-processing calibration.
# Every checkpoint is forwarded exactly once; all 21 calibration combinations
# reuse the same cached candidates and official raster-IoU matrices.
VARIANT="${VARIANT:-full}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-175000 200000 225000 250000 278000}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.30 0.35 0.40 0.45 0.50 0.55 0.60}"
QUALITY_POWERS="${QUALITY_POWERS:-0.25 0.50 0.75}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95}"
SELECTION_IOU="${SELECTION_IOU:-0.50}"
TIE_BREAK_IOU="${TIE_BREAK_IOU:-0.70}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUIRE_ALL="${REQUIRE_ALL:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"

case "${VARIANT}" in
  full)
    RUN_NAME="culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_full_seed3407_278k"
    CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_full_seed3407_278k.yaml"
    EXPECTED_INTRA="True"
    EXPECTED_INTER="True"
    ;;
  no_intra)
    RUN_NAME="culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_no_intra_seed3407_278k"
    CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_no_intra_seed3407_278k.yaml"
    EXPECTED_INTRA="False"
    EXPECTED_INTER="True"
    ;;
  no_inter)
    RUN_NAME="culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_no_inter_seed3407_278k"
    CONFIG="dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_no_inter_seed3407_278k.yaml"
    EXPECTED_INTRA="True"
    EXPECTED_INTER="False"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected full, no_intra, or no_inter." >&2
    exit 2
    ;;
esac

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SWEEP_DIR="${RUN_DIR}/val_joint_sweep_seed3407"
CACHE_DIR="${OUTPUT_ROOT}/diagnostic_cache/aaai_seed3407_joint_grid/${VARIANT}"
mkdir -p "${SWEEP_DIR}" "${CACHE_DIR}"

echo "variant: ${VARIANT}"
echo "checkpoint candidates: ${CHECKPOINT_ITERS}"
echo "score thresholds: ${SCORE_THRESHOLDS}"
echo "quality powers: ${QUALITY_POWERS}"
echo "selection: max val F1@${SELECTION_IOU}; tie mF1, then F1@${TIE_BREAK_IOU}"
echo "candidate inference: exact FP32, batch=${EVAL_BATCH_SIZE}, workers=${CACHE_NUM_WORKERS}"

for iteration in ${CHECKPOINT_ITERS}; do
  printf -v ckpt_name 'iter_%07d.pt' "${iteration}"
  ckpt="${RUN_DIR}/${ckpt_name}"
  if [[ ! -f "${ckpt}" ]]; then
    if [[ "${REQUIRE_ALL}" == "1" ]]; then
      echo "Missing required checkpoint: ${ckpt}" >&2
      exit 1
    fi
    echo "Skipping missing checkpoint: ${ckpt}" >&2
    continue
  fi

  read -r embedded_iter embedded_intra embedded_inter embedded_seed < <(
    "${PYTHON_BIN}" -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu"); c=p.get("cfg", {}); s=c.get("model", {}).get("structured_query", {}); print(int(p.get("iteration", 0)), s.get("use_intra_attention", True), s.get("use_inter_attention", True), c.get("seed"))' "${ckpt}"
  )
  if [[ "${embedded_iter}" != "${iteration}" || "${embedded_intra}" != "${EXPECTED_INTRA}" || "${embedded_inter}" != "${EXPECTED_INTER}" || "${embedded_seed}" != "3407" ]]; then
    echo "Refusing checkpoint ${ckpt}: iter=${embedded_iter}, intra=${embedded_intra}, inter=${embedded_inter}, seed=${embedded_seed}; expected ${iteration}/${EXPECTED_INTRA}/${EXPECTED_INTER}/3407." >&2
    exit 1
  fi

  out_json="${SWEEP_DIR}/${VARIANT}_iter_$(printf '%07d' "${iteration}")_val_sweep.json"
  out_txt="${SWEEP_DIR}/${VARIANT}_iter_$(printf '%07d' "${iteration}")_val_sweep.txt"

  echo "============================================================"
  echo "validation sweep: ${ckpt_name}"
  echo "output: ${out_json}"
  echo "============================================================"

  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.sweep_cached_culane_thresholds \
    --config "${CONFIG}" \
    --checkpoint "${ckpt}" \
    --split val \
    --list-path dataset/list/val.txt \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_DIR}" \
    --reuse-cache \
    --no-pretrained-init \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --cache-num-workers "${CACHE_NUM_WORKERS}" \
    --stage main \
    --top-k 4 \
    --nms-distance-thresh-px 20.0 \
    --nms-min-overlap-points 5 \
    --score-thresholds ${SCORE_THRESHOLDS} \
    --quality-powers ${QUALITY_POWERS} \
    --iou-thresholds ${IOU_THRESHOLDS} \
    --selection-iou "${SELECTION_IOU}" \
    --tie-break-iou "${TIE_BREAK_IOU}" \
    --output-json "${out_json}" \
    --output-txt "${out_txt}"
done

SWEEP_DIR="${SWEEP_DIR}" \
VARIANT="${VARIANT}" \
CHECKPOINT_ITERS="${CHECKPOINT_ITERS}" \
SCORE_THRESHOLDS="${SCORE_THRESHOLDS}" \
QUALITY_POWERS="${QUALITY_POWERS}" \
SELECTION_IOU="${SELECTION_IOU}" \
TIE_BREAK_IOU="${TIE_BREAK_IOU}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

sweep_dir = Path(os.environ["SWEEP_DIR"])
variant = os.environ["VARIANT"]
iterations = [int(value) for value in os.environ["CHECKPOINT_ITERS"].split()]
selection_key = f"{float(os.environ['SELECTION_IOU']):.2f}"
tie_key = f"{float(os.environ['TIE_BREAK_IOU']):.2f}"

rows = []
for iteration in iterations:
    result_path = sweep_dir / f"{variant}_iter_{iteration:07d}_val_sweep.json"
    if not result_path.exists():
        continue
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    best = payload.get("best")
    if best is None:
        continue
    primary = best["metrics"][selection_key]
    tie = best["metrics"][tie_key]
    rows.append(
        {
            "iteration": iteration,
            "score_threshold": best["score_threshold"],
            "quality_power": best["quality_power"],
            "F1@0.50": best["metrics"]["0.50"]["F1"],
            "F1@0.70": best["metrics"]["0.70"]["F1"],
            "mF1": best["mF1"],
            "Precision@0.50": best["metrics"]["0.50"]["Precision"],
            "Recall@0.50": best["metrics"]["0.50"]["Recall"],
            "sweep_path": str(result_path),
            "_primary": primary["F1"],
            "_tie": tie["F1"],
        }
    )

if not rows:
    raise SystemExit("No completed validation sweeps found.")

rows.sort(
    key=lambda row: (
        row["_primary"],
        row["mF1"],
        row["_tie"],
        -row["score_threshold"],
    ),
    reverse=True,
)
for row in rows:
    row.pop("_primary")
    row.pop("_tie")

summary = {
    "selection_split": "val",
    "variant": variant,
    "checkpoint_candidates": iterations,
    "score_threshold_grid": [float(v) for v in os.environ["SCORE_THRESHOLDS"].split()],
    "quality_power_grid": [float(v) for v in os.environ["QUALITY_POWERS"].split()],
    "selection_rule": "max F1@0.50; tie-break by mF1, then F1@0.70, then lower score threshold",
    "best": rows[0],
    "rows_sorted": rows,
}
summary_path = sweep_dir / f"{variant}_val_joint_selection_seed3407.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print("\nJoint validation ranking (each checkpoint at its own val-selected calibration)")
print("iteration  score  quality  F1@0.50  F1@0.70  mF1      P@0.50   R@0.50")
for row in rows:
    print(
        f"{row['iteration']:9d}  "
        f"{row['score_threshold']:5.2f}  "
        f"{row['quality_power']:7.2f}  "
        f"{100.0 * row['F1@0.50']:8.4f}  "
        f"{100.0 * row['F1@0.70']:8.4f}  "
        f"{100.0 * row['mF1']:8.4f}  "
        f"{100.0 * row['Precision@0.50']:8.4f}  "
        f"{100.0 * row['Recall@0.50']:8.4f}"
    )
print(
    f"\nSelected from validation: iter_{rows[0]['iteration']:07d}.pt, "
    f"score={rows[0]['score_threshold']:.2f}, quality={rows[0]['quality_power']:.2f}"
)
print(f"Summary: {summary_path}")
PY
