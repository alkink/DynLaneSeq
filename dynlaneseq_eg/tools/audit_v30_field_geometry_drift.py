from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
)


ROW_BANDS = (
    ("top_0_39", 0, 40),
    ("upper_middle_40_79", 40, 80),
    ("lower_middle_80_119", 80, 120),
    ("bottom_120_159", 120, 160),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose paired V7/V30 support and final-selection transitions "
            "into row-position and range errors."
        )
    )
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"invalid diagnostic cache: {path}")
    return payload


def _image_key(record: dict[str, Any]) -> str:
    parts = Path(str(record.get("image_id", ""))).parts
    if len(parts) < 3:
        raise ValueError("invalid image ID in diagnostic cache")
    return Path(*parts[-3:]).as_posix()


def _stage(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    stage = record.get("stages", {}).get("main")
    required = (
        "pred_x_rows",
        "range_norm",
        "official_iou",
        "official_candidate_valid",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
        "selection_slot_official_iou",
        "selection_slot_official_candidate_valid",
        "selection_slot_active",
    )
    if not isinstance(stage, dict) or not all(
        isinstance(stage.get(name), torch.Tensor) for name in required
    ):
        raise ValueError("diagnostic cache lacks required full-stage tensors")
    return stage


def _valid_target_rows(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    target = record["target"]
    x_rows = target["x_rows"].float()
    mask = target["valid_mask"].bool() & torch.isfinite(x_rows)
    valid_gt = mask.sum(dim=-1) >= 5
    return x_rows[valid_gt], mask[valid_gt]


def _row_masks(
    x_rows: torch.Tensor,
    range_norm: torch.Tensor,
    *,
    input_h: int,
    input_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    synthetic = {"pred_x_rows": x_rows, "range_norm": range_norm}
    curves, masks, _valid = candidate_row_masks(
        synthetic,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=5,
    )
    return curves, masks


def _selected_ids(stage: dict[str, torch.Tensor]) -> list[int]:
    active = stage["selection_slot_active"].bool()
    valid = stage["selection_slot_official_candidate_valid"].bool()
    return (active & valid).nonzero(as_tuple=False).flatten().tolist()


def _support_tp_gt(stage: dict[str, torch.Tensor], threshold: float) -> set[int]:
    quality = stage["official_iou"].float()
    valid = stage["official_candidate_valid"].bool()
    assignment = cardinality_oracle_assignment(
        quality,
        threshold=float(threshold),
        top_k=4,
        candidate_valid=valid,
    )
    return {int(gt) for gt, _proposal in assignment.pairs}


def _selection_tp_gt(stage: dict[str, torch.Tensor], threshold: float) -> set[int]:
    assignment = evaluator_hungarian_assignment(
        stage["selection_slot_official_iou"].float(),
        _selected_ids(stage),
        threshold=float(threshold),
    )
    return {int(gt) for gt, _slot in assignment.pairs}


def _best_curve(
    record: dict[str, Any],
    gt_index: int,
    *,
    population: str,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    stage = _stage(record)
    meta = record["meta"]
    input_h = int(meta["input_h"])
    input_w = int(meta["input_w"])
    if population == "support":
        quality = stage["official_iou"][gt_index].float().clone()
        valid = stage["official_candidate_valid"].bool()
        quality[~valid] = -1.0
        if not bool(valid.any()):
            return None
        index = int(quality.argmax())
        curves, masks = candidate_row_masks(
            stage,
            input_h=input_h,
            input_w=input_w,
            min_valid_rows=5,
        )[:2]
        return curves[index], masks[index], float(quality[index])
    if population == "selection":
        ids = _selected_ids(stage)
        if not ids:
            return None
        quality = stage["selection_slot_official_iou"][gt_index, ids].float()
        index = int(ids[int(quality.argmax())])
        curves, masks = _row_masks(
            stage["selection_slot_pred_x_rows"].float(),
            stage["selection_slot_range_norm"].float(),
            input_h=input_h,
            input_w=input_w,
        )
        return curves[index], masks[index], float(
            stage["selection_slot_official_iou"][gt_index, index]
        )
    raise ValueError(f"unknown population: {population}")


def _new_curve_stats() -> dict[str, Any]:
    return {
        "lanes": 0,
        "quality": [],
        "start_error_rows": [],
        "end_error_rows": [],
        "bands": {
            name: {
                "gt_rows": 0,
                "overlap_rows": 0,
                "missing_rows": 0,
                "absolute_errors_px": [],
                "signed_error_sum_px": 0.0,
                "squared_error_sum_px2": 0.0,
            }
            for name, _start, _end in ROW_BANDS
        },
    }


def _add_curve(
    stats: dict[str, Any],
    curve: tuple[torch.Tensor, torch.Tensor, float] | None,
    target_x: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    if curve is None:
        return
    pred_x, pred_mask, quality = curve
    stats["lanes"] += 1
    stats["quality"].append(float(quality))
    gt_ids = target_mask.nonzero(as_tuple=False).flatten()
    pred_ids = pred_mask.nonzero(as_tuple=False).flatten()
    if gt_ids.numel() and pred_ids.numel():
        stats["start_error_rows"].append(abs(int(pred_ids[0]) - int(gt_ids[0])))
        stats["end_error_rows"].append(abs(int(pred_ids[-1]) - int(gt_ids[-1])))
    for name, start, end in ROW_BANDS:
        band = stats["bands"][name]
        selector = torch.zeros_like(target_mask)
        selector[start:end] = True
        gt = target_mask & selector
        overlap = gt & pred_mask
        missing = gt & ~pred_mask
        errors = pred_x[overlap] - target_x[overlap]
        band["gt_rows"] += int(gt.sum())
        band["overlap_rows"] += int(overlap.sum())
        band["missing_rows"] += int(missing.sum())
        if errors.numel():
            absolute = errors.abs().tolist()
            band["absolute_errors_px"].extend(float(value) for value in absolute)
            band["signed_error_sum_px"] += float(errors.sum())
            band["squared_error_sum_px2"] += float(errors.square().sum())


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _finish_curve_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result = {
        "lanes": int(stats["lanes"]),
        "quality": _summary(stats["quality"]),
        "start_error_rows": _summary(stats["start_error_rows"]),
        "end_error_rows": _summary(stats["end_error_rows"]),
        "bands": {},
    }
    for name, band in stats["bands"].items():
        overlap = int(band["overlap_rows"])
        gt_rows = int(band["gt_rows"])
        absolute = _summary(band["absolute_errors_px"])
        result["bands"][name] = {
            "gt_rows": gt_rows,
            "overlap_rows": overlap,
            "missing_rows": int(band["missing_rows"]),
            "row_coverage": overlap / gt_rows if gt_rows else 0.0,
            "mean_absolute_error_px": float(absolute["mean"]),
            "p50_absolute_error_px": float(absolute["p50"]),
            "p90_absolute_error_px": float(absolute["p90"]),
            "signed_bias_px": (
                float(band["signed_error_sum_px"]) / overlap if overlap else 0.0
            ),
            "rmse_px": (
                float(band["squared_error_sum_px2"]) / overlap
            )
            ** 0.5
            if overlap
            else 0.0,
        }
    return result


def _new_cohort() -> dict[str, Any]:
    return {
        "gt_lanes": 0,
        "source": _new_curve_stats(),
        "candidate": _new_curve_stats(),
    }


def main() -> None:
    args = _parse_args()
    source = _load(args.source_cache)
    candidate = _load(args.candidate_cache)
    source_by_image = {_image_key(record): record for record in source["records"]}
    candidate_by_image = {
        _image_key(record): record for record in candidate["records"]
    }
    if source_by_image.keys() != candidate_by_image.keys():
        raise ValueError("paired caches do not contain the same images")

    thresholds = [float(value) for value in args.thresholds]
    cohorts: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        tag = f"{threshold:.2f}"
        for population in ("support", "selection"):
            cohorts[f"{population}_cross_down_{tag}"] = _new_cohort()
            cohorts[f"{population}_cross_up_{tag}"] = _new_cohort()

    for image_id in sorted(source_by_image):
        source_record = source_by_image[image_id]
        candidate_record = candidate_by_image[image_id]
        source_target_x, source_target_mask = _valid_target_rows(source_record)
        candidate_target_x, candidate_target_mask = _valid_target_rows(candidate_record)
        if not torch.equal(source_target_mask, candidate_target_mask) or not torch.allclose(
            source_target_x, candidate_target_x
        ):
            raise ValueError(f"paired target mismatch for {image_id}")
        source_stage = _stage(source_record)
        candidate_stage = _stage(candidate_record)
        if int(source_stage["official_iou"].shape[0]) != int(source_target_x.shape[0]):
            raise ValueError("source official-IoU GT count mismatch")
        if int(candidate_stage["official_iou"].shape[0]) != int(source_target_x.shape[0]):
            raise ValueError("candidate official-IoU GT count mismatch")

        for threshold in thresholds:
            tag = f"{threshold:.2f}"
            for population, membership in (
                ("support", _support_tp_gt),
                ("selection", _selection_tp_gt),
            ):
                source_tp = membership(source_stage, threshold)
                candidate_tp = membership(candidate_stage, threshold)
                transitions = (
                    (f"{population}_cross_down_{tag}", source_tp - candidate_tp),
                    (f"{population}_cross_up_{tag}", candidate_tp - source_tp),
                )
                for cohort_name, gt_ids in transitions:
                    cohort = cohorts[cohort_name]
                    for gt_index in sorted(gt_ids):
                        cohort["gt_lanes"] += 1
                        source_curve = _best_curve(
                            source_record, gt_index, population=population
                        )
                        candidate_curve = _best_curve(
                            candidate_record, gt_index, population=population
                        )
                        _add_curve(
                            cohort["source"],
                            source_curve,
                            source_target_x[gt_index],
                            source_target_mask[gt_index],
                        )
                        _add_curve(
                            cohort["candidate"],
                            candidate_curve,
                            source_target_x[gt_index],
                            source_target_mask[gt_index],
                        )

    finished: dict[str, Any] = {}
    for name, cohort in cohorts.items():
        source_stats = _finish_curve_stats(cohort["source"])
        candidate_stats = _finish_curve_stats(cohort["candidate"])
        band_deltas = {
            band: {
                "candidate_minus_source_mae_px": (
                    candidate_stats["bands"][band]["mean_absolute_error_px"]
                    - source_stats["bands"][band]["mean_absolute_error_px"]
                ),
                "candidate_minus_source_row_coverage": (
                    candidate_stats["bands"][band]["row_coverage"]
                    - source_stats["bands"][band]["row_coverage"]
                ),
            }
            for band, _start, _end in ROW_BANDS
        }
        finished[name] = {
            "gt_lanes": int(cohort["gt_lanes"]),
            "source": source_stats,
            "candidate": candidate_stats,
            "candidate_minus_source": {
                "mean_quality": (
                    float(candidate_stats["quality"]["mean"])
                    - float(source_stats["quality"]["mean"])
                ),
                "mean_start_error_rows": (
                    float(candidate_stats["start_error_rows"]["mean"])
                    - float(source_stats["start_error_rows"]["mean"])
                ),
                "mean_end_error_rows": (
                    float(candidate_stats["end_error_rows"]["mean"])
                    - float(source_stats["end_error_rows"]["mean"])
                ),
                "bands": band_deltas,
            },
        }

    payload = {
        "experiment": "V30 Field-only versus exact V7 row/range drift audit",
        "source_cache": args.source_cache,
        "candidate_cache": args.candidate_cache,
        "images": len(source_by_image),
        "row_bands": [
            {"name": name, "start_inclusive": start, "end_exclusive": end}
            for name, start, end in ROW_BANDS
        ],
        "cohorts": finished,
        "notes": {
            "cross_down": "GT is a TP for source but not candidate",
            "cross_up": "GT is a TP for candidate but not source",
            "selection_curve": (
                "closest active final slot for that GT; set-level TP membership "
                "still uses evaluator Hungarian assignment"
            ),
            "support_curve": "highest-official-IoU proposal for that GT",
            "test_split_used": False,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
