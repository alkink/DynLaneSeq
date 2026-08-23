from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Factor a paired four-slot cache into activity-mask and "
            "conditional route/geometry effects."
        )
    )
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75]
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("records"), list
    ):
        raise ValueError(f"invalid diagnostic cache: {path}")
    return payload


def _stage(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    stage = record.get("stages", {}).get("main")
    if not isinstance(stage, dict):
        raise ValueError("cache record has no main stage")
    required = (
        "selection_slot_active",
        "selection_slot_official_iou",
        "selection_slot_official_candidate_valid",
    )
    if not all(isinstance(stage.get(name), torch.Tensor) for name in required):
        raise ValueError("cache lacks refined four-slot official-IoU tensors")
    return stage


def _image_key(record: dict[str, Any]) -> str:
    parts = Path(str(record.get("image_id", ""))).parts
    if len(parts) < 3:
        raise ValueError("cache record has an invalid image ID")
    return Path(*parts[-3:]).as_posix()


def _empty_counter() -> dict[str, int]:
    return {"images": 0, "gt": 0, "selected": 0, "tp": 0, "fp": 0, "fn": 0}


def _update(
    counter: dict[str, int],
    *,
    activity_stage: dict[str, torch.Tensor],
    geometry_stage: dict[str, torch.Tensor],
    threshold: float,
) -> None:
    iou = geometry_stage["selection_slot_official_iou"].float()
    valid = geometry_stage[
        "selection_slot_official_candidate_valid"
    ].bool()
    active = activity_stage["selection_slot_active"].bool()
    if active.shape != valid.shape:
        raise ValueError("source/candidate slot shapes differ")
    selected = (active & valid).nonzero(as_tuple=False).flatten().tolist()
    assignment = evaluator_hungarian_assignment(iou, selected, threshold)
    hits = int(assignment.hit_count)
    gt = int(iou.shape[0])
    counter["images"] += 1
    counter["gt"] += gt
    counter["selected"] += len(selected)
    counter["tp"] += hits
    counter["fp"] += len(selected) - hits
    counter["fn"] += gt - hits


def _finish(counter: dict[str, int]) -> dict[str, float | int]:
    tp = int(counter["tp"])
    fp = int(counter["fp"])
    fn = int(counter["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        **counter,
        "prediction_count": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()
    source = _load(args.source_cache)
    candidate = _load(args.candidate_cache)
    source_records = source["records"]
    candidate_records = candidate["records"]
    source_by_image = {
        _image_key(record): record for record in source_records
    }
    candidate_by_image = {
        _image_key(record): record for record in candidate_records
    }
    if len(source_by_image) != len(source_records) or len(
        candidate_by_image
    ) != len(candidate_records):
        raise ValueError("paired cache contains duplicate image IDs")
    if source_by_image.keys() != candidate_by_image.keys():
        raise ValueError("paired caches contain different image sets")

    pairs: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
    changed_activity_images = 0
    changed_activity_slots = 0
    for image_id in source_by_image:
        source_record = source_by_image[image_id]
        candidate_record = candidate_by_image[image_id]
        source_stage = _stage(source_record)
        candidate_stage = _stage(candidate_record)
        source_gt = int(source_stage["selection_slot_official_iou"].shape[0])
        candidate_gt = int(
            candidate_stage["selection_slot_official_iou"].shape[0]
        )
        if source_gt != candidate_gt:
            raise ValueError("paired cache GT counts differ")
        activity_delta = (
            source_stage["selection_slot_active"].bool()
            != candidate_stage["selection_slot_active"].bool()
        )
        changed_activity_images += int(bool(activity_delta.any()))
        changed_activity_slots += int(activity_delta.sum())
        pairs.append((source_stage, candidate_stage))

    by_threshold: dict[str, Any] = {}
    for threshold in (float(value) for value in args.iou_thresholds):
        counters = {
            "source_activity_source_geometry": _empty_counter(),
            "candidate_activity_source_geometry": _empty_counter(),
            "source_activity_candidate_geometry": _empty_counter(),
            "candidate_activity_candidate_geometry": _empty_counter(),
        }
        for source_stage, candidate_stage in pairs:
            _update(
                counters["source_activity_source_geometry"],
                activity_stage=source_stage,
                geometry_stage=source_stage,
                threshold=threshold,
            )
            _update(
                counters["candidate_activity_source_geometry"],
                activity_stage=candidate_stage,
                geometry_stage=source_stage,
                threshold=threshold,
            )
            _update(
                counters["source_activity_candidate_geometry"],
                activity_stage=source_stage,
                geometry_stage=candidate_stage,
                threshold=threshold,
            )
            _update(
                counters["candidate_activity_candidate_geometry"],
                activity_stage=candidate_stage,
                geometry_stage=candidate_stage,
                threshold=threshold,
            )
        cells = {name: _finish(value) for name, value in counters.items()}
        baseline = float(cells["source_activity_source_geometry"]["f1"])
        activity_only = float(
            cells["candidate_activity_source_geometry"]["f1"]
        ) - baseline
        conditional_geometry_only = float(
            cells["source_activity_candidate_geometry"]["f1"]
        ) - baseline
        total = float(
            cells["candidate_activity_candidate_geometry"]["f1"]
        ) - baseline
        by_threshold[str(threshold)] = {
            "cells": cells,
            "effects": {
                "candidate_activity_on_source_geometry_delta_f1": activity_only,
                "candidate_geometry_on_source_activity_delta_f1": (
                    conditional_geometry_only
                ),
                "full_candidate_delta_f1": total,
                "interaction_delta_f1": (
                    total - activity_only - conditional_geometry_only
                ),
            },
        }

    payload = {
        "experiment": "V30 activity/count versus conditional route/geometry factorial audit",
        "diagnostic_only": True,
        "test_set_used": False,
        "source_cache": args.source_cache,
        "candidate_cache": args.candidate_cache,
        "images": len(pairs),
        "changed_activity_images": changed_activity_images,
        "changed_activity_slots": changed_activity_slots,
        "interpretation_contract": (
            "Slot identities are paired by fixed four-slot index. Geometry "
            "includes proposal routing and slot refinement; activity is only "
            "the final per-slot active mask."
        ),
        "thresholds": by_threshold,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
