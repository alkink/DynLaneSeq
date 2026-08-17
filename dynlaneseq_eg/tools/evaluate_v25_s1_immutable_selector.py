from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import eval_predictions
from dynlaneseq_eg.modeling.v25_s1_immutable_selector import (
    ImmutableBankSingleEditSelector,
)
from dynlaneseq_eg.tools.evaluate_v25_g1b_multi_path_capacity import (
    _model_lane_to_original,
)
from dynlaneseq_eg.tools.evaluate_v25_v7_top3_union_oracle import (
    _quantize_writer_lane,
    _write_lane_file,
)


POLICIES = ("source_v7", "treatment", "geometry_control", "wrong_image_evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed V25-S1 selectors on full official validation.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--treatment-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--metric-workers", type=int, default=20)
    parser.add_argument("--metric-chunksize", type=int, default=32)
    return parser.parse_args()


def _load_selector(path: str, device: torch.device) -> ImmutableBankSingleEditSelector:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ImmutableBankSingleEditSelector(hidden_dim=64, dropout=0.1)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.requires_grad_(False).eval().to(device)


def _feature_batch(arrays, indices: slice, *, wrong: bool = False):
    prefix = "wrong_" if wrong else ""
    return {
        "row_evidence": torch.from_numpy(np.asarray(arrays[f"{prefix}row_evidence"][indices])).float(),
        "row_geometry": torch.from_numpy(np.asarray(arrays["row_geometry"][indices])).float(),
        "global_evidence": torch.from_numpy(np.asarray(arrays[f"{prefix}global_evidence"][indices])).float(),
        "global_geometry": torch.from_numpy(np.asarray(arrays["global_geometry"][indices])).float(),
        "action_valid": torch.from_numpy(np.asarray(arrays["action_valid"][indices])),
    }


def _relative_prediction_path(meta: dict[str, Any]) -> Path:
    image = Path(str(meta["image_path"]))
    return Path(*image.parts[-3:]).with_suffix(".lines.txt")


def _selected_geometry(arrays, index: int, action: int):
    x = np.asarray(arrays["source_x"][index]).copy()
    ranges = np.asarray(arrays["source_range"][index]).copy()
    active = np.asarray(arrays["source_active"][index]).copy()
    if action > 0:
        edit = action - 1
        slot, path = divmod(edit, 3)
        if not bool(arrays["action_valid"][index, edit]):
            raise RuntimeError("selector emitted an invalid immutable-bank action")
        x[slot] = arrays["hypotheses"][index, slot, path]
        ranges[slot] = arrays["hypothesis_range"][index, slot]
    return x, ranges, active


def _write_policy(root: Path, arrays, metadata, actions: np.ndarray) -> int:
    predictions = 0
    for index, (meta, action) in enumerate(zip(metadata, actions)):
        x, ranges, active = _selected_geometry(arrays, index, int(action))
        lanes = []
        for slot in range(4):
            if not active[slot]:
                continue
            lane = _model_lane_to_original(
                torch.from_numpy(x[slot]), torch.from_numpy(ranges[slot]), meta
            )
            if len(lane) < 2:
                raise RuntimeError("a valid immutable-bank action changed V7 prediction count")
            lanes.append(_quantize_writer_lane(lane))
        _write_lane_file(root / _relative_prediction_path(meta), lanes)
        predictions += len(lanes)
    return predictions


def _outcome(arrays, actions: np.ndarray) -> dict[str, Any]:
    source50 = np.asarray(arrays["source_tp50"], dtype=np.int64)
    source75 = np.asarray(arrays["source_tp75"], dtype=np.int64)
    result50 = source50.copy()
    result75 = source75.copy()
    lost50 = np.zeros_like(source50)
    lost75 = np.zeros_like(source75)
    edit = actions > 0
    rows = np.nonzero(edit)[0]
    columns = actions[rows] - 1
    result50[rows] = arrays["action_tp50"][rows, columns]
    result75[rows] = arrays["action_tp75"][rows, columns]
    lost50[rows] = arrays["action_lost50"][rows, columns]
    lost75[rows] = arrays["action_lost75"][rows, columns]
    return {
        "images": int(actions.size),
        "edited_images": int(edit.sum()),
        "tp50_delta": int((result50 - source50).sum()),
        "tp75_delta": int((result75 - source75).sum()),
        "tp50_improved_images": int((result50 > source50).sum()),
        "tp50_worsened_images": int((result50 < source50).sum()),
        "tp75_improved_images": int((result75 > source75).sum()),
        "tp75_worsened_images": int((result75 < source75).sum()),
        "source_correct_tp50_lost": int(lost50.sum()),
        "source_correct_tp75_lost": int(lost75.sum()),
        "source_correct_tp50_total": int(source50.sum()),
        "source_correct_tp75_total": int(source75.sum()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cache = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    names = (
        "row_evidence", "row_geometry", "global_evidence", "global_geometry",
        "wrong_row_evidence", "wrong_global_evidence", "action_valid",
        "action_tp50", "action_tp75", "action_lost50", "action_lost75",
        "source_tp50", "source_tp75", "source_x", "source_range", "source_active",
        "hypotheses", "hypothesis_range",
    )
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}
    count = len(metadata)
    if any(int(value.shape[0]) != count for value in arrays.values()):
        raise ValueError("official validation cache population drifted")
    device = torch.device(args.device)
    treatment = _load_selector(args.treatment_checkpoint, device)
    control = _load_selector(args.control_checkpoint, device)
    actions = {
        "source_v7": np.zeros((count,), dtype=np.int64),
        "treatment": np.zeros((count,), dtype=np.int64),
        "geometry_control": np.zeros((count,), dtype=np.int64),
        "wrong_image_evidence": np.zeros((count,), dtype=np.int64),
    }
    started = time.perf_counter()
    for start in range(0, count, int(args.batch_size)):
        end = min(start + int(args.batch_size), count)
        section = slice(start, end)
        correct = {name: value.to(device, non_blocking=True) for name, value in _feature_batch(arrays, section).items()}
        wrong = {name: value.to(device, non_blocking=True) for name, value in _feature_batch(arrays, section, wrong=True).items()}
        actions["treatment"][section] = treatment(correct, evidence_enabled=True)["selected_action"].cpu().numpy()
        actions["geometry_control"][section] = control(correct, evidence_enabled=False)["selected_action"].cpu().numpy()
        actions["wrong_image_evidence"][section] = treatment(wrong, evidence_enabled=True)["selected_action"].cpu().numpy()

    prediction_roots = {policy: output_dir / "predictions" / policy for policy in POLICIES}
    prediction_counts = {
        policy: _write_policy(prediction_roots[policy], arrays, metadata, actions[policy])
        for policy in POLICIES
    }
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metrics = {
        policy: {
            str(threshold): values
            for threshold, values in eval_predictions(
                pred_dir=prediction_roots[policy],
                anno_dir=dataset_root,
                list_path=dataset_root / "list/val.txt",
                iou_thresholds=(0.50, 0.75),
                width=30,
                official=True,
                sequential=False,
                num_workers=args.metric_workers,
                chunksize=args.metric_chunksize,
            ).items()
        }
        for policy in POLICIES
    }
    outcomes = {policy: _outcome(arrays, actions[policy]) for policy in POLICIES}
    source50 = float(metrics["source_v7"]["0.5"]["F1"])
    source75 = float(metrics["source_v7"]["0.75"]["F1"])
    treatment50 = float(metrics["treatment"]["0.5"]["F1"])
    treatment75 = float(metrics["treatment"]["0.75"]["F1"])
    control50 = float(metrics["geometry_control"]["0.5"]["F1"])
    wrong50 = float(metrics["wrong_image_evidence"]["0.5"]["F1"])
    treatment_outcome = outcomes["treatment"]
    source_loss_rate = float(treatment_outcome["source_correct_tp50_lost"]) / float(
        max(treatment_outcome["source_correct_tp50_total"], 1)
    )
    checks = {
        "f1_50_at_least_v7_plus_0_80_points": 100.0 * (treatment50 - source50) >= 0.80,
        "f1_75_non_regression": treatment75 >= source75,
        "prediction_count_exact_v7": len(set(prediction_counts.values())) == 1,
        "source_correct_loss_below_1_percent": source_loss_rate < 0.01,
        "improved_images_at_least_twice_worsened": treatment_outcome["tp50_improved_images"] >= 2 * treatment_outcome["tp50_worsened_images"],
        "treatment_beats_geometry_control_by_0_30_points": 100.0 * (treatment50 - control50) >= 0.30,
        "correct_beats_wrong_evidence_by_0_25_points": 100.0 * (treatment50 - wrong50) >= 0.25,
    }
    report = {
        "experiment": "V25-S1 out-of-fold immutable-bank single-edit selector",
        "metrics": metrics,
        "outcomes": outcomes,
        "prediction_counts": prediction_counts,
        "selected_action_histograms": {
            policy: np.bincount(value, minlength=13).tolist() for policy, value in actions.items()
        },
        "gains_f1_points": {
            "treatment_vs_v7_50": 100.0 * (treatment50 - source50),
            "treatment_vs_v7_75": 100.0 * (treatment75 - source75),
            "treatment_vs_geometry_control_50": 100.0 * (treatment50 - control50),
            "treatment_vs_wrong_evidence_50": 100.0 * (treatment50 - wrong50),
        },
        "gate": {"checks": checks, "pass": all(checks.values())},
        "checkpoints": {
            "treatment": str(Path(args.treatment_checkpoint).resolve()),
            "treatment_sha256": sha256_file(args.treatment_checkpoint),
            "geometry_control": str(Path(args.control_checkpoint).resolve()),
            "geometry_control_sha256": sha256_file(args.control_checkpoint),
        },
        "contract": {
            "official_validation_rows": count,
            "validation_subset": False,
            "hard_categorical_selection": True,
            "coordinate_blending": False,
            "geometry_mutation": False,
            "activity_count": "exact V7",
            "checkpoint_selection": False,
            "threshold_selection": False,
            "test_set_used": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "v25_s1_official_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "gate": report["gate"], "gains": report["gains_f1_points"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
