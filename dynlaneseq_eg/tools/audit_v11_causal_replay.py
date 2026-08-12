from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
from typing import Any

import cv2
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
    official_proposal_gt_iou_matrix,
    sha256_file,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.train import seed_everything


COUNT_MODES = ("neural_active", "writer_valid")
PRIMARY_POLICIES = (
    "v7_geometry__v7_activity",
    "v11_coarse__v7_activity",
    "v11_final__v7_activity",
    "v7_geometry__v11_activity",
    "v11_final__v11_activity",
)

# x source, range source, activity source.  The names are intentionally
# explicit because this report is used to authorize or reject a new training
# graph; an ambiguous G0/G1 label is too easy to misread later.
FACTORIAL_AXES = OrderedDict(
    (
        ("v7_geometry__v7_activity", ("v7", "v7", "v7")),
        ("v7_geometry__v11_activity", ("v7", "v7", "v11")),
        ("v11_coarse__v7_activity", ("coarse", "coarse", "v7")),
        ("v11_coarse__v11_activity", ("coarse", "coarse", "v11")),
        ("v11_final__v7_activity", ("final", "final", "v7")),
        ("v11_final__v11_activity", ("final", "final", "v11")),
        ("v11_final_x_v7_range__v7_activity", ("final", "v7", "v7")),
        ("v11_final_x_v7_range__v11_activity", ("final", "v7", "v11")),
        ("v7_x_v11_final_range__v7_activity", ("v7", "final", "v7")),
        ("v7_x_v11_final_range__v11_activity", ("v7", "final", "v11")),
        (
            "v11_final_x_coarse_range__v7_activity",
            ("final", "coarse", "v7"),
        ),
        (
            "v11_final_x_coarse_range__v11_activity",
            ("final", "coarse", "v11"),
        ),
        (
            "v11_coarse_x_final_range__v7_activity",
            ("coarse", "final", "v7"),
        ),
        (
            "v11_coarse_x_final_range__v11_activity",
            ("coarse", "final", "v11"),
        ),
        ("v11_coarse_x_v7_range__v7_activity", ("coarse", "v7", "v7")),
        ("v11_coarse_x_v7_range__v11_activity", ("coarse", "v7", "v11")),
        ("v7_x_v11_coarse_range__v7_activity", ("v7", "coarse", "v7")),
        ("v7_x_v11_coarse_range__v11_activity", ("v7", "coarse", "v11")),
    )
)

EVIDENCE_VARIANTS = (
    "p2_zero_content",
    "p2_wrong_image",
    "p2_x_reversed",
    "p2_row_reversed",
    "proposal_tokens_zero",
    "proposal_tokens_wrong_image",
    "proposal_all_wrong_image",
    "legacy_route_prior_zero",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free V11 causal replay: V7/V11 geometry x activity, "
            "coarse-to-final, x/range factorials, and P2/proposal evidence "
            "interventions under the official raster IoU evaluator."
        )
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--v11-config", required=True)
    parser.add_argument("--v11-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--list-path", required=True)
    parser.add_argument(
        "--sample-strategy",
        choices=("sequential", "uniform"),
        default="sequential",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.75),
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--skip-evidence-ablations", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _prepare_config(path: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg.setdefault("dataset", {}).setdefault("lists", {})[args.split] = str(
        Path(args.list_path).expanduser().resolve()
    )
    dataloader = cfg.setdefault("dataloader", {})
    dataloader["eval_batch_size"] = int(args.eval_batch_size)
    dataloader["num_workers"] = int(args.num_workers)
    dataloader["persistent_workers"] = bool(int(args.num_workers) > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _plain_meta(meta: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "image_path",
        "anno_path",
        "orig_h",
        "orig_w",
        "input_h",
        "input_w",
        "scale_x",
        "scale_y",
        "crop_x",
        "crop_y",
    ):
        value = meta.get(key)
        if isinstance(value, torch.Tensor):
            value = (
                value.item()
                if value.numel() == 1
                else value.detach().cpu().tolist()
            )
        if value is not None:
            result[key] = (
                str(value) if key in {"image_path", "anno_path"} else value
            )
    return result


def _image_id(meta: dict[str, Any], fallback: str) -> str:
    value = meta.get("image_path")
    return str(value) if value is not None else fallback


def _max_batches(args: argparse.Namespace) -> int:
    if int(args.max_images) <= 0:
        return 0
    return math.ceil(int(args.max_images) / int(args.eval_batch_size))


def _loader(
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, list[int]]:
    loader = build_dataloader(cfg, split=args.split, training=False)
    return select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=_max_batches(args),
        num_workers=int(args.num_workers),
    )


def _inference(model: torch.nn.Module, images: torch.Tensor) -> dict[str, Any]:
    if bool(getattr(model, "supports_inference_only", False)):
        return model(images, inference_only=True)
    return model(images)


def _required_output(
    outputs: dict[str, Any],
    name: str,
) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"required model output is unavailable: {name}")
    return value


@torch.no_grad()
def _collect_source(
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[int], int]:
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.source_checkpoint, model, strict=False))
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    model.eval()
    loader, sampled_indices = _loader(cfg, args)
    records: list[dict[str, Any]] = []
    fields = (
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
        "selection_slot_active",
        "selection_slot_active_logits",
        "selection_slot_geometry_route_indices",
        "selection_slot_real_route_logits",
    )
    for batch_index, (images, _targets, metas) in enumerate(
        tqdm(loader, desc="V7 source replay", ncols=90)
    ):
        images = images.to(device, non_blocking=True)
        outputs = _inference(model, images)
        cpu = {
            name: _required_output(outputs, name).detach().float().cpu()
            if _required_output(outputs, name).dtype.is_floating_point
            else _required_output(outputs, name).detach().cpu()
            for name in fields
        }
        remaining = (
            len(metas)
            if int(args.max_images) <= 0
            else max(int(args.max_images) - len(records), 0)
        )
        take = min(len(metas), remaining)
        for bi, meta in enumerate(metas[:take]):
            records.append(
                {
                    "image_id": _image_id(
                        meta,
                        f"source_{batch_index:06d}_{bi}",
                    ),
                    "meta": _plain_meta(meta),
                    "outputs": {name: value[bi].clone() for name, value in cpu.items()},
                }
            )
        if int(args.max_images) > 0 and len(records) >= int(args.max_images):
            break
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if int(args.max_images) > 0:
        sampled_indices = sampled_indices[: len(records)]
    return records, sampled_indices, iteration


def _roll_batch(value: torch.Tensor) -> torch.Tensor:
    if int(value.shape[0]) < 2:
        raise ValueError("wrong-image interventions require batch size >= 2")
    return value.roll(shifts=1, dims=0)


def _variant_kwargs(
    captured: dict[str, torch.Tensor],
    variant: str,
) -> dict[str, torch.Tensor]:
    kwargs = dict(captured)
    if variant == "p2_zero_content":
        kwargs["row_value_features"] = torch.zeros_like(
            captured["row_value_features"]
        )
    elif variant == "p2_wrong_image":
        kwargs["row_value_features"] = _roll_batch(
            captured["row_value_features"]
        )
    elif variant == "p2_x_reversed":
        kwargs["row_value_features"] = captured["row_value_features"].flip(2)
    elif variant == "p2_row_reversed":
        kwargs["row_value_features"] = captured["row_value_features"].flip(1)
    elif variant == "proposal_tokens_zero":
        kwargs["proposal_row_tokens"] = torch.zeros_like(
            captured["proposal_row_tokens"]
        )
    elif variant == "proposal_tokens_wrong_image":
        kwargs["proposal_row_tokens"] = _roll_batch(
            captured["proposal_row_tokens"]
        )
    elif variant == "proposal_all_wrong_image":
        for name in (
            "proposal_row_tokens",
            "proposal_x_rows",
            "proposal_range_norm",
            "legacy_route_logits",
            "candidate_valid",
        ):
            kwargs[name] = _roll_batch(captured[name])
    elif variant == "legacy_route_prior_zero":
        kwargs["legacy_route_logits"] = torch.zeros_like(
            captured["legacy_route_logits"]
        )
    else:
        raise KeyError(f"unknown evidence intervention: {variant}")
    return kwargs


def _factorial_policies(
    source: dict[str, torch.Tensor],
    v11: dict[str, torch.Tensor],
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    x_values = {
        "v7": source["selection_slot_pred_x_rows"],
        "coarse": v11["selection_slot_unified_base_x_rows"],
        "final": v11["selection_slot_pred_x_rows"],
    }
    range_values = {
        "v7": source["selection_slot_range_norm"],
        "coarse": v11["selection_slot_unified_base_range_norm"],
        "final": v11["selection_slot_range_norm"],
    }
    activity_values = {
        "v7": source["selection_slot_active"].bool(),
        "v11": v11["selection_slot_active"].bool(),
    }
    return OrderedDict(
        (
            name,
            {
                "x": x_values[x_name],
                "range": range_values[range_name],
                "active": activity_values[activity_name],
            },
        )
        for name, (x_name, range_name, activity_name) in FACTORIAL_AXES.items()
    )


def _new_metric_tree(
    policies: tuple[str, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    return {
        policy: {
            mode: {
                "images": 0,
                "count_exact": 0,
                "count_under": 0,
                "count_over": 0,
                "count_absolute_error": 0,
                "thresholds": {
                    f"{threshold:.2f}": {
                        "tp": 0,
                        "predictions": 0,
                        "gt": 0,
                    }
                    for threshold in thresholds
                },
            }
            for mode in COUNT_MODES
        }
        for policy in policies
    }


def _metric_row(tp: int, predictions: int, gt: int) -> dict[str, int | float]:
    fp = int(predictions) - int(tp)
    fn = int(gt) - int(tp)
    precision = float(tp) / float(max(int(predictions), 1))
    recall = float(tp) / float(max(int(gt), 1))
    denominator = 2 * int(tp) + fp + fn
    return {
        "tp": int(tp),
        "fp": fp,
        "fn": fn,
        "predictions": int(predictions),
        "gt": int(gt),
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if denominator <= 0 else float(2 * int(tp)) / denominator,
    }


def _finalize_metric_tree(tree: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy, policy_data in tree.items():
        result[policy] = {}
        for mode, mode_data in policy_data.items():
            images = int(mode_data["images"])
            result[policy][mode] = {
                "cardinality": {
                    "images": images,
                    "exact": int(mode_data["count_exact"]),
                    "exact_fraction": float(mode_data["count_exact"])
                    / float(max(images, 1)),
                    "under": int(mode_data["count_under"]),
                    "over": int(mode_data["count_over"]),
                    "mae": float(mode_data["count_absolute_error"])
                    / float(max(images, 1)),
                },
                "thresholds": {
                    threshold: _metric_row(
                        int(row["tp"]),
                        int(row["predictions"]),
                        int(row["gt"]),
                    )
                    for threshold, row in mode_data["thresholds"].items()
                },
            }
    return result


def _accumulate_record(
    tree: dict[str, Any],
    evaluated: dict[str, torch.Tensor],
    layout: dict[str, tuple[int, int]],
    active_by_policy: dict[str, torch.Tensor],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    quality = evaluated["quality"].float()
    valid = evaluated["valid"].bool()
    gt_count = int(quality.shape[0])
    primary: dict[str, Any] = {}
    for policy, (start, stop) in layout.items():
        local_quality = quality[:, start:stop]
        local_valid = valid[start:stop]
        active = active_by_policy[policy].bool()
        for mode in COUNT_MODES:
            selected = active if mode == "neural_active" else active & local_valid
            selected_ids = torch.nonzero(
                selected,
                as_tuple=False,
            ).flatten().tolist()
            prediction_count = len(selected_ids)
            count_error = prediction_count - gt_count
            mode_row = tree[policy][mode]
            mode_row["images"] += 1
            mode_row["count_exact"] += int(count_error == 0)
            mode_row["count_under"] += int(count_error < 0)
            mode_row["count_over"] += int(count_error > 0)
            mode_row["count_absolute_error"] += abs(count_error)
            for threshold in thresholds:
                assignment = evaluator_hungarian_assignment(
                    local_quality,
                    selected_ids,
                    threshold=float(threshold),
                )
                row = mode_row["thresholds"][f"{threshold:.2f}"]
                row["tp"] += int(assignment.hit_count)
                row["predictions"] += prediction_count
                row["gt"] += gt_count
                if policy in PRIMARY_POLICIES and mode == "writer_valid":
                    primary.setdefault(policy, {})[f"{threshold:.2f}"] = {
                        "tp": int(assignment.hit_count),
                        "predictions": prediction_count,
                        "gt": gt_count,
                    }
    return primary


def _evaluate_records(
    records: list[dict[str, Any]],
    *,
    line_width: float,
    min_valid_rows: int,
    workers: int,
) -> list[dict[str, torch.Tensor]]:
    def evaluate(item: dict[str, Any]) -> dict[str, torch.Tensor]:
        matrix, valid = official_proposal_gt_iou_matrix(
            item["record"],
            "combined",
            line_width=float(line_width),
            min_valid_rows=int(min_valid_rows),
            row_visibility_thresh=0.0,
        )
        return {"quality": matrix, "valid": valid}

    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        count = max(int(workers), 1)
        if count == 1:
            return [
                evaluate(item)
                for item in tqdm(records, desc="official causal replay", ncols=90)
            ]
        with ThreadPoolExecutor(max_workers=count) as executor:
            return list(
                tqdm(
                    executor.map(evaluate, records),
                    total=len(records),
                    desc="official causal replay",
                    ncols=90,
                )
            )
    finally:
        cv2.setNumThreads(previous_threads)


def _mean(values: list[float]) -> float:
    return float(sum(values)) / float(max(len(values), 1))


def _metric_delta(
    metrics: dict[str, Any],
    treatment: str,
    control: str,
    mode: str,
    threshold: str,
) -> dict[str, float | int]:
    lhs = metrics[treatment][mode]["thresholds"][threshold]
    rhs = metrics[control][mode]["thresholds"][threshold]
    return {
        "delta_tp": int(lhs["tp"]) - int(rhs["tp"]),
        "delta_fp": int(lhs["fp"]) - int(rhs["fp"]),
        "delta_fn": int(lhs["fn"]) - int(rhs["fn"]),
        "delta_predictions": int(lhs["predictions"])
        - int(rhs["predictions"]),
        "delta_f1": float(lhs["f1"]) - float(rhs["f1"]),
        "delta_f1_points": 100.0 * (float(lhs["f1"]) - float(rhs["f1"])),
    }


def _factorial_decomposition(
    metrics: dict[str, Any],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    names = {
        "g0a0": "v7_geometry__v7_activity",
        "g1a0": "v11_final__v7_activity",
        "g0a1": "v7_geometry__v11_activity",
        "g1a1": "v11_final__v11_activity",
    }
    result: dict[str, Any] = {}
    for mode in COUNT_MODES:
        result[mode] = {}
        for value in thresholds:
            threshold = f"{value:.2f}"
            rows = {
                key: metrics[name][mode]["thresholds"][threshold]
                for key, name in names.items()
            }
            decomposition: dict[str, Any] = {}
            for field in ("tp", "predictions", "f1"):
                values = {key: float(row[field]) for key, row in rows.items()}
                geometry = 0.5 * (
                    (values["g1a0"] - values["g0a0"])
                    + (values["g1a1"] - values["g0a1"])
                )
                activity = 0.5 * (
                    (values["g0a1"] - values["g0a0"])
                    + (values["g1a1"] - values["g1a0"])
                )
                interaction = (
                    values["g1a1"]
                    - values["g1a0"]
                    - values["g0a1"]
                    + values["g0a0"]
                )
                scale = 100.0 if field == "f1" else 1.0
                decomposition[field] = {
                    "geometry_shapley": scale * geometry,
                    "activity_shapley": scale * activity,
                    "interaction": scale * interaction,
                    "joint_delta": scale
                    * (values["g1a1"] - values["g0a0"]),
                    "geometry_at_v7_activity": scale
                    * (values["g1a0"] - values["g0a0"]),
                    "activity_at_v7_geometry": scale
                    * (values["g0a1"] - values["g0a0"]),
                    "activity_at_v11_geometry": scale
                    * (values["g1a1"] - values["g1a0"]),
                }
            result[mode][threshold] = decomposition
    return result


def _named_deltas(
    metrics: dict[str, Any],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    comparisons = OrderedDict(
        (
            (
                "v11_final_fixed_activity_minus_v7",
                ("v11_final__v7_activity", "v7_geometry__v7_activity"),
            ),
            (
                "v11_activity_on_v7_geometry_minus_v7",
                ("v7_geometry__v11_activity", "v7_geometry__v7_activity"),
            ),
            (
                "joint_v11_minus_v7",
                ("v11_final__v11_activity", "v7_geometry__v7_activity"),
            ),
            (
                "v11_coarse_fixed_activity_minus_v7",
                ("v11_coarse__v7_activity", "v7_geometry__v7_activity"),
            ),
            (
                "v11_final_minus_coarse_fixed_activity",
                ("v11_final__v7_activity", "v11_coarse__v7_activity"),
            ),
            (
                "v11_final_x_only_fixed_activity_minus_v7",
                (
                    "v11_final_x_v7_range__v7_activity",
                    "v7_geometry__v7_activity",
                ),
            ),
            (
                "v11_final_range_only_fixed_activity_minus_v7",
                (
                    "v7_x_v11_final_range__v7_activity",
                    "v7_geometry__v7_activity",
                ),
            ),
        )
    )
    result: dict[str, Any] = {}
    for label, (treatment, control) in comparisons.items():
        result[label] = {
            mode: {
                f"{threshold:.2f}": _metric_delta(
                    metrics,
                    treatment,
                    control,
                    mode,
                    f"{threshold:.2f}",
                )
                for threshold in thresholds
            }
            for mode in COUNT_MODES
        }
    return result


def _single_domain_diagnosis(
    metrics: dict[str, Any],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    mode = "writer_valid"
    fixed_geometry = []
    activity = []
    for value in thresholds:
        threshold = f"{value:.2f}"
        fixed_geometry.append(
            _metric_delta(
                metrics,
                "v11_final__v7_activity",
                "v7_geometry__v7_activity",
                mode,
                threshold,
            )["delta_f1_points"]
        )
        activity.append(
            _metric_delta(
                metrics,
                "v11_final__v11_activity",
                "v11_final__v7_activity",
                mode,
                threshold,
            )["delta_f1_points"]
        )
    if all(float(value) >= 0.0 for value in fixed_geometry):
        geometry_status = "positive_with_v7_activity"
    elif all(float(value) < 0.0 for value in fixed_geometry):
        geometry_status = "negative_even_with_v7_activity"
    else:
        geometry_status = "mixed_across_iou_thresholds"
    if all(float(value) >= 0.0 for value in activity):
        activity_status = "nonnegative"
    elif all(float(value) < 0.0 for value in activity):
        activity_status = "harmful"
    else:
        activity_status = "mixed"
    return {
        "geometry_status": geometry_status,
        "activity_status": activity_status,
        "fixed_activity_geometry_delta_f1_points": fixed_geometry,
        "learned_activity_delta_f1_points_on_v11_geometry": activity,
        "training_authorized": False,
        "reason": (
            "This is a causal autopsy of the already-trained cold-start V11, "
            "not a V11.1 generalization gate. Combine held-out-clip and "
            "validation reports before authorizing a parity-anchored arm."
        ),
    }


@torch.no_grad()
def _collect_v11_and_policies(
    cfg: dict[str, Any],
    source_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[int], int, dict[str, Any]]:
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.v11_checkpoint, model, strict=False))
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    model.eval()
    selector = model.structured_query_head.set_selection_head
    decoder = getattr(selector, "unified_slot_decoder", None)
    if decoder is None:
        raise ValueError("V11 causal replay requires unified_slot_decoder")
    loader, sampled_indices = _loader(cfg, args)
    records: list[dict[str, Any]] = []
    route_mismatches = 0
    route_logit_max_abs = 0.0
    legacy_activity_reconstruction_max_abs = 0.0
    decoder_replay_max_abs = 0.0
    ablation_diagnostics: dict[str, dict[str, list[float]]] = {
        name: {
            "x_shift_from_correct_px": [],
            "range_shift_from_correct": [],
            "proposal_entropy": [],
            "visual_entropy": [],
        }
        for name in ("correct", *EVIDENCE_VARIANTS)
    }
    source_cursor = 0
    for batch_index, (images, _targets, metas) in enumerate(
        tqdm(loader, desc="V11 causal replay", ncols=90)
    ):
        images = images.to(device, non_blocking=True)
        captured: dict[str, torch.Tensor] = {}

        def capture(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            captured.update(kwargs)

        handle = decoder.register_forward_pre_hook(capture, with_kwargs=True)
        outputs = _inference(model, images)
        handle.remove()
        if not captured:
            raise RuntimeError("unified decoder pre-hook captured no inputs")
        correct_replay = decoder(**captured)
        decoder_replay_max_abs = max(
            decoder_replay_max_abs,
            float(
                (
                    correct_replay["selection_slot_pred_x_rows"]
                    - _required_output(outputs, "selection_slot_pred_x_rows")
                )
                .abs()
                .max()
                .cpu()
            ),
            float(
                (
                    correct_replay["selection_slot_range_norm"]
                    - _required_output(outputs, "selection_slot_range_norm")
                )
                .abs()
                .max()
                .cpu()
            ),
        )
        correct_proposal_attention = correct_replay[
            "selection_slot_unified_proposal_attention"
        ].float()
        correct_proposal_entropy = -(
            correct_proposal_attention
            * correct_proposal_attention.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        correct_visual_attention = correct_replay[
            "selection_slot_unified_visual_attention"
        ].float()
        correct_visual_entropy = -(
            correct_visual_attention
            * correct_visual_attention.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        ablation_diagnostics["correct"]["x_shift_from_correct_px"].append(0.0)
        ablation_diagnostics["correct"]["range_shift_from_correct"].append(0.0)
        ablation_diagnostics["correct"]["proposal_entropy"].append(
            float(correct_proposal_entropy.mean().cpu())
        )
        ablation_diagnostics["correct"]["visual_entropy"].append(
            float(correct_visual_entropy.mean().cpu())
        )
        variants: dict[str, dict[str, torch.Tensor]] = {}
        if not args.skip_evidence_ablations:
            for name in EVIDENCE_VARIANTS:
                if "wrong_image" in name and int(images.shape[0]) < 2:
                    continue
                value = decoder(**_variant_kwargs(captured, name))
                variants[name] = value
                ablation_diagnostics[name]["x_shift_from_correct_px"].append(
                    float(
                        (
                            value["selection_slot_pred_x_rows"]
                            - correct_replay["selection_slot_pred_x_rows"]
                        )
                        .abs()
                        .mean()
                        .cpu()
                    )
                )
                ablation_diagnostics[name][
                    "range_shift_from_correct"
                ].append(
                    float(
                        (
                            value["selection_slot_range_norm"]
                            - correct_replay["selection_slot_range_norm"]
                        )
                        .abs()
                        .mean()
                        .cpu()
                    )
                )
                attention = value[
                    "selection_slot_unified_proposal_attention"
                ].float()
                entropy = -(
                    attention * attention.clamp_min(1.0e-12).log()
                ).sum(dim=-1)
                ablation_diagnostics[name]["proposal_entropy"].append(
                    float(entropy.mean().cpu())
                )
                visual_attention = value[
                    "selection_slot_unified_visual_attention"
                ].float()
                visual_entropy = -(
                    visual_attention
                    * visual_attention.clamp_min(1.0e-12).log()
                ).sum(dim=-1)
                ablation_diagnostics[name]["visual_entropy"].append(
                    float(visual_entropy.mean().cpu())
                )

        batch_size = int(images.shape[0])
        take = min(batch_size, len(source_records) - source_cursor)
        if take <= 0:
            break
        for bi, meta in enumerate(metas[:take]):
            source_record = source_records[source_cursor + bi]
            image_id = _image_id(meta, f"v11_{batch_index:06d}_{bi}")
            if image_id != source_record["image_id"]:
                raise ValueError(
                    "source/V11 image order mismatch: "
                    f"{source_record['image_id']} != {image_id}"
                )
            source = source_record["outputs"]
            v11 = {
                "selection_slot_pred_x_rows": _required_output(
                    outputs,
                    "selection_slot_pred_x_rows",
                )[bi].detach().float().cpu(),
                "selection_slot_range_norm": _required_output(
                    outputs,
                    "selection_slot_range_norm",
                )[bi].detach().float().cpu(),
                "selection_slot_active": _required_output(
                    outputs,
                    "selection_slot_active",
                )[bi].detach().cpu().bool(),
                "selection_slot_active_logits": _required_output(
                    outputs,
                    "selection_slot_active_logits",
                )[bi].detach().float().cpu(),
                "selection_slot_unified_activity_residual": _required_output(
                    outputs,
                    "selection_slot_unified_activity_residual",
                )[bi].detach().float().cpu(),
                "selection_slot_unified_base_x_rows": _required_output(
                    outputs,
                    "selection_slot_unified_base_x_rows",
                )[bi].detach().float().cpu(),
                "selection_slot_unified_base_range_norm": _required_output(
                    outputs,
                    "selection_slot_unified_base_range_norm",
                )[bi].detach().float().cpu(),
            }
            current_routes = _required_output(
                outputs,
                "selection_slot_geometry_route_indices",
            )[bi].detach().cpu().long()
            route_mismatches += int(
                (current_routes != source["selection_slot_geometry_route_indices"])
                .sum()
                .item()
            )
            route_logit_max_abs = max(
                route_logit_max_abs,
                float(
                    (
                        _required_output(
                            outputs,
                            "selection_slot_real_route_logits",
                        )[bi]
                        .detach()
                        .float()
                        .cpu()
                        - source["selection_slot_real_route_logits"].float()
                    )
                    .abs()
                    .max()
                ),
            )
            reconstructed_legacy_activity = (
                v11["selection_slot_active_logits"]
                - v11["selection_slot_unified_activity_residual"]
            )
            legacy_activity_reconstruction_max_abs = max(
                legacy_activity_reconstruction_max_abs,
                float(
                    (
                        reconstructed_legacy_activity
                        - source["selection_slot_active_logits"].float()
                    )
                    .abs()
                    .max()
                ),
            )
            policies = _factorial_policies(source, v11)
            for name, value in variants.items():
                policies[f"ablation_{name}__v7_activity"] = {
                    "x": value["selection_slot_pred_x_rows"][bi]
                    .detach()
                    .float()
                    .cpu(),
                    "range": value["selection_slot_range_norm"][bi]
                    .detach()
                    .float()
                    .cpu(),
                    "active": source["selection_slot_active"].bool(),
                }
            geometry: list[torch.Tensor] = []
            ranges: list[torch.Tensor] = []
            layout: dict[str, tuple[int, int]] = {}
            active_by_policy: dict[str, torch.Tensor] = {}
            cursor = 0
            for name, policy in policies.items():
                x = policy["x"]
                lane_range = policy["range"]
                if int(x.shape[0]) != int(lane_range.shape[0]):
                    raise ValueError(f"policy lane count mismatch: {name}")
                geometry.append(x)
                ranges.append(lane_range)
                layout[name] = (cursor, cursor + int(x.shape[0]))
                active_by_policy[name] = policy["active"].bool()
                cursor += int(x.shape[0])
            records.append(
                {
                    "image_id": image_id,
                    "record": {
                        "meta": source_record["meta"],
                        "stages": {
                            "combined": {
                                "pred_x_rows": torch.cat(geometry, dim=0),
                                "range_norm": torch.cat(ranges, dim=0),
                            }
                        },
                    },
                    "layout": layout,
                    "active_by_policy": active_by_policy,
                }
            )
        source_cursor += take
        if source_cursor >= len(source_records):
            break
    if int(args.max_images) > 0:
        sampled_indices = sampled_indices[: len(records)]
    if len(records) != len(source_records):
        raise ValueError(
            f"source/V11 image count mismatch: {len(source_records)} vs {len(records)}"
        )
    diagnostics = {
        "route_index_mismatch_count": route_mismatches,
        "route_logit_max_abs_error": route_logit_max_abs,
        "legacy_activity_reconstruction_max_abs_error": (
            legacy_activity_reconstruction_max_abs
        ),
        "decoder_replay_max_abs_error": decoder_replay_max_abs,
        "slot_alignment_passed": route_mismatches == 0
        and route_logit_max_abs <= 1.0e-4
        and legacy_activity_reconstruction_max_abs <= 1.0e-4,
        "evidence_ablation_tensor_effects": {
            name: {metric: _mean(values) for metric, values in rows.items()}
            for name, rows in ablation_diagnostics.items()
            if any(rows.values())
        },
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, sampled_indices, iteration, diagnostics


def main() -> None:
    args = parse_args()
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    if not thresholds:
        raise ValueError("at least one IoU threshold is required")
    if int(args.eval_batch_size) < 2 and not args.skip_evidence_ablations:
        raise ValueError("evidence ablations require eval batch size >= 2")
    list_path = Path(args.list_path).expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f"evaluation list does not exist: {list_path}")
    source_cfg = _prepare_config(args.source_config, args)
    v11_cfg = _prepare_config(args.v11_config, args)
    seed = int(v11_cfg.get("training", {}).get("seed", 3407))
    seed_everything(seed)
    source_records, source_indices, source_iteration = _collect_source(
        source_cfg,
        args,
    )
    seed_everything(seed)
    records, v11_indices, v11_iteration, diagnostics = (
        _collect_v11_and_policies(
            v11_cfg,
            source_records,
            args,
        )
    )
    if source_indices != v11_indices:
        raise ValueError("source/V11 diagnostic sample indices differ")
    if not diagnostics["slot_alignment_passed"]:
        raise RuntimeError(
            "V7/V11 slot alignment failed; factorial replay would be invalid"
        )
    evaluated = _evaluate_records(
        records,
        line_width=float(args.line_width),
        min_valid_rows=int(args.min_valid_rows),
        workers=int(args.metric_workers),
    )
    policy_names = tuple(records[0]["layout"]) if records else tuple()
    tree = _new_metric_tree(policy_names, thresholds)
    per_image_primary: list[dict[str, Any]] = []
    for item, result in zip(records, evaluated):
        primary = _accumulate_record(
            tree,
            result,
            item["layout"],
            item["active_by_policy"],
            thresholds,
        )
        per_image_primary.append(
            {"image_id": item["image_id"], "policies": primary}
        )
    metrics = _finalize_metric_tree(tree)
    report = {
        "experiment": "V11 training-free causal replay",
        "source_config": str(Path(args.source_config).expanduser().resolve()),
        "source_checkpoint": str(
            Path(args.source_checkpoint).expanduser().resolve()
        ),
        "source_iteration": source_iteration,
        "v11_config": str(Path(args.v11_config).expanduser().resolve()),
        "v11_checkpoint": str(Path(args.v11_checkpoint).expanduser().resolve()),
        "v11_iteration": v11_iteration,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "split": args.split,
        "list_path": str(list_path),
        "list_sha256": sha256_file(list_path),
        "sample_strategy": args.sample_strategy,
        "sampled_indices": source_indices,
        "images": len(records),
        "thresholds": list(thresholds),
        "line_width": float(args.line_width),
        "min_valid_rows": int(args.min_valid_rows),
        "test_set_used": False,
        "optimizer_steps": 0,
        "evidence_ablations_run": not args.skip_evidence_ablations,
        "alignment_and_replay_contract": diagnostics,
        "policy_axes": {
            name: {
                "x": axes[0],
                "range": axes[1],
                "activity": axes[2],
            }
            for name, axes in FACTORIAL_AXES.items()
        },
        "metrics": metrics,
        "named_deltas": _named_deltas(metrics, thresholds),
        "geometry_activity_shapley": _factorial_decomposition(
            metrics,
            thresholds,
        ),
        "diagnosis": _single_domain_diagnosis(metrics, thresholds),
        "per_image_primary": per_image_primary,
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output.resolve()),
                "images": len(records),
                "source_iteration": source_iteration,
                "v11_iteration": v11_iteration,
                "slot_alignment_passed": diagnostics[
                    "slot_alignment_passed"
                ],
                "diagnosis": report["diagnosis"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
