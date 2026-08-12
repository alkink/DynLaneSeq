from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
from itertools import permutations
import json
import math
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    official_proposal_gt_iou_matrix,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.losses.loss_s0 import build_four_slot_cluster_targets
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.train import seed_everything


POLICIES = (
    "current_hard",
    "predicted_soft",
    "target_soft",
    "target_hard",
)
STAGES = ("reference", "refined")
COUNT_MODES = ("fixed_neural_active", "writer_valid")
ERROR_CLASSES = (
    "CORRECT_ID",
    "IN_SUPPORT_WRONG_MEMBER",
    "OUTSIDE_SUPPORT",
    "PATH_OR_ACTIVITY_ERROR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free V7 route-support decomposition, hard/soft reference "
            "counterfactuals, forced existing-refiner replay, and optional V8 "
            "forced-global-mix audit."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=(0.5, 0.75))
    parser.add_argument("--v8-config", default="")
    parser.add_argument("--v8-checkpoint", default="")
    parser.add_argument(
        "--forced-mixes",
        type=float,
        nargs="+",
        default=(0.0, 0.0188, 0.05, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _prepare_config(path: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root).expanduser())
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(int(args.num_workers) > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _plain_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
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
            value = value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        if value is not None:
            out[key] = str(value) if key in {"image_path", "anno_path"} else value
    return out


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "median": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
    }


def _f1(tp: int, predictions: int, gt: int) -> dict[str, float | int]:
    fp = int(predictions) - int(tp)
    fn = int(gt) - int(tp)
    precision = float(tp) / float(max(predictions, 1))
    recall = float(tp) / float(max(gt, 1))
    denominator = 2 * int(tp) + fp + fn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "predictions": int(predictions),
        "gt": int(gt),
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if denominator <= 0 else float(2 * int(tp)) / float(denominator),
    }


def _hungarian_hits(quality: torch.Tensor, threshold: float) -> int:
    if int(quality.shape[0]) == 0 or int(quality.shape[1]) == 0:
        return 0
    gt_ids, pred_ids = linear_sum_assignment(1.0 - quality.detach().cpu().numpy())
    return int((quality[gt_ids, pred_ids] >= float(threshold)).sum())


def _hard_min_slots(
    active_logits: torch.Tensor,
    route_logits: torch.Tensor,
    targets: torch.Tensor,
) -> list[int]:
    """Return the exact detached hard-min slot index for every GT row."""

    slots = int(active_logits.numel())
    gt_count = int(targets.shape[0])
    if gt_count == 0:
        return []
    log_probability = F.log_softmax(route_logits.float(), dim=-1)
    active_cost = F.softplus(-active_logits.float())
    inactive_cost = F.softplus(active_logits.float())
    candidate_cost = -torch.einsum("sn,gn->sg", log_probability, targets.float())
    candidate_cost = candidate_cost + active_cost.unsqueeze(-1)
    best_cost = math.inf
    best_slots: tuple[int, ...] | None = None
    for assigned_slots in permutations(range(slots), gt_count):
        assigned = set(int(value) for value in assigned_slots)
        cost = sum(
            float(candidate_cost[int(slot), gt].detach())
            for gt, slot in enumerate(assigned_slots)
        )
        cost += sum(
            float(inactive_cost[slot].detach())
            for slot in range(slots)
            if slot not in assigned
        )
        if cost < best_cost:
            best_cost = cost
            best_slots = tuple(int(value) for value in assigned_slots)
    if best_slots is None:
        raise RuntimeError("hard-min slot assignment has no path")
    return list(best_slots)


def _target_hard_unique_routes(
    target_rows: torch.Tensor,
    slots_for_gt: list[int],
    current_routes: torch.Tensor,
    current_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> tuple[torch.Tensor, list[int], int]:
    """Prioritize one unique target member per GT, then fill unused slots."""

    routes = current_routes.clone().long()
    gt_count, candidates = target_rows.shape
    target_argmax = target_rows.argmax(dim=-1).tolist() if gt_count else []
    raw_collision = len(target_argmax) - len(set(int(v) for v in target_argmax))
    if gt_count:
        valid_ids = torch.nonzero(candidate_valid.bool(), as_tuple=False).flatten()
        score = target_rows[:, valid_ids].clamp_min(1.0e-30).log()
        gt_ids, local_candidate_ids = linear_sum_assignment(-score.detach().cpu().numpy())
        chosen: dict[int, int] = {
            int(gt): int(valid_ids[int(local)])
            for gt, local in zip(gt_ids.tolist(), local_candidate_ids.tolist())
        }
    else:
        chosen = {}
    reserved: set[int] = set()
    assigned_slots = set(int(value) for value in slots_for_gt)
    for gt, slot in enumerate(slots_for_gt):
        candidate = int(chosen[gt])
        routes[int(slot)] = candidate
        reserved.add(candidate)
    for slot in range(int(routes.numel())):
        if slot in assigned_slots:
            continue
        order = current_logits[slot].argsort(descending=True)
        replacement = next(
            (
                int(candidate)
                for candidate in order.tolist()
                if bool(candidate_valid[int(candidate)]) and int(candidate) not in reserved
            ),
            int(routes[slot]),
        )
        routes[slot] = replacement
        reserved.add(replacement)
    return routes, [int(value) for value in target_argmax], int(raw_collision)


def _target_soft_weights(
    target_rows: torch.Tensor,
    slots_for_gt: list[int],
    fallback_routes: torch.Tensor,
    candidates: int,
) -> torch.Tensor:
    weight = target_rows.new_zeros((int(fallback_routes.numel()), candidates))
    weight.scatter_(1, fallback_routes.clamp(min=0).unsqueeze(-1), 1.0)
    for gt, slot in enumerate(slots_for_gt):
        weight[int(slot)] = target_rows[gt]
    return weight


@contextmanager
def _refiner_mode(
    refiner: torch.nn.Module,
    *,
    reference_mode: str,
    structured_unique: bool | None = None,
) -> Iterator[None]:
    old_mode = str(refiner.reference_mode)
    old_structured = bool(refiner.structured_unique_routing)
    refiner.reference_mode = str(reference_mode)
    if structured_unique is not None:
        refiner.structured_unique_routing = bool(structured_unique)
    try:
        yield
    finally:
        refiner.reference_mode = old_mode
        refiner.structured_unique_routing = old_structured


def _run_refiner(
    refiner: torch.nn.Module,
    captured: dict[str, torch.Tensor],
    *,
    reference_mode: str,
    route_indices: torch.Tensor | None = None,
    route_logits: torch.Tensor | None = None,
    structured_unique: bool | None = None,
) -> dict[str, torch.Tensor]:
    kwargs = dict(captured)
    if route_indices is not None:
        kwargs["route_indices"] = route_indices
    if route_logits is not None:
        kwargs["route_logits"] = route_logits
    with _refiner_mode(
        refiner,
        reference_mode=reference_mode,
        structured_unique=structured_unique,
    ):
        return refiner(**kwargs)


def _metric_template(
    policies: tuple[str, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    return {
        policy: {
            stage: {
                count_mode: {
                    f"{threshold:.2f}": {"tp": 0, "predictions": 0, "gt": 0}
                    for threshold in thresholds
                }
                for count_mode in COUNT_MODES
            }
            for stage in STAGES
        }
        for policy in policies
    }


def _evaluate_combined_record(
    record: dict[str, Any],
    *,
    thresholds: tuple[float, ...],
    line_width: float,
    min_valid_rows: int,
) -> dict[str, Any]:
    matrix, valid = official_proposal_gt_iou_matrix(
        record,
        "combined",
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=0.0,
    )
    return {"quality": matrix, "valid": valid}


def _accumulate_policy_metrics(
    aggregate: dict[str, Any],
    evaluated: dict[str, Any],
    layout: dict[str, tuple[int, int]],
    active: torch.Tensor,
    *,
    thresholds: tuple[float, ...],
) -> None:
    quality = evaluated["quality"]
    valid = evaluated["valid"].bool()
    gt_count = int(quality.shape[0])
    for key, (start, stop) in layout.items():
        policy, stage = key.split("/", 1)
        local_quality = quality[:, start:stop]
        local_valid = valid[start:stop]
        for count_mode in COUNT_MODES:
            selected = active.bool().clone()
            if count_mode == "writer_valid":
                selected &= local_valid
            selected_quality = local_quality[:, selected]
            prediction_count = int(selected.sum())
            for threshold in thresholds:
                row = aggregate[policy][stage][count_mode][f"{threshold:.2f}"]
                row["tp"] += _hungarian_hits(selected_quality, threshold)
                row["predictions"] += prediction_count
                row["gt"] += gt_count


def _finalize_metric_tree(tree: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for policy, policy_data in tree.items():
        out[policy] = {}
        for stage, stage_data in policy_data.items():
            out[policy][stage] = {}
            for count_mode, threshold_data in stage_data.items():
                out[policy][stage][count_mode] = {
                    threshold: _f1(
                        int(row["tp"]),
                        int(row["predictions"]),
                        int(row["gt"]),
                    )
                    for threshold, row in threshold_data.items()
                }
    return out


@torch.no_grad()
def _collect_v7(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    thresholds: tuple[float, ...],
) -> tuple[dict[str, Any], list[int], int]:
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    selector = model.structured_query_head.set_selection_head
    refiner = selector.slot_refinement
    if refiner is None:
        raise ValueError("V7 audit requires a four-slot refiner")
    loader = build_dataloader(cfg, split="val", training=False)
    max_batches = math.ceil(int(args.max_images) / int(args.eval_batch_size))
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy="uniform",
        max_batches=max_batches,
        num_workers=int(args.num_workers),
    )
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})
    metric_tree = _metric_template(POLICIES, thresholds)
    records: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    replay_max_error = 0.0
    target_argmax_collision_count = 0

    for images, targets, metas in tqdm(loader, desc="V7 reference policies", ncols=90):
        images = images.to(device, non_blocking=True)
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, _args, kwargs):
            captured.update(kwargs)

        handle = refiner.register_forward_pre_hook(capture, with_kwargs=True)
        outputs = model(images)
        handle.remove()
        target_data = build_four_slot_cluster_targets(
            outputs,
            targets,
            num_slots=int(outputs["selection_slot_real_route_logits"].shape[1]),
            input_h=int(model_cfg.get("input_h", 288)),
            line_width=float(loss_cfg.get("four_slot_line_width", 30.0)),
            min_valid_rows=int(loss_cfg.get("four_slot_min_valid_rows", 5)),
            representable_min=float(loss_cfg.get("four_slot_representable_min", 0.0)),
            cluster_min=float(loss_cfg.get("four_slot_cluster_min", 0.0)),
            cluster_delta=float(loss_cfg.get("four_slot_cluster_delta", 0.10)),
            temperature=float(loss_cfg.get("four_slot_cluster_temperature", 0.03)),
            target_mode=str(loss_cfg.get("four_slot_target_mode", "all_gt")),
        )
        target_rows_batch = target_data["rows"]
        current_replay = _run_refiner(refiner, captured, reference_mode="hard_st")
        replay_max_error = max(
            replay_max_error,
            float(
                (
                    current_replay["selection_slot_pred_x_rows"]
                    - outputs["selection_slot_pred_x_rows"]
                ).abs().max()
            ),
        )

        batch_size = int(images.shape[0])
        forced_routes = outputs["selection_slot_geometry_route_indices"].clone()
        target_soft_logits = outputs["selection_slot_real_route_logits"].new_full(
            outputs["selection_slot_real_route_logits"].shape,
            -69.0,
        )
        batch_assignments: list[list[int]] = []
        batch_target_argmax: list[list[int]] = []
        for bi in range(batch_size):
            target_rows = target_rows_batch[bi][..., :-1].float()
            slots_for_gt = _hard_min_slots(
                outputs["selection_slot_active_logits"][bi],
                outputs["selection_slot_real_route_logits"][bi],
                target_rows,
            )
            routes, target_argmax, collisions = _target_hard_unique_routes(
                target_rows,
                slots_for_gt,
                outputs["selection_slot_geometry_route_indices"][bi],
                outputs["selection_slot_real_route_logits"][bi],
                outputs["selection_slot_candidate_valid"][bi],
            )
            target_argmax_collision_count += collisions
            forced_routes[bi] = routes
            weight = _target_soft_weights(
                target_rows,
                slots_for_gt,
                routes,
                int(outputs["selection_slot_real_route_logits"].shape[-1]),
            )
            target_soft_logits[bi] = weight.clamp_min(1.0e-30).log()
            batch_assignments.append(slots_for_gt)
            batch_target_argmax.append(target_argmax)

        predicted_soft = _run_refiner(
            refiner,
            captured,
            reference_mode="soft",
            structured_unique=True,
        )
        target_soft = _run_refiner(
            refiner,
            captured,
            reference_mode="soft",
            route_indices=forced_routes,
            route_logits=target_soft_logits,
            structured_unique=False,
        )
        target_hard = _run_refiner(
            refiner,
            captured,
            reference_mode="hard_st",
            route_indices=forced_routes,
        )
        policy_outputs = {
            "current_hard": current_replay,
            "predicted_soft": predicted_soft,
            "target_soft": target_soft,
            "target_hard": target_hard,
        }

        for bi in range(batch_size):
            geometry: list[torch.Tensor] = []
            ranges: list[torch.Tensor] = []
            layout: dict[str, tuple[int, int]] = {}
            cursor = 0
            # Proposal geometry is included once so per-slot official current
            # and target-member IoUs share the exact same raster pass.
            proposal_count = int(outputs["pred_x_rows"].shape[1])
            geometry.append(outputs["pred_x_rows"][bi].detach().cpu())
            ranges.append(outputs["range_norm"][bi].detach().cpu())
            layout["proposal/reference"] = (cursor, cursor + proposal_count)
            cursor += proposal_count
            for policy in POLICIES:
                value = policy_outputs[policy]
                for stage, x_key, range_key in (
                    (
                        "reference",
                        "selection_slot_input_reference_x_rows",
                        "selection_slot_input_range_norm",
                    ),
                    (
                        "refined",
                        "selection_slot_pred_x_rows",
                        "selection_slot_range_norm",
                    ),
                ):
                    x = value[x_key][bi].detach().cpu()
                    lane_range = value[range_key][bi].detach().cpu()
                    geometry.append(x)
                    ranges.append(lane_range)
                    layout[f"{policy}/{stage}"] = (cursor, cursor + int(x.shape[0]))
                    cursor += int(x.shape[0])
            records.append(
                {
                    "record": {
                        "meta": _plain_meta(metas[bi]),
                        "stages": {
                            "combined": {
                                "pred_x_rows": torch.cat(geometry, dim=0),
                                "range_norm": torch.cat(ranges, dim=0),
                            }
                        },
                    },
                    "layout": layout,
                    "active": outputs["selection_slot_active"][bi].detach().cpu().bool(),
                    "target_rows": target_rows_batch[bi][..., :-1].detach().cpu(),
                    "slots_for_gt": batch_assignments[bi],
                    "target_argmax": batch_target_argmax[bi],
                    "current_routes": outputs[
                        "selection_slot_geometry_route_indices"
                    ][bi].detach().cpu().long(),
                    "route_probability": torch.softmax(
                        outputs["selection_slot_real_route_logits"][bi].float(),
                        dim=-1,
                    ).detach().cpu(),
                    "image_id": str(metas[bi].get("image_path", "")),
                }
            )

    evaluated_rows = _evaluate_records(
        records,
        thresholds=thresholds,
        line_width=float(loss_cfg.get("four_slot_line_width", 30.0)),
        min_valid_rows=int(loss_cfg.get("four_slot_min_valid_rows", 5)),
        workers=int(args.metric_workers),
        description="official policy IoU",
    )

    error_counts = {name: 0 for name in ERROR_CLASSES}
    support_mass: list[float] = []
    support_mass_by_error = {name: [] for name in ERROR_CLASSES}
    target_margin: list[float] = []
    predicted_margin: list[float] = []
    target_rank: list[float] = []
    kl_values: list[float] = []
    for item, evaluated in zip(records, evaluated_rows):
        policy_layout = {
            key: value
            for key, value in item["layout"].items()
            if not key.startswith("proposal/")
        }
        _accumulate_policy_metrics(
            metric_tree,
            evaluated,
            policy_layout,
            item["active"],
            thresholds=thresholds,
        )
        proposal_start, proposal_stop = item["layout"]["proposal/reference"]
        proposal_quality = evaluated["quality"][:, proposal_start:proposal_stop]
        route_probability = item["route_probability"]
        for gt, slot in enumerate(item["slots_for_gt"]):
            target = item["target_rows"][gt]
            support = target > 0.0
            probability = route_probability[int(slot)]
            current_id = int(item["current_routes"][int(slot)])
            target_id = int(item["target_argmax"][gt])
            mass = float(probability[support].sum())
            if not bool(item["active"][int(slot)]):
                error_class = "PATH_OR_ACTIVITY_ERROR"
            elif current_id == target_id:
                error_class = "CORRECT_ID"
            elif 0 <= current_id < int(support.numel()) and bool(support[current_id]):
                error_class = "IN_SUPPORT_WRONG_MEMBER"
            else:
                error_class = "OUTSIDE_SUPPORT"
            error_counts[error_class] += 1
            support_mass.append(mass)
            support_mass_by_error[error_class].append(mass)
            target_values = target[support].sort(descending=True).values
            t_margin = float(target_values[0]) - (
                float(target_values[1]) if int(target_values.numel()) > 1 else 0.0
            )
            predicted_values = probability.sort(descending=True).values
            p_margin = float(predicted_values[0] - predicted_values[1])
            rank = int((probability > probability[target_id]).sum()) + 1
            kl = float(
                (
                    target[support]
                    * (
                        target[support].clamp_min(1.0e-12).log()
                        - probability[support].clamp_min(1.0e-12).log()
                    )
                ).sum()
            )
            target_margin.append(t_margin)
            predicted_margin.append(p_margin)
            target_rank.append(float(rank))
            kl_values.append(kl)
            slot_rows.append(
                {
                    "image_id": item["image_id"],
                    "slot_id": int(slot),
                    "assigned_gt": int(gt),
                    "error_class": error_class,
                    "support_size": int(support.sum()),
                    "target_entropy": float(
                        -(target[support] * target[support].clamp_min(1.0e-12).log()).sum()
                    ),
                    "target_top1_mass": float(target.max()),
                    "target_top1_minus_top2": t_margin,
                    "predicted_support_mass": mass,
                    "predicted_target_id_rank": rank,
                    "predicted_top1_minus_top2": p_margin,
                    "kl_target_to_predicted_on_support": kl,
                    "current_id": current_id,
                    "target_id": target_id,
                    "current_in_support": bool(
                        0 <= current_id < int(support.numel()) and support[current_id]
                    ),
                    "active": bool(item["active"][int(slot)]),
                    "official_iou_current": (
                        float(proposal_quality[gt, current_id]) if current_id >= 0 else 0.0
                    ),
                    "official_iou_target": float(proposal_quality[gt, target_id]),
                }
            )

    total = sum(error_counts.values())
    result = {
        "iteration": int(iteration),
        "sampled_indices": sampled_indices[: int(args.max_images)],
        "images": len(records),
        "replay_max_abs_error": replay_max_error,
        "target_argmax_raw_collision_count": target_argmax_collision_count,
        "support_mass_decomposition": {
            "total_assigned_gt": total,
            "counts": error_counts,
            "fractions": {
                name: float(value) / float(max(total, 1))
                for name, value in error_counts.items()
            },
            "predicted_support_mass": _summary(support_mass),
            "predicted_support_mass_by_error": {
                name: _summary(values) for name, values in support_mass_by_error.items()
            },
            "fraction_support_mass_ge_0p5": float(
                np.mean(np.asarray(support_mass) >= 0.5)
            ) if support_mass else 0.0,
            "fraction_support_mass_ge_0p8": float(
                np.mean(np.asarray(support_mass) >= 0.8)
            ) if support_mass else 0.0,
            "target_top1_margin": _summary(target_margin),
            "predicted_top1_margin": _summary(predicted_margin),
            "predicted_target_id_rank": _summary(target_rank),
            "kl_target_to_predicted_on_support": _summary(kl_values),
        },
        "reference_policies": _finalize_metric_tree(metric_tree),
        "per_slot": slot_rows,
    }
    return result, sampled_indices, int(iteration)


@torch.no_grad()
def _collect_v8_forced_mix(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    thresholds: tuple[float, ...],
    expected_indices: list[int],
) -> dict[str, Any]:
    if not args.v8_config or not args.v8_checkpoint:
        return {"enabled": False}
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = load_checkpoint(args.v8_checkpoint, model, strict=False)
    model.eval()
    refiner = model.structured_query_head.set_selection_head.slot_refinement
    if refiner is None or refiner.neighborhood_mix is None:
        raise ValueError("V8 forced-mix audit requires a neighborhood refiner")
    learned_mix = float(torch.tanh(refiner.neighborhood_mix.detach()))
    requested_policies = [("mix_learned", learned_mix)]
    requested_policies.extend(
        (f"mix_{float(value):g}", float(value)) for value in args.forced_mixes
    )
    labels = tuple(label for label, _value in requested_policies)
    metric_tree = _metric_template(labels, thresholds)
    loader = build_dataloader(cfg, split="val", training=False)
    max_batches = math.ceil(int(args.max_images) / int(args.eval_batch_size))
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy="uniform",
        max_batches=max_batches,
        num_workers=int(args.num_workers),
    )
    if sampled_indices[: int(args.max_images)] != expected_indices[: int(args.max_images)]:
        raise ValueError("V7 and V8 diagnostic indices differ")
    records: list[dict[str, Any]] = []
    effective: dict[str, float] = {}
    parity_max_error = 0.0
    for images, _targets, metas in tqdm(loader, desc="V8 forced mixes", ncols=90):
        images = images.to(device, non_blocking=True)
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, _args, kwargs):
            captured.update(kwargs)

        handle = refiner.register_forward_pre_hook(capture, with_kwargs=True)
        outputs = model(images)
        handle.remove()
        policy_outputs: dict[str, dict[str, torch.Tensor]] = {}
        original = refiner.neighborhood_mix.detach().clone()
        for label, requested_mix in requested_policies:
            if label == "mix_learned":
                refiner.neighborhood_mix.copy_(original)
            else:
                clipped = max(min(float(requested_mix), 0.999999), -0.999999)
                raw = math.atanh(clipped)
                refiner.neighborhood_mix.copy_(
                    refiner.neighborhood_mix.new_tensor(raw)
                )
            value = _run_refiner(refiner, captured, reference_mode="neighborhood_soft")
            policy_outputs[label] = value
            effective[label] = float(torch.tanh(refiner.neighborhood_mix.detach()))
            if label == "mix_learned":
                parity_max_error = max(
                    parity_max_error,
                    float(
                        (
                            value["selection_slot_pred_x_rows"]
                            - outputs["selection_slot_pred_x_rows"]
                        ).abs().max()
                    ),
                )
        refiner.neighborhood_mix.copy_(original)
        for bi in range(int(images.shape[0])):
            geometry: list[torch.Tensor] = []
            ranges: list[torch.Tensor] = []
            layout: dict[str, tuple[int, int]] = {}
            cursor = 0
            for label in labels:
                value = policy_outputs[label]
                for stage, x_key, range_key in (
                    (
                        "reference",
                        "selection_slot_input_reference_x_rows",
                        "selection_slot_input_range_norm",
                    ),
                    (
                        "refined",
                        "selection_slot_pred_x_rows",
                        "selection_slot_range_norm",
                    ),
                ):
                    x = value[x_key][bi].detach().cpu()
                    lane_range = value[range_key][bi].detach().cpu()
                    geometry.append(x)
                    ranges.append(lane_range)
                    layout[f"{label}/{stage}"] = (cursor, cursor + int(x.shape[0]))
                    cursor += int(x.shape[0])
            records.append(
                {
                    "record": {
                        "meta": _plain_meta(metas[bi]),
                        "stages": {
                            "combined": {
                                "pred_x_rows": torch.cat(geometry, dim=0),
                                "range_norm": torch.cat(ranges, dim=0),
                            }
                        },
                    },
                    "layout": layout,
                    "active": outputs["selection_slot_active"][bi].detach().cpu().bool(),
                }
            )
    loss_cfg = cfg.get("loss", {})
    evaluated_rows = _evaluate_records(
        records,
        thresholds=thresholds,
        line_width=float(loss_cfg.get("four_slot_line_width", 30.0)),
        min_valid_rows=int(loss_cfg.get("four_slot_min_valid_rows", 5)),
        workers=int(args.metric_workers),
        description="official forced-mix IoU",
    )
    for item, evaluated in zip(records, evaluated_rows):
        _accumulate_policy_metrics(
            metric_tree,
            evaluated,
            item["layout"],
            item["active"],
            thresholds=thresholds,
        )
    return {
        "enabled": True,
        "iteration": int(iteration),
        "learned_mix": learned_mix,
        "requested_to_effective_mix": effective,
        "learned_mix_replay_max_abs_error": parity_max_error,
        "metrics": _finalize_metric_tree(metric_tree),
    }


def _evaluate_records(
    records: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...],
    line_width: float,
    min_valid_rows: int,
    workers: int,
    description: str,
) -> list[dict[str, Any]]:
    def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        return _evaluate_combined_record(
            item["record"],
            thresholds=thresholds,
            line_width=line_width,
            min_valid_rows=min_valid_rows,
        )

    worker_count = max(int(workers), 1)
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        if worker_count == 1:
            return [
                evaluate(item)
                for item in tqdm(records, desc=description, ncols=90)
            ]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(
                tqdm(
                    executor.map(evaluate, records),
                    total=len(records),
                    desc=description,
                    ncols=90,
                )
            )
    finally:
        cv2.setNumThreads(previous_cv_threads)


def _decision(payload: dict[str, Any]) -> dict[str, Any]:
    decomposition = payload["v7"]["support_mass_decomposition"]
    fractions = decomposition["fractions"]
    inside_wrong = float(fractions["IN_SUPPORT_WRONG_MEMBER"])
    outside = float(fractions["OUTSIDE_SUPPORT"])
    policies = payload["v7"]["reference_policies"]
    current = policies["current_hard"]["refined"]["fixed_neural_active"]
    predicted_soft = policies["predicted_soft"]["refined"]["fixed_neural_active"]
    target_soft = policies["target_soft"]["refined"]["fixed_neural_active"]
    target_hard = policies["target_hard"]["refined"]["fixed_neural_active"]
    deltas: dict[str, dict[str, float]] = {}
    for threshold in ("0.50", "0.75"):
        deltas[threshold] = {
            "predicted_soft_minus_current_f1": float(predicted_soft[threshold]["f1"])
            - float(current[threshold]["f1"]),
            "target_soft_minus_current_f1": float(target_soft[threshold]["f1"])
            - float(current[threshold]["f1"]),
            "target_hard_minus_current_f1": float(target_hard[threshold]["f1"])
            - float(current[threshold]["f1"]),
            "target_hard_minus_target_soft_f1": float(target_hard[threshold]["f1"])
            - float(target_soft[threshold]["f1"]),
        }
    if outside > inside_wrong:
        diagnosis = "cluster_discovery_dominates"
        next_action = "stop_anchor_local_v8_and_build_global_soft_memory_or_direct_p2_slots"
    elif deltas["0.50"]["target_hard_minus_target_soft_f1"] > 0.01:
        diagnosis = "within_support_hard_member_error_with_barycentric_blur"
        next_action = "build_soft_context_but_slot_owned_row_geometry_not_coordinate_averaging"
    elif deltas["0.50"]["predicted_soft_minus_current_f1"] > 0.003:
        diagnosis = "hard_argmax_interface_dominates_and_predicted_soft_is_viable"
        next_action = "run_refiner_only_control_then_one_short_soft_memory_arm"
    else:
        diagnosis = "within_support_error_but_current_soft_distribution_is_not_deployable"
        next_action = "build_slot_owned_geometry_with_geometry_gradient_to_slot_context"
    return {
        "diagnosis": diagnosis,
        "deltas": deltas,
        "next_action": next_action,
        "long_training_authorized": False,
    }


def main() -> None:
    args = parse_args()
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    seed_everything(3407)
    v7_cfg = _prepare_config(args.config, args)
    v7, sampled_indices, iteration = _collect_v7(v7_cfg, args, thresholds)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    v8_cfg = _prepare_config(args.v8_config, args) if args.v8_config else {}
    v8 = _collect_v8_forced_mix(v8_cfg, args, thresholds, sampled_indices)
    payload = {
        "experiment": "V7/V8 route-support and reference-policy zero-step audit",
        "diagnostic_only": True,
        "test_set_used": False,
        "training_steps": 0,
        "protocol": {
            "split": "val",
            "sample_strategy": "uniform",
            "max_images": int(args.max_images),
            "eval_batch_size": int(args.eval_batch_size),
            "amp_dtype": "none",
            "score_threshold": 0.0,
            "top_k": 4,
            "nms": 0.0,
            "iou_thresholds": list(thresholds),
        },
        "v7_config": str(Path(args.config).resolve()),
        "v7_checkpoint": str(Path(args.checkpoint).resolve()),
        "v7_iteration": int(iteration),
        "v7": v7,
        "v8_forced_mix": v8,
    }
    payload["decision"] = _decision(payload)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    compact_v8 = {"enabled": bool(v8.get("enabled", False))}
    if compact_v8["enabled"]:
        compact_v8.update({
            "iteration": v8["iteration"],
            "learned_mix": v8["learned_mix"],
            "learned_mix_replay_max_abs_error": v8[
                "learned_mix_replay_max_abs_error"
            ],
        })
    print(json.dumps({
        "experiment": payload["experiment"],
        "protocol": payload["protocol"],
        "support_mass_decomposition": v7["support_mass_decomposition"],
        "v8_forced_mix": compact_v8,
        "decision": payload["decision"],
    }, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
