from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import gc
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.audit_v11_causal_replay import (
    _accumulate_record,
    _evaluate_records,
    _finalize_metric_tree,
    _image_id,
    _new_metric_tree,
    _plain_meta,
)
from dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_official import (
    _required,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v19_semantic_coverage_core import (
    RouteEvaluation,
    evaluate_routes,
    exact_injective_routes,
    exact_joint_injective_routes,
    select_source_slot_conditioned_routes,
    select_unique_by_score,
)


THRESHOLDS = (0.50, 0.75)
FIXED_POLICIES = (
    "source_v7",
    "v19_deployed",
    "perfect_independent_iou",
    "predicted_iou_only",
    "v7_plus_predicted_iou_logit",
)
EDIT_BUDGETS = (0, 1, 2, 3, 4)
CONSOLIDATED_POLICIES = (
    "source_slot_conditioned",
    "joint_lexicographic",
    *(f"joint_edit_budget_{budget}" for budget in EDIT_BUDGETS),
)
ALL_POLICIES = FIXED_POLICIES + CONSOLIDATED_POLICIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free exact V19 route-change, independent-fidelity and "
            "joint/slot-conditioned/minimum-edit semantic-coverage autopsy."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle-device", default="cuda")
    parser.add_argument("--oracle-chunk-size", type=int, default=262_144)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--reference-json")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _config(path: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg["dataset"].setdefault("lists", {})[args.split] = str(
        Path(args.list_path).resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _gather_counterfactual(
    value: torch.Tensor,
    route: torch.Tensor,
) -> torch.Tensor:
    if value.ndim not in (4, 5):
        raise ValueError("counterfactual tensor has an unsupported rank")
    safe = route.clamp(min=0)
    suffix = value.shape[3:]
    index = safe.view(*safe.shape, 1, *([1] * len(suffix))).expand(
        *safe.shape, 1, *suffix
    )
    return value.gather(2, index).squeeze(2)


def _new_policy_accumulator() -> dict[str, dict[str, float | int]]:
    return {
        f"{threshold:.2f}": {
            "tp": 0,
            "predictions": 0,
            "gt": 0,
            "iou_sum": 0.0,
            "semantic_collision_excess": 0,
            "images_with_collision": 0,
            "improved_images_vs_source": 0,
            "worsened_images_vs_source": 0,
            "tied_images_vs_source": 0,
        }
        for threshold in THRESHOLDS
    }


def _metric_row(row: dict[str, float | int]) -> dict[str, float | int]:
    tp = int(row["tp"])
    predictions = int(row["predictions"])
    gt = int(row["gt"])
    fp = predictions - tp
    fn = gt - tp
    denominator = 2 * tp + fp + fn
    return {
        **row,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(predictions, 1),
        "recall": tp / max(gt, 1),
        "f1": 0.0 if denominator <= 0 else 2 * tp / denominator,
    }


def _accumulate_policy(
    accumulator: dict[str, dict[str, float | int]],
    result: RouteEvaluation,
    source: RouteEvaluation,
    threshold: float,
) -> None:
    row = accumulator[f"{threshold:.2f}"]
    row["tp"] = int(row["tp"]) + int(result.hit_count)
    row["predictions"] = int(row["predictions"]) + int(
        result.prediction_count
    )
    row["gt"] = int(row["gt"]) + int(result.gt_count)
    row["iou_sum"] = float(row["iou_sum"]) + float(result.iou_sum)
    row["semantic_collision_excess"] = int(
        row["semantic_collision_excess"]
    ) + int(result.semantic_collision_excess)
    row["images_with_collision"] = int(row["images_with_collision"]) + int(
        result.semantic_collision_excess > 0
    )
    delta = int(result.hit_count) - int(source.hit_count)
    key = (
        "improved_images_vs_source"
        if delta > 0
        else "worsened_images_vs_source"
        if delta < 0
        else "tied_images_vs_source"
    )
    row[key] = int(row[key]) + 1


def _route_change_decomposition(
    quality: torch.Tensor,
    source_route: torch.Tensor,
    endpoint_route: torch.Tensor,
    active: torch.Tensor,
    source_result: RouteEvaluation,
    endpoint_result: RouteEvaluation,
    threshold: float,
) -> tuple[Counter[str], dict[str, int]]:
    classes: Counter[str] = Counter()
    source_covered = set(source_result.covered_gt)
    endpoint_covered = set(endpoint_result.covered_gt)
    coverage = {
        "newly_covered_gt": len(endpoint_covered - source_covered),
        "abandoned_source_gt": len(source_covered - endpoint_covered),
        "preserved_source_gt": len(source_covered & endpoint_covered),
        "tp_delta": endpoint_result.hit_count - source_result.hit_count,
    }
    gt_count = int(quality.shape[0])
    for slot in torch.nonzero(active.bool(), as_tuple=False).flatten().tolist():
        old = int(source_route[slot])
        new = int(endpoint_route[slot])
        if old == new:
            continue
        classes["ACTIVE_ROUTE_CHANGED"] += 1
        if gt_count == 0 or old < 0 or new < 0:
            classes["BACKGROUND_OR_INVALID"] += 1
            continue
        old_value, old_gt = quality[:, slot, old].max(dim=0)
        new_value, new_gt = quality[:, slot, new].max(dim=0)
        old_q = float(old_value)
        new_q = float(new_value)
        old_gt_id = int(old_gt)
        new_gt_id = int(new_gt)
        if (old_q > threshold) == (new_q > threshold):
            classes["THRESHOLD_STATUS_NEUTRAL"] += 1
        else:
            classes["THRESHOLD_STATUS_CHANGED"] += 1
        if new_q <= threshold:
            classes["BACKGROUND_OR_NEAR_MISS"] += 1
        elif new_gt_id == old_gt_id:
            if new_q > old_q + 1.0e-6:
                classes["SAME_GT_BETTER_MEMBER"] += 1
            elif new_q < old_q - 1.0e-6:
                classes["SAME_GT_WORSE_MEMBER"] += 1
            else:
                classes["SAME_GT_TIED_MEMBER"] += 1
        elif new_gt_id in source_covered:
            classes["ALREADY_COVERED_GT"] += 1
        else:
            classes["NEW_UNCOVERED_GT_TARGET"] += 1
        if old_q > threshold and old_gt_id not in endpoint_covered:
            classes["ROUTE_ABANDONED_OLD_GT"] += 1
    return classes, coverage


def _reference_reproduction(
    actual: dict[str, Any],
    reference_path: str | None,
) -> dict[str, Any]:
    if not reference_path:
        return {"provided": False, "exact": None}
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    exact = True
    for policy in ("source_v7", "v19_deployed"):
        rows[policy] = {}
        for threshold in ("0.50", "0.75"):
            expected = reference["metrics"][policy]["writer_valid"][
                "thresholds"
            ][threshold]
            observed = actual[policy]["writer_valid"]["thresholds"][threshold]
            expected_triplet = [
                int(expected[name]) for name in ("tp", "fp", "fn")
            ]
            observed_triplet = [
                int(observed[name]) for name in ("tp", "fp", "fn")
            ]
            same = expected_triplet == observed_triplet
            exact &= same
            rows[policy][threshold] = {
                "expected_tp_fp_fn": expected_triplet,
                "observed_tp_fp_fn": observed_triplet,
                "exact": same,
            }
    return {
        "provided": True,
        "path": str(Path(reference_path).resolve()),
        "exact": exact,
        "rows": rows,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = _config(args.config, args)
    source_cfg = _config(args.source_config, args)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    source_model = build_model(source_cfg).to(device)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    source_model.eval()
    loader = build_dataloader(cfg, split=args.split, training=False)

    records: list[dict[str, Any]] = []
    contract: dict[str, float | int] = {
        "source_route_mismatch": 0,
        "active_mask_mismatch": 0,
        "source_counterfactual_x_max_difference": 0.0,
        "source_counterfactual_range_max_difference": 0.0,
        "v19_counterfactual_x_max_difference": 0.0,
        "v19_counterfactual_range_max_difference": 0.0,
    }
    for batch_index, (images, _targets, metas) in enumerate(
        tqdm(loader, desc="V19 semantic autopsy forward", ncols=90)
    ):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        source_outputs = source_model(images)
        source_route = _required(
            source_outputs, "selection_slot_geometry_route_indices"
        )
        v7_route = _required(
            outputs, "selection_slot_v19_v7_geometry_route_indices"
        )
        v19_route = _required(outputs, "selection_slot_geometry_route_indices")
        source_active = _required(source_outputs, "selection_slot_active").bool()
        endpoint_active = _required(outputs, "selection_slot_active").bool()
        contract["source_route_mismatch"] = int(
            contract["source_route_mismatch"]
        ) + int((source_route != v7_route).sum().cpu())
        contract["active_mask_mismatch"] = int(
            contract["active_mask_mismatch"]
        ) + int((source_active != endpoint_active).sum().cpu())

        counterfactual_x = _required(
            outputs, "selection_slot_v19_counterfactual_x_rows"
        )
        counterfactual_range = _required(
            outputs, "selection_slot_v19_counterfactual_range_norm"
        )
        source_cf_x = _gather_counterfactual(counterfactual_x, v7_route)
        source_cf_range = _gather_counterfactual(
            counterfactual_range, v7_route
        )
        endpoint_cf_x = _gather_counterfactual(counterfactual_x, v19_route)
        endpoint_cf_range = _gather_counterfactual(
            counterfactual_range, v19_route
        )
        for key, difference in (
            (
                "source_counterfactual_x_max_difference",
                source_cf_x
                - _required(source_outputs, "selection_slot_pred_x_rows"),
            ),
            (
                "source_counterfactual_range_max_difference",
                source_cf_range
                - _required(source_outputs, "selection_slot_range_norm"),
            ),
            (
                "v19_counterfactual_x_max_difference",
                endpoint_cf_x - _required(outputs, "selection_slot_pred_x_rows"),
            ),
            (
                "v19_counterfactual_range_max_difference",
                endpoint_cf_range - _required(outputs, "selection_slot_range_norm"),
            ),
        ):
            contract[key] = max(
                float(contract[key]), float(difference.abs().max().cpu())
            )

        for item, meta in enumerate(metas):
            source_x = _required(
                source_outputs, "selection_slot_pred_x_rows"
            )[item]
            source_range = _required(
                source_outputs, "selection_slot_range_norm"
            )[item]
            endpoint_x = _required(outputs, "selection_slot_pred_x_rows")[item]
            endpoint_range = _required(outputs, "selection_slot_range_norm")[
                item
            ]
            slots, candidates, rows = counterfactual_x[item].shape
            geometry = torch.cat(
                (
                    source_x.detach().float().cpu(),
                    endpoint_x.detach().float().cpu(),
                    counterfactual_x[item]
                    .reshape(slots * candidates, rows)
                    .detach()
                    .float()
                    .cpu(),
                ),
                dim=0,
            )
            ranges = torch.cat(
                (
                    source_range.detach().float().cpu(),
                    endpoint_range.detach().float().cpu(),
                    counterfactual_range[item]
                    .reshape(slots * candidates, 2)
                    .detach()
                    .float()
                    .cpu(),
                ),
                dim=0,
            )
            records.append(
                {
                    "image_id": _image_id(
                        meta, f"v19_semantic_{batch_index:06d}_{item}"
                    ),
                    "record": {
                        "meta": _plain_meta(meta),
                        "stages": {
                            "combined": {
                                "pred_x_rows": geometry,
                                "range_norm": ranges,
                            }
                        },
                    },
                    "layout": {
                        "source_v7": (0, slots),
                        "v19_deployed": (slots, 2 * slots),
                        "counterfactual": (
                            2 * slots,
                            2 * slots + slots * candidates,
                        ),
                    },
                    "active_by_policy": {
                        "source_v7": source_active[item].detach().cpu(),
                        "v19_deployed": endpoint_active[item].detach().cpu(),
                    },
                    "counterfactual_valid": _required(
                        outputs,
                        "selection_slot_v19_counterfactual_valid",
                    )[item]
                    .detach()
                    .bool()
                    .cpu(),
                    "predicted_iou": _required(
                        outputs, "selection_slot_v19_expected_iou"
                    )[item]
                    .detach()
                    .float()
                    .cpu(),
                    "predicted_fidelity": _required(
                        outputs, "selection_slot_v19_fidelity_delta"
                    )[item]
                    .detach()
                    .float()
                    .cpu(),
                    "legacy_logits": _required(
                        outputs, "selection_slot_v19_v7_real_route_logits"
                    )[item]
                    .detach()
                    .float()
                    .cpu(),
                    "v7_route": v7_route[item].detach().long().cpu(),
                    "v19_route": v19_route[item].detach().long().cpu(),
                    "active": source_active[item].detach().bool().cpu(),
                }
            )

    del model, source_model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    evaluated = _evaluate_records(
        records,
        line_width=30.0,
        min_valid_rows=5,
        workers=int(args.metric_workers),
    )
    actual_tree = _new_metric_tree(
        ("source_v7", "v19_deployed"), THRESHOLDS
    )
    policy_rows = {
        policy: _new_policy_accumulator()
        for policy in ALL_POLICIES
    }
    policy_rows.update(
        {
            "perfect_injective_50": _new_policy_accumulator(),
            "perfect_injective_75": _new_policy_accumulator(),
        }
    )
    decomposition = {
        f"{threshold:.2f}": {
            "exclusive_and_flag_counts": Counter(),
            "coverage_totals": Counter(),
        }
        for threshold in THRESHOLDS
    }
    oracle_assignment_counts: Counter[int] = Counter()
    route_activity = Counter()
    source_slot_ownership = Counter()
    source_slot_edit_histogram: Counter[int] = Counter()
    joint_selected_edit_histogram: Counter[int] = Counter()
    budget_edit_histograms = {
        budget: Counter() for budget in EDIT_BUDGETS
    }

    iterator = zip(records, evaluated)
    for item, result in tqdm(
        iterator,
        total=len(records),
        desc="V19 exact semantic oracle",
        ncols=90,
    ):
        _accumulate_record(
            actual_tree,
            result,
            {
                name: item["layout"][name]
                for name in ("source_v7", "v19_deployed")
            },
            item["active_by_policy"],
            THRESHOLDS,
        )
        quality_all = result["quality"].float()
        valid_all = result["valid"].bool()
        start, stop = item["layout"]["counterfactual"]
        slots, candidates = item["predicted_iou"].shape
        gt_count = int(quality_all.shape[0])
        quality = quality_all[:, start:stop].reshape(
            gt_count, slots, candidates
        )
        valid = valid_all[start:stop].reshape(slots, candidates)
        valid &= item["counterfactual_valid"]
        active = item["active"]
        changed_route = item["v7_route"] != item["v19_route"]
        route_activity["all_slots"] += int(changed_route.numel())
        route_activity["all_changed"] += int(changed_route.sum())
        route_activity["active_slots"] += int(active.sum())
        route_activity["active_changed"] += int((changed_route & active).sum())
        route_activity["inactive_slots"] += int((~active).sum())
        route_activity["inactive_changed"] += int(
            (changed_route & ~active).sum()
        )
        route_activity["images_with_active_change"] += int(
            bool((changed_route & active).any())
        )
        if gt_count:
            independent_score = quality.max(dim=0).values
        else:
            independent_score = torch.zeros(slots, candidates)
        predicted_iou = item["predicted_iou"].clamp(1.0e-6, 1.0 - 1.0e-6)
        slot_conditioned_route, ownership = (
            select_source_slot_conditioned_routes(
                quality,
                valid,
                item["v7_route"],
                active,
            )
        )
        owned = (ownership >= 0) & active
        source_slot_ownership["active_slots"] += int(active.sum())
        source_slot_ownership["owned_slots"] += int(owned.sum())
        source_slot_ownership["unowned_slots"] += int((active & ~owned).sum())
        source_slot_ownership["images_with_unowned_slot"] += int(
            bool((active & ~owned).any())
        )
        source_slot_edits = int(
            ((slot_conditioned_route != item["v7_route"]) & active).sum()
        )
        source_slot_edit_histogram[source_slot_edits] += 1
        routes = {
            "source_v7": item["v7_route"],
            "v19_deployed": item["v19_route"],
            "perfect_independent_iou": select_unique_by_score(
                independent_score, valid, active
            ),
            "predicted_iou_only": select_unique_by_score(
                predicted_iou, valid, active
            ),
            "v7_plus_predicted_iou_logit": select_unique_by_score(
                item["legacy_logits"] + torch.logit(predicted_iou),
                valid,
                active,
            ),
            "source_slot_conditioned": slot_conditioned_route,
        }
        injective = exact_injective_routes(
            quality,
            valid,
            active,
            thresholds=THRESHOLDS,
            device=args.oracle_device,
            chunk_size=int(args.oracle_chunk_size),
        )
        oracle_assignment_counts[
            injective[THRESHOLDS[0]].assignments_evaluated
        ] += 1
        joint = exact_joint_injective_routes(
            quality,
            valid,
            active,
            item["v7_route"],
            device=args.oracle_device,
            chunk_size=int(args.oracle_chunk_size),
        )
        routes["joint_lexicographic"] = torch.tensor(
            joint.best.routes, dtype=torch.long
        )
        joint_selected_edit_histogram[joint.best.edit_count] += 1
        active_count = int(active.sum())
        for budget in EDIT_BUDGETS:
            result_for_budget = joint.by_max_edits[min(budget, active_count)]
            routes[f"joint_edit_budget_{budget}"] = torch.tensor(
                result_for_budget.routes, dtype=torch.long
            )
            budget_edit_histograms[budget][result_for_budget.edit_count] += 1

        evaluations: dict[str, dict[float, RouteEvaluation]] = {}
        for policy, route in routes.items():
            evaluations[policy] = {
                threshold: evaluate_routes(
                    quality,
                    valid,
                    route,
                    active,
                    threshold=threshold,
                )
                for threshold in THRESHOLDS
            }
        for threshold in THRESHOLDS:
            source_result = evaluations["source_v7"][threshold]
            for policy in ALL_POLICIES:
                _accumulate_policy(
                    policy_rows[policy],
                    evaluations[policy][threshold],
                    source_result,
                    threshold,
                )
            injective_name = f"perfect_injective_{int(threshold * 100)}"
            injective_route = torch.tensor(
                injective[threshold].routes, dtype=torch.long
            )
            injective_result = evaluate_routes(
                quality,
                valid,
                injective_route,
                active,
                threshold=threshold,
            )
            _accumulate_policy(
                policy_rows[injective_name],
                injective_result,
                source_result,
                threshold,
            )
            classes, coverage = _route_change_decomposition(
                quality,
                item["v7_route"],
                item["v19_route"],
                active,
                source_result,
                evaluations["v19_deployed"][threshold],
                threshold,
            )
            key = f"{threshold:.2f}"
            decomposition[key]["exclusive_and_flag_counts"].update(classes)
            decomposition[key]["coverage_totals"].update(coverage)

    actual_metrics = _finalize_metric_tree(actual_tree)
    metrics = {
        policy: {
            threshold: _metric_row(row)
            for threshold, row in thresholds.items()
        }
        for policy, thresholds in policy_rows.items()
    }
    comparisons: dict[str, Any] = {}
    for threshold in ("0.50", "0.75"):
        source_tp = int(metrics["source_v7"][threshold]["tp"])
        v19_tp = int(metrics["v19_deployed"][threshold]["tp"])
        independent_tp = int(
            metrics["perfect_independent_iou"][threshold]["tp"]
        )
        injective_name = f"perfect_injective_{int(float(threshold) * 100)}"
        injective_tp = int(metrics[injective_name][threshold]["tp"])
        comparisons[threshold] = {
            "v19_minus_source_tp": v19_tp - source_tp,
            "perfect_independent_minus_v19_tp": independent_tp - v19_tp,
            "perfect_independent_minus_source_tp": independent_tp - source_tp,
            "perfect_injective_minus_independent_tp": (
                injective_tp - independent_tp
            ),
            "perfect_injective_minus_source_tp": injective_tp - source_tp,
            "source_slot_conditioned_minus_source_tp": (
                int(metrics["source_slot_conditioned"][threshold]["tp"])
                - source_tp
            ),
            "joint_lexicographic_minus_source_tp": (
                int(metrics["joint_lexicographic"][threshold]["tp"])
                - source_tp
            ),
            "joint_minus_source_slot_conditioned_tp": (
                int(metrics["joint_lexicographic"][threshold]["tp"])
                - int(metrics["source_slot_conditioned"][threshold]["tp"])
            ),
            "joint_edit_budget_minus_source_tp": {
                str(budget): (
                    int(
                        metrics[f"joint_edit_budget_{budget}"][threshold][
                            "tp"
                        ]
                    )
                    - source_tp
                )
                for budget in EDIT_BUDGETS
            },
        }

    reference = _reference_reproduction(actual_metrics, args.reference_json)
    contract_passed = (
        int(contract["source_route_mismatch"]) == 0
        and int(contract["active_mask_mismatch"]) == 0
        and float(contract["source_counterfactual_x_max_difference"]) <= 1.0e-5
        and float(contract["source_counterfactual_range_max_difference"])
        <= 1.0e-6
        and float(contract["v19_counterfactual_x_max_difference"]) <= 1.0e-5
        and float(contract["v19_counterfactual_range_max_difference"])
        <= 1.0e-6
        and reference.get("exact") is not False
    )
    report = {
        "experiment": "V19 training-free exact semantic-coverage autopsy",
        "config": str(Path(args.config).resolve()),
        "source_config": str(Path(args.source_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "iteration": iteration,
        "source_iteration": source_iteration,
        "split": args.split,
        "list_path": str(Path(args.list_path).resolve()),
        "list_sha256": sha256_file(Path(args.list_path)),
        "images": len(records),
        "contract": {**contract, "passed": contract_passed},
        "reference_reproduction": reference,
        "actual_policy_metrics": actual_metrics,
        "counterfactual_policy_metrics": metrics,
        "oracle_comparisons": comparisons,
        "route_change_decomposition": {
            threshold: {
                "exclusive_and_flag_counts": dict(
                    value["exclusive_and_flag_counts"]
                ),
                "coverage_totals": dict(value["coverage_totals"]),
            }
            for threshold, value in decomposition.items()
        },
        "route_change_activity": {
            **dict(route_activity),
            "all_changed_fraction": int(route_activity["all_changed"])
            / max(int(route_activity["all_slots"]), 1),
            "active_changed_fraction": int(route_activity["active_changed"])
            / max(int(route_activity["active_slots"]), 1),
            "inactive_changed_fraction": int(route_activity["inactive_changed"])
            / max(int(route_activity["inactive_slots"]), 1),
        },
        "source_slot_ownership": {
            **dict(source_slot_ownership),
            "owned_fraction": int(source_slot_ownership["owned_slots"])
            / max(int(source_slot_ownership["active_slots"]), 1),
            "route_edit_histogram": {
                str(key): value
                for key, value in sorted(source_slot_edit_histogram.items())
            },
        },
        "joint_selected_edit_histogram": {
            str(key): value
            for key, value in sorted(joint_selected_edit_histogram.items())
        },
        "joint_edit_budget_usage": {
            str(budget): {
                str(key): value
                for key, value in sorted(histogram.items())
            }
            for budget, histogram in budget_edit_histograms.items()
        },
        "oracle_assignments_evaluated_histogram": {
            str(key): value
            for key, value in sorted(oracle_assignment_counts.items())
        },
        "interpretation_contract": {
            "perfect_independent": (
                "Exact max-over-any-GT candidate fidelity with proposal-ID "
                "uniqueness, but no semantic-GT uniqueness."
            ),
            "perfect_injective": (
                "Exhaustive distinct proposal-ID search, scored by the exact "
                "official maximum-total-IoU assignment and threshold TP count."
            ),
            "predicted_iou_only": (
                "V19 expected-IoU output alone; diagnostic only, no score sweep."
            ),
            "v7_plus_predicted_iou_logit": (
                "Immutable V7 unary plus unit-scale expected-IoU logit; "
                "diagnostic only, no scale sweep."
            ),
            "source_slot_conditioned": (
                "Source V7 active slots are assigned injectively to GT by "
                "maximum total official IoU; each slot then chooses the best "
                "valid proposal only for its fixed owned GT."
            ),
            "joint_lexicographic": (
                "One deployable route per image selected by exact TP@.50, "
                "then TP@.75, then total official IoU, then minimum active "
                "route edits relative to V7. GT-conditioned diagnostic only."
            ),
            "joint_edit_budget": (
                "Exact joint-lexicographic oracle constrained to at most K "
                "active V7 route replacements; no inactive edits are counted."
            ),
        },
        "optimizer_steps_during_audit": 0,
        "checkpoint_selection_performed": False,
        "score_scale_search_performed": False,
        "threshold_search_performed": False,
        "nms_search_performed": False,
        "new_training_authorized": False,
        "full_validation_executed": False,
        "test_set_used": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
