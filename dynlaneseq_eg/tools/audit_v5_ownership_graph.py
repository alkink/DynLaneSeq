from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any, Iterable

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    official_proposal_gt_iou_matrix,
)
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    _frozen_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V5 ownership assignment consistency without optimizer "
            "updates. The audit separates within-forward layer churn, "
            "checkpoint churn conditional on a fixed official winner, and "
            "ownership-loss gradient conflict."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--gradient-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--stable-iou-floor", type=float, default=0.50)
    parser.add_argument("--no-lane-weight", type=float, default=0.10)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, name: str):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(name)
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg.setdefault("model", {}).setdefault("structured_query", {})[
        "intermediate_supervision"
    ] = True
    return cfg


def _stage(output: dict[str, Any], batch_index: int) -> dict[str, torch.Tensor]:
    fields = ("pred_x_rows", "range_norm", "exist_logits", "quality_logits")
    return {
        name: output[name][batch_index].detach().float().cpu()
        for name in fields
        if isinstance(output.get(name), torch.Tensor)
    }


def _owners_from_match(match: dict[str, torch.Tensor], gt_count: int) -> list[int]:
    owners = [-1 for _ in range(int(gt_count))]
    for pred_index, gt_index in zip(
        match["pred_indices"].detach().cpu().tolist(),
        match["gt_indices"].detach().cpu().tolist(),
    ):
        if 0 <= int(gt_index) < int(gt_count):
            owners[int(gt_index)] = int(pred_index)
    return owners


def _official_owners(
    iou: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> tuple[list[int], list[float]]:
    gt_count, candidate_count = iou.shape
    owners = [-1 for _ in range(int(gt_count))]
    qualities = [0.0 for _ in range(int(gt_count))]
    valid_ids = [
        index
        for index in range(int(candidate_count))
        if bool(candidate_valid[index])
    ]
    if not valid_ids or int(gt_count) == 0:
        return owners, qualities
    gt_ids, local_ids = linear_sum_assignment(
        1.0 - iou[:, valid_ids].detach().cpu().numpy()
    )
    for gt_index, local_index in zip(gt_ids, local_ids):
        candidate_index = int(valid_ids[int(local_index)])
        owners[int(gt_index)] = candidate_index
        qualities[int(gt_index)] = float(iou[int(gt_index), candidate_index])
    return owners, qualities


def _new_transition() -> dict[str, int]:
    return {
        "same_gt_same_query": 0,
        "same_gt_different_query": 0,
        "matched_to_unmatched": 0,
        "unmatched_to_matched": 0,
        "unmatched_both": 0,
    }


def _update_transition(
    counts: dict[str, int],
    left: Iterable[int],
    right: Iterable[int],
) -> None:
    for owner_left, owner_right in zip(left, right):
        if int(owner_left) < 0 and int(owner_right) < 0:
            counts["unmatched_both"] += 1
        elif int(owner_left) < 0:
            counts["unmatched_to_matched"] += 1
        elif int(owner_right) < 0:
            counts["matched_to_unmatched"] += 1
        elif int(owner_left) == int(owner_right):
            counts["same_gt_same_query"] += 1
        else:
            counts["same_gt_different_query"] += 1


def _finish_transition(counts: dict[str, int]) -> dict[str, Any]:
    matched_both = counts["same_gt_same_query"] + counts[
        "same_gt_different_query"
    ]
    total = sum(counts.values())
    return {
        **counts,
        "gt_total": int(total),
        "matched_both": int(matched_both),
        "same_query_given_matched_both": counts["same_gt_same_query"]
        / float(max(matched_both, 1)),
        "any_target_state_change": (
            counts["same_gt_different_query"]
            + counts["matched_to_unmatched"]
            + counts["unmatched_to_matched"]
        )
        / float(max(total, 1)),
    }


def _label_disagreement(
    left: list[int],
    right: list[int],
    candidate_count: int,
) -> tuple[int, int]:
    left_positive = {int(value) for value in left if int(value) >= 0}
    right_positive = {int(value) for value in right if int(value) >= 0}
    return len(left_positive.symmetric_difference(right_positive)), int(
        candidate_count
    )


@torch.no_grad()
def _collect_checkpoint(
    model: torch.nn.Module,
    matcher: Any,
    cfg: dict[str, Any],
    checkpoint: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
    channels_last: bool,
) -> dict[str, Any]:
    iteration = load_checkpoint(checkpoint, model, strict=False)
    if hasattr(matcher, "set_iteration"):
        matcher.set_iteration(iteration)
    model.eval()
    assert model.structured_query_head is not None
    model.structured_query_head.intermediate_supervision = True
    base_loader = build_dataloader(cfg, split="val", training=False)
    loader, dataset_indices = select_diagnostic_loader(
        base_loader,
        strategy="uniform",
        max_batches=int(args.max_batches),
        num_workers=int(args.num_workers),
    )

    records: list[dict[str, Any]] = []
    layer_transitions: list[dict[str, int]] | None = None
    label_disagreement: list[list[int]] | None = None
    for images, batch_targets, metas in tqdm(
        loader,
        desc=f"V5 graph trace {Path(checkpoint).stem}",
        ncols=90,
    ):
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        with _amp_context(device, args.amp_dtype):
            outputs = _frozen_outputs(model, images)
        auxiliary = outputs.get("aux_outputs")
        if not isinstance(auxiliary, (list, tuple)) or not auxiliary:
            raise ValueError("V5 graph audit requires intermediate aux_outputs")
        layers = [*auxiliary, outputs]
        targets = nested_to_device(batch_targets, device)
        matches = matcher.match_many(tuple(layers), targets)
        if layer_transitions is None:
            layer_transitions = [_new_transition() for _ in layers[:-1]]
            label_disagreement = [[0, 0] for _ in layers[:-1]]

        for batch_index, meta in enumerate(metas):
            gt_count = int(batch_targets[batch_index]["x_rows"].shape[0])
            layer_owners = [
                _owners_from_match(layer_matches[batch_index], gt_count)
                for layer_matches in matches
            ]
            final_owners = layer_owners[-1]
            candidate_count = int(layers[-1]["pred_x_rows"].shape[1])
            assert layer_transitions is not None
            assert label_disagreement is not None
            for layer_index, owners in enumerate(layer_owners[:-1]):
                _update_transition(
                    layer_transitions[layer_index], owners, final_owners
                )
                different, total = _label_disagreement(
                    owners, final_owners, candidate_count
                )
                label_disagreement[layer_index][0] += int(different)
                label_disagreement[layer_index][1] += int(total)

            official_by_layer = []
            official_quality_by_layer = []
            for layer in layers:
                stage = _stage(layer, batch_index)
                record = {"stages": {"main": stage}, "meta": meta}
                official_iou, candidate_valid = official_proposal_gt_iou_matrix(
                    record,
                    "main",
                    line_width=float(args.line_width),
                    min_valid_rows=int(args.min_valid_rows),
                    row_visibility_thresh=float(args.row_visibility_thresh),
                )
                owners, qualities = _official_owners(
                    official_iou, candidate_valid
                )
                official_by_layer.append(owners)
                official_quality_by_layer.append(qualities)
            records.append(
                {
                    "image_path": str(meta.get("image_path", "")),
                    "training_owners_by_layer": layer_owners,
                    "official_owners_by_layer": official_by_layer,
                    "official_qualities_by_layer": official_quality_by_layer,
                }
            )

    if layer_transitions is None or label_disagreement is None:
        raise ValueError("diagnostic loader produced no images")
    layer_summary = []
    final_index = len(layer_transitions)
    for index, counts in enumerate(layer_transitions):
        different, total = label_disagreement[index]
        layer_summary.append(
            {
                "layer_index": int(index),
                "final_layer_index": int(final_index),
                "assignment": _finish_transition(counts),
                "binary_owner_target_disagreement_fraction": different
                / float(max(total, 1)),
            }
        )
    return {
        "checkpoint": str(checkpoint),
        "iteration": int(iteration),
        "dataset_indices": list(dataset_indices),
        "records": records,
        "within_forward": layer_summary,
    }


def _maximum_query_alignment(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    stable_iou_floor: float,
    num_queries: int,
) -> tuple[list[int], torch.Tensor]:
    counts = torch.zeros((num_queries, num_queries), dtype=torch.float64)
    for record_a, record_b in zip(before["records"], after["records"]):
        owners_a = record_a["official_owners_by_layer"][-1]
        owners_b = record_b["official_owners_by_layer"][-1]
        quality_a = record_a["official_qualities_by_layer"][-1]
        quality_b = record_b["official_qualities_by_layer"][-1]
        for left, right, qa, qb in zip(
            owners_a, owners_b, quality_a, quality_b
        ):
            if (
                int(left) >= 0
                and int(right) >= 0
                and min(float(qa), float(qb)) >= float(stable_iou_floor)
            ):
                counts[int(left), int(right)] += 1.0
    rows, cols = linear_sum_assignment(-counts.numpy())
    after_to_before = list(range(num_queries))
    for before_id, after_id in zip(rows, cols):
        after_to_before[int(after_id)] = int(before_id)
    return after_to_before, counts


def _cross_checkpoint(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    stable_iou_floor: float,
    num_queries: int,
) -> dict[str, Any]:
    if before["dataset_indices"] != after["dataset_indices"]:
        raise ValueError("checkpoint traces use different dataset indices")
    mapping, cooccurrence = _maximum_query_alignment(
        before,
        after,
        stable_iou_floor=stable_iou_floor,
        num_queries=num_queries,
    )
    counts = {
        "eligible": 0,
        "official_best_fixed": 0,
        "training_retained_when_official_fixed": 0,
        "training_comparable_when_official_fixed": 0,
        "raw_training_retained": 0,
        "aligned_training_retained": 0,
        "training_comparable": 0,
        "raw_official_retained": 0,
        "aligned_official_retained": 0,
    }
    for record_a, record_b in zip(before["records"], after["records"]):
        official_a = record_a["official_owners_by_layer"][-1]
        official_b = record_b["official_owners_by_layer"][-1]
        quality_a = record_a["official_qualities_by_layer"][-1]
        quality_b = record_b["official_qualities_by_layer"][-1]
        training_a = record_a["training_owners_by_layer"][-1]
        training_b = record_b["training_owners_by_layer"][-1]
        for oa, ob, qa, qb, ta, tb in zip(
            official_a,
            official_b,
            quality_a,
            quality_b,
            training_a,
            training_b,
        ):
            if (
                int(oa) < 0
                or int(ob) < 0
                or min(float(qa), float(qb)) < float(stable_iou_floor)
            ):
                continue
            counts["eligible"] += 1
            counts["raw_official_retained"] += int(int(oa) == int(ob))
            counts["aligned_official_retained"] += int(
                int(oa) == int(mapping[int(ob)])
            )
            if int(ta) >= 0 and int(tb) >= 0:
                counts["training_comparable"] += 1
                counts["raw_training_retained"] += int(int(ta) == int(tb))
                counts["aligned_training_retained"] += int(
                    int(ta) == int(mapping[int(tb)])
                )
            if int(oa) != int(ob):
                continue
            counts["official_best_fixed"] += 1
            if int(ta) >= 0 and int(tb) >= 0:
                counts["training_comparable_when_official_fixed"] += 1
                counts["training_retained_when_official_fixed"] += int(
                    int(ta) == int(tb)
                )
    return {
        "before_iteration": int(before["iteration"]),
        "after_iteration": int(after["iteration"]),
        **counts,
        "official_best_fixed_fraction": counts["official_best_fixed"]
        / float(max(counts["eligible"], 1)),
        "training_owner_retention_given_official_best_fixed": counts[
            "training_retained_when_official_fixed"
        ]
        / float(max(counts["training_comparable_when_official_fixed"], 1)),
        "raw_training_owner_retention": counts["raw_training_retained"]
        / float(max(counts["training_comparable"], 1)),
        "permutation_aligned_training_owner_retention": counts[
            "aligned_training_retained"
        ]
        / float(max(counts["training_comparable"], 1)),
        "raw_official_owner_retention": counts["raw_official_retained"]
        / float(max(counts["eligible"], 1)),
        "permutation_aligned_official_owner_retention": counts[
            "aligned_official_retained"
        ]
        / float(max(counts["eligible"], 1)),
        "query_permutation_after_to_before": mapping,
        "alignment_support": int(cooccurrence.sum().item()),
    }


def _owner_targets(
    logits: torch.Tensor,
    matches: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    batch, candidates, _classes = logits.shape
    target = torch.ones(
        (batch, candidates), dtype=torch.long, device=logits.device
    )
    for batch_index, match in enumerate(matches):
        pred = match["pred_indices"].to(logits.device)
        if int(pred.numel()):
            target[batch_index, pred] = 0
    return target


def _ownership_loss(
    output: dict[str, Any],
    matches: list[dict[str, torch.Tensor]],
    *,
    no_lane_weight: float,
) -> torch.Tensor:
    logits = output["exist_logits"].float()
    target = _owner_targets(logits, matches)
    weights = logits.new_tensor([1.0, float(no_lane_weight)])
    return F.cross_entropy(
        logits.reshape(-1, 2), target.reshape(-1), weight=weights
    )


def _parameter_groups(
    model: torch.nn.Module,
) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, list[int]]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if (
            "structured_query_head.ownership_tokens." in name
            or "structured_query_head.ownership_layers." in name
            or "structured_query_head.exist." in name
            or "structured_query_head.decision_norm." in name
        ):
            parameter.requires_grad_(True)
            selected.append((name, parameter))
        else:
            parameter.requires_grad_(False)
    if not selected:
        raise ValueError("no V5 ownership parameters were found")
    groups: dict[str, list[int]] = {
        "all_ownership": list(range(len(selected))),
        "ownership_tokens": [],
        "first_ownership_layer": [],
        "exist_head": [],
    }
    for index, (name, _parameter) in enumerate(selected):
        if ".ownership_tokens." in name:
            groups["ownership_tokens"].append(index)
        if ".ownership_layers.0." in name:
            groups["first_ownership_layer"].append(index)
        if ".exist." in name or ".decision_norm." in name:
            groups["exist_head"].append(index)
    return selected, groups


def _empty_pair_stats(layer_count: int, group_names: Iterable[str]):
    return {
        group: {
            (left, right): [0.0, 0.0, 0.0]
            for left in range(layer_count)
            for right in range(layer_count)
        }
        for group in group_names
    }


def _gradient_audit(
    model: torch.nn.Module,
    matcher: Any,
    cfg: dict[str, Any],
    checkpoint: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
    channels_last: bool,
) -> dict[str, Any]:
    iteration = load_checkpoint(checkpoint, model, strict=False)
    if hasattr(matcher, "set_iteration"):
        matcher.set_iteration(iteration)
    model.eval()
    assert model.structured_query_head is not None
    model.structured_query_head.intermediate_supervision = True
    named_parameters, groups = _parameter_groups(model)
    parameters = [parameter for _name, parameter in named_parameters]
    base_loader = build_dataloader(cfg, split="val", training=False)
    loader, _indices = select_diagnostic_loader(
        base_loader,
        strategy="uniform",
        max_batches=max(int(args.gradient_batches), 1),
        num_workers=int(args.num_workers),
    )
    pair_stats = None
    loss_sums: list[float] | None = None
    used_batches = 0
    for batch_index, (images, batch_targets, _metas) in enumerate(loader):
        if batch_index >= int(args.gradient_batches):
            break
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        targets = nested_to_device(batch_targets, device)
        with _amp_context(device, args.amp_dtype):
            outputs = _frozen_outputs(model, images)
            auxiliary = outputs.get("aux_outputs")
            if not isinstance(auxiliary, (list, tuple)) or not auxiliary:
                raise ValueError("gradient audit requires intermediate outputs")
            layers = [*auxiliary, outputs]
            matches = matcher.match_many(tuple(layers), targets)
            losses = [
                _ownership_loss(
                    layer,
                    layer_matches,
                    no_lane_weight=float(args.no_lane_weight),
                )
                for layer, layer_matches in zip(layers, matches)
            ]
        gradients: list[list[torch.Tensor | None]] = []
        for layer_index, loss in enumerate(losses):
            gradients.append(
                list(
                    torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=layer_index < len(losses) - 1,
                        allow_unused=True,
                    )
                )
            )
        if pair_stats is None:
            pair_stats = _empty_pair_stats(len(layers), groups)
            loss_sums = [0.0 for _ in layers]
        assert loss_sums is not None
        for index, loss in enumerate(losses):
            loss_sums[index] += float(loss.detach())
        for group_name, parameter_ids in groups.items():
            for left in range(len(layers)):
                for right in range(len(layers)):
                    dot = 0.0
                    left_sq = 0.0
                    right_sq = 0.0
                    for parameter_index in parameter_ids:
                        grad_left = gradients[left][parameter_index]
                        grad_right = gradients[right][parameter_index]
                        if grad_left is not None:
                            left_sq += float(
                                grad_left.detach().float().square().sum()
                            )
                        if grad_right is not None:
                            right_sq += float(
                                grad_right.detach().float().square().sum()
                            )
                        if grad_left is not None and grad_right is not None:
                            dot += float(
                                (
                                    grad_left.detach().float()
                                    * grad_right.detach().float()
                                ).sum()
                            )
                    values = pair_stats[group_name][(left, right)]
                    values[0] += dot
                    values[1] += left_sq
                    values[2] += right_sq
        used_batches += 1
        del outputs, losses, gradients
    if pair_stats is None or loss_sums is None:
        raise ValueError("gradient diagnostic loader produced no batches")
    cosine = {}
    for group_name, values_by_pair in pair_stats.items():
        matrix = []
        for left in range(len(loss_sums)):
            row = []
            for right in range(len(loss_sums)):
                dot, left_sq, right_sq = values_by_pair[(left, right)]
                denominator = math.sqrt(max(left_sq * right_sq, 0.0))
                row.append(dot / denominator if denominator > 0.0 else 0.0)
            matrix.append(row)
        cosine[group_name] = matrix
    return {
        "checkpoint": str(checkpoint),
        "iteration": int(iteration),
        "batches": int(used_batches),
        "layers_include_final_last": True,
        "mean_owner_ce_by_layer": [
            value / float(max(used_batches, 1)) for value in loss_sums
        ],
        "parameter_groups": {
            group_name: {
                "parameter_tensors": len(parameter_ids),
                "parameters": sum(
                    int(parameters[index].numel()) for index in parameter_ids
                ),
            }
            for group_name, parameter_ids in groups.items()
        },
        "gradient_cosine_by_layer": cosine,
    }


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) < 1:
        raise ValueError("at least one checkpoint is required")
    if min(args.eval_batch_size, args.max_batches) < 1:
        raise ValueError("eval batch size and max batches must be positive")
    if int(args.gradient_batches) < 0 or int(args.num_workers) < 0:
        raise ValueError("gradient batches and workers must be non-negative")
    cfg = _prepare_config(args)
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()
    if model.structured_query_head is None:
        raise ValueError("V5 ownership graph audit requires structured queries")
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    matcher = build_matcher(cfg)

    traces = [
        _collect_checkpoint(
            model,
            matcher,
            cfg,
            checkpoint,
            args=args,
            device=device,
            channels_last=channels_last,
        )
        for checkpoint in args.checkpoints
    ]
    traces.sort(key=lambda row: int(row["iteration"]))
    num_queries = int(
        cfg.get("model", {})
        .get("structured_query", {})
        .get("num_instances", 32)
    )
    consecutive = [
        _cross_checkpoint(
            left,
            right,
            stable_iou_floor=float(args.stable_iou_floor),
            num_queries=num_queries,
        )
        for left, right in zip(traces[:-1], traces[1:])
    ]
    gradient = None
    if int(args.gradient_batches) > 0:
        gradient = _gradient_audit(
            model,
            matcher,
            cfg,
            traces[-1]["checkpoint"],
            args=args,
            device=device,
            channels_last=channels_last,
        )
    public_traces = [
        {
            "checkpoint": trace["checkpoint"],
            "iteration": trace["iteration"],
            "within_forward": trace["within_forward"],
        }
        for trace in traces
    ]
    result = {
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "warning": (
            "This is a fixed-checkpoint assignment/gradient audit. Backward "
            "is used only for gradient measurement; no optimizer step occurs."
        ),
        "config": args.config,
        "settings": {
            "sample_strategy": "uniform",
            "images": len(traces[0]["records"]),
            "stable_iou_floor": float(args.stable_iou_floor),
            "no_lane_weight": float(args.no_lane_weight),
            "gradient_batches": int(args.gradient_batches),
        },
        "checkpoints": public_traces,
        "cross_checkpoint": consecutive,
        "gradient_audit": gradient,
        "decision_fields": {
            "layer_target_conflict": (
                "binary_owner_target_disagreement_fraction and layer-to-final "
                "assignment transitions"
            ),
            "true_temporal_instability": (
                "training_owner_retention_given_official_best_fixed"
            ),
            "global_query_relabeling": (
                "raw versus permutation_aligned owner retention"
            ),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
