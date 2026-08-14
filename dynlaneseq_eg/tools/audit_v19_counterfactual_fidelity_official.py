from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    evaluator_hungarian_assignment,
    sha256_file,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.losses.loss_s0 import _padded_lane_targets
from dynlaneseq_eg.losses.range_aware_iou import (
    batched_pairwise_range_aware_row_strip_iou,
)
from dynlaneseq_eg.tools.audit_v11_causal_replay import (
    _accumulate_record,
    _evaluate_records,
    _finalize_metric_tree,
    _image_id,
    _metric_delta,
    _new_metric_tree,
    _plain_meta,
)
from dynlaneseq_eg.tools.train import seed_everything


POLICIES = ("source_v7", "v19_deployed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official V19 selection and proposal-fidelity audit."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
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


def _required(outputs: dict[str, Any], name: str) -> torch.Tensor:
    value = outputs.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"missing V19 output: {name}")
    return value


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return 0.0
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1.0e-12))


def _ordinal_rank(value: torch.Tensor) -> torch.Tensor:
    order = value.argsort(stable=True)
    rank = torch.empty_like(order, dtype=torch.float32)
    rank[order] = torch.arange(
        value.numel(), dtype=torch.float32, device=value.device
    )
    return rank


def _metric_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        mode: {
            threshold: _metric_delta(
                metrics, "v19_deployed", "source_v7", mode, threshold
            )
            for threshold in ("0.50", "0.75")
        }
        for mode in ("neural_active", "writer_valid")
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
    selector = model.structured_query_head.set_selection_head
    if selector.counterfactual_fidelity is None:
        raise ValueError("V19 official audit requires fidelity module")
    source_model = build_model(source_cfg).to(device)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    source_model.eval()
    loader = build_dataloader(cfg, split=args.split, training=False)

    records: list[dict[str, Any]] = []
    proposal_tensor_difference = {"x": 0.0, "range": 0.0, "valid": 0}
    count_equal_images = 0
    total_images = 0
    route_changed = 0
    route_total = 0
    for batch_index, (images, targets, metas) in enumerate(
        tqdm(loader, desc="V19 official", ncols=90)
    ):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        source_outputs = source_model(images)
        proposal_tensor_difference["x"] = max(
            proposal_tensor_difference["x"],
            float(
                (
                    _required(outputs, "pred_x_rows")
                    - _required(source_outputs, "pred_x_rows")
                )
                .abs()
                .max()
                .cpu()
            ),
        )
        proposal_tensor_difference["range"] = max(
            proposal_tensor_difference["range"],
            float(
                (
                    _required(outputs, "range_norm")
                    - _required(source_outputs, "range_norm")
                )
                .abs()
                .max()
                .cpu()
            ),
        )
        proposal_tensor_difference["valid"] += int(
            (
                _required(outputs, "selection_slot_candidate_valid")
                != _required(source_outputs, "selection_slot_candidate_valid")
            )
            .sum()
            .cpu()
        )
        source_active = _required(source_outputs, "selection_slot_active").bool()
        endpoint_active = _required(outputs, "selection_slot_active").bool()
        count_equal_images += int(
            (source_active.sum(dim=-1) == endpoint_active.sum(dim=-1))
            .sum()
            .cpu()
        )
        total_images += int(images.shape[0])
        v7_route = _required(
            outputs, "selection_slot_v19_v7_geometry_route_indices"
        )
        v19_route = _required(outputs, "selection_slot_geometry_route_indices")
        route_changed += int((v7_route != v19_route).sum().cpu())
        route_total += int(v19_route.numel())

        cf_batch_x = _required(
            outputs, "selection_slot_v19_counterfactual_x_rows"
        )
        cf_batch_range = _required(
            outputs, "selection_slot_v19_counterfactual_range_norm"
        )
        cf_batch, cf_slots, cf_candidates, cf_rows = cf_batch_x.shape
        padded_x, padded_valid = _padded_lane_targets(
            targets,
            device=cf_batch_x.device,
            dtype=torch.float32,
            rows=int(cf_rows),
        )
        surrogate_matrix, _surrogate_valid, _gt_valid = (
            batched_pairwise_range_aware_row_strip_iou(
                cf_batch_x.reshape(
                    cf_batch, cf_slots * cf_candidates, cf_rows
                ),
                cf_batch_range.reshape(
                    cf_batch, cf_slots * cf_candidates, 2
                ),
                padded_x,
                padded_valid,
                input_h=int(cfg["model"]["input_h"]),
                line_width=float(
                    cfg.get("loss", {}).get("four_slot_line_width", 30.0)
                ),
                min_valid_rows=int(
                    cfg.get("loss", {}).get("four_slot_min_valid_rows", 5)
                ),
            )
        )
        if surrogate_matrix.shape[-1] > 0:
            surrogate_target = surrogate_matrix.max(dim=-1).values.reshape(
                cf_batch, cf_slots, cf_candidates
            )
        else:
            surrogate_target = cf_batch_x.new_zeros(
                (cf_batch, cf_slots, cf_candidates)
            )

        for item, meta in enumerate(metas):
            geometry = []
            ranges = []
            layout: dict[str, tuple[int, int]] = {}
            active_by_policy: dict[str, torch.Tensor] = {}
            cursor = 0

            policy_values = OrderedDict(
                (
                    (
                        "source_v7",
                        (
                            _required(
                                source_outputs,
                                "selection_slot_pred_x_rows",
                            )[item],
                            _required(
                                source_outputs,
                                "selection_slot_range_norm",
                            )[item],
                            source_active[item],
                        ),
                    ),
                    (
                        "v19_deployed",
                        (
                            _required(
                                outputs, "selection_slot_pred_x_rows"
                            )[item],
                            _required(outputs, "selection_slot_range_norm")[
                                item
                            ],
                            endpoint_active[item],
                        ),
                    ),
                )
            )
            for name, (x_rows, lane_range, active) in policy_values.items():
                geometry.append(x_rows.detach().float().cpu())
                ranges.append(lane_range.detach().float().cpu())
                count = int(x_rows.shape[0])
                layout[name] = (cursor, cursor + count)
                active_by_policy[name] = active.detach().cpu()
                cursor += count

            for name, owner in (
                ("source_proposals", source_outputs),
                ("v19_proposals", outputs),
            ):
                x_rows = _required(owner, "pred_x_rows")[item]
                lane_range = _required(owner, "range_norm")[item]
                geometry.append(x_rows.detach().float().cpu())
                ranges.append(lane_range.detach().float().cpu())
                layout[name] = (cursor, cursor + int(x_rows.shape[0]))
                cursor += int(x_rows.shape[0])

            counterfactual_x = _required(
                outputs, "selection_slot_v19_counterfactual_x_rows"
            )[item]
            counterfactual_range = _required(
                outputs, "selection_slot_v19_counterfactual_range_norm"
            )[item]
            slots, candidates, rows = counterfactual_x.shape
            cf_x = counterfactual_x.reshape(slots * candidates, rows)
            cf_range = counterfactual_range.reshape(slots * candidates, 2)
            geometry.append(cf_x.detach().float().cpu())
            ranges.append(cf_range.detach().float().cpu())
            layout["counterfactual"] = (
                cursor,
                cursor + int(cf_x.shape[0]),
            )

            records.append(
                {
                    "image_id": _image_id(
                        meta, f"v19_{batch_index:06d}_{item}"
                    ),
                    "record": {
                        "meta": _plain_meta(meta),
                        "stages": {
                            "combined": {
                                "pred_x_rows": torch.cat(geometry, dim=0),
                                "range_norm": torch.cat(ranges, dim=0),
                            }
                        },
                    },
                    "layout": layout,
                    "active_by_policy": active_by_policy,
                    "counterfactual_valid": _required(
                        outputs,
                        "selection_slot_v19_counterfactual_valid",
                    )[item]
                    .detach()
                    .cpu(),
                    "predicted_fidelity": _required(
                        outputs, "selection_slot_v19_fidelity_delta"
                    )[item]
                    .detach()
                    .float()
                    .cpu(),
                    "predicted_iou": _required(
                        outputs, "selection_slot_v19_expected_iou"
                    )[item]
                    .detach()
                    .float()
                    .cpu(),
                    "surrogate_target": surrogate_target[item]
                    .detach()
                    .float()
                    .cpu(),
                    "v7_route": v7_route[item].detach().cpu(),
                    "v19_route": v19_route[item].detach().cpu(),
                }
            )

    evaluated = _evaluate_records(
        records,
        line_width=30.0,
        min_valid_rows=5,
        workers=int(args.metric_workers),
    )
    tree = _new_metric_tree(POLICIES, (0.50, 0.75))
    paired = {
        threshold: {"improved": 0, "worsened": 0, "tied": 0}
        for threshold in ("0.50", "0.75")
    }
    source_correct = {"0.50": 0, "0.75": 0}
    source_correct_lost = {"0.50": 0, "0.75": 0}
    source_oracle = {"0.50": 0, "0.75": 0}
    endpoint_oracle = {"0.50": 0, "0.75": 0}
    predicted_values: list[torch.Tensor] = []
    predicted_iou_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    surrogate_values: list[torch.Tensor] = []
    pair_correct = 0
    pair_total = 0
    crossing_correct = 0
    crossing_total = 0
    selected_quality_sum = 0.0
    v7_quality_sum = 0.0
    oracle_quality_sum = 0.0
    selection_slots = 0

    for item, result in zip(records, evaluated):
        _accumulate_record(
            tree,
            result,
            {name: item["layout"][name] for name in POLICIES},
            item["active_by_policy"],
            (0.50, 0.75),
        )
        quality = result["quality"].float()
        valid = result["valid"].bool()
        for threshold in (0.50, 0.75):
            assignments = {}
            for name in POLICIES:
                start, stop = item["layout"][name]
                selected = item["active_by_policy"][name].bool() & valid[
                    start:stop
                ]
                ids = torch.nonzero(
                    selected, as_tuple=False
                ).flatten().tolist()
                assignments[name] = evaluator_hungarian_assignment(
                    quality[:, start:stop], ids, threshold=threshold
                )
            source_assignment = assignments["source_v7"]
            endpoint_assignment = assignments["v19_deployed"]
            delta = endpoint_assignment.hit_count - source_assignment.hit_count
            label = "improved" if delta > 0 else "worsened" if delta < 0 else "tied"
            paired[f"{threshold:.2f}"][label] += 1
            source_gt = {gt for gt, _pred in source_assignment.pairs}
            endpoint_gt = {gt for gt, _pred in endpoint_assignment.pairs}
            source_correct[f"{threshold:.2f}"] += len(source_gt)
            source_correct_lost[f"{threshold:.2f}"] += len(
                source_gt - endpoint_gt
            )
            for name, accumulator in (
                ("source_proposals", source_oracle),
                ("v19_proposals", endpoint_oracle),
            ):
                start, stop = item["layout"][name]
                ids = torch.nonzero(
                    valid[start:stop], as_tuple=False
                ).flatten().tolist()
                accumulator[f"{threshold:.2f}"] += (
                    evaluator_hungarian_assignment(
                        quality[:, start:stop], ids, threshold=threshold
                    ).hit_count
                )

        start, stop = item["layout"]["counterfactual"]
        cf_quality = quality[:, start:stop]
        cf_valid = valid[start:stop] & item["counterfactual_valid"].reshape(-1)
        slots, candidates = item["predicted_fidelity"].shape
        if cf_quality.shape[0] > 0:
            target, best_gt = cf_quality.max(dim=0)
        else:
            target = torch.zeros(slots * candidates)
            best_gt = torch.full((slots * candidates,), -1, dtype=torch.long)
        pred = item["predicted_fidelity"].reshape(-1)
        pred_iou = item["predicted_iou"].reshape(-1)
        predicted_values.append(pred[cf_valid])
        predicted_iou_values.append(pred_iou[cf_valid])
        target_values.append(target[cf_valid])
        surrogate_values.append(item["surrogate_target"].reshape(-1)[cf_valid])

        target_slot = target.reshape(slots, candidates)
        gt_slot = best_gt.reshape(slots, candidates)
        valid_slot = cf_valid.reshape(slots, candidates)
        pred_slot = item["predicted_fidelity"]
        for slot in range(slots):
            q = target_slot[slot]
            p = pred_slot[slot]
            same = gt_slot[slot, :, None] == gt_slot[slot, None, :]
            usable = valid_slot[slot, :, None] & valid_slot[slot, None, :]
            upper = torch.triu(
                torch.ones(candidates, candidates, dtype=torch.bool),
                diagonal=1,
            )
            gap = q[:, None] - q[None, :]
            pair = usable & same & upper & (gap.abs() >= 0.02)
            signed = gap.sign() * (p[:, None] - p[None, :])
            pair_correct += int((signed[pair] > 0).sum())
            pair_total += int(pair.sum())
            crossing = pair & (
                ((q[:, None] >= 0.50) != (q[None, :] >= 0.50))
                | ((q[:, None] >= 0.75) != (q[None, :] >= 0.75))
            )
            crossing_correct += int((signed[crossing] > 0).sum())
            crossing_total += int(crossing.sum())

            if not bool(valid_slot[slot].any()):
                continue
            v19_id = int(item["v19_route"][slot])
            v7_id = int(item["v7_route"][slot])
            if v19_id < 0 or v7_id < 0:
                continue
            selected_quality_sum += float(q[v19_id])
            v7_quality_sum += float(q[v7_id])
            oracle_quality_sum += float(q[valid_slot[slot]].max())
            selection_slots += 1

    metrics = _finalize_metric_tree(tree)
    predicted_all = torch.cat(predicted_values) if predicted_values else torch.zeros(0)
    predicted_iou_all = (
        torch.cat(predicted_iou_values)
        if predicted_iou_values
        else torch.zeros(0)
    )
    target_all = torch.cat(target_values) if target_values else torch.zeros(0)
    surrogate_all = (
        torch.cat(surrogate_values) if surrogate_values else torch.zeros(0)
    )
    source_mean = v7_quality_sum / max(selection_slots, 1)
    endpoint_mean = selected_quality_sum / max(selection_slots, 1)
    oracle_mean = oracle_quality_sum / max(selection_slots, 1)
    report = {
        "experiment": "V19 frozen counterfactual proposal fidelity official audit",
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
        "metrics": metrics,
        "v19_minus_source": _metric_comparison(metrics),
        "paired_image_effects": paired,
        "source_correct_degradation": {
            threshold: {
                "source_correct": source_correct[threshold],
                "lost_by_v19": source_correct_lost[threshold],
                "fraction": source_correct_lost[threshold]
                / max(source_correct[threshold], 1),
            }
            for threshold in ("0.50", "0.75")
        },
        "proposal_oracle": {
            "source_tp": source_oracle,
            "v19_tp": endpoint_oracle,
            "exact_nonregression": source_oracle == endpoint_oracle,
            "proposal_tensor_max_difference": proposal_tensor_difference,
        },
        "cardinality": {
            "exact_source_count_images": count_equal_images,
            "images": total_images,
            "exact_fraction": count_equal_images / max(total_images, 1),
        },
        "official_counterfactual_fidelity": {
            "candidate_values": int(target_all.numel()),
            "fidelity_pearson": _pearson(predicted_all, target_all),
            "predicted_iou_pearson": _pearson(predicted_iou_all, target_all),
            "training_surrogate_pearson": _pearson(
                surrogate_all, target_all
            ),
            "training_surrogate_spearman_ordinal": _pearson(
                _ordinal_rank(surrogate_all), _ordinal_rank(target_all)
            ),
            "training_surrogate_threshold_agreement_50": float(
                ((surrogate_all >= 0.50) == (target_all >= 0.50))
                .float()
                .mean()
            ) if target_all.numel() else 0.0,
            "training_surrogate_threshold_agreement_75": float(
                ((surrogate_all >= 0.75) == (target_all >= 0.75))
                .float()
                .mean()
            ) if target_all.numel() else 0.0,
            "fidelity_spearman_ordinal": _pearson(
                _ordinal_rank(predicted_all), _ordinal_rank(target_all)
            ),
            "same_gt_pair_accuracy": pair_correct / max(pair_total, 1),
            "same_gt_pairs": pair_total,
            "threshold_crossing_pair_accuracy": crossing_correct
            / max(crossing_total, 1),
            "threshold_crossing_pairs": crossing_total,
            "source_selected_mean_iou": source_mean,
            "v19_selected_mean_iou": endpoint_mean,
            "oracle_mean_iou": oracle_mean,
            "source_regret": oracle_mean - source_mean,
            "v19_regret": oracle_mean - endpoint_mean,
        },
        "route_changed_fraction": route_changed / max(route_total, 1),
        "optimizer_steps_during_audit": 0,
        "checkpoint_selection_performed": False,
        "threshold_search_performed": False,
        "nms_search_performed": False,
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
