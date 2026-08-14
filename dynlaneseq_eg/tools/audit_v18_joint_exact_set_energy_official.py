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
from dynlaneseq_eg.losses.loss_s0 import (
    _padded_lane_targets,
    batched_pairwise_range_aware_row_strip_iou,
)
from dynlaneseq_eg.modeling.v18_joint_exact_set_energy import (
    exact_ordered_set_energies,
    exact_unordered_set_rewards,
    unordered_set_log_scores,
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


POLICIES = (
    "source_v7",
    "endpoint_legacy_v7",
    "v18_anchor",
    "v18_refine_always",
    "v18_deployed",
    "cross_clip_wrong_image",
    "cross_clip_wrong_proposal_memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official V18 source/endpoint and causal intervention audit."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--wrong-list-path", required=True)
    parser.add_argument("--cross-clip-report", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _config(path: str, args: argparse.Namespace, list_path: str) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root).expanduser())
    cfg["dataset"].setdefault("lists", {})[args.split] = str(
        Path(list_path).resolve()
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
        raise KeyError(f"missing V18 output: {name}")
    return value


def _capture_selector_inputs(storage: dict[str, Any]):
    def hook(_module, args, kwargs):
        if args:
            base_outputs = args[0]
        else:
            base_outputs = kwargs["outputs"]
        storage["outputs"] = dict(base_outputs)
        storage["row_value_features"] = kwargs.get("row_value_features")
        scales = kwargs.get("multi_scale_features")
        storage["multi_scale_features"] = (
            dict(scales) if isinstance(scales, dict) else scales
        )

    return hook


def _selector_replay(
    selector,
    captured: dict[str, Any],
    *,
    row_value_features: torch.Tensor | None = None,
    multi_scale_features: dict[str, torch.Tensor] | None = None,
    proposal_rows: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    outputs = dict(captured["outputs"])
    if proposal_rows is not None:
        outputs["structured_row_tokens"] = proposal_rows
    rows = (
        captured["row_value_features"]
        if row_value_features is None
        else row_value_features
    )
    scales = (
        captured["multi_scale_features"]
        if multi_scale_features is None
        else multi_scale_features
    )
    return selector(
        outputs,
        row_value_features=rows,
        multi_scale_features=scales,
    )


def _deltas(metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons = OrderedDict(
        (
            ("v18_minus_source", ("v18_deployed", "source_v7")),
            ("endpoint_legacy_minus_source", ("endpoint_legacy_v7", "source_v7")),
            ("v18_minus_endpoint_legacy", ("v18_deployed", "endpoint_legacy_v7")),
            ("anchor_minus_source", ("v18_anchor", "source_v7")),
            ("refine_always_minus_anchor", ("v18_refine_always", "v18_anchor")),
            (
                "correct_minus_cross_clip_wrong_image",
                ("v18_deployed", "cross_clip_wrong_image"),
            ),
            (
                "correct_minus_cross_clip_wrong_proposal_memory",
                ("v18_deployed", "cross_clip_wrong_proposal_memory"),
            ),
        )
    )
    return {
        label: {
            mode: {
                threshold: _metric_delta(
                    metrics, treatment, control, mode, threshold
                )
                for threshold in ("0.50", "0.75")
            }
            for mode in ("neural_active", "writer_valid")
        }
        for label, (treatment, control) in comparisons.items()
    }


def _mean_regret(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    module,
    *,
    input_h: int,
    line_width: float,
    min_valid_rows: int,
    use_v18_scores: bool,
) -> tuple[float, int]:
    proposal_x = outputs["pred_x_rows"].detach().float()
    proposal_range = outputs["range_norm"].detach().float()
    padded_x, padded_valid = _padded_lane_targets(
        targets,
        device=proposal_x.device,
        dtype=torch.float32,
        rows=int(proposal_x.shape[-1]),
    )
    quality, quality_valid, gt_valid = batched_pairwise_range_aware_row_strip_iou(
        proposal_x,
        proposal_range,
        padded_x,
        padded_valid,
        input_h=int(input_h),
        line_width=float(line_width),
        min_valid_rows=int(min_valid_rows),
    )
    candidate_valid = outputs["selection_slot_candidate_valid"].detach().bool()
    reward, valid = exact_unordered_set_rewards(
        quality,
        gt_valid,
        candidate_valid & quality_valid,
        module.combination_table,
    )
    if use_v18_scores:
        scores = outputs["selection_slot_v18_unordered_set_scores"].detach().float()
        valid = valid & outputs["selection_slot_v18_valid_set"].detach().bool()
    else:
        unary = outputs["selection_slot_real_route_logits"].detach().float()
        pair = unary.new_zeros(
            unary.shape[0], 6, unary.shape[-1], unary.shape[-1]
        )
        ordered, valid_energy = exact_ordered_set_energies(
            unary,
            pair,
            outputs["selection_slot_active_logits"].detach().float(),
            candidate_valid,
            module.ordered_assignments,
            module.slot_pairs,
        )
        scores = unordered_set_log_scores(
            ordered,
            valid_energy,
            permutation_temperature=float(module.permutation_temperature),
        )
        valid = valid & valid_energy
    values: list[torch.Tensor] = []
    for batch_index in range(int(scores.shape[0])):
        keep = valid[batch_index]
        if not bool(keep.any()):
            continue
        local_score = scores[batch_index, keep]
        local_reward = reward[batch_index, keep]
        values.append(local_reward.max() - local_reward[local_score.argmax()])
    if not values:
        return 0.0, 0
    return float(torch.stack(values).sum().cpu()), len(values)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    crossclip = json.loads(Path(args.cross_clip_report).read_text())
    if not (
        crossclip.get("passed") is True
        and int(crossclip.get("same_image_partner_count", -1)) == 0
        and int(crossclip.get("same_clip_partner_count", -1)) == 0
    ):
        raise ValueError("invalid cross-clip negative-control report")
    cfg = _config(args.config, args, args.list_path)
    source_cfg = _config(args.source_config, args, args.list_path)
    wrong_cfg = _config(args.config, args, args.wrong_list_path)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    selector = model.structured_query_head.set_selection_head
    module = selector.joint_exact_set_energy
    if module is None:
        raise ValueError("V18 official audit requires exact-set module")
    source_model = build_model(source_cfg).to(device)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    source_model.eval()

    loader = build_dataloader(cfg, split=args.split, training=False)
    wrong_loader = build_dataloader(wrong_cfg, split=args.split, training=False)
    if len(loader.dataset) != len(wrong_loader.dataset):
        raise ValueError("cross-clip lists are not aligned")

    records: list[dict[str, Any]] = []
    same_image = 0
    same_clip = 0
    intervention_change = {"wrong_image": [], "wrong_proposal_memory": []}
    source_regret_sum = 0.0
    endpoint_regret_sum = 0.0
    regret_images = 0
    route_changed = 0
    route_total = 0
    refine_slots = 0
    active_slots = 0

    for batch_index, (correct_batch, wrong_batch) in enumerate(
        tqdm(
            zip(loader, wrong_loader),
            total=len(loader),
            desc="V18 official",
            ncols=90,
        )
    ):
        images, targets, metas = correct_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        images = images.to(device, non_blocking=True)
        wrong_images = wrong_images.to(device, non_blocking=True)
        targets_device = [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        correct_capture: dict[str, Any] = {}
        wrong_capture: dict[str, Any] = {}
        handle = selector.register_forward_pre_hook(
            _capture_selector_inputs(correct_capture), with_kwargs=True
        )
        outputs = model(images)
        handle.remove()
        wrong_handle = selector.register_forward_pre_hook(
            _capture_selector_inputs(wrong_capture), with_kwargs=True
        )
        model(wrong_images)
        wrong_handle.remove()
        source_outputs = source_model(images)
        if not correct_capture or not wrong_capture:
            raise RuntimeError("V18 audit failed to capture selector inputs")

        for meta, wrong_meta in zip(metas, wrong_metas):
            left = str(meta.get("image_path", ""))
            right = str(wrong_meta.get("image_path", ""))
            same_image += int(left == right)
            same_clip += int(str(Path(left).parent) == str(Path(right).parent))

        saved_module = selector.joint_exact_set_energy
        selector.joint_exact_set_energy = None
        endpoint_legacy = _selector_replay(selector, correct_capture)
        selector.joint_exact_set_energy = saved_module
        wrong_image = _selector_replay(
            selector,
            correct_capture,
            row_value_features=wrong_capture["row_value_features"],
            multi_scale_features=wrong_capture["multi_scale_features"],
        )
        wrong_proposal = _selector_replay(
            selector,
            correct_capture,
            proposal_rows=wrong_capture["outputs"]["structured_row_tokens"],
        )

        final_x = _required(outputs, "selection_slot_pred_x_rows")
        intervention_change["wrong_image"].append(
            float(
                (wrong_image["selection_slot_pred_x_rows"] - final_x)
                .abs()
                .mean()
                .cpu()
            )
        )
        intervention_change["wrong_proposal_memory"].append(
            float(
                (wrong_proposal["selection_slot_pred_x_rows"] - final_x)
                .abs()
                .mean()
                .cpu()
            )
        )
        source_regret, source_count = _mean_regret(
            source_outputs,
            targets_device,
            module,
            input_h=int(cfg["model"]["input_h"]),
            line_width=float(cfg["loss"].get("four_slot_line_width", 30.0)),
            min_valid_rows=int(cfg["loss"].get("four_slot_min_valid_rows", 5)),
            use_v18_scores=False,
        )
        endpoint_regret, endpoint_count = _mean_regret(
            outputs,
            targets_device,
            module,
            input_h=int(cfg["model"]["input_h"]),
            line_width=float(cfg["loss"].get("four_slot_line_width", 30.0)),
            min_valid_rows=int(cfg["loss"].get("four_slot_min_valid_rows", 5)),
            use_v18_scores=True,
        )
        if source_count != endpoint_count:
            raise ValueError("source/endpoint set-regret image counts differ")
        source_regret_sum += source_regret
        endpoint_regret_sum += endpoint_regret
        regret_images += source_count

        legacy_ids = _required(outputs, "selection_slot_v18_v7_geometry_route_indices")
        v18_ids = _required(outputs, "selection_slot_geometry_route_indices")
        route_changed += int((legacy_ids != v18_ids).sum().cpu())
        route_total += int(v18_ids.numel())
        policy = _required(outputs, "selection_slot_v18_policy")
        deployed_active = _required(outputs, "selection_slot_active").bool()
        refine_slots += int(((policy == 1) & deployed_active).sum().cpu())
        active_slots += int(deployed_active.sum().cpu())

        policies = OrderedDict(
            (
                (
                    "source_v7",
                    (
                        _required(source_outputs, "selection_slot_pred_x_rows"),
                        _required(source_outputs, "selection_slot_range_norm"),
                        _required(source_outputs, "selection_slot_active").bool(),
                    ),
                ),
                (
                    "endpoint_legacy_v7",
                    (
                        _required(endpoint_legacy, "selection_slot_pred_x_rows"),
                        _required(endpoint_legacy, "selection_slot_range_norm"),
                        _required(endpoint_legacy, "selection_slot_active").bool(),
                    ),
                ),
                (
                    "v18_anchor",
                    (
                        _required(outputs, "selection_slot_v18_anchor_x_rows"),
                        _required(outputs, "selection_slot_v18_anchor_range_norm"),
                        deployed_active,
                    ),
                ),
                (
                    "v18_refine_always",
                    (
                        _required(outputs, "selection_slot_v18_refined_x_rows"),
                        _required(outputs, "selection_slot_v18_refined_range_norm"),
                        deployed_active,
                    ),
                ),
                (
                    "v18_deployed",
                    (
                        final_x,
                        _required(outputs, "selection_slot_range_norm"),
                        deployed_active,
                    ),
                ),
                (
                    "cross_clip_wrong_image",
                    (
                        _required(wrong_image, "selection_slot_pred_x_rows"),
                        _required(wrong_image, "selection_slot_range_norm"),
                        _required(wrong_image, "selection_slot_active").bool(),
                    ),
                ),
                (
                    "cross_clip_wrong_proposal_memory",
                    (
                        _required(wrong_proposal, "selection_slot_pred_x_rows"),
                        _required(wrong_proposal, "selection_slot_range_norm"),
                        _required(wrong_proposal, "selection_slot_active").bool(),
                    ),
                ),
            )
        )
        for batch_item, meta in enumerate(metas):
            geometry_rows: list[torch.Tensor] = []
            range_rows: list[torch.Tensor] = []
            layout: dict[str, tuple[int, int]] = {}
            active_by_policy: dict[str, torch.Tensor] = {}
            cursor = 0
            for name, (x_rows, lane_range, active) in policies.items():
                geometry_rows.append(x_rows[batch_item].detach().float().cpu())
                range_rows.append(lane_range[batch_item].detach().float().cpu())
                count = int(x_rows.shape[1])
                layout[name] = (cursor, cursor + count)
                active_by_policy[name] = active[batch_item].detach().cpu()
                cursor += count
            records.append(
                {
                    "image_id": _image_id(
                        meta, f"v18_{batch_index:06d}_{batch_item}"
                    ),
                    "record": {
                        "meta": _plain_meta(meta),
                        "stages": {
                            "combined": {
                                "pred_x_rows": torch.cat(geometry_rows, dim=0),
                                "range_norm": torch.cat(range_rows, dim=0),
                            }
                        },
                    },
                    "layout": layout,
                    "active_by_policy": active_by_policy,
                    "policy": policy[batch_item].detach().cpu(),
                }
            )
    if same_image or same_clip:
        raise ValueError("runtime cross-clip pairing was contaminated")

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
    refine_decisions = 0
    refine_improved = 0
    refine_nonworse = 0
    for item, result in zip(records, evaluated):
        _accumulate_record(
            tree,
            result,
            item["layout"],
            item["active_by_policy"],
            (0.50, 0.75),
        )
        quality = result["quality"].float()
        valid = result["valid"].bool()
        assignments: dict[str, dict[str, Any]] = {}
        for threshold in (0.50, 0.75):
            for policy_name in ("source_v7", "v18_deployed"):
                start, stop = item["layout"][policy_name]
                selected = item["active_by_policy"][policy_name].bool() & valid[
                    start:stop
                ]
                ids = torch.nonzero(selected, as_tuple=False).flatten().tolist()
                assignments[f"{policy_name}_{threshold:.2f}"] = (
                    evaluator_hungarian_assignment(
                        quality[:, start:stop], ids, threshold=float(threshold)
                    )
                )
            source_assignment = assignments[f"source_v7_{threshold:.2f}"]
            endpoint_assignment = assignments[f"v18_deployed_{threshold:.2f}"]
            delta = endpoint_assignment.hit_count - source_assignment.hit_count
            label = "improved" if delta > 0 else "worsened" if delta < 0 else "tied"
            paired[f"{threshold:.2f}"][label] += 1
            source_gt = {gt for gt, _pred in source_assignment.pairs}
            endpoint_gt = {gt for gt, _pred in endpoint_assignment.pairs}
            source_correct[f"{threshold:.2f}"] += len(source_gt)
            source_correct_lost[f"{threshold:.2f}"] += len(source_gt - endpoint_gt)

        final_start, final_stop = item["layout"]["v18_deployed"]
        final_selected = item["active_by_policy"]["v18_deployed"].bool() & valid[
            final_start:final_stop
        ]
        final_ids = torch.nonzero(final_selected, as_tuple=False).flatten().tolist()
        final_assignment = evaluator_hungarian_assignment(
            quality[:, final_start:final_stop], final_ids, threshold=0.0
        )
        anchor_start, _ = item["layout"]["v18_anchor"]
        refine_start, _ = item["layout"]["v18_refine_always"]
        for gt_id, slot_id in final_assignment.pairs:
            if int(item["policy"][slot_id]) != 1:
                continue
            q0 = float(quality[gt_id, anchor_start + slot_id])
            q1 = float(quality[gt_id, refine_start + slot_id])
            refine_decisions += 1
            refine_improved += int(q1 > q0 + 1.0e-9)
            refine_nonworse += int(q1 + 1.0e-9 >= q0)

    metrics = _finalize_metric_tree(tree)
    source_mean_regret = source_regret_sum / max(regret_images, 1)
    endpoint_mean_regret = endpoint_regret_sum / max(regret_images, 1)
    regret_reduction = (
        (source_mean_regret - endpoint_mean_regret) / source_mean_regret
        if source_mean_regret > 0.0
        else 0.0
    )
    report = {
        "experiment": "V18 joint exact set-energy official causal replay",
        "config": str(Path(args.config).resolve()),
        "source_config": str(Path(args.source_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "iteration": iteration,
        "source_iteration": source_iteration,
        "split": args.split,
        "list_path": str(Path(args.list_path).resolve()),
        "list_sha256": sha256_file(Path(args.list_path)),
        "wrong_list_path": str(Path(args.wrong_list_path).resolve()),
        "images": len(records),
        "cross_clip_runtime_same_image": same_image,
        "cross_clip_runtime_same_clip": same_clip,
        "metrics": metrics,
        "deltas": _deltas(metrics),
        "paired_image_effects": paired,
        "source_correct_degradation": {
            threshold: {
                "source_correct": source_correct[threshold],
                "lost_by_v18": source_correct_lost[threshold],
                "fraction": source_correct_lost[threshold]
                / max(source_correct[threshold], 1),
            }
            for threshold in ("0.50", "0.75")
        },
        "chosen_set_regret": {
            "images": regret_images,
            "source_mean": source_mean_regret,
            "endpoint_mean": endpoint_mean_regret,
            "relative_reduction": regret_reduction,
        },
        "refine_policy": {
            "active_refine_fraction": refine_slots / max(active_slots, 1),
            "matched_refine_decisions": refine_decisions,
            "precision_improved": refine_improved / max(refine_decisions, 1),
            "precision_nonworse": refine_nonworse / max(refine_decisions, 1),
        },
        "route_changed_fraction": route_changed / max(route_total, 1),
        "intervention_mean_abs_x_change_px": {
            name: sum(values) / max(len(values), 1)
            for name, values in intervention_change.items()
        },
        "optimizer_steps_during_audit": 0,
        "checkpoint_selection_performed": False,
        "threshold_search_performed": False,
        "nms_search_performed": False,
        "full_validation_executed": False,
        "test_set_used": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_json": str(destination),
                "deltas": report["deltas"],
                "chosen_set_regret": report["chosen_set_regret"],
                "refine_policy": report["refine_policy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
