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
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.signed_lane_registration import (
    SignedRegistrationResult,
    gather_whole_proposals,
    signed_curve_registration,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.v16_candidate_groups import (
    build_v16_anchor_candidate_groups,
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
    "v7_anchor_reference",
    "v7_refined",
    "registration_correct_p2",
    "registration_cross_clip_wrong_p2",
    "registration_zero_content_p2",
    "registration_position_only_p2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free V14-posterior/V16-group signed registration replay."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--wrong-list-path", required=True)
    parser.add_argument("--cross-clip-report", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--inspect-index", type=int, default=151)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Debug-only early stop; zero evaluates the complete fixed list.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _config(args: argparse.Namespace, list_path: str) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg["dataset"].setdefault("lists", {})[args.split] = str(
        Path(list_path).expanduser().resolve()
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
        raise KeyError(f"missing V16.1 output: {name}")
    return value


def _selected_diagnostic(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return values.gather(2, indices.unsqueeze(-1)).squeeze(-1)


def _deltas(metrics: dict[str, Any]) -> dict[str, Any]:
    comparisons = OrderedDict(
        (
            (
                "correct_minus_v7_anchor_reference",
                ("registration_correct_p2", "v7_anchor_reference"),
            ),
            (
                "correct_minus_v7_refined",
                ("registration_correct_p2", "v7_refined"),
            ),
            (
                "correct_minus_cross_clip_wrong_p2",
                (
                    "registration_correct_p2",
                    "registration_cross_clip_wrong_p2",
                ),
            ),
            (
                "correct_minus_zero_content_p2",
                (
                    "registration_correct_p2",
                    "registration_zero_content_p2",
                ),
            ),
            (
                "correct_minus_position_only_p2",
                (
                    "registration_correct_p2",
                    "registration_position_only_p2",
                ),
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


def _registration(
    *,
    visual_logits: torch.Tensor,
    module_inputs: dict[str, torch.Tensor],
    anchor_range: torch.Tensor,
    group: dict[str, torch.Tensor],
    writer_valid: torch.Tensor,
    input_w: int,
) -> SignedRegistrationResult:
    return signed_curve_registration(
        visual_logits=visual_logits,
        proposal_x_rows=module_inputs["proposal_x_rows"],
        proposal_range_norm=module_inputs["proposal_range_norm"],
        anchor_range_norm=anchor_range,
        proposal_visible=group["proposal_visible"],
        group_mask=group["group_mask"],
        writer_valid=writer_valid,
        input_w=input_w,
        min_valid_rows=5,
    )


def _inspect_item(
    *,
    image_id: str,
    batch_item: int,
    anchor_ids: torch.Tensor,
    writer_valid: torch.Tensor,
    group_mask: torch.Tensor,
    variants: dict[str, SignedRegistrationResult],
) -> dict[str, Any]:
    slots: list[dict[str, Any]] = []
    for slot in range(int(anchor_ids.shape[1])):
        if not bool(writer_valid[batch_item, slot]):
            continue
        candidate_ids = group_mask[batch_item, slot].nonzero(
            as_tuple=False
        ).flatten()
        slot_report: dict[str, Any] = {
            "slot": slot,
            "anchor_id": int(anchor_ids[batch_item, slot]),
            "group_candidate_ids": [int(value) for value in candidate_ids],
            "variants": {},
        }
        for name, result in variants.items():
            chosen = int(result.selected_indices[batch_item, slot])
            slot_report["variants"][name] = {
                "selected_id": chosen,
                "candidate_scores": {
                    str(int(candidate)): float(
                        result.scores[batch_item, slot, candidate]
                    )
                    for candidate in candidate_ids
                },
                "candidate_mean_signed_displacement_px": {
                    str(int(candidate)): float(
                        result.mean_signed_displacement_px[
                            batch_item, slot, candidate
                        ]
                    )
                    for candidate in candidate_ids
                },
                "candidate_p90_absolute_displacement_px": {
                    str(int(candidate)): float(
                        result.p90_absolute_displacement_px[
                            batch_item, slot, candidate
                        ]
                    )
                    for candidate in candidate_ids
                },
            }
        slots.append(slot_report)
    return {"image_id": image_id, "slots": slots}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    crossclip = json.loads(Path(args.cross_clip_report).expanduser().read_text())
    if not (
        crossclip.get("passed") is True
        and int(crossclip.get("same_image_partner_count", -1)) == 0
        and int(crossclip.get("same_clip_partner_count", -1)) == 0
    ):
        raise ValueError("invalid V16.1 cross-clip negative-control report")
    cfg = _config(args, args.list_path)
    wrong_cfg = _config(args, args.wrong_list_path)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    input_w = int(cfg.get("model", {}).get("input_w", 0))
    if input_w <= 1:
        raise ValueError("V16.1 requires a resolved positive model.input_w")
    selector = model.structured_query_head.set_selection_head
    module = selector.corrected_visual_first_association
    if module is None:
        raise ValueError("V16.1 requires the V14 Stage-A association module")
    loader = build_dataloader(cfg, split=args.split, training=False)
    wrong_loader = build_dataloader(wrong_cfg, split=args.split, training=False)
    if len(loader.dataset) != len(wrong_loader.dataset):
        raise ValueError("V16.1 correct/wrong lists are not aligned")

    records: list[dict[str, Any]] = []
    inspect_records: list[dict[str, Any]] = []
    same_image = 0
    same_clip = 0
    global_image_index = 0
    writer_slots = 0
    group_sizes: list[float] = []
    outside_group_count = 0
    duplicate_counts = {name: 0 for name in (
        "correct", "cross_clip_wrong", "zero_content", "position_only"
    )}
    changed_from_anchor = {name: 0 for name in duplicate_counts}
    changed_from_correct = {name: 0 for name in (
        "cross_clip_wrong", "zero_content", "position_only"
    )}
    selected_signed: dict[str, list[float]] = {
        name: [] for name in duplicate_counts
    }
    selected_p90: dict[str, list[float]] = {
        name: [] for name in duplicate_counts
    }
    exact_gather_max_abs = 0.0

    for batch_index, (correct_batch, wrong_batch) in enumerate(
        tqdm(
            zip(loader, wrong_loader),
            total=len(loader),
            desc="V16.1 signed registration",
            ncols=100,
        )
    ):
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        images, _targets, metas = correct_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        images = images.to(device, non_blocking=True)
        wrong_images = wrong_images.to(device, non_blocking=True)
        correct_inputs: dict[str, torch.Tensor] = {}
        wrong_inputs: dict[str, torch.Tensor] = {}

        def capture(_module, _args, values):
            correct_inputs.update(values)

        def capture_wrong(_module, _args, values):
            wrong_inputs.update(values)

        handle = module.register_forward_pre_hook(capture, with_kwargs=True)
        outputs = model(images)
        handle.remove()
        wrong_handle = module.register_forward_pre_hook(
            capture_wrong, with_kwargs=True
        )
        model(wrong_images)
        wrong_handle.remove()
        if not correct_inputs or not wrong_inputs:
            raise RuntimeError("V16.1 failed to capture V14 module inputs")

        for meta, wrong_meta in zip(metas, wrong_metas):
            left = str(meta.get("image_path", ""))
            right = str(wrong_meta.get("image_path", ""))
            same_image += int(left == right)
            same_clip += int(str(Path(left).parent) == str(Path(right).parent))

        wrong_variant = module(
            **{
                **correct_inputs,
                "row_value_features": wrong_inputs["row_value_features"],
            }
        )
        zero_variant = module(**correct_inputs, feature_policy="zero_content")
        position_variant = module(**correct_inputs, feature_policy="position_only")

        anchor_ids = _required(
            outputs, "selection_slot_geometry_route_indices"
        ).long()
        writer_valid = _required(
            outputs, "selection_slot_v14_writer_valid"
        ).bool()
        anchor_x = _required(outputs, "selection_slot_v14_anchor_x_rows")
        anchor_range = _required(
            outputs, "selection_slot_v14_anchor_range_norm"
        )
        group = build_v16_anchor_candidate_groups(
            proposal_x_rows=correct_inputs["proposal_x_rows"],
            proposal_range_norm=correct_inputs["proposal_range_norm"],
            candidate_valid=correct_inputs["candidate_valid"],
            anchor_indices=anchor_ids,
            anchor_active=correct_inputs["anchor_active"],
            input_w=input_w,
        )
        correct_registration = _registration(
            visual_logits=_required(outputs, "selection_slot_v14_visual_logits"),
            module_inputs=correct_inputs,
            anchor_range=anchor_range,
            group=group,
            writer_valid=writer_valid,
            input_w=input_w,
        )
        wrong_registration = _registration(
            visual_logits=_required(
                wrong_variant, "selection_slot_v14_visual_logits"
            ),
            module_inputs=correct_inputs,
            anchor_range=anchor_range,
            group=group,
            writer_valid=writer_valid,
            input_w=input_w,
        )
        zero_registration = _registration(
            visual_logits=_required(
                zero_variant, "selection_slot_v14_visual_logits"
            ),
            module_inputs=correct_inputs,
            anchor_range=anchor_range,
            group=group,
            writer_valid=writer_valid,
            input_w=input_w,
        )
        position_registration = _registration(
            visual_logits=_required(
                position_variant, "selection_slot_v14_visual_logits"
            ),
            module_inputs=correct_inputs,
            anchor_range=anchor_range,
            group=group,
            writer_valid=writer_valid,
            input_w=input_w,
        )
        variants = {
            "correct": correct_registration,
            "cross_clip_wrong": wrong_registration,
            "zero_content": zero_registration,
            "position_only": position_registration,
        }
        group_mask = group["group_mask"].bool()
        writer_slots += int(writer_valid.sum())
        group_sizes.extend(
            group["group_size"][writer_valid].float().cpu().tolist()
        )
        for name, result in variants.items():
            ids = result.selected_indices
            selected_in_group = group_mask.gather(
                2, ids.unsqueeze(-1)
            ).squeeze(-1)
            outside_group_count += int((writer_valid & ~selected_in_group).sum())
            changed_from_anchor[name] += int(
                (writer_valid & (ids != anchor_ids)).sum()
            )
            for image_ids, active_ids in zip(ids, writer_valid):
                active_values = image_ids[active_ids]
                duplicate_counts[name] += int(
                    active_values.numel() - active_values.unique().numel()
                )
            selected_signed[name].extend(
                _selected_diagnostic(
                    result.mean_signed_displacement_px, ids
                )[writer_valid].cpu().tolist()
            )
            selected_p90[name].extend(
                _selected_diagnostic(
                    result.p90_absolute_displacement_px, ids
                )[writer_valid].cpu().tolist()
            )
        for name in changed_from_correct:
            changed_from_correct[name] += int(
                (
                    writer_valid
                    & (
                        correct_registration.selected_indices
                        != variants[name].selected_indices
                    )
                ).sum()
            )

        proposal_x = correct_inputs["proposal_x_rows"].detach().float()
        proposal_range = correct_inputs["proposal_range_norm"].detach().float()
        raw_anchor_x = gather_whole_proposals(proposal_x, anchor_ids)
        raw_anchor_range = gather_whole_proposals(proposal_range, anchor_ids)
        selected_geometry: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, result in variants.items():
            gathered_x = gather_whole_proposals(
                proposal_x, result.selected_indices
            )
            gathered_range = gather_whole_proposals(
                proposal_range, result.selected_indices
            )
            batch_ids = torch.arange(
                int(proposal_x.shape[0]), device=proposal_x.device
            ).view(-1, 1)
            independently_indexed_x = proposal_x[
                batch_ids, result.selected_indices
            ]
            independently_indexed_range = proposal_range[
                batch_ids, result.selected_indices
            ]
            exact_gather_max_abs = max(
                exact_gather_max_abs,
                float((gathered_x - independently_indexed_x).abs().max()),
                float(
                    (gathered_range - independently_indexed_range).abs().max()
                ),
            )
            selected_geometry[name] = (gathered_x, gathered_range)

        policies = OrderedDict(
            (
                ("v7_anchor_reference", (raw_anchor_x, raw_anchor_range)),
                ("v7_refined", (anchor_x, anchor_range)),
                (
                    "registration_correct_p2",
                    selected_geometry["correct"],
                ),
                (
                    "registration_cross_clip_wrong_p2",
                    selected_geometry["cross_clip_wrong"],
                ),
                (
                    "registration_zero_content_p2",
                    selected_geometry["zero_content"],
                ),
                (
                    "registration_position_only_p2",
                    selected_geometry["position_only"],
                ),
            )
        )
        for batch_item, meta in enumerate(metas):
            image_id = _image_id(
                meta, f"v16_1_{batch_index:06d}_{batch_item}"
            )
            if global_image_index == int(args.inspect_index):
                inspect_records.append(
                    _inspect_item(
                        image_id=image_id,
                        batch_item=batch_item,
                        anchor_ids=anchor_ids,
                        writer_valid=writer_valid,
                        group_mask=group_mask,
                        variants=variants,
                    )
                )
            geometry_rows = []
            range_rows = []
            layout = {}
            active_by_policy = {}
            cursor = 0
            for policy, (x_rows, lane_range) in policies.items():
                geometry_rows.append(x_rows[batch_item].float().cpu())
                range_rows.append(lane_range[batch_item].float().cpu())
                count = int(x_rows.shape[1])
                layout[policy] = (cursor, cursor + count)
                active_by_policy[policy] = writer_valid[batch_item].cpu()
                cursor += count
            records.append(
                {
                    "image_id": image_id,
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
                }
            )
            global_image_index += 1

    if same_image or same_clip:
        raise ValueError("runtime V16.1 cross-clip pairing was contaminated")
    if outside_group_count:
        raise RuntimeError("V16.1 selected a proposal outside its fixed group")
    evaluated = _evaluate_records(
        records,
        line_width=30.0,
        min_valid_rows=5,
        workers=int(args.metric_workers),
    )
    tree = _new_metric_tree(POLICIES, (0.50, 0.75))
    for item, result in zip(records, evaluated):
        _accumulate_record(
            tree,
            result,
            item["layout"],
            item["active_by_policy"],
            (0.50, 0.75),
        )
    metrics = _finalize_metric_tree(tree)

    def mean(values: list[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    report = {
        "experiment": "V16.1 proposal-independent signed registration replay",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "iteration": iteration,
        "split": args.split,
        "list_path": str(Path(args.list_path).resolve()),
        "list_sha256": sha256_file(Path(args.list_path)),
        "wrong_list_path": str(Path(args.wrong_list_path).resolve()),
        "images": len(records),
        "max_batches_debug": int(args.max_batches),
        "complete_fixed_list": int(args.max_batches) == 0,
        "input_w": input_w,
        "cross_clip_runtime_same_image": same_image,
        "cross_clip_runtime_same_clip": same_clip,
        "writer_valid_slots": writer_slots,
        "group_size": {
            "mean": mean(group_sizes),
            "min": min(group_sizes) if group_sizes else 0.0,
            "max": max(group_sizes) if group_sizes else 0.0,
        },
        "selection_changed_from_anchor_fraction": {
            name: float(value) / float(max(writer_slots, 1))
            for name, value in changed_from_anchor.items()
        },
        "selection_changed_from_correct_fraction": {
            name: float(value) / float(max(writer_slots, 1))
            for name, value in changed_from_correct.items()
        },
        "selected_duplicate_count": duplicate_counts,
        "selected_outside_group_count": outside_group_count,
        "exact_whole_proposal_gather_max_abs": exact_gather_max_abs,
        "selected_registration_diagnostics": {
            name: {
                "mean_signed_displacement_px": mean(selected_signed[name]),
                "mean_p90_absolute_displacement_px": mean(selected_p90[name]),
            }
            for name in selected_signed
        },
        "score_contract": {
            "row_distribution": "V14 log_softmax over full-width x bins",
            "row_value": "bilinear sampled relative log probability",
            "row_weight": "0.10 + 0.90*y^3",
            "aggregation": "0.5 weighted mean + 0.5 weighted q10",
            "learned_or_tuned_score_parameters": False,
        },
        "inspect_records": inspect_records,
        "metrics": metrics,
        "deltas": _deltas(metrics),
        "coordinate_averaging_used": False,
        "polynomial_fit_used": False,
        "whole_proposal_hard_selection": True,
        "target_or_gt_used_in_selection_forward": False,
        "optimizer_steps_during_audit": 0,
        "checkpoint_selection_performed": False,
        "threshold_or_nms_search_performed": False,
        "test_set_used": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output_json": str(destination), "deltas": report["deltas"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
