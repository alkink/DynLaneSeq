from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import (
    build_criterion,
    build_dataloader,
    build_matcher,
    build_model,
)
from dynlaneseq_eg.losses.loss_s0 import _padded_lane_targets
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.train import seed_everything


POLICIES = (
    "correct_p2",
    "cross_clip_wrong_p2",
    "zero_content",
    "position_only",
    "zero_content_zero_position",
    "x_reversed",
    "row_reversed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V14 association identifiability under exact fixed V7 "
            "assignment and deterministic P2 interventions."
        )
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
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _configure(
    path: str,
    *,
    dataset_root: str,
    split: str,
    list_path: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(path)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(dataset_root).expanduser()
    )
    cfg["dataset"].setdefault("lists", {})[split] = str(
        Path(list_path).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _state_digest(
    state: dict[str, torch.Tensor], names: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for name in names:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _clip(path: str) -> str:
    return str(Path(path).parent)


def _valid_visual_rows(
    matches: list[dict[str, torch.Tensor]],
    padded_valid: torch.Tensor,
) -> int:
    count = 0
    for batch_index, match in enumerate(matches):
        if match["gt_indices"].numel():
            count += int(
                padded_valid[batch_index, match["gt_indices"]].sum().item()
            )
    return count


def _association_rows(
    probability: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    hard_ids: torch.Tensor | None = None,
) -> tuple[list[float], list[float], list[int]]:
    representable = target["representable_match"].bool()
    support = target["target_support"].bool()
    target_id = target["target_id"].long()
    candidates = int(support.shape[-1])
    real = probability[..., :candidates].float()
    support_mass = (real * support.float()).sum(dim=-1)
    if hard_ids is None:
        hard_ids = probability.float().argmax(dim=-1)
    correct = hard_ids.long() == target_id
    return (
        support_mass[representable].cpu().tolist(),
        correct[representable].float().cpu().tolist(),
        hard_ids[representable].long().cpu().tolist(),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    correct_cfg = _configure(
        args.config,
        dataset_root=args.dataset_root,
        split=args.split,
        list_path=args.list_path,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    wrong_cfg = _configure(
        args.config,
        dataset_root=args.dataset_root,
        split=args.split,
        list_path=args.wrong_list_path,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    source_cfg = _configure(
        args.source_config,
        dataset_root=args.dataset_root,
        split=args.split,
        list_path=args.list_path,
        batch_size=args.eval_batch_size,
        num_workers=0,
    )
    cross_clip = json.loads(
        Path(args.cross_clip_report).read_text(encoding="utf-8")
    )
    if not (
        cross_clip.get("passed") is True
        and int(cross_clip.get("same_image_partner_count", -1)) == 0
        and int(cross_clip.get("same_clip_partner_count", -1)) == 0
    ):
        raise ValueError("cross-clip report does not prove an exact derangement")

    seed_everything(int(correct_cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(correct_cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    matcher = build_matcher(correct_cfg)
    criterion = build_criterion(correct_cfg).to(device)
    criterion.set_iteration(iteration)
    selector = model.structured_query_head.set_selection_head
    module = selector.corrected_visual_first_association
    if module is None:
        raise ValueError("V14 corrected visual-first module is absent")

    # Exact state equality proves the public graph still contains the frozen
    # V7 source weights at the endpoint, rather than merely similar outputs on
    # this diagnostic subset.
    source_model = build_model(source_cfg)
    source_iteration = int(
        load_checkpoint(args.source_checkpoint, source_model, strict=False)
    )
    source_names = tuple(sorted(source_model.state_dict()))
    endpoint_state = model.state_dict()
    missing_source_names = [
        name for name in source_names if name not in endpoint_state
    ]
    source_digest = _state_digest(source_model.state_dict(), source_names)
    endpoint_legacy_digest = (
        _state_digest(endpoint_state, source_names)
        if not missing_source_names
        else ""
    )
    legacy_state_exact = (
        not missing_source_names and source_digest == endpoint_legacy_digest
    )
    del source_model

    correct_loader = build_dataloader(
        correct_cfg, split=args.split, training=False
    )
    wrong_loader = build_dataloader(wrong_cfg, split=args.split, training=False)
    if len(correct_loader.dataset) != len(wrong_loader.dataset):
        raise ValueError("correct and wrong-P2 lists have different lengths")

    metrics: dict[str, list[float]] = {
        "v7_support_mass": [],
        "v7_hard_target_top1": [],
        "v7_raw_argmax_target_top1": [],
        "v7_route_logit_abs_mean": [],
        "v14_real_logit_abs_mean": [],
        "v7_v14_real_argmax_agreement": [],
        "v14_target_id_rank": [],
        "v14_target_id_top2": [],
        "v14_target_id_top4": [],
    }
    for policy in POLICIES:
        metrics[f"{policy}_support_mass"] = []
        metrics[f"{policy}_hard_target_top1"] = []
        metrics[f"{policy}_visual_dfl"] = []
        metrics[f"{policy}_visual_mae_px"] = []
        metrics[f"{policy}_proposal_attention_change"] = []
        metrics[f"{policy}_visual_attention_change"] = []

    visual_loss_sums = {policy: 0.0 for policy in POLICIES}
    visual_mae_sums = {policy: 0.0 for policy in POLICIES}
    visual_row_counts = {policy: 0 for policy in POLICIES}
    per_slot: list[dict[str, object]] = []
    actual_same_image = 0
    actual_same_clip = 0
    image_count = 0
    matched_count = 0
    representable_count = 0
    deployment_anchor_error = 0.0
    tensor_shapes: dict[str, list[int]] = {}

    iterator = zip(correct_loader, wrong_loader)
    for correct_batch, wrong_batch in tqdm(
        iterator,
        total=len(correct_loader),
        desc=f"V14 state {args.split}",
        ncols=90,
    ):
        images, targets, metas = correct_batch
        wrong_images, wrong_targets, wrong_metas = wrong_batch
        if int(images.shape[0]) != int(wrong_images.shape[0]):
            raise ValueError("aligned cross-clip batches have different sizes")
        for meta, wrong_meta in zip(metas, wrong_metas):
            left = str(meta.get("image_path", ""))
            right = str(wrong_meta.get("image_path", ""))
            actual_same_image += int(left == right)
            actual_same_clip += int(_clip(left) == _clip(right))

        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        wrong_images = wrong_images.to(device, non_blocking=True)
        wrong_targets = nested_to_device(wrong_targets, device)
        captured: dict[str, torch.Tensor] = {}
        wrong_captured: dict[str, torch.Tensor] = {}

        def capture(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            captured.update(kwargs)

        def capture_wrong(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            wrong_captured.update(kwargs)

        handle = module.register_forward_pre_hook(capture, with_kwargs=True)
        outputs, _ = forward_with_matches(
            model, images, targets, matcher, correct_cfg, iteration
        )
        handle.remove()
        handle = module.register_forward_pre_hook(
            capture_wrong, with_kwargs=True
        )
        forward_with_matches(
            model,
            wrong_images,
            wrong_targets,
            matcher,
            wrong_cfg,
            iteration,
        )
        handle.remove()
        if not captured or not wrong_captured:
            raise RuntimeError("failed to capture V14 module inputs")
        if not tensor_shapes:
            tensor_shapes = {
                "images": list(images.shape),
                "slot_states": list(captured["slot_states"].shape),
                "proposal_row_tokens": list(
                    captured["proposal_row_tokens"].shape
                ),
                "proposal_x_rows": list(captured["proposal_x_rows"].shape),
                "p2_row_grid": list(captured["row_value_features"].shape),
                "visual_attention": list(
                    outputs["selection_slot_v14_visual_attention"].shape
                ),
                "proposal_transport": list(
                    outputs["selection_slot_v14_proposal_attention"].shape
                ),
            }

        deployment_anchor_error = max(
            deployment_anchor_error,
            float(
                (
                    outputs["selection_slot_pred_x_rows"].float()
                    - outputs["selection_slot_v14_anchor_x_rows"].float()
                )
                .abs()
                .max()
                .cpu()
            ),
            float(
                (
                    outputs["selection_slot_range_norm"].float()
                    - outputs["selection_slot_v14_anchor_range_norm"].float()
                )
                .abs()
                .max()
                .cpu()
            ),
        )

        padded = _padded_lane_targets(
            targets,
            device=device,
            dtype=torch.float32,
            rows=int(outputs["pred_x_rows"].shape[-1]),
        )
        matches, _quality, counts = criterion._match_four_slot_v14_anchor(
            outputs, targets, padded
        )
        target = criterion._four_slot_v14_joint_targets(
            outputs, targets, matches, padded
        )
        matched_count += int(counts.sum().item())
        representable_count += int(target["representable_match"].sum().item())

        variants: dict[str, dict[str, torch.Tensor]] = {
            "correct_p2": outputs
        }
        wrong_kwargs = dict(captured)
        wrong_kwargs["row_value_features"] = wrong_captured[
            "row_value_features"
        ]
        variants["cross_clip_wrong_p2"] = {
            **outputs,
            **module(**wrong_kwargs, feature_policy="correct"),
        }
        for policy in (
            "zero_content",
            "position_only",
            "zero_content_zero_position",
            "x_reversed",
            "row_reversed",
        ):
            variants[policy] = {
                **outputs,
                **module(**captured, feature_policy=policy),
            }

        v7_probability = torch.softmax(
            outputs["selection_slot_real_route_logits"].float(), dim=-1
        )
        v7_hard = outputs["selection_slot_geometry_route_indices"].long()
        v7_mass, v7_top1, _ = _association_rows(
            v7_probability, target, hard_ids=v7_hard
        )
        _, v7_raw_top1, _ = _association_rows(v7_probability, target)
        metrics["v7_support_mass"].extend(v7_mass)
        metrics["v7_hard_target_top1"].extend(v7_top1)
        metrics["v7_raw_argmax_target_top1"].extend(v7_raw_top1)

        v7_logits = outputs["selection_slot_real_route_logits"].float()
        v14_logits = outputs["selection_slot_v14_proposal_logits"].float()
        metrics["v7_route_logit_abs_mean"].append(
            float(v7_logits.abs().mean().cpu())
        )
        metrics["v14_real_logit_abs_mean"].append(
            float(v14_logits.abs().mean().cpu())
        )
        metrics["v7_v14_real_argmax_agreement"].extend(
            (v7_logits.argmax(dim=-1) == v14_logits.argmax(dim=-1))
            .float()
            .cpu()
            .reshape(-1)
            .tolist()
        )
        representable = target["representable_match"].bool()
        target_id = target["target_id"].long()
        ranking = v14_logits.argsort(dim=-1, descending=True)
        rank = (
            ranking == target_id.clamp_min(0).unsqueeze(-1)
        ).float().argmax(dim=-1)
        if bool(representable.any()):
            selected_rank = rank[representable]
            metrics["v14_target_id_rank"].extend(
                (selected_rank + 1).float().cpu().tolist()
            )
            metrics["v14_target_id_top2"].extend(
                (selected_rank < 2).float().cpu().tolist()
            )
            metrics["v14_target_id_top4"].extend(
                (selected_rank < 4).float().cpu().tolist()
            )

        valid_rows = _valid_visual_rows(matches, padded[1])
        correct_attention = outputs[
            "selection_slot_v14_proposal_attention"
        ].float()
        correct_visual = outputs[
            "selection_slot_v14_visual_attention"
        ].float()
        policy_mass: dict[str, list[float]] = {}
        policy_top1: dict[str, list[float]] = {}
        for policy, variant in variants.items():
            probability = variant[
                "selection_slot_v14_proposal_attention"
            ].float()
            mass, top1, _hard = _association_rows(probability, target)
            policy_mass[policy] = mass
            policy_top1[policy] = top1
            metrics[f"{policy}_support_mass"].extend(mass)
            metrics[f"{policy}_hard_target_top1"].extend(top1)
            visual_loss, visual_mae = (
                criterion._four_slot_visual_distribution_loss(
                    variant["selection_slot_v14_visual_logits"],
                    matches,
                    padded[0],
                    padded[1],
                )
            )
            visual_loss_sums[policy] += float(visual_loss.cpu()) * valid_rows
            visual_mae_sums[policy] += float(visual_mae.cpu()) * valid_rows
            visual_row_counts[policy] += valid_rows
            metrics[f"{policy}_proposal_attention_change"].append(
                float((probability - correct_attention).abs().mean().cpu())
            )
            metrics[f"{policy}_visual_attention_change"].append(
                float(
                    (
                        variant["selection_slot_v14_visual_attention"].float()
                        - correct_visual
                    )
                    .abs()
                    .mean()
                    .cpu()
                )
            )

        representable = target["representable_match"].bool()
        support = target["target_support"].bool()
        target_id = target["target_id"].long()
        for batch_index, match in enumerate(matches):
            image_id = str(metas[batch_index].get("image_path", ""))
            for slot_tensor, gt_tensor in zip(
                match["pred_indices"], match["gt_indices"]
            ):
                slot = int(slot_tensor)
                gt = int(gt_tensor)
                row: dict[str, object] = {
                    "image_id": image_id,
                    "slot_id": slot,
                    "gt_id": gt,
                    "representable": bool(representable[batch_index, slot]),
                    "target_id": int(target_id[batch_index, slot]),
                    "support_size": int(support[batch_index, slot].sum()),
                    "v7_hard_id": int(v7_hard[batch_index, slot]),
                }
                if bool(representable[batch_index, slot]):
                    for policy, variant in variants.items():
                        probability = variant[
                            "selection_slot_v14_proposal_attention"
                        ][batch_index, slot].float()
                        row[f"{policy}_support_mass"] = float(
                            (
                                probability[: support.shape[-1]]
                                * support[batch_index, slot].float()
                            ).sum().cpu()
                        )
                        row[f"{policy}_hard_id"] = int(probability.argmax())
                per_slot.append(row)
        image_count += int(images.shape[0])

    for policy in POLICIES:
        count = visual_row_counts[policy]
        metrics[f"{policy}_visual_dfl"] = [
            visual_loss_sums[policy] / float(max(count, 1))
        ]
        metrics[f"{policy}_visual_mae_px"] = [
            visual_mae_sums[policy] / float(max(count, 1))
        ]
    summaries = {name: _summary(values) for name, values in metrics.items()}

    def mean(name: str) -> float:
        return float(summaries[name]["mean"])

    correct_support = mean("correct_p2_support_mass")
    correct_top1 = mean("correct_p2_hard_target_top1")
    correct_dfl = mean("correct_p2_visual_dfl")
    wrong_dfl = mean("cross_clip_wrong_p2_visual_dfl")
    position_dfl = mean("position_only_visual_dfl")
    deployment_parity = legacy_state_exact and deployment_anchor_error == 0.0
    report = {
        "experiment": "V14 corrected visual-first Stage-A state audit",
        "iteration": iteration,
        "source_iteration": source_iteration,
        "split": args.split,
        "list_path": str(Path(args.list_path).resolve()),
        "wrong_list_path": str(Path(args.wrong_list_path).resolve()),
        "image_count": image_count,
        "matched_count": matched_count,
        "representable_count": representable_count,
        "summaries": summaries,
        "tensor_shapes": tensor_shapes,
        "logit_contract": {
            "legacy_route_prior_coefficient": 0.0,
            "legacy_route_logits_consumed_by_v14": False,
            "v7_route_logit_abs_mean": mean(
                "v7_route_logit_abs_mean"
            ),
            "v14_real_logit_abs_mean": mean(
                "v14_real_logit_abs_mean"
            ),
            "v7_v14_real_argmax_agreement": mean(
                "v7_v14_real_argmax_agreement"
            ),
            "v14_target_id_mean_rank": mean("v14_target_id_rank"),
            "v14_target_id_top2": mean("v14_target_id_top2"),
            "v14_target_id_top4": mean("v14_target_id_top4"),
        },
        "correct_minus_v7": {
            "support_mass": correct_support - mean("v7_support_mass"),
            "hard_target_top1": correct_top1
            - mean("v7_hard_target_top1"),
        },
        "correct_minus_cross_clip_wrong": {
            "support_mass": correct_support
            - mean("cross_clip_wrong_p2_support_mass"),
            "hard_target_top1": correct_top1
            - mean("cross_clip_wrong_p2_hard_target_top1"),
        },
        "correct_minus_zero_content": {
            "support_mass": correct_support
            - mean("zero_content_support_mass"),
        },
        "correct_minus_position_only": {
            "support_mass": correct_support
            - mean("position_only_support_mass"),
        },
        "visual_dfl_ratios": {
            "correct_over_cross_clip_wrong": correct_dfl
            / max(wrong_dfl, 1.0e-12),
            "correct_over_position_only": correct_dfl
            / max(position_dfl, 1.0e-12),
        },
        "deployment": {
            "mode": "exact_v7",
            "legacy_state_exact": legacy_state_exact,
            "source_state_sha256": source_digest,
            "endpoint_legacy_state_sha256": endpoint_legacy_digest,
            "missing_source_state_names": missing_source_names,
            "max_anchor_error": deployment_anchor_error,
            "exact": deployment_parity,
        },
        "cross_clip": {
            "report": cross_clip,
            "actual_same_image_count": actual_same_image,
            "actual_same_clip_count": actual_same_clip,
            "exact": actual_same_image == 0 and actual_same_clip == 0,
        },
        "per_slot": per_slot,
        "optimizer_steps_during_audit": 0,
        "test_set_used": False,
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_slot"}, indent=2))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
