from __future__ import annotations

import argparse
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
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.train import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure V12 visual-first association and correct/wrong/zero P2 "
            "causality without changing deployment outputs."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument(
        "--sample-strategy", choices=("uniform", "sequential"), default="uniform"
    )
    parser.add_argument("--max-images", type=int, default=256)
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


def _association_rows(
    probability: torch.Tensor,
    matches: list[dict[str, torch.Tensor]],
    target_probability: torch.Tensor,
    support: torch.Tensor,
) -> tuple[list[float], list[float]]:
    support_mass: list[float] = []
    target_top1: list[float] = []
    for batch_index, match in enumerate(matches):
        pred_ids = match["pred_indices"]
        gt_ids = match["gt_indices"]
        if pred_ids.numel() == 0:
            continue
        predicted = probability[batch_index, pred_ids].float()
        target = target_probability[batch_index, gt_ids].float()
        target_support = support[batch_index, gt_ids]
        support_mass.extend(
            (predicted * target_support.float()).sum(dim=-1).cpu().tolist()
        )
        target_top1.extend(
            (predicted.argmax(dim=-1) == target.argmax(dim=-1))
            .float()
            .cpu()
            .tolist()
        )
    return support_mass, target_top1


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    if args.list_path:
        cfg.setdefault("dataset", {}).setdefault("lists", {})[
            args.split
        ] = str(Path(args.list_path).expanduser().resolve())
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(args.num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model.eval()
    selector = model.structured_query_head.set_selection_head
    module = selector.visual_first_association
    if module is None:
        raise ValueError("checkpoint/config has no V12 association module")
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg).to(device)
    criterion.set_iteration(iteration)
    loader = build_dataloader(cfg, split=args.split, training=False)
    max_batches = (
        0
        if int(args.max_images) <= 0
        else math.ceil(int(args.max_images) / int(args.eval_batch_size))
    )
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=args.sample_strategy,
        max_batches=max_batches,
        num_workers=int(args.num_workers),
    )

    values: dict[str, list[float]] = {
        key: []
        for policy in ("v7_route", "correct_p2", "wrong_p2", "zero_p2")
        for key in (
            f"{policy}_support_mass",
            f"{policy}_target_top1",
        )
    }
    for policy in ("correct_p2", "wrong_p2", "zero_p2"):
        values[f"{policy}_first_visual_loss"] = []
        values[f"{policy}_final_visual_loss"] = []
        values[f"{policy}_first_visual_mae_px"] = []
        values[f"{policy}_final_visual_mae_px"] = []
    proposal_changes: list[float] = []
    visual_changes: list[float] = []
    matched_counts: list[float] = []
    image_count = 0

    for images, targets, _metas in tqdm(
        loader, desc="V12 visual-first state", ncols=90
    ):
        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        captured: dict[str, torch.Tensor] = {}

        def capture(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            captured.update(kwargs)

        handle = module.register_forward_pre_hook(capture, with_kwargs=True)
        outputs, matches_from_model = forward_with_matches(
            model, images, targets, matcher, cfg, iteration
        )
        handle.remove()
        if not captured:
            raise RuntimeError("V12 state audit captured no module inputs")
        matches, _quality, match_count = (
            criterion._match_four_slot_visual_first_anchor(outputs, targets)
        )
        matched_counts.extend(match_count.cpu().tolist())
        target_probability, support = (
            criterion._four_slot_unified_proposal_targets(outputs, targets)
        )
        v7_probability = torch.softmax(
            outputs["selection_slot_real_route_logits"].float(), dim=-1
        )
        support_rows, top1_rows = _association_rows(
            v7_probability, matches, target_probability, support
        )
        values["v7_route_support_mass"].extend(support_rows)
        values["v7_route_target_top1"].extend(top1_rows)

        variants: dict[str, dict[str, torch.Tensor]] = {
            "correct_p2": {
                name: value for name, value in outputs.items()
            }
        }
        for policy, features in (
            (
                "wrong_p2",
                torch.roll(captured["row_value_features"], shifts=1, dims=0),
            ),
            ("zero_p2", torch.zeros_like(captured["row_value_features"])),
        ):
            kwargs = dict(captured)
            kwargs["row_value_features"] = features
            variants[policy] = {**outputs, **module(**kwargs)}

        correct_proposal = outputs[
            "selection_slot_v12_proposal_attention"
        ].float()
        correct_visual = outputs["selection_slot_v12_visual_attention"].float()
        wrong_proposal = variants["wrong_p2"][
            "selection_slot_v12_proposal_attention"
        ].float()
        wrong_visual = variants["wrong_p2"][
            "selection_slot_v12_visual_attention"
        ].float()
        proposal_changes.append(
            float((correct_proposal - wrong_proposal).abs().mean().cpu())
        )
        visual_changes.append(
            float((correct_visual - wrong_visual).abs().mean().cpu())
        )

        for policy, variant_outputs in variants.items():
            probability = variant_outputs[
                "selection_slot_v12_proposal_attention"
            ].float()
            support_rows, top1_rows = _association_rows(
                probability, matches, target_probability, support
            )
            values[f"{policy}_support_mass"].extend(support_rows)
            values[f"{policy}_target_top1"].extend(top1_rows)
            policy_losses = criterion.compute_four_slot_visual_first_loss(
                variant_outputs, targets
            )
            batch = int(images.shape[0])
            for metric, loss_key in (
                ("first_visual_loss", "first_visual"),
                ("final_visual_loss", "final_visual"),
                ("first_visual_mae_px", "mean_first_visual_mae_px"),
                ("final_visual_mae_px", "mean_final_visual_mae_px"),
            ):
                values[f"{policy}_{metric}"].extend(
                    [float(policy_losses[loss_key].cpu())] * batch
                )

        # The model-level matcher is intentionally unrelated to the V12
        # source-stable assignment.  It is retained only to exercise the exact
        # production forward contract in this inference-only audit.
        del matches_from_model
        image_count += int(images.shape[0])

    summaries = {name: _summary(rows) for name, rows in values.items()}
    correct_support = float(summaries["correct_p2_support_mass"]["mean"])
    wrong_support = float(summaries["wrong_p2_support_mass"]["mean"])
    zero_support = float(summaries["zero_p2_support_mass"]["mean"])
    v7_support = float(summaries["v7_route_support_mass"]["mean"])
    correct_top1 = float(summaries["correct_p2_target_top1"]["mean"])
    v7_top1 = float(summaries["v7_route_target_top1"]["mean"])
    correct_visual_loss = float(
        summaries["correct_p2_final_visual_loss"]["mean"]
    )
    wrong_visual_loss = float(
        summaries["wrong_p2_final_visual_loss"]["mean"]
    )
    report = {
        "experiment": "V12 visual-first Stage-A association state",
        "iteration": iteration,
        "split": args.split,
        "list_path": str(Path(args.list_path).resolve()) if args.list_path else "",
        "sample_strategy": args.sample_strategy,
        "sampled_indices": sampled_indices,
        "image_count": image_count,
        "summaries": summaries,
        "matched_count": _summary(matched_counts),
        "correct_minus_v7": {
            "support_mass": correct_support - v7_support,
            "target_top1": correct_top1 - v7_top1,
        },
        "correct_minus_wrong_p2": {
            "support_mass": correct_support - wrong_support,
            "final_visual_loss_advantage": wrong_visual_loss
            - correct_visual_loss,
        },
        "correct_minus_zero_p2": {
            "support_mass": correct_support - zero_support,
        },
        "wrong_p2_mean_proposal_attention_change": _summary(proposal_changes),
        "wrong_p2_mean_visual_attention_change": _summary(visual_changes),
        "deployment_mode": "exact_v7",
        "legacy_route_logits_used_by_v12": False,
        "optimizer_steps": 0,
        "test_set_used": False,
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
