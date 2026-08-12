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
        description="Measure V11 unified attention/state without training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
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
        name: []
        for name in (
            "target_support_mass",
            "base_quality",
            "final_quality",
            "quality_gain",
            "matched",
            "active_probability",
            "active_per_image",
            "proposal_entropy",
            "proposal_top1_mass",
            "visual_entropy",
            "delta_mean_abs_px",
            "delta_max_abs_px",
            "delta_boundary_mass",
            "activity_residual_abs",
            "final_base_shift_px",
        )
    }
    image_count = 0
    for images, targets, _metas in tqdm(
        loader,
        desc="V11 checkpoint state",
        ncols=90,
    ):
        images = images.to(device, non_blocking=True)
        targets = nested_to_device(targets, device)
        outputs, matches = forward_with_matches(
            model,
            images,
            targets,
            matcher,
            cfg,
            iteration,
        )
        losses = criterion(outputs, targets, matches)
        batch = int(images.shape[0])
        image_count += batch
        scalar_map = {
            "target_support_mass": "four_slot_unified_mean_target_support_mass",
            "base_quality": "four_slot_unified_mean_base_quality",
            "final_quality": "four_slot_unified_mean_final_quality",
            "quality_gain": "four_slot_unified_mean_quality_gain",
            "matched": "four_slot_unified_mean_matched",
            "active_probability": "four_slot_unified_mean_active_probability",
        }
        for output_name, loss_name in scalar_map.items():
            values[output_name].extend(
                [float(losses[loss_name].detach().cpu())] * batch
            )
        active = outputs["selection_slot_active"].detach().float()
        values["active_per_image"].extend(
            active.sum(dim=-1).cpu().tolist()
        )
        proposal_attention = outputs[
            "selection_slot_unified_proposal_attention"
        ].detach().float()
        proposal_entropy = -(
            proposal_attention
            * proposal_attention.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        values["proposal_entropy"].extend(
            proposal_entropy.mean(dim=-1).cpu().tolist()
        )
        values["proposal_top1_mass"].extend(
            proposal_attention.amax(dim=-1).mean(dim=-1).cpu().tolist()
        )
        visual_entropy = outputs[
            "selection_slot_unified_visual_entropy"
        ].detach().float()
        values["visual_entropy"].extend(visual_entropy.cpu().tolist())
        for output_name, tensor_name in (
            ("delta_mean_abs_px", "selection_slot_delta_mean_abs"),
            ("delta_max_abs_px", "selection_slot_delta_max_abs"),
            ("delta_boundary_mass", "selection_slot_delta_boundary_mass"),
        ):
            values[output_name].extend(
                outputs[tensor_name].detach().float().cpu().tolist()
            )
        residual = outputs[
            "selection_slot_unified_activity_residual"
        ].detach().float()
        values["activity_residual_abs"].extend(
            residual.abs().mean(dim=-1).cpu().tolist()
        )
        shift = (
            outputs["selection_slot_pred_x_rows"].detach().float()
            - outputs["selection_slot_unified_base_x_rows"].detach().float()
        ).abs().mean(dim=(1, 2))
        values["final_base_shift_px"].extend(shift.cpu().tolist())

    if int(args.max_images) > 0 and image_count > int(args.max_images):
        # Diagnostic sampler rounds to complete batches.  Retain the exact
        # sampled-index contract in the report rather than silently claiming
        # an exact image limit.
        sampled_indices = sampled_indices[: int(args.max_images)]
    report = {
        "experiment": "V11 unified slot-row checkpoint state",
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "iteration": iteration,
        "split": args.split,
        "list_path": (
            str(Path(args.list_path).expanduser().resolve())
            if args.list_path
            else ""
        ),
        "sample_strategy": args.sample_strategy,
        "images": image_count,
        "sampled_indices": sampled_indices,
        "test_set_used": False,
        "statistics": {name: _summary(rows) for name, rows in values.items()},
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
