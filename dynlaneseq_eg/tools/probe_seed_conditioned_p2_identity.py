from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import soft_expected_x
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_frozen_p2_lane_discovery import (
    TeacherSeedConditionMetrics,
    _best_iou_per_gt,
    _paired_teacher_seed_ious,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a GT-supplied lower-lane seed can identify the same "
            "lane across rows in frozen P2. The base detector remains frozen."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-max-batches", type=int, default=16)
    parser.add_argument(
        "--sample-strategy",
        choices=("sequential", "uniform"),
        default="uniform",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--x-bins", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--point-loss-weight", type=float, default=5.0)
    parser.add_argument("--point-beta", type=float, default=0.01)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-probe", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


class SeedConditionedP2IdentityProbe(nn.Module):
    """Dense query-to-P2 correlation conditioned by one lower-lane seed."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_rows: int,
        x_bins: int,
        input_w: int,
    ) -> None:
        super().__init__()
        if hidden_dim % 8 != 0:
            raise ValueError("hidden_dim must be divisible by 8")
        self.hidden_dim = int(hidden_dim)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.input_w = int(input_w)
        self.tower = nn.Sequential(
            nn.Conv2d(int(in_dim) + 2, int(hidden_dim), 3, padding=1, bias=False),
            nn.GroupNorm(8, int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), 3, padding=1, bias=False),
            nn.GroupNorm(8, int(hidden_dim)),
            nn.GELU(),
        )
        self.query_projection = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.key_projection = nn.Conv2d(int(hidden_dim), int(hidden_dim), 1)
        self.seed_coordinate = nn.Sequential(
            nn.Linear(2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.row_embedding = nn.Embedding(int(num_rows), int(hidden_dim))
        self.query_norm = nn.LayerNorm(int(hidden_dim))
        self.key_norm = nn.LayerNorm(int(hidden_dim))
        nn.init.normal_(self.row_embedding.weight, std=0.02)

    def encode(self, p2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resized = F.interpolate(
            p2,
            size=(self.num_rows, self.x_bins),
            mode="bilinear",
            align_corners=False,
        )
        batch = int(resized.shape[0])
        yy = torch.linspace(-1.0, 1.0, self.num_rows, device=p2.device, dtype=resized.dtype)
        xx = torch.linspace(-1.0, 1.0, self.x_bins, device=p2.device, dtype=resized.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack((grid_x, grid_y), dim=0)
        coordinates = coordinates.unsqueeze(0).expand(batch, -1, -1, -1)
        seed_features = self.tower(torch.cat((resized, coordinates), dim=1))
        keys = self.key_projection(seed_features).permute(0, 2, 3, 1).contiguous()
        return seed_features, self.key_norm(keys)

    def seed_queries(
        self,
        seed_features: torch.Tensor,
        seed_yx: torch.Tensor,
        *,
        feature_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seed_features.shape[0] != 1:
            raise ValueError("seed_queries expects one image")
        seed_y = seed_yx[:, 0].long().clamp(0, self.num_rows - 1)
        seed_x = seed_yx[:, 1].long().clamp(0, self.x_bins - 1)
        if feature_override is None:
            sampled = seed_features[0, :, seed_y, seed_x].transpose(0, 1).contiguous()
        else:
            sampled = feature_override
        normalized_x = seed_x.float() / float(max(self.x_bins - 1, 1)) * 2.0 - 1.0
        normalized_y = seed_y.float() / float(max(self.num_rows - 1, 1)) * 2.0 - 1.0
        coordinates = torch.stack((normalized_x, normalized_y), dim=-1)
        query = self.query_projection(sampled)
        query = query + self.seed_coordinate(coordinates.to(dtype=query.dtype))
        row = self.row_embedding.weight.to(device=query.device, dtype=query.dtype)
        return self.query_norm(query[:, None, :] + row[None, :, :])

    def score(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if keys.shape[0] != 1:
            raise ValueError("score expects one image")
        return torch.einsum(
            "lrd,rxd->lrx",
            queries,
            keys[0],
        ) / math.sqrt(float(self.hidden_dim))

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return soft_expected_x(
            logits,
            input_w=self.input_w,
            x_bins=self.x_bins,
        )


def _lane_seed_coordinates(
    target: dict[str, torch.Tensor],
    *,
    num_rows: int,
    x_bins: int,
    input_w: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    x_rows = target["x_rows"].to(device=device, dtype=torch.float32)
    valid = target["valid_mask"].to(device=device).bool()
    coordinates: list[list[int]] = []
    lane_indices: list[int] = []
    source_rows = int(x_rows.shape[-1])
    for lane_index in range(int(x_rows.shape[0])):
        visible = valid[lane_index].nonzero(as_tuple=False).flatten()
        if visible.numel() == 0:
            continue
        source_y = int(visible[-1])
        y = int(round(source_y * float(num_rows - 1) / float(max(source_rows - 1, 1))))
        x = int(
            round(
                float(x_rows[lane_index, source_y])
                * float(x_bins - 1)
                / float(input_w)
            )
        )
        coordinates.append(
            [
                max(0, min(num_rows - 1, y)),
                max(0, min(x_bins - 1, x)),
            ]
        )
        lane_indices.append(lane_index)
    if not coordinates:
        return torch.zeros((0, 2), device=device, dtype=torch.long), lane_indices
    return torch.tensor(coordinates, device=device, dtype=torch.long), lane_indices


def _seed_curve_loss(
    probe: SeedConditionedP2IdentityProbe,
    seed_features: torch.Tensor,
    keys: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    *,
    input_w: float,
    point_loss_weight: float,
    point_beta: float,
) -> tuple[torch.Tensor, int, int]:
    total_ce = keys.sum() * 0.0
    total_point = keys.sum() * 0.0
    row_count = 0
    lane_count = 0
    for batch_index, target in enumerate(targets):
        seed_yx, lane_indices = _lane_seed_coordinates(
            target,
            num_rows=probe.num_rows,
            x_bins=probe.x_bins,
            input_w=input_w,
            device=keys.device,
        )
        if not lane_indices:
            continue
        queries = probe.seed_queries(
            seed_features[batch_index : batch_index + 1],
            seed_yx,
        )
        logits = probe.score(queries, keys[batch_index : batch_index + 1])
        predictions = probe.decode(logits)
        target_x = target["x_rows"].to(device=keys.device, dtype=predictions.dtype)
        target_valid = target["valid_mask"].to(device=keys.device).bool()
        source_rows = int(target_x.shape[-1])
        row_indices = torch.round(
            torch.linspace(
                0,
                source_rows - 1,
                probe.num_rows,
                device=keys.device,
            )
        ).long()
        for proposal_index, lane_index in enumerate(lane_indices):
            visible = target_valid[lane_index, row_indices]
            if not bool(visible.any()):
                continue
            x = target_x[lane_index, row_indices]
            bins = torch.round(
                x / float(input_w) * float(probe.x_bins - 1)
            ).long().clamp(0, probe.x_bins - 1)
            total_ce = total_ce + F.cross_entropy(
                logits[proposal_index, visible],
                bins[visible],
                reduction="sum",
            )
            total_point = total_point + F.smooth_l1_loss(
                predictions[proposal_index, visible] / float(input_w),
                x[visible] / float(input_w),
                beta=float(point_beta),
                reduction="sum",
            )
            row_count += int(visible.sum())
            lane_count += 1
    ce = total_ce / max(row_count, 1)
    point = total_point / max(row_count, 1)
    return ce + float(point_loss_weight) * point, lane_count, row_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    probe: SeedConditionedP2IdentityProbe,
    loader: Iterable,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
    group_size: int,
    input_w: float,
    line_width: float,
) -> tuple[dict[str, dict[str, float | int]], int]:
    condition_names = (
        "correct_p2",
        "wrong_image_keys",
        "wrong_seed_feature",
        "zero_seed_feature",
        "horizontal_mean_keys",
    )
    metrics = {
        name: TeacherSeedConditionMetrics()
        for name in condition_names
    }
    model.eval()
    probe.eval()
    images_seen = 0
    for images, targets, _metas in tqdm(
        loader,
        desc="seed-conditioned P2 identity eval",
        ncols=106,
    ):
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last
                if channels_last
                else torch.contiguous_format
            ),
        )
        with _amp_context(device, amp_dtype):
            encoded = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )
            outputs = model.structured_query_head(
                encoded["features"],
                inference_only=False,
            )
        seed_features, keys = probe.encode(encoded["features"].float())
        base_candidates = outputs["pred_x_rows"][:, :group_size].float()
        for sample_index, target in enumerate(targets):
            seed_yx, lane_indices = _lane_seed_coordinates(
                target,
                num_rows=probe.num_rows,
                x_bins=probe.x_bins,
                input_w=input_w,
                device=device,
            )
            queries = probe.seed_queries(
                seed_features[sample_index : sample_index + 1],
                seed_yx,
            )
            donor_index = (sample_index + 1) % int(images.shape[0])
            donor_seed_features = seed_features[
                donor_index : donor_index + 1
            ]
            donor_sampled = donor_seed_features[
                0,
                :,
                seed_yx[:, 0],
                seed_yx[:, 1],
            ].transpose(0, 1).contiguous()
            wrong_seed_queries = probe.seed_queries(
                seed_features[sample_index : sample_index + 1],
                seed_yx,
                feature_override=donor_sampled,
            )
            zero_seed_queries = probe.seed_queries(
                seed_features[sample_index : sample_index + 1],
                seed_yx,
                feature_override=torch.zeros_like(donor_sampled),
            )
            current_keys = keys[sample_index : sample_index + 1]
            donor_keys = keys[donor_index : donor_index + 1]
            mean_keys = current_keys.mean(dim=2, keepdim=True).expand_as(current_keys)
            logits_by_condition = {
                "correct_p2": probe.score(queries, current_keys),
                "wrong_image_keys": probe.score(queries, donor_keys),
                "wrong_seed_feature": probe.score(wrong_seed_queries, current_keys),
                "zero_seed_feature": probe.score(zero_seed_queries, current_keys),
                "horizontal_mean_keys": probe.score(queries, mean_keys),
            }
            base_iou = _best_iou_per_gt(
                base_candidates[sample_index],
                target,
                line_width=line_width,
            )
            for name, logits in logits_by_condition.items():
                candidates = probe.decode(logits).float()
                paired_iou = _paired_teacher_seed_ious(
                    candidates,
                    lane_indices,
                    target,
                    line_width=line_width,
                )
                metrics[name].update(base_iou, paired_iou)
        images_seen += int(images.shape[0])
    return {
        name: value.summary()
        for name, value in metrics.items()
    }, images_seen


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["persistent_workers"] = bool(int(args.num_workers) > 0)
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg["training"]["seed"] = int(args.seed)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    if (
        amp_dtype == torch.bfloat16
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.float16
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )

    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    probe = SeedConditionedP2IdentityProbe(
        in_dim=int(model_cfg.get("dim", 256)),
        hidden_dim=int(args.hidden_dim),
        num_rows=int(model_cfg.get("num_rows", 72)),
        x_bins=int(args.x_bins),
        input_w=int(model_cfg.get("input_w", 800)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_loader = build_dataloader(cfg, split="train", training=True)
    train_iterator = iter(train_loader)
    running_loss = 0.0
    running_lanes = 0
    running_rows = 0
    for step in range(1, int(args.train_steps) + 1):
        try:
            images, targets, _metas = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, targets, _metas = next(train_iterator)
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last
                if channels_last
                else torch.contiguous_format
            ),
        )
        with torch.no_grad(), _amp_context(device, amp_dtype):
            p2 = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )["features"]
        seed_features, keys = probe.encode(p2.float())
        loss, lane_count, row_count = _seed_curve_loss(
            probe,
            seed_features,
            keys,
            targets,
            input_w=float(model_cfg.get("input_w", 800)),
            point_loss_weight=float(args.point_loss_weight),
            point_beta=float(args.point_beta),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(probe.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += float(loss.detach())
        running_lanes += int(lane_count)
        running_rows += int(row_count)
        if step % int(args.log_interval) == 0 or step == int(args.train_steps):
            interval = (
                int(args.log_interval)
                if step % int(args.log_interval) == 0
                else step % int(args.log_interval)
            )
            print(
                f"identity probe step {step:05d}/{int(args.train_steps):05d} | "
                f"loss {running_loss / max(interval, 1):.4f} | "
                f"lanes {running_lanes} | rows {running_rows}",
                flush=True,
            )
            running_loss = 0.0
            running_lanes = 0
            running_rows = 0

    eval_loader = build_dataloader(cfg, split="val", training=False)
    eval_loader, sampled_indices = select_diagnostic_loader(
        eval_loader,
        strategy=str(args.sample_strategy),
        max_batches=int(args.eval_max_batches),
        num_workers=int(args.num_workers),
    )
    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(
        structured_cfg.get("num_instances", model_cfg.get("num_slots", 0))
    )
    num_groups = int(structured_cfg.get("num_groups", 1))
    results, images_seen = evaluate(
        model,
        probe,
        eval_loader,
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        group_size=num_instances // max(num_groups, 1),
        input_w=float(model_cfg.get("input_w", 800)),
        line_width=float(args.line_width),
    )
    correct_miss = results["correct_p2"]
    control_miss_iou = max(
        float(results[name]["base_miss_paired_mean_iou"])
        for name in (
            "wrong_image_keys",
            "wrong_seed_feature",
            "zero_seed_feature",
            "horizontal_mean_keys",
        )
    )
    control_miss_recall = max(
        float(results[name]["base_miss_paired_recall_050"])
        for name in (
            "wrong_image_keys",
            "wrong_seed_feature",
            "zero_seed_feature",
            "horizontal_mean_keys",
        )
    )
    positive_gate = bool(
        float(correct_miss["base_miss_paired_mean_iou"])
        >= control_miss_iou + 0.03
        and float(correct_miss["base_miss_paired_recall_050"])
        >= control_miss_recall + 0.05
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "GT supplies one lower-lane seed per curve. This is an identity/"
            "association diagnostic, not a deployable proposal mechanism."
        ),
        "question": (
            "Does frozen P2 preserve lane identity strongly enough for a lower "
            "seed feature to retrieve the same lane across rows?"
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "seed": int(args.seed),
        "train_steps": int(args.train_steps),
        "probe_parameters": int(
            sum(parameter.numel() for parameter in probe.parameters())
        ),
        "images": int(images_seen),
        "sample_strategy": str(args.sample_strategy),
        "sampled_dataset_indices": sampled_indices,
        "conditions": results,
        "positive_gate_definition": (
            "On base misses, correct P2 must exceed every image/seed/position "
            "control by >=0.03 mean IoU and >=5 recall@0.50 points."
        ),
        "positive_gate": positive_gate,
    }
    compact = dict(payload)
    compact.pop("sampled_dataset_indices")
    print(json.dumps(compact, indent=2))
    if args.save_probe:
        probe_path = Path(args.save_probe)
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"probe": probe.state_dict(), "metadata": payload}, probe_path)
        print(f"probe_checkpoint: {probe_path}")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
