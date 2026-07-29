from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.analyze_missed_lane_dense_support import (
    _best_iou,
    _empty_bucket,
    _row_peaks,
    _summarize,
    _update_bucket,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_independent_p2_curve_proposals import (
    extract_frozen_feature_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train two tiny dense centerline heads on frozen P2 features. One uses "
            "the production BCE objective; the other adds direct per-lane row-peak "
            "supervision. Comparing them with the checkpoint head and wrong-image "
            "controls separates feature availability from head/loss/acquisition failure."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--peak-top-k", type=int, default=8)
    parser.add_argument("--peak-nms-radius-bins", type=int, default=4)
    parser.add_argument("--sample-strategy", choices=("uniform", "sequential"), default="uniform")
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-probes", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FrozenP2CenterlineProbe(nn.Module):
    """A modest dense reader with no lane queries, references, or assignments."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_rows: int,
        x_bins: int,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim % 8 != 0:
            raise ValueError("hidden_dim must be divisible by 8")
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_dim), hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        logits = self.net(p2)
        if logits.shape[-2:] != (self.num_rows, self.x_bins):
            logits = F.interpolate(
                logits,
                size=(self.num_rows, self.x_bins),
                mode="bilinear",
                align_corners=False,
            )
        return logits


def build_centerline_target(
    targets: list[dict[str, torch.Tensor]],
    *,
    num_rows: int,
    x_bins: int,
    input_w: float,
    sigma_bins: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    target_map = torch.zeros(
        (len(targets), 1, int(num_rows), int(x_bins)),
        device=device,
        dtype=dtype,
    )
    grid = torch.arange(int(x_bins), device=device, dtype=dtype).view(1, 1, -1)
    bin_width = float(input_w) / float(x_bins)
    sigma = max(float(sigma_bins), 1e-3)
    for batch_index, target in enumerate(targets):
        x_rows = target["x_rows"].to(device=device, dtype=dtype)
        valid = target["valid_mask"].to(device=device).bool()
        if x_rows.numel() == 0:
            continue
        row_count = min(int(x_rows.shape[1]), int(num_rows))
        x_rows = x_rows[:, :row_count]
        valid = valid[:, :row_count]
        centers = (x_rows / bin_width).clamp(0.0, float(x_bins - 1))
        valid = (
            valid
            & torch.isfinite(centers)
            & (x_rows >= 0.0)
            & (x_rows <= float(input_w))
        )
        if not bool(valid.any()):
            continue
        gaussian = torch.exp(
            -0.5 * ((grid - centers.unsqueeze(-1)) / sigma).pow(2)
        )
        gaussian = gaussian * valid.unsqueeze(-1).to(dtype=dtype)
        target_map[batch_index, 0, :row_count] = gaussian.amax(dim=0)
    return target_map


def lane_center_nll(
    logits: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    *,
    input_w: float,
) -> torch.Tensor:
    """Directly reward a peak at every visible GT lane/row center."""

    _, _, num_rows, x_bins = logits.shape
    bin_width = float(input_w) / float(x_bins)
    log_probabilities = logits[:, 0].float().log_softmax(dim=-1)
    terms: list[torch.Tensor] = []
    for batch_index, target in enumerate(targets):
        x_rows = target["x_rows"].to(device=logits.device, dtype=torch.float32)
        valid = target["valid_mask"].to(device=logits.device).bool()
        if x_rows.numel() == 0:
            continue
        row_count = min(int(x_rows.shape[1]), int(num_rows))
        x_rows = x_rows[:, :row_count]
        valid = (
            valid[:, :row_count]
            & torch.isfinite(x_rows)
            & (x_rows >= 0.0)
            & (x_rows <= float(input_w))
        )
        lane_indices, row_indices = valid.nonzero(as_tuple=True)
        if row_indices.numel() == 0:
            continue
        target_bins = torch.floor(x_rows[lane_indices, row_indices] / bin_width)
        target_bins = target_bins.long().clamp(0, int(x_bins) - 1)
        terms.append(-log_probabilities[batch_index, row_indices, target_bins])
    if not terms:
        return logits.sum() * 0.0
    return torch.cat(terms).mean()


def production_bce_loss(
    logits: torch.Tensor,
    target_map: torch.Tensor,
    *,
    pos_weight: float,
) -> torch.Tensor:
    weight = torch.tensor(
        [float(pos_weight)],
        device=logits.device,
        dtype=logits.dtype,
    )
    return F.binary_cross_entropy_with_logits(
        logits,
        target_map.to(dtype=logits.dtype),
        pos_weight=weight,
    )


def discovery_loss(
    logits: torch.Tensor,
    target_map: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    *,
    input_w: float,
    pos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = production_bce_loss(logits, target_map, pos_weight=pos_weight)
    peak_nll = lane_center_nll(logits, targets, input_w=input_w)
    total = peak_nll + 0.25 * bce
    return total, peak_nll, bce


def _evaluate_probability_map(
    buckets: dict[str, dict[str, Any]],
    *,
    row_probability: torch.Tensor,
    target: dict[str, torch.Tensor],
    candidates_group: torch.Tensor,
    candidates_all: torch.Tensor,
    input_w: float,
    line_width: float,
    peak_top_k: int,
    peak_nms_radius_bins: int,
) -> None:
    peak_indices, _ = _row_peaks(
        row_probability,
        top_k=int(peak_top_k),
        radius=int(peak_nms_radius_bins),
    )
    bins = int(row_probability.shape[-1])
    bin_scale = float(input_w) / float(bins)
    gt_rows = target["x_rows"].to(
        device=row_probability.device,
        dtype=torch.float32,
    )
    gt_valid = target["valid_mask"].to(device=row_probability.device).bool()
    for lane_index in range(int(gt_rows.shape[0])):
        valid = gt_valid[lane_index]
        if int(valid.sum()) < 5:
            continue
        gt_x = gt_rows[lane_index]
        best_group_iou = _best_iou(
            candidates_group,
            gt_x,
            valid,
            line_width=float(line_width),
        )
        best_all_iou = _best_iou(
            candidates_all,
            gt_x,
            valid,
            line_width=float(line_width),
        )
        valid_rows = valid.nonzero(as_tuple=False).flatten()
        gt_bin_float = (gt_x[valid_rows] / bin_scale).clamp(0.0, float(bins - 1))
        gt_bin_nearest = gt_bin_float.round().long()
        gt_probabilities = row_probability[valid_rows, gt_bin_nearest]
        selected_peaks = peak_indices[valid_rows]
        peak_distances_bins = (
            selected_peaks.float() - gt_bin_float[:, None]
        ).abs()
        nearest_peak_position = selected_peaks.gather(
            1,
            peak_distances_bins.argmin(dim=1, keepdim=True),
        ).squeeze(1)
        peak_distances_px = (
            nearest_peak_position.float() - gt_bin_float
        ).abs() * bin_scale
        dense_curve = gt_x.new_zeros(gt_x.shape)
        dense_curve[valid_rows] = (
            nearest_peak_position.float() + 0.5
        ) * bin_scale
        dense_iou = _best_iou(
            dense_curve.unsqueeze(0),
            gt_x,
            valid,
            line_width=float(line_width),
        )
        values = {
            "best_group_iou": best_group_iou,
            "best_all_iou": best_all_iou,
            "gt_probabilities": gt_probabilities,
            "peak_distances_px": peak_distances_px,
            "dense_peak_oracle_iou": dense_iou,
        }
        _update_bucket(buckets["all"], **values)
        _update_bucket(
            buckets[
                "group_hit_050"
                if best_group_iou >= 0.5
                else "group_miss_050"
            ],
            **values,
        )
        _update_bucket(
            buckets[
                "group_hit_070"
                if best_group_iou >= 0.7
                else "group_miss_070"
            ],
            **values,
        )


@torch.no_grad()
def evaluate(
    base_model: nn.Module,
    bce_probe: FrozenP2CenterlineProbe,
    discovery_probe: FrozenP2CenterlineProbe,
    loader: Iterable,
    *,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    canonical_channels: int,
    group_size: int,
    input_w: float,
    line_width: float,
    peak_top_k: int,
    peak_nms_radius_bins: int,
) -> dict[str, Any]:
    mode_names = (
        "checkpoint_correct",
        "checkpoint_wrong_image",
        "fresh_production_bce_correct",
        "fresh_production_bce_wrong_image",
        "fresh_discovery_correct",
        "fresh_discovery_wrong_image",
        "fresh_discovery_horizontal_mean",
    )
    mode_buckets = {
        mode: {
            name: _empty_bucket()
            for name in (
                "all",
                "group_hit_050",
                "group_miss_050",
                "group_hit_070",
                "group_miss_070",
            )
        }
        for mode in mode_names
    }
    base_model.eval()
    bce_probe.eval()
    discovery_probe.eval()
    checkpoint_head = base_model.encoder.centerline_aux_head
    if checkpoint_head is None:
        raise RuntimeError("The checkpoint/config has no centerline auxiliary head")
    images_seen = 0
    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    for images, targets, _metas in tqdm(
        loader,
        ncols=96,
        desc="frozen-P2 centerline separability eval",
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
        context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if autocast_enabled
            else nullcontext()
        )
        with context:
            p2, _ = extract_frozen_feature_source(
                base_model,
                images,
                feature_source="p2",
                canonical_channels=int(canonical_channels),
            )
            base_outputs = base_model.structured_query_head(
                p2,
                inference_only=True,
            )
        p2_float = p2.float()
        wrong_indices = torch.roll(
            torch.arange(int(p2.shape[0]), device=device),
            shifts=1,
        )
        horizontal_mean = p2_float.mean(dim=-1, keepdim=True).expand_as(p2_float)
        logits_by_mode = {
            "checkpoint_correct": checkpoint_head(p2_float),
            "checkpoint_wrong_image": checkpoint_head(p2_float[wrong_indices]),
            "fresh_production_bce_correct": bce_probe(p2_float),
            "fresh_production_bce_wrong_image": bce_probe(p2_float[wrong_indices]),
            "fresh_discovery_correct": discovery_probe(p2_float),
            "fresh_discovery_wrong_image": discovery_probe(p2_float[wrong_indices]),
            "fresh_discovery_horizontal_mean": discovery_probe(horizontal_mean),
        }
        probabilities_by_mode = {
            name: torch.sigmoid(logits[:, 0].float())
            for name, logits in logits_by_mode.items()
        }
        images_seen += int(images.shape[0])
        for sample_index, target in enumerate(targets):
            candidates_all = base_outputs["pred_x_rows"][sample_index].float()
            candidates_group = candidates_all[: int(group_size)]
            for mode_name, probability_batch in probabilities_by_mode.items():
                _evaluate_probability_map(
                    mode_buckets[mode_name],
                    row_probability=probability_batch[sample_index],
                    target=target,
                    candidates_group=candidates_group,
                    candidates_all=candidates_all,
                    input_w=float(input_w),
                    line_width=float(line_width),
                    peak_top_k=int(peak_top_k),
                    peak_nms_radius_bins=int(peak_nms_radius_bins),
                )

    summary = {
        mode: {
            bucket_name: _summarize(bucket)
            for bucket_name, bucket in buckets.items()
        }
        for mode, buckets in mode_buckets.items()
    }
    miss_key = "group_miss_050"
    correct = summary["fresh_discovery_correct"][miss_key]
    wrong = summary["fresh_discovery_wrong_image"][miss_key]
    production = summary["checkpoint_correct"][miss_key]
    fresh_bce = summary["fresh_production_bce_correct"][miss_key]
    p2_globally_readable = (
        float(correct["dense_peak_oracle_recall_050"]) >= 0.20
        and (
            float(correct["dense_peak_oracle_recall_050"])
            - float(wrong["dense_peak_oracle_recall_050"])
        )
        >= 0.10
        and (
            float(correct["row_peak_recall_15px"])
            - float(wrong["row_peak_recall_15px"])
        )
        >= 0.15
    )
    production_objective_suspect = (
        float(correct["dense_peak_oracle_recall_050"])
        - float(fresh_bce["dense_peak_oracle_recall_050"])
        >= 0.10
    )
    joint_head_optimization_suspect = (
        float(fresh_bce["dense_peak_oracle_recall_050"])
        - float(production["dense_peak_oracle_recall_050"])
        >= 0.10
    )
    return {
        "images": int(images_seen),
        "gate_definitions": {
            "p2_globally_readable": (
                "On base misses at IoU 0.50: fresh discovery dense-oracle recall "
                ">= 0.20, correct-minus-wrong dense-oracle recall >= 0.10, and "
                "correct-minus-wrong 15px row-peak recall >= 0.15."
            ),
            "production_objective_suspect": (
                "Fresh discovery exceeds the identically initialized fresh "
                "production-BCE probe by >= 0.10 dense-oracle recall on base misses."
            ),
            "joint_head_optimization_suspect": (
                "Fresh production-BCE probe exceeds the checkpoint head by >= 0.10 "
                "dense-oracle recall on base misses."
            ),
        },
        "gates": {
            "p2_globally_readable": bool(p2_globally_readable),
            "production_objective_suspect": bool(production_objective_suspect),
            "joint_head_optimization_suspect": bool(joint_head_optimization_suspect),
        },
        "modes": summary,
    }


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    dataloader_cfg = cfg.setdefault("dataloader", {})
    dataloader_cfg["num_workers"] = int(args.num_workers)
    dataloader_cfg["eval_batch_size"] = int(args.eval_batch_size)
    dataloader_cfg["persistent_workers"] = bool(int(args.num_workers) > 0)
    training_cfg = cfg.setdefault("training", {})
    training_cfg["batch_size"] = int(args.batch_size)
    training_cfg["seed"] = int(args.seed)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    channels_last = (
        bool(training_cfg.get("channels_last", False))
        and device.type == "cuda"
    )
    base_model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(
        args.checkpoint,
        base_model,
        strict=False,
    )
    base_model.requires_grad_(False)
    base_model = base_model.to(device).eval()
    if channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)

    input_w = float(model_cfg.get("input_w", 800))
    num_rows = int(model_cfg.get("num_rows", 72))
    x_bins = int(model_cfg.get("x_bins", 200))
    canonical_channels = int(model_cfg.get("dim", 256))
    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(
        structured_cfg.get(
            "num_instances",
            model_cfg.get("num_slots", 0),
        )
    )
    num_groups = int(structured_cfg.get("num_groups", 1))
    if num_groups < 1 or num_instances % num_groups != 0:
        raise ValueError("Invalid structured-query grouping")
    group_size = num_instances // num_groups
    loss_cfg = cfg.get("loss", {})
    sigma_bins = float(loss_cfg.get("centerline_sigma_bins", 1.5))
    pos_weight = float(loss_cfg.get("centerline_pos_weight", 1.0))

    bce_probe = FrozenP2CenterlineProbe(
        in_dim=canonical_channels,
        hidden_dim=int(args.hidden_dim),
        num_rows=num_rows,
        x_bins=x_bins,
    ).to(device)
    discovery_probe = deepcopy(bce_probe).to(device)
    optimizer = torch.optim.AdamW(
        list(bce_probe.parameters()) + list(discovery_probe.parameters()),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    train_loader = build_dataloader(cfg, split="train", training=True)
    raw_eval_loader = build_dataloader(cfg, split="val", training=False)
    eval_loader, sampled_indices = select_diagnostic_loader(
        raw_eval_loader,
        strategy=str(args.sample_strategy),
        max_batches=int(args.eval_max_batches),
        num_workers=int(args.num_workers),
    )

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
    autocast_enabled = amp_dtype is not None and device.type == "cuda"

    bce_probe.train()
    discovery_probe.train()
    train_iterator = iter(train_loader)
    running = {
        "production_bce": 0.0,
        "discovery_total": 0.0,
        "discovery_peak_nll": 0.0,
        "discovery_bce": 0.0,
    }
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
        with torch.no_grad():
            context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if autocast_enabled
                else nullcontext()
            )
            with context:
                p2, _ = extract_frozen_feature_source(
                    base_model,
                    images,
                    feature_source="p2",
                    canonical_channels=canonical_channels,
                )
        p2 = p2.float()
        target_map = build_centerline_target(
            targets,
            num_rows=num_rows,
            x_bins=x_bins,
            input_w=input_w,
            sigma_bins=sigma_bins,
            device=device,
        )
        bce_logits = bce_probe(p2)
        discovery_logits = discovery_probe(p2)
        bce_loss = production_bce_loss(
            bce_logits,
            target_map,
            pos_weight=pos_weight,
        )
        discovery_total, discovery_peak, discovery_bce = discovery_loss(
            discovery_logits,
            target_map,
            targets,
            input_w=input_w,
            pos_weight=pos_weight,
        )
        total = bce_loss + discovery_total
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(
            list(bce_probe.parameters()) + list(discovery_probe.parameters()),
            max_norm=5.0,
        )
        optimizer.step()
        values = {
            "production_bce": bce_loss,
            "discovery_total": discovery_total,
            "discovery_peak_nll": discovery_peak,
            "discovery_bce": discovery_bce,
        }
        for name, value in values.items():
            running[name] += float(value.detach())
        if (
            step % int(args.log_interval) == 0
            or step == int(args.train_steps)
        ):
            window = int(args.log_interval)
            if (
                step == int(args.train_steps)
                and step % int(args.log_interval) != 0
            ):
                window = step % int(args.log_interval)
            print(
                f"frozen-P2 centerline step {step:05d}/{int(args.train_steps):05d}"
                + " | "
                + " | ".join(
                    f"{name} {running[name] / max(window, 1):.4f}"
                    for name in running
                ),
                flush=True,
            )
            running = {name: 0.0 for name in running}

    evaluation = evaluate(
        base_model,
        bce_probe,
        discovery_probe,
        eval_loader,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        canonical_channels=canonical_channels,
        group_size=group_size,
        input_w=input_w,
        line_width=float(args.line_width),
        peak_top_k=int(args.peak_top_k),
        peak_nms_radius_bins=int(args.peak_nms_radius_bins),
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "The base detector and P2 features are frozen. Dense peak curves use "
            "GT association independently at each visible row and are oracle "
            "evidence diagnostics, not deployable predictions."
        ),
        "question": (
            "Do frozen P2 features contain globally decodable centerline evidence "
            "for lanes missed by the production structured decoder?"
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(checkpoint_iteration),
        "seed": int(args.seed),
        "train_steps": int(args.train_steps),
        "train_batch_size": int(args.batch_size),
        "eval_split": "val",
        "sample_strategy": str(args.sample_strategy),
        "sampled_dataset_indices": sampled_indices,
        "probe_parameters_each": int(
            sum(parameter.numel() for parameter in bce_probe.parameters())
        ),
        "matched_initialization": True,
        "probe_inputs": (
            "frozen projected P2 only; no lane queries, row states, decoder "
            "outputs, predicted references, or matcher assignments"
        ),
        "objectives": {
            "fresh_production_bce": (
                f"same Gaussian centerline target and BCE pos_weight={pos_weight:g} "
                "used by the production auxiliary objective"
            ),
            "fresh_discovery": (
                "direct per-visible-lane row-center log-softmax NLL plus 0.25 "
                "times the production BCE"
            ),
            "centerline_sigma_bins": sigma_bins,
        },
        "evaluation": evaluation,
    }
    print(json.dumps(payload, indent=2))

    if args.save_probes:
        path = Path(args.save_probes)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "production_bce_probe": bce_probe.state_dict(),
                "discovery_probe": discovery_probe.state_dict(),
                "metadata": payload,
            },
            path,
        )
        print(f"probe_checkpoint: {path}")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output_json: {path}")


if __name__ == "__main__":
    main()
