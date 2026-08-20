"""Causal observability probe for refined V7 source-vs-candidate decisions.

This is deliberately not a deployable selector.  It asks one narrower question:
given an exact refined V7 lane and a counterfactual-refined proposal from the
same physical lane, does image evidence between the two curves reveal which
curve has the better official raster quality on unseen clips?

Two fixed 256-image mechanisms sets are used in both directions.  Every probe
is trained on one set and evaluated on the other.  Geometry-only, P2 point,
continuous P2 corridor, native stride-two DLA evidence, wrong-image stride-two
evidence, and RGB corridor arms share the exact same classifier.  Candidate
order is antisymmetrized, so a source/candidate position shortcut cannot solve
the task.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.engine.checkpoint import _torch_load, load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.v19_counterfactual_fidelity import (
    frozen_v7_counterfactual_anchors,
)
from dynlaneseq_eg.tools.audit_v11_causal_replay import _image_id
from dynlaneseq_eg.tools.train import seed_everything


ARMS = (
    "geometry_only",
    "p2_center",
    "p2_corridor",
    "p1_corridor",
    "wrong_p1_corridor",
    "rgb_corridor",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PairSpec:
    slot: int
    candidate: int
    owner_gt: int
    label: int
    source_quality: float
    candidate_quality: float
    threshold_kind: str
    mean_abs_dx: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen P2 and native stride-two pair-corridor evidence "
            "on exact same-owner refined proposal decisions."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--list-a", required=True)
    parser.add_argument("--quality-cache-a", required=True)
    parser.add_argument("--list-b", required=True)
    parser.add_argument("--quality-cache-b", required=True)
    parser.add_argument("--device", default="cuda")
    # Fixed mechanism lists can contain up to five consecutive frames from the
    # same clip.  Eight guarantees an in-batch cross-clip negative on the
    # predeclared 256-image sets while every model forward remains singleton.
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ribbon-points", type=int, default=25)
    parser.add_argument("--ribbon-margin-px", type=float, default=16.0)
    parser.add_argument("--vertical-bands", type=int, default=8)
    parser.add_argument("--profile-channels", type=int, default=64)
    parser.add_argument("--max-pairs-per-class-slot", type=int, default=3)
    parser.add_argument("--max-mean-abs-dx", type=float, default=64.0)
    parser.add_argument("--min-quality-gap", type=float, default=0.03)
    parser.add_argument("--probe-steps", type=int, default=600)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--feature-cache-a")
    parser.add_argument("--feature-cache-b")
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _checkpoint_cfg(
    checkpoint: str,
    dataset_root: str,
    list_path: str,
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    payload = _torch_load(checkpoint)
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint does not contain its training config")
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("dataset", {})["root"] = str(
        Path(dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})["val"] = str(
        Path(list_path).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _load_quality_cache(path: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    records = {str(row["image_id"]): row for row in payload["records"]}
    if len(records) != len(payload["records"]):
        raise ValueError("quality cache contains duplicate image IDs")
    return payload["metadata"], records


def _clip_id(image_id: str) -> str:
    """Return the video/clip path, excluding the frame filename."""

    return Path(str(image_id)).parent.as_posix()


def _source_ownership(record: dict[str, Any]) -> dict[int, int]:
    source_quality = record["source_quality"].float()
    source_valid = record["source_valid"].bool()
    active = record["active"].bool() & source_valid
    active_slots = torch.nonzero(active, as_tuple=False).flatten().tolist()
    if source_quality.shape[0] == 0 or not active_slots:
        return {}
    matrix = source_quality[:, active_slots].numpy().astype(np.float64, copy=False)
    gt_ids, local_predictions = linear_sum_assignment(-matrix)
    return {
        int(active_slots[local]): int(gt)
        for gt, local in zip(gt_ids.tolist(), local_predictions.tolist())
    }


def _quality_key(value: float) -> tuple[int, int, float]:
    return int(value > 0.50), int(value > 0.75), float(value)


def _threshold_kind(source: float, candidate: float) -> str:
    if (source > 0.50) != (candidate > 0.50):
        return "cross_50"
    if (source > 0.75) != (candidate > 0.75):
        return "cross_75"
    return "quality_only"


def _valid_rows(
    curve: torch.Tensor,
    range_norm: torch.Tensor,
    *,
    rows: int,
) -> torch.Tensor:
    y = torch.arange(rows, device=curve.device, dtype=torch.float32) / float(rows)
    lo = torch.minimum(range_norm[0], range_norm[1])
    hi = torch.maximum(range_norm[0], range_norm[1])
    return torch.isfinite(curve) & (y >= lo) & (y <= hi)


def select_pair_specs(
    record: dict[str, Any],
    source_x: torch.Tensor,
    source_range: torch.Tensor,
    refined_x: torch.Tensor,
    refined_range: torch.Tensor,
    *,
    max_pairs_per_class_slot: int,
    max_mean_abs_dx: float,
    min_quality_gap: float,
) -> list[PairSpec]:
    """Select balanced hard same-owner alternatives without using image features."""

    ownership = _source_ownership(record)
    source_routes = record["source_routes"].long()
    source_quality = record["source_quality"].float()
    refined_quality = record["refined_quality"].float()
    refined_valid = record["refined_valid"].bool()
    rows = int(source_x.shape[-1])
    selected: list[PairSpec] = []
    for slot, owner_gt in ownership.items():
        source_route = int(source_routes[slot])
        source_q = float(source_quality[owner_gt, slot])
        source_rows = _valid_rows(source_x[slot], source_range[slot], rows=rows)
        by_label: dict[int, list[PairSpec]] = defaultdict(list)
        for candidate in range(int(refined_x.shape[1])):
            if candidate == source_route or not bool(refined_valid[slot, candidate]):
                continue
            candidate_column = refined_quality[:, slot, candidate]
            if int(candidate_column.argmax()) != int(owner_gt):
                continue
            candidate_q = float(candidate_column[owner_gt])
            source_key = _quality_key(source_q)
            candidate_key = _quality_key(candidate_q)
            if candidate_key == source_key:
                continue
            kind = _threshold_kind(source_q, candidate_q)
            if kind == "quality_only" and abs(candidate_q - source_q) < float(min_quality_gap):
                continue
            candidate_rows = _valid_rows(
                refined_x[slot, candidate],
                refined_range[slot, candidate],
                rows=rows,
            )
            common = source_rows & candidate_rows
            if int(common.sum()) < 5:
                continue
            mean_abs_dx = float(
                (
                    refined_x[slot, candidate, common]
                    - source_x[slot, common]
                ).abs().mean()
            )
            if mean_abs_dx > float(max_mean_abs_dx):
                continue
            label = int(candidate_key > source_key)
            by_label[label].append(
                PairSpec(
                    slot=int(slot),
                    candidate=int(candidate),
                    owner_gt=int(owner_gt),
                    label=label,
                    source_quality=source_q,
                    candidate_quality=candidate_q,
                    threshold_kind=kind,
                    mean_abs_dx=mean_abs_dx,
                )
            )
        for label, values in by_label.items():
            # Threshold-changing pairs come first.  Positive pairs then prefer
            # larger real gains; negative pairs prefer high-quality, close
            # distractors rather than easy background proposals.
            if label == 1:
                values.sort(
                    key=lambda row: (
                        row.threshold_kind == "cross_50",
                        row.threshold_kind == "cross_75",
                        row.candidate_quality - row.source_quality,
                        -row.mean_abs_dx,
                    ),
                    reverse=True,
                )
            else:
                values.sort(
                    key=lambda row: (
                        row.threshold_kind == "cross_50",
                        row.threshold_kind == "cross_75",
                        row.candidate_quality,
                        -row.mean_abs_dx,
                    ),
                    reverse=True,
                )
            selected.extend(values[: max(int(max_pairs_per_class_slot), 1)])
    return selected


def pair_ribbon_x(
    source: torch.Tensor,
    candidate: torch.Tensor,
    *,
    points: int,
    margin_px: float,
) -> torch.Tensor:
    """Return an oriented ribbon with source/candidate fixed at 1/4 and 3/4."""

    if source.shape != candidate.shape or source.ndim != 2:
        raise ValueError("source/candidate curves must share [N,R]")
    u = torch.linspace(0.0, 1.0, int(points), device=source.device)
    u = u.view(1, 1, -1)
    delta = candidate - source
    global_direction = torch.sign(torch.nanmedian(delta, dim=-1).values)
    global_direction = torch.where(
        global_direction == 0,
        torch.ones_like(global_direction),
        global_direction,
    ).view(-1, 1, 1)
    source_e = source.unsqueeze(-1)
    candidate_e = candidate.unsqueeze(-1)
    left = source_e + ((u - 0.25) / 0.25) * float(margin_px) * global_direction
    middle = source_e + ((u - 0.25) / 0.50) * (candidate_e - source_e)
    right = candidate_e + ((u - 0.75) / 0.25) * float(margin_px) * global_direction
    return torch.where(u < 0.25, left, torch.where(u > 0.75, right, middle))


def _sample_feature(
    feature: torch.Tensor,
    x: torch.Tensor,
    *,
    input_w: int,
    input_h: int,
) -> torch.Tensor:
    """Sample [C,H,W] on [N,R,K] image coordinates -> [N,R,K,C]."""

    feature = feature.float()
    x = x.float()
    pairs, rows, points = x.shape
    y = torch.arange(rows, device=x.device, dtype=x.dtype)
    y = y * (float(input_h) / float(rows))
    y = y.view(1, rows, 1).expand(pairs, rows, points)
    grid_x = 2.0 * x.clamp(0.0, float(input_w - 1)) / float(input_w - 1) - 1.0
    grid_y = 2.0 * y.clamp(0.0, float(input_h - 1)) / float(input_h - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    sampled = F.grid_sample(
        feature.unsqueeze(0),
        grid.reshape(1, pairs * rows, points, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return (
        sampled.squeeze(0)
        .permute(1, 2, 0)
        .reshape(pairs, rows, points, feature.shape[0])
    )


def _project_or_pad(
    profile: torch.Tensor,
    channels: int,
    *,
    seed: int,
) -> torch.Tensor:
    source_channels = int(profile.shape[-1])
    if source_channels == int(channels):
        return profile
    if source_channels < int(channels):
        return F.pad(profile, (0, int(channels) - source_channels))
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 1009 * source_channels + int(channels)
    )
    matrix = torch.randn(source_channels, int(channels), generator=generator)
    matrix = torch.linalg.qr(matrix, mode="reduced").Q.to(
        device=profile.device,
        dtype=profile.dtype,
    )
    return profile @ matrix


def _pool_vertical(
    profile: torch.Tensor,
    valid: torch.Tensor,
    *,
    bands: int,
) -> torch.Tensor:
    pairs, rows, _points, _channels = profile.shape
    edges = torch.linspace(0, rows, int(bands) + 1).round().long().tolist()
    output = []
    for band in range(int(bands)):
        start = int(edges[band])
        end = max(int(edges[band + 1]), start + 1)
        weight = valid[:, start:end].float()
        denominator = weight.sum(dim=-1).clamp_min(1.0).view(pairs, 1, 1)
        pooled = (
            profile[:, start:end]
            * weight[:, :, None, None]
        ).sum(dim=1) / denominator
        output.append(pooled)
    return torch.stack(output, dim=1)


def geometry_features(
    source: torch.Tensor,
    candidate: torch.Tensor,
    source_range: torch.Tensor,
    candidate_range: torch.Tensor,
    valid: torch.Tensor,
    *,
    input_w: int,
) -> torch.Tensor:
    rows = int(source.shape[-1])
    result = []
    for index in range(int(source.shape[0])):
        mask = valid[index]
        delta = (candidate[index] - source[index]) / float(input_w)
        values = delta[mask]
        if values.numel() == 0:
            values = delta.new_zeros((1,))
        quantiles = torch.quantile(
            values,
            values.new_tensor((0.10, 0.25, 0.50, 0.75, 0.90)),
        )
        thirds = []
        for start, end in ((0, rows // 3), (rows // 3, 2 * rows // 3), (2 * rows // 3, rows)):
            band = mask[start:end]
            band_values = delta[start:end][band]
            thirds.append(
                band_values.mean() if band_values.numel() else delta.new_zeros(())
            )
        slope = delta[1:] - delta[:-1]
        curvature = slope[1:] - slope[:-1]
        vector = torch.stack(
            (
                values.mean(),
                values.std(unbiased=False),
                values.abs().mean(),
                *quantiles.unbind(),
                *thirds,
                slope.mean(),
                slope.std(unbiased=False),
                curvature.abs().mean(),
                curvature.std(unbiased=False),
                candidate_range[index, 0] - source_range[index, 0],
                candidate_range[index, 1] - source_range[index, 1],
                mask.float().mean(),
            )
        )
        result.append(vector)
    return torch.stack(result)


def _pair_population_summary(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = defaultdict(int)
    by_kind = defaultdict(int)
    distances = []
    for row in metadata:
        by_label[str(row["label"])] += 1
        by_kind[str(row["threshold_kind"])] += 1
        distances.append(float(row["mean_abs_dx"]))
    return {
        "pairs": len(metadata),
        "labels": dict(sorted(by_label.items())),
        "threshold_kind": dict(sorted(by_kind.items())),
        "mean_abs_dx": {
            "mean": float(np.mean(distances)) if distances else 0.0,
            "p50": float(np.quantile(distances, 0.50)) if distances else 0.0,
            "p90": float(np.quantile(distances, 0.90)) if distances else 0.0,
        },
    }


@torch.inference_mode()
def collect_feature_set(
    model: nn.Module,
    checkpoint_cfg: dict[str, Any],
    quality_records: dict[str, dict[str, Any]],
    *,
    list_path: str,
    args: argparse.Namespace,
    description: str,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(checkpoint_cfg))
    cfg["dataset"]["lists"]["val"] = str(Path(list_path).expanduser().resolve())
    loader = build_dataloader(cfg, split="val", training=False)
    device = torch.device(args.device)
    selection_head = model.structured_query_head.set_selection_head
    refiner = selection_head.slot_refinement
    if refiner is None:
        raise RuntimeError("pair-corridor audit requires the frozen V7 refiner")
    captured: dict[str, torch.Tensor] = {}

    def capture_p1(_module, _inputs, output):
        captured["p1"] = output.detach()

    def capture_refiner(_module, _inputs, kwargs):
        captured.update(
            {
                f"refiner_{key}": value
                for key, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            }
        )

    p1_handle = model.encoder.backbone.level1.register_forward_hook(capture_p1)
    refiner_handle = refiner.register_forward_pre_hook(capture_refiner, with_kwargs=True)
    profiles: dict[str, list[torch.Tensor]] = {name: [] for name in ARMS}
    geometry_rows: list[torch.Tensor] = []
    reverse_geometry_rows: list[torch.Tensor] = []
    labels: list[int] = []
    groups: list[int] = []
    pair_metadata: list[dict[str, Any]] = []
    input_w = int(cfg["model"]["input_w"])
    input_h = int(cfg["model"]["input_h"])
    images_seen = 0
    try:
        for batch_index, (images, _targets, metas) in enumerate(
            tqdm(loader, desc=description, ncols=96)
        ):
            if args.max_images is not None and images_seen >= int(args.max_images):
                break
            images = images.to(device, non_blocking=True)
            mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            rgb_batch = (images.float() * std + mean).clamp(0.0, 1.0)
            batch_size = int(images.shape[0])
            if batch_size < 2:
                raise RuntimeError("wrong-image control requires eval batch size >= 2")
            # The fixed quality cache was generated at batch size one.  V7 has
            # a few exact route-score ties whose proposal ID changes under a
            # different CUDA batch shape.  Forward every image independently
            # so geometry/labels remain bit-for-bit on the cached endpoint;
            # the other singleton in this loader batch supplies wrong-image P1.
            states: list[dict[str, Any]] = []
            for batch_item, meta in enumerate(metas):
                image_id = _image_id(meta, f"pair_{batch_index:06d}_{batch_item}")
                captured.clear()
                outputs = model(images[batch_item : batch_item + 1])
                required = {
                    "refiner_slot_states",
                    "refiner_proposal_row_tokens",
                    "refiner_proposal_x_rows",
                    "refiner_proposal_range_norm",
                    "refiner_candidate_valid",
                    "refiner_row_value_features",
                    "p1",
                }
                missing = required - set(captured)
                if missing:
                    raise RuntimeError(f"feature hooks missed: {sorted(missing)}")
                counterfactual = frozen_v7_counterfactual_anchors(
                    refiner,
                    slot_states=captured["refiner_slot_states"],
                    proposal_row_tokens=captured["refiner_proposal_row_tokens"],
                    proposal_x_rows=captured["refiner_proposal_x_rows"],
                    proposal_range_norm=captured["refiner_proposal_range_norm"],
                    candidate_valid=captured["refiner_candidate_valid"],
                    row_value_features=captured["refiner_row_value_features"],
                )
                states.append(
                    {
                        "meta": meta,
                        "image_id": image_id,
                        "clip_id": _clip_id(image_id),
                        "routes": outputs[
                            "selection_slot_geometry_route_indices"
                        ][0].detach().cpu().long(),
                        "active": outputs["selection_slot_active"][0]
                        .detach()
                        .cpu()
                        .bool(),
                        "source_x": outputs["selection_slot_pred_x_rows"][0].float(),
                        "source_range": outputs["selection_slot_range_norm"][0].float(),
                        "refined_x": counterfactual["x_rows"][0].float(),
                        "refined_range": counterfactual["range_norm"][0].float(),
                        "p2": captured["refiner_row_value_features"][0]
                        .permute(2, 0, 1)
                        .contiguous(),
                        "p1": captured["p1"][0],
                        "rgb": rgb_batch[batch_item],
                    }
                )
            for batch_item, state in enumerate(states):
                if args.max_images is not None and images_seen >= int(args.max_images):
                    break
                meta = state["meta"]
                image_id = state["image_id"]
                record = quality_records.get(image_id)
                if record is None:
                    raise KeyError(f"quality cache lacks image: {image_id}")
                observed_routes = state["routes"]
                observed_active = state["active"]
                if not torch.equal(observed_routes, record["source_routes"].long()):
                    raise RuntimeError(
                        f"V7 route parity failed for {image_id}: "
                        f"observed={observed_routes.tolist()} "
                        f"cached={record['source_routes'].long().tolist()}"
                    )
                if not torch.equal(observed_active, record["active"].bool()):
                    raise RuntimeError(f"V7 activity parity failed for {image_id}")
                source_x_all = state["source_x"]
                source_range_all = state["source_range"]
                refined_x_all = state["refined_x"]
                refined_range_all = state["refined_range"]
                specs = select_pair_specs(
                    record,
                    source_x_all,
                    source_range_all,
                    refined_x_all,
                    refined_range_all,
                    max_pairs_per_class_slot=int(args.max_pairs_per_class_slot),
                    max_mean_abs_dx=float(args.max_mean_abs_dx),
                    min_quality_gap=float(args.min_quality_gap),
                )
                if not specs:
                    images_seen += 1
                    continue
                source = torch.stack([source_x_all[row.slot] for row in specs])
                candidate = torch.stack(
                    [refined_x_all[row.slot, row.candidate] for row in specs]
                )
                source_range = torch.stack(
                    [source_range_all[row.slot] for row in specs]
                )
                candidate_range = torch.stack(
                    [refined_range_all[row.slot, row.candidate] for row in specs]
                )
                rows = int(source.shape[-1])
                source_valid = torch.stack(
                    [_valid_rows(source[i], source_range[i], rows=rows) for i in range(len(specs))]
                )
                candidate_valid = torch.stack(
                    [_valid_rows(candidate[i], candidate_range[i], rows=rows) for i in range(len(specs))]
                )
                common_valid = source_valid & candidate_valid
                ribbon = pair_ribbon_x(
                    source,
                    candidate,
                    points=int(args.ribbon_points),
                    margin_px=float(args.ribbon_margin_px),
                )
                p2 = _sample_feature(
                    state["p2"],
                    ribbon,
                    input_w=input_w,
                    input_h=input_h,
                )
                p1 = _sample_feature(
                    state["p1"],
                    ribbon,
                    input_w=input_w,
                    input_h=input_h,
                )
                wrong_index = next(
                    (
                        index
                        for index, other in enumerate(states)
                        if index != batch_item and other["clip_id"] != state["clip_id"]
                    ),
                    None,
                )
                if wrong_index is None:
                    raise RuntimeError(
                        "wrong-image control could not find an in-batch image "
                        f"from a different clip for {image_id}; increase "
                        "--eval-batch-size"
                    )
                wrong_p1 = _sample_feature(
                    states[wrong_index]["p1"],
                    ribbon,
                    input_w=input_w,
                    input_h=input_h,
                )
                rgb = _sample_feature(
                    state["rgb"],
                    ribbon,
                    input_w=input_w,
                    input_h=input_h,
                )
                projected = {
                    "p2_corridor": _project_or_pad(
                        p2,
                        int(args.profile_channels),
                        seed=int(args.seed),
                    ),
                    "p1_corridor": _project_or_pad(
                        p1,
                        int(args.profile_channels),
                        seed=int(args.seed),
                    ),
                    "wrong_p1_corridor": _project_or_pad(
                        wrong_p1,
                        int(args.profile_channels),
                        seed=int(args.seed),
                    ),
                    "rgb_corridor": _project_or_pad(
                        rgb,
                        int(args.profile_channels),
                        seed=int(args.seed),
                    ),
                }
                center = torch.zeros_like(projected["p2_corridor"])
                source_index = int(round(0.25 * (int(args.ribbon_points) - 1)))
                candidate_index = int(round(0.75 * (int(args.ribbon_points) - 1)))
                center[:, :, source_index] = projected["p2_corridor"][:, :, source_index]
                center[:, :, candidate_index] = projected["p2_corridor"][:, :, candidate_index]
                projected["p2_center"] = center
                zero = torch.zeros_like(projected["p2_corridor"])
                projected["geometry_only"] = zero
                for arm in ARMS:
                    pooled = _pool_vertical(
                        projected[arm],
                        common_valid,
                        bands=int(args.vertical_bands),
                    )
                    profiles[arm].append(
                        pooled.permute(0, 3, 1, 2).cpu().to(torch.float16)
                    )
                geometry_rows.append(
                    geometry_features(
                        source,
                        candidate,
                        source_range,
                        candidate_range,
                        common_valid,
                        input_w=input_w,
                    ).cpu()
                )
                reverse_geometry_rows.append(
                    geometry_features(
                        candidate,
                        source,
                        candidate_range,
                        source_range,
                        common_valid,
                        input_w=input_w,
                    ).cpu()
                )
                for row in specs:
                    labels.append(int(row.label))
                    groups.append(images_seen)
                    pair_metadata.append(
                        {
                            "image_id": image_id,
                            "wrong_image_id": states[wrong_index]["image_id"],
                            "slot": row.slot,
                            "candidate": row.candidate,
                            "owner_gt": row.owner_gt,
                            "label": row.label,
                            "source_quality": row.source_quality,
                            "candidate_quality": row.candidate_quality,
                            "quality_delta": row.candidate_quality - row.source_quality,
                            "threshold_kind": row.threshold_kind,
                            "mean_abs_dx": row.mean_abs_dx,
                        }
                    )
                images_seen += 1
    finally:
        p1_handle.remove()
        refiner_handle.remove()
    if not labels:
        raise RuntimeError("no same-owner source/candidate pairs were collected")
    return {
        "profiles": {name: torch.cat(values, dim=0) for name, values in profiles.items()},
        "geometry": torch.cat(geometry_rows, dim=0).float(),
        "reverse_geometry": torch.cat(reverse_geometry_rows, dim=0).float(),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "groups": torch.tensor(groups, dtype=torch.long),
        "pair_metadata": pair_metadata,
        "population": _pair_population_summary(pair_metadata),
        "images": images_seen,
    }


class PairRibbonProbe(nn.Module):
    def __init__(self, channels: int, geometry_dim: int) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(int(channels), 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.geometry = nn.Sequential(
            nn.LayerNorm(int(geometry_dim)),
            nn.Linear(int(geometry_dim), 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(16 * 4 * 4 + 32, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def raw_score(self, profile: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        visual = self.visual(profile).flatten(1)
        geometric = self.geometry(geometry)
        return self.head(torch.cat((visual, geometric), dim=-1)).squeeze(-1)

    def forward(
        self,
        profile: torch.Tensor,
        geometry: torch.Tensor,
        reverse_geometry: torch.Tensor,
    ) -> torch.Tensor:
        forward = self.raw_score(profile, geometry)
        reverse = self.raw_score(profile.flip(-1), reverse_geometry)
        return 0.5 * (forward - reverse)


def _auc(labels: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor) -> float | None:
    y = labels[mask].numpy()
    value = scores[mask].numpy()
    if y.size == 0 or np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, value))


def _metrics(
    labels: torch.Tensor,
    scores: torch.Tensor,
    metadata: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    threshold_mask = torch.tensor(
        [row["threshold_kind"] != "quality_only" for row in metadata],
        dtype=torch.bool,
    )
    prediction = scores > 0.0
    safe_prediction = scores > float(threshold)
    negative = labels == 0
    positive = labels == 1
    return {
        "pairs": int(labels.numel()),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "auc_all": _auc(labels, scores, torch.ones_like(positive)),
        "auc_threshold_changing": _auc(labels, scores, threshold_mask),
        "balanced_accuracy_at_zero": float(
            balanced_accuracy_score(labels.numpy(), prediction.numpy())
        ),
        "calibrated_threshold": float(threshold),
        "harmful_switch_rate": (
            float(safe_prediction[negative].float().mean())
            if bool(negative.any())
            else None
        ),
        "beneficial_recall_at_risk_threshold": (
            float(safe_prediction[positive].float().mean())
            if bool(positive.any())
            else None
        ),
        "selected_fraction": float(safe_prediction.float().mean()),
    }


def _scores(
    probe: PairRibbonProbe,
    profile: torch.Tensor,
    geometry: torch.Tensor,
    reverse_geometry: torch.Tensor,
    *,
    profile_mean: torch.Tensor,
    profile_std: torch.Tensor,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    output = []
    probe.eval()
    with torch.inference_mode():
        for start in range(0, int(profile.shape[0]), int(batch_size)):
            x = profile[start : start + int(batch_size)].to(device).float()
            g = geometry[start : start + int(batch_size)].to(device).float()
            rg = reverse_geometry[start : start + int(batch_size)].to(device).float()
            x = (x - profile_mean) / profile_std
            g = (g - geometry_mean) / geometry_std
            rg = (rg - geometry_mean) / geometry_std
            output.append(probe(x, g, rg).cpu())
    return torch.cat(output)


def train_and_evaluate_arm(
    train_set: dict[str, Any],
    test_set: dict[str, Any],
    arm: str,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    device = torch.device(args.device)
    labels = train_set["labels"]
    groups = train_set["groups"]
    unique_groups = groups.unique(sorted=True)
    calibration_groups = unique_groups[torch.arange(unique_groups.numel()) % 5 == 0]
    calibration = (groups[:, None] == calibration_groups[None]).any(dim=1)
    fit = ~calibration
    if labels[fit].unique().numel() < 2 or labels[calibration].unique().numel() < 2:
        raise RuntimeError("fit/calibration split must contain both pair classes")
    train_profile = train_set["profiles"][arm]
    profile_mean = train_profile[fit].float().mean(dim=(0, 2, 3), keepdim=True).to(device)
    profile_std = train_profile[fit].float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1.0e-4).to(device)
    geometry_mean = train_set["geometry"][fit].mean(dim=0, keepdim=True).to(device)
    geometry_std = train_set["geometry"][fit].std(dim=0, keepdim=True).clamp_min(1.0e-4).to(device)
    seed_everything(int(seed))
    probe = PairRibbonProbe(
        int(train_profile.shape[1]),
        int(train_set["geometry"].shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1.0e-4,
    )
    positives = float(labels[fit].sum())
    negatives = float(fit.sum()) - positives
    positive_weight = torch.tensor(
        negatives / max(positives, 1.0),
        device=device,
    )
    fit_ids = torch.nonzero(fit, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(int(seed))
    probe.train()
    last_loss = 0.0
    for _step in range(max(int(args.probe_steps), 1)):
        take = fit_ids[
            torch.randint(
                0,
                int(fit_ids.numel()),
                (min(int(args.probe_batch_size), int(fit_ids.numel())),),
                generator=generator,
            )
        ]
        x = train_profile[take].to(device).float()
        g = train_set["geometry"][take].to(device).float()
        rg = train_set["reverse_geometry"][take].to(device).float()
        y = labels[take].to(device)
        x = (x - profile_mean) / profile_std
        g = (g - geometry_mean) / geometry_std
        rg = (rg - geometry_mean) / geometry_std
        logits = probe(x, g, rg)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=positive_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    calibration_scores = _scores(
        probe,
        train_profile[calibration],
        train_set["geometry"][calibration],
        train_set["reverse_geometry"][calibration],
        profile_mean=profile_mean,
        profile_std=profile_std,
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
        device=device,
        batch_size=int(args.probe_batch_size),
    )
    calibration_labels = labels[calibration]
    negative_scores = calibration_scores[calibration_labels == 0]
    if negative_scores.numel() < 10:
        raise RuntimeError("too few calibration negatives for the 1% risk gate")
    sorted_negative = negative_scores.sort().values
    rank = min(
        max(int(math.ceil(0.99 * int(sorted_negative.numel()))) - 1, 0),
        int(sorted_negative.numel()) - 1,
    )
    risk_threshold = float(sorted_negative[rank])
    test_scores = _scores(
        probe,
        test_set["profiles"][arm],
        test_set["geometry"],
        test_set["reverse_geometry"],
        profile_mean=profile_mean,
        profile_std=profile_std,
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
        device=device,
        batch_size=int(args.probe_batch_size),
    )
    calibration_metadata = [
        row for index, row in enumerate(train_set["pair_metadata"]) if bool(calibration[index])
    ]
    return {
        "arm": arm,
        "parameters": sum(parameter.numel() for parameter in probe.parameters()),
        "fit_pairs": int(fit.sum()),
        "calibration_pairs": int(calibration.sum()),
        "final_train_loss": last_loss,
        "calibration": _metrics(
            calibration_labels,
            calibration_scores,
            calibration_metadata,
            risk_threshold,
        ),
        "test": _metrics(
            test_set["labels"],
            test_scores,
            test_set["pair_metadata"],
            risk_threshold,
        ),
    }


def _load_or_collect(
    cache_path: str | None,
    *,
    reuse: bool,
    collect,
) -> dict[str, Any]:
    if cache_path and reuse and Path(cache_path).expanduser().is_file():
        return torch.load(Path(cache_path).expanduser(), map_location="cpu", weights_only=False)
    value = collect()
    if cache_path:
        path = Path(cache_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
    return value


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    seed_everything(int(args.seed))
    metadata_a, records_a = _load_quality_cache(args.quality_cache_a)
    metadata_b, records_b = _load_quality_cache(args.quality_cache_b)
    for metadata, list_path in ((metadata_a, args.list_a), (metadata_b, args.list_b)):
        if Path(metadata["list_path"]).resolve() != Path(list_path).expanduser().resolve():
            raise ValueError("quality cache/list path mismatch")
    cfg = _checkpoint_cfg(
        args.checkpoint,
        args.dataset_root,
        args.list_a,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=True)
    model.eval()
    set_a = _load_or_collect(
        args.feature_cache_a,
        reuse=bool(args.reuse_feature_cache),
        collect=lambda: collect_feature_set(
            model,
            cfg,
            records_a,
            list_path=args.list_a,
            args=args,
            description="collect pair corridor set A",
        ),
    )
    set_b = _load_or_collect(
        args.feature_cache_b,
        reuse=bool(args.reuse_feature_cache),
        collect=lambda: collect_feature_set(
            model,
            cfg,
            records_b,
            list_path=args.list_b,
            args=args,
            description="collect pair corridor set B",
        ),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    directions = []
    for direction_index, (name, train_set, test_set) in enumerate(
        (
            ("A_to_B", set_a, set_b),
            ("B_to_A", set_b, set_a),
        )
    ):
        arms = []
        # All evidence arms in a direction use the exact same initialization.
        # This makes arm-to-arm differences attributable to the evidence tensor,
        # rather than to probe initialization noise.
        for arm in ARMS:
            result = train_and_evaluate_arm(
                train_set,
                test_set,
                arm,
                args=args,
                seed=int(args.seed) + 100 * direction_index,
            )
            arms.append(result)
            print(
                f"{name}/{arm}: AUC={result['test']['auc_all']} "
                f"risk-recall={result['test']['beneficial_recall_at_risk_threshold']}"
            )
        directions.append({"direction": name, "arms": arms})
    by_direction = {
        row["direction"]: {arm["arm"]: arm for arm in row["arms"]}
        for row in directions
    }
    gate_rows = []
    for name, arms in by_direction.items():
        p1 = arms["p1_corridor"]["test"]
        p2 = arms["p2_corridor"]["test"]
        geometry = arms["geometry_only"]["test"]
        wrong = arms["wrong_p1_corridor"]["test"]
        gate_rows.append(
            {
                "direction": name,
                "p1_auc_threshold_changing": p1["auc_threshold_changing"],
                "p1_minus_p2_auc_all": (
                    None
                    if p1["auc_all"] is None or p2["auc_all"] is None
                    else p1["auc_all"] - p2["auc_all"]
                ),
                "p1_minus_geometry_auc_all": (
                    None
                    if p1["auc_all"] is None or geometry["auc_all"] is None
                    else p1["auc_all"] - geometry["auc_all"]
                ),
                "p1_minus_wrong_auc_all": (
                    None
                    if p1["auc_all"] is None or wrong["auc_all"] is None
                    else p1["auc_all"] - wrong["auc_all"]
                ),
                "harmful_switch_rate": p1["harmful_switch_rate"],
                "beneficial_recall": p1["beneficial_recall_at_risk_threshold"],
            }
        )
    passed = all(
        row["p1_auc_threshold_changing"] is not None
        and row["p1_auc_threshold_changing"] >= 0.70
        and row["p1_minus_p2_auc_all"] is not None
        and row["p1_minus_p2_auc_all"] >= 0.08
        and row["p1_minus_geometry_auc_all"] is not None
        and row["p1_minus_geometry_auc_all"] >= 0.08
        and row["p1_minus_wrong_auc_all"] is not None
        and row["p1_minus_wrong_auc_all"] >= 0.08
        and row["harmful_switch_rate"] is not None
        and row["harmful_switch_rate"] <= 0.01
        and row["beneficial_recall"] is not None
        and row["beneficial_recall"] >= 0.20
        for row in gate_rows
    )
    payload = {
        "experiment": "V27 minimal source-vs-candidate pair-corridor observability",
        "diagnostic_only": True,
        "test_set_used": False,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "sets": {
            "A": {
                "list": str(Path(args.list_a).expanduser().resolve()),
                "quality_cache": str(Path(args.quality_cache_a).expanduser().resolve()),
                "population": set_a["population"],
            },
            "B": {
                "list": str(Path(args.list_b).expanduser().resolve()),
                "quality_cache": str(Path(args.quality_cache_b).expanduser().resolve()),
                "population": set_b["population"],
            },
        },
        "contract": {
            "candidate_geometry": "exact refined V7 source vs frozen counterfactual-refined proposal",
            "ownership": "source-Hungarian GT; candidate best GT must equal source owner",
            "pair_order": "antisymmetric source/candidate scoring",
            "visual_arms": ARMS,
            "profile_channels": int(args.profile_channels),
            "vertical_bands": int(args.vertical_bands),
            "ribbon_points": int(args.ribbon_points),
            "ribbon_margin_px": float(args.ribbon_margin_px),
            "train_eval": "A->B and B->A; calibration is image-disjoint within the training side",
            "risk_threshold": "fixed on training-side calibration negatives at empirical <=1% FPR",
        },
        "directions": directions,
        "gate": {
            "rows": gate_rows,
            "passed": bool(passed),
            "requirements": {
                "p1_threshold_changing_auc": 0.70,
                "p1_auc_gain_over_p2_geometry_wrong": 0.08,
                "harmful_switch_rate_max": 0.01,
                "beneficial_recall_min": 0.20,
                "must_pass_both_directions": True,
            },
        },
        "interpretation_guardrail": (
            "PASS supports observability, not guaranteed selector F1. FAIL closes "
            "this frozen late high-resolution visual-selector mechanism; it does "
            "not prove that all jointly trained detectors are impossible."
        ),
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_ready(payload["gate"]), indent=2, sort_keys=True))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
