"""Measure whether frozen S0 tokens encode the oracle affine correction.

The detector is never trained. Exact official-raster near-miss identities are
read from an existing candidate cache. Tiny probes are trained on a disjoint
set of images and evaluated on held-out images, preventing same-image leakage.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    diagnostic_iou_matrix,
    evaluator_hungarian_assignment,
    override_eval_list,
    trace_postprocess,
)
from dynlaneseq_eg.evaluation.near_miss_oracles import (
    fit_polynomial_delta,
    row_lane_iou,
)
from dynlaneseq_eg.evaluation.proposal_recall import collect_prediction_stages
from dynlaneseq_eg.factory import build_dataloader, build_model


@dataclass(frozen=True)
class NearMissRef:
    proposal_id: int
    gt_id: int
    official_iou: float


@dataclass
class ImageProbeRecord:
    image_id: str
    refs: list[NearMissRef]
    gt_x: torch.Tensor
    gt_mask: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage", default="main")
    parser.add_argument("--score-thresh", type=float, default=0.40)
    parser.add_argument("--quality-power", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--near-min-iou", type=float, default=0.30)
    parser.add_argument("--near-max-iou", type=float, default=0.50)
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--max-affine-displacement", type=float, default=64.0)
    parser.add_argument("--row-segments", type=int, default=8)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--direction-deadband-px", type=float, default=2.0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--probe-seeds",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025, 2026],
        help="Probe initialization/shuffle seeds; the image split remains fixed.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load_torch(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _stage_name(record: dict[str, Any], requested: str) -> str:
    if requested in record["stages"]:
        return requested
    if requested == "main":
        for candidate in ("final", "coarse", "stage2"):
            if candidate in record["stages"]:
                return candidate
    raise KeyError(f"Stage {requested!r} not found for {record['image_id']}")


def _canonical_id(value: str | Path) -> str:
    parts = Path(value).as_posix().split("/")
    return "/".join(parts[-3:]) if len(parts) >= 3 else "/".join(parts)


def collect_near_miss_records(
    cache: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, ImageProbeRecord], dict[str, int]]:
    model_meta = cache.get("metadata", {})
    input_h = int(model_meta.get("input_h", 288))
    input_w = int(model_meta.get("input_w", 800))
    post = model_meta.get("postprocess", {})
    nms_distance = float(post.get("lane_nms_distance_thresh_px", 20.0))
    nms_overlap = int(post.get("lane_nms_min_overlap_points", 5))

    records: dict[str, ImageProbeRecord] = {}
    counts = {
        "cache_images": 0,
        "near_miss_images": 0,
        "near_misses": 0,
        "skipped_gt_alignment_images": 0,
    }
    for record in cache["records"]:
        counts["cache_images"] += 1
        stage_name = _stage_name(record, args.stage)
        stage = record["stages"][stage_name]
        official, _valid_gt, _candidate_valid = diagnostic_iou_matrix(
            record,
            stage_name,
            use_official=True,
            input_h=input_h,
            input_w=input_w,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        trace = trace_postprocess(
            stage,
            input_h=input_h,
            input_w=input_w,
            score_thresh=args.score_thresh,
            quality_power=args.quality_power,
            min_valid_rows=args.min_valid_rows,
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_overlap,
            top_k=args.top_k,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        selected = list(trace["selected_ids"])
        assignment = evaluator_hungarian_assignment(official, selected, args.iou_thresh)
        tp_ids = set(assignment.proposal_ids)

        target_x = record["target"]["x_rows"].float()
        target_mask = record["target"]["valid_mask"].bool() & torch.isfinite(target_x)
        valid_gt = target_mask.sum(dim=-1) >= args.min_valid_rows
        target_x = target_x[valid_gt]
        target_mask = target_mask[valid_gt]
        if target_x.shape[0] != official.shape[0]:
            counts["skipped_gt_alignment_images"] += 1
            continue

        refs: list[NearMissRef] = []
        for proposal_id in selected:
            if proposal_id in tp_ids or official.shape[0] == 0:
                continue
            value, gt_id = official[:, proposal_id].max(dim=0)
            if args.near_min_iou <= float(value) <= args.near_max_iou:
                refs.append(
                    NearMissRef(
                        proposal_id=int(proposal_id),
                        gt_id=int(gt_id),
                        official_iou=float(value),
                    )
                )
        if refs:
            key = _canonical_id(record["image_id"])
            if key in records:
                raise RuntimeError(f"Canonical image ID collision: {key}")
            records[key] = ImageProbeRecord(
                image_id=str(record["image_id"]),
                refs=refs,
                gt_x=target_x,
                gt_mask=target_mask,
            )
            counts["near_miss_images"] += 1
            counts["near_misses"] += len(refs)
    return records, counts


def split_images(
    records: dict[str, ImageProbeRecord], args: argparse.Namespace
) -> tuple[set[str], set[str]]:
    ids = sorted(records)
    rng = random.Random(args.split_seed)
    rng.shuffle(ids)
    if args.max_images > 0:
        ids = ids[: args.max_images]
    if len(ids) < 2:
        raise RuntimeError(f"At least two near-miss images are required, found {len(ids)}")
    split_at = max(1, min(len(ids) - 1, round(len(ids) * args.train_fraction)))
    return set(ids[:split_at]), set(ids[split_at:])


def _chunk_means(tokens: torch.Tensor, segments: int) -> torch.Tensor:
    return torch.cat([chunk.mean(dim=0) for chunk in torch.tensor_split(tokens, segments, dim=0)])


def _geometry_feature(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    stage: dict[str, torch.Tensor],
    proposal_id: int,
    input_w: int,
    segments: int,
) -> torch.Tensor:
    x_values: list[torch.Tensor] = []
    visibility: list[torch.Tensor] = []
    for row_ids in torch.tensor_split(torch.arange(pred_x.numel()), segments):
        mask = pred_mask[row_ids]
        values = pred_x[row_ids]
        x_values.append(values[mask].mean() if bool(mask.any()) else values.mean())
        visibility.append(mask.float().mean())
    extras: list[torch.Tensor] = []
    for key in ("exist_logits", "quality_logits", "range_norm"):
        value = stage.get(key)
        if value is None:
            extras.append(torch.zeros(1))
        else:
            extras.append(value[proposal_id].float().reshape(-1))
    return torch.cat(
        [
            torch.stack(x_values) / max(float(input_w - 1), 1.0),
            torch.stack(visibility),
            *extras,
        ]
    )


def extract_features(
    records: dict[str, ImageProbeRecord],
    train_images: set[str],
    test_images: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = override_eval_list(load_config(args.config), args.split, args.list_path or None)
    input_h = int(cfg.get("model", {}).get("input_h", 288))
    input_w = int(cfg.get("model", {}).get("input_w", 800))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = build_dataloader(cfg, split=args.split, training=False)

    wanted = train_images | test_images
    lists: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "geometry",
            "q_ins",
            "pooled",
            "segmented",
            "target",
            "pred_x",
            "pred_mask",
            "gt_x",
            "gt_mask",
            "official_iou",
            "saturated",
            "split",
        )
    }
    seen: set[str] = set()
    saturated_samples = 0
    skipped_index = 0

    for images, _targets, metas in tqdm(loader, ncols=90, desc="extracting frozen probe features"):
        batch_ids = [_canonical_id(meta["image_path"]) for meta in metas]
        selected = [index for index, image_id in enumerate(batch_ids) if image_id in wanted]
        if not selected:
            continue
        selected_ids = [batch_ids[index] for index in selected]
        with torch.inference_mode():
            outputs = model(images[selected].to(device, non_blocking=True))
        stages = collect_prediction_stages(outputs)
        if args.stage in stages:
            batch_stage = stages[args.stage]
        elif args.stage == "main" and len(stages) == 1:
            batch_stage = next(iter(stages.values()))
        else:
            raise KeyError(f"Stage {args.stage!r} missing; available={list(stages)}")
        if "queries" not in batch_stage or "structured_row_tokens" not in batch_stage:
            raise KeyError(
                "Selected stage must expose queries and structured_row_tokens; "
                f"available={sorted(batch_stage)}"
            )

        for local_id, image_id in enumerate(selected_ids):
            seen.add(image_id)
            image_record = records[image_id]
            stage = {
                key: value[local_id].detach().cpu()
                for key, value in batch_stage.items()
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == len(selected)
            }
            pred_x_all, pred_mask_all, _ = candidate_row_masks(
                stage,
                input_h=input_h,
                input_w=input_w,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            queries = stage["queries"].float()
            row_tokens = stage["structured_row_tokens"].float()
            for ref in image_record.refs:
                if ref.proposal_id >= pred_x_all.shape[0] or ref.gt_id >= image_record.gt_x.shape[0]:
                    skipped_index += 1
                    continue
                pred_x = pred_x_all[ref.proposal_id]
                pred_mask = pred_mask_all[ref.proposal_id]
                gt_x = image_record.gt_x[ref.gt_id]
                gt_mask = image_record.gt_mask[ref.gt_id]
                coeff = fit_polynomial_delta(pred_x, pred_mask, gt_x, gt_mask, degree=1)
                if coeff is None:
                    skipped_index += 1
                    continue
                y = torch.linspace(-1.0, 1.0, pred_x.numel())
                saturated = bool(
                    float((coeff[0] + coeff[1] * y).abs().max())
                    > args.max_affine_displacement
                )
                saturated_samples += int(saturated)

                q_ins = queries[ref.proposal_id]
                q_geo = row_tokens[ref.proposal_id]
                lists["geometry"].append(
                    _geometry_feature(
                        pred_x,
                        pred_mask,
                        stage,
                        ref.proposal_id,
                        input_w,
                        args.row_segments,
                    )
                )
                lists["q_ins"].append(q_ins)
                lists["pooled"].append(
                    torch.cat([q_ins, q_geo.mean(dim=0), q_geo.amax(dim=0)])
                )
                lists["segmented"].append(
                    torch.cat([q_ins, _chunk_means(q_geo, args.row_segments)])
                )
                lists["target"].append(coeff.float())
                lists["pred_x"].append(pred_x.float())
                lists["pred_mask"].append(pred_mask.bool())
                lists["gt_x"].append(gt_x.float())
                lists["gt_mask"].append(gt_mask.bool())
                lists["official_iou"].append(torch.tensor(ref.official_iou))
                lists["saturated"].append(torch.tensor(saturated))
                lists["split"].append(torch.tensor(0 if image_id in train_images else 1))
        if seen >= wanted:
            break

    missing = wanted - seen
    if missing:
        raise RuntimeError(
            f"Could not align {len(missing)} cache image IDs with the dataloader; "
            f"examples={sorted(missing)[:5]}"
        )
    if not lists["target"]:
        raise RuntimeError("No valid probe samples were extracted")
    data: dict[str, Any] = {key: torch.stack(values) for key, values in lists.items()}
    data["metadata"] = {
        "selected_images": len(wanted),
        "train_images": len(train_images),
        "test_images": len(test_images),
        "source_near_misses": sum(len(records[key].refs) for key in wanted),
        "saturated_samples": saturated_samples,
        "skipped_index_or_fit": skipped_index,
        "input_w": input_w,
        "split_seed": args.split_seed,
    }
    return data


class Probe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, 2)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    return 0.0 if float(denom) <= 1e-12 else float((x * y).sum() / denom)


def evaluate_coefficients(
    predicted: torch.Tensor,
    data: dict[str, Any],
    test_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    predicted = predicted.float().cpu()
    target = data["target"][test_mask].float()
    mae = (predicted - target).abs().mean(dim=0)
    sse = (predicted - target).square().sum(dim=0)
    sst = (target - target.mean(dim=0)).square().sum(dim=0).clamp_min(1e-12)
    direction: dict[str, float | None] = {}
    for index, name in enumerate(("a0", "a1")):
        active = target[:, index].abs() >= args.direction_deadband_px
        direction[name] = (
            float((predicted[active, index].sign() == target[active, index].sign()).float().mean())
            if bool(active.any())
            else None
        )

    sample_ids = test_mask.nonzero(as_tuple=False).flatten()

    def row_metrics(local_mask: torch.Tensor) -> dict[str, Any]:
        chosen_local = local_mask.nonzero(as_tuple=False).flatten()
        baseline_ious: list[float] = []
        predicted_ious: list[float] = []
        oracle_ious: list[float] = []
        input_w = int(data.get("metadata", {}).get("input_w", 800))
        bound = float(args.max_affine_displacement)
        for local_id in chosen_local.tolist():
            sample_id = int(sample_ids[local_id])
            pred_x = data["pred_x"][sample_id]
            pred_mask = data["pred_mask"][sample_id]
            gt_x = data["gt_x"][sample_id]
            gt_mask = data["gt_mask"][sample_id]
            y = torch.linspace(-1.0, 1.0, pred_x.numel())
            pred_delta = (predicted[local_id, 0] + predicted[local_id, 1] * y).clamp(-bound, bound)
            oracle_delta = (target[local_id, 0] + target[local_id, 1] * y).clamp(-bound, bound)
            pred_corrected = (pred_x + pred_delta).clamp(0.0, float(input_w - 1))
            oracle_corrected = (pred_x + oracle_delta).clamp(0.0, float(input_w - 1))
            baseline_ious.append(
                float(row_lane_iou(pred_x, pred_mask, gt_x, gt_mask, line_width=args.line_width))
            )
            predicted_ious.append(
                float(
                    row_lane_iou(
                        pred_corrected,
                        pred_mask,
                        gt_x,
                        gt_mask,
                        line_width=args.line_width,
                    )
                )
            )
            oracle_ious.append(
                float(
                    row_lane_iou(
                        oracle_corrected,
                        pred_mask,
                        gt_x,
                        gt_mask,
                        line_width=args.line_width,
                    )
                )
            )
        if not predicted_ious:
            return {
                "count": 0,
                "baseline_mean_iou": None,
                "predicted_mean_iou": None,
                "oracle_mean_iou": None,
                "predicted_rescue_rate_at_0p5": None,
                "oracle_rescue_rate_at_0p5": None,
            }
        pred_iou = torch.tensor(predicted_ious)
        oracle_iou = torch.tensor(oracle_ious)
        return {
            "count": len(predicted_ious),
            "baseline_mean_iou": float(torch.tensor(baseline_ious).mean()),
            "predicted_mean_iou": float(pred_iou.mean()),
            "oracle_mean_iou": float(oracle_iou.mean()),
            "predicted_rescue_rate_at_0p5": float((pred_iou > args.iou_thresh).float().mean()),
            "oracle_rescue_rate_at_0p5": float((oracle_iou > args.iou_thresh).float().mean()),
        }

    saturated = data["saturated"][test_mask].bool()
    all_rows = row_metrics(torch.ones_like(saturated, dtype=torch.bool))
    return {
        "mae_px": {"a0": float(mae[0]), "a1": float(mae[1])},
        "pearson": {
            "a0": _pearson(predicted[:, 0], target[:, 0]),
            "a1": _pearson(predicted[:, 1], target[:, 1]),
        },
        "r2": {
            "a0": float(1.0 - sse[0] / sst[0]),
            "a1": float(1.0 - sse[1] / sst[1]),
        },
        "direction_accuracy": direction,
        "row_space": all_rows,
        "row_space_subgroups": {
            "unsaturated": row_metrics(~saturated),
            "saturated": row_metrics(saturated),
        },
    }


def train_probe(
    name: str,
    features: torch.Tensor,
    data: dict[str, Any],
    args: argparse.Namespace,
    hidden_dim: int | None,
    seed: int,
) -> dict[str, Any]:
    train_mask = data["split"] == 0
    test_mask = data["split"] == 1
    train_x = features[train_mask].float()
    test_x = features[test_mask].float()
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    target_scale = float(args.max_affine_displacement)
    train_coeff = data["target"][train_mask].float() / target_scale
    train_valid = (data["pred_mask"][train_mask] & data["gt_mask"][train_mask]).bool()

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_coeff, train_valid),
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=generator,
    )
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = Probe(train_x.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(beta=0.1)
    model.train()
    for _epoch in range(args.epochs):
        for batch_x, batch_coeff, batch_valid in loader:
            batch_x = batch_x.to(device)
            batch_coeff = batch_coeff.to(device)
            batch_valid = batch_valid.to(device)
            prediction = model(batch_x)
            rows = int(batch_valid.shape[1])
            y = torch.linspace(-1.0, 1.0, rows, device=device).view(1, rows)
            predicted_curve = (
                prediction[:, 0:1] + prediction[:, 1:2] * y
            ).clamp(-1.0, 1.0)
            target_curve = (
                batch_coeff[:, 0:1] + batch_coeff[:, 1:2] * y
            ).clamp(-1.0, 1.0)
            if bool(batch_valid.any()):
                loss = criterion(predicted_curve[batch_valid], target_curve[batch_valid])
            else:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(test_x.to(device)).cpu() * target_scale
    metrics = evaluate_coefficients(prediction, data, test_mask, args)
    metrics.update(
        {
            "name": name,
            "probe_type": "mlp" if hidden_dim is not None else "linear",
            "seed": int(seed),
            "input_dim": int(features.shape[1]),
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
        }
    )
    return metrics


def _mean_std(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def aggregate_probe_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["name"]), []).append(run)
    summaries: dict[str, Any] = {}
    for name, items in grouped.items():
        summary: dict[str, Any] = {
            "probe_type": items[0]["probe_type"],
            "input_dim": items[0]["input_dim"],
            "seeds": [int(item["seed"]) for item in items],
            "pearson": {},
            "r2": {},
            "direction_accuracy": {},
            "mae_px": {},
            "row_space": {},
            "row_space_subgroups": {},
        }
        for metric in ("pearson", "r2", "direction_accuracy", "mae_px"):
            for coefficient in ("a0", "a1"):
                values = [item[metric][coefficient] for item in items]
                values = [float(value) for value in values if value is not None]
                summary[metric][coefficient] = _mean_std(values) if values else None
        for metric in (
            "predicted_mean_iou",
            "predicted_rescue_rate_at_0p5",
        ):
            summary["row_space"][metric] = _mean_std(
                [float(item["row_space"][metric]) for item in items]
            )
        for subgroup in ("unsaturated", "saturated"):
            summary["row_space_subgroups"][subgroup] = {}
            for metric in (
                "predicted_mean_iou",
                "predicted_rescue_rate_at_0p5",
            ):
                values = [
                    item["row_space_subgroups"][subgroup][metric]
                    for item in items
                    if item["row_space_subgroups"][subgroup][metric] is not None
                ]
                summary["row_space_subgroups"][subgroup][metric] = (
                    _mean_std([float(value) for value in values]) if values else None
                )
        summaries[name] = summary
    return summaries


def main() -> None:
    args = parse_args()
    random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    feature_path = Path(args.feature_cache)

    source_counts: dict[str, int] | None = None
    if args.reuse_feature_cache and feature_path.exists():
        data = _load_torch(feature_path)
    else:
        cache = _load_torch(args.candidate_cache)
        records, source_counts = collect_near_miss_records(cache, args)
        train_images, test_images = split_images(records, args)
        data = extract_features(records, train_images, test_images, args)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, feature_path)

    train_mask = data["split"] == 0
    test_mask = data["split"] == 1
    if int(train_mask.sum()) < 20 or int(test_mask.sum()) < 20:
        raise RuntimeError(
            f"Insufficient samples: train={int(train_mask.sum())}, test={int(test_mask.sum())}"
        )

    test_count = int(test_mask.sum())
    train_mean = data["target"][train_mask].mean(dim=0)
    baselines = {
        "zero": evaluate_coefficients(torch.zeros((test_count, 2)), data, test_mask, args),
        "train_mean": evaluate_coefficients(train_mean.repeat(test_count, 1), data, test_mask, args),
    }
    probe_specs = (
        ("geometry_linear", data["geometry"], None),
        ("geometry_mlp", data["geometry"], args.hidden_dim),
        ("q_ins_linear", data["q_ins"], None),
        ("q_ins_mlp", data["q_ins"], args.hidden_dim),
        ("pooled_linear", data["pooled"], None),
        ("pooled_mlp", data["pooled"], args.hidden_dim),
        ("segmented_linear", data["segmented"], None),
        ("segmented_mlp", data["segmented"], args.hidden_dim),
    )
    probe_runs: list[dict[str, Any]] = []
    for seed in args.probe_seeds:
        for name, features, hidden_dim in probe_specs:
            probe_runs.append(
                train_probe(name, features, data, args, hidden_dim, seed=int(seed))
            )
    probe_summaries = aggregate_probe_runs(probe_runs)

    test_target = data["target"][test_mask]
    report = {
        "metadata": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "candidate_cache": args.candidate_cache,
            "feature_cache": str(feature_path),
            "source_counts": source_counts,
            "extraction": data.get("metadata", {}),
            "train_samples": int(train_mask.sum()),
            "test_samples": test_count,
            "image_level_split": True,
            "split_seed": args.split_seed,
            "probe_seeds": args.probe_seeds,
            "bounded_curve_training": True,
            "saturated_samples_included": True,
        },
        "test_target": {
            "a0_mean": float(test_target[:, 0].mean()),
            "a0_std": float(test_target[:, 0].std()),
            "a1_mean": float(test_target[:, 1].mean()),
            "a1_std": float(test_target[:, 1].std()),
        },
        "baselines": baselines,
        "probe_summaries": probe_summaries,
        "probe_runs": probe_runs,
        "interpretation": {
            "geometry_only": "Control for dataset/shape priors; it must not be mistaken for visual evidence.",
            "q_ins_vs_segmented": (
                "Weak q_ins with strong segmented tokens means row evidence exists but pooling loses it. "
                "Both near the baselines means the frozen representation lacks affine-direction evidence."
            ),
            "scope": (
                "This is a frozen-representation diagnostic on held-out validation images, not an "
                "end-to-end F1 result and not proof that a full correction head will train safely."
            ),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"probe samples: train={int(train_mask.sum())} test={test_count}")
    for name, result in probe_summaries.items():
        rho_a0 = result["pearson"]["a0"]
        rho_a1 = result["pearson"]["a1"]
        dir_a0 = result["direction_accuracy"]["a0"]
        dir_a1 = result["direction_accuracy"]["a1"]
        rescue = result["row_space"]["predicted_rescue_rate_at_0p5"]
        print(
            f"{name:>20}: "
            f"rho={rho_a0['mean']:.3f}+/-{rho_a0['std']:.3f}/"
            f"{rho_a1['mean']:.3f}+/-{rho_a1['std']:.3f} "
            f"dir={dir_a0['mean']:.3f}/{dir_a1['mean']:.3f} "
            f"rescue={rescue['mean']:.3f}+/-{rescue['std']:.3f}"
        )
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
