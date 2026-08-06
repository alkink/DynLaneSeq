from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import sort_range_norm
from dynlaneseq_eg.tools.audit_v4_pointer_large_representation import (
    _feature_moments,
    _state_to_cpu,
    evaluate_scores,
    make_image_disjoint_splits,
    summarize_runs,
)
from dynlaneseq_eg.tools.audit_v4_pointer_root_causes import (
    _autocast_kwargs,
    _git_commit,
    _root_selector,
    _sha256,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.train import seed_everything


ROW_CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Determine whether V4 representative quality was lost by pooled "
            "candidate descriptors or is absent from the frozen row-level "
            "geometry/evidence representation itself."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--row-cache", required=True)
    parser.add_argument("--reuse-row-cache", action="store_true")
    parser.add_argument("--development-images", type=int, default=4096)
    parser.add_argument("--holdout-images", type=int, default=1024)
    parser.add_argument("--early-stop-images", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--row-samples", type=int, default=24)
    parser.add_argument("--channel-projection", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=8849)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3407, 5419, 7823],
    )
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--row-batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--candidate-lr", type=float, default=1e-3)
    parser.add_argument("--row-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--candidate-hidden", type=int, default=128)
    parser.add_argument("--row-hidden", type=int, default=128)
    parser.add_argument("--row-layers", type=int, default=2)
    parser.add_argument("--row-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--representable-min", type=float, default=0.50)
    parser.add_argument("--cluster-quality-min", type=float, default=1e-4)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--pair-min-quality-gap", type=float, default=0.02)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--residual-l2-weight", type=float, default=1e-4)
    parser.add_argument("--split-seed", type=int, default=1907)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_base_cache(
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    required = (
        "raw_descriptor",
        "relation_hidden",
        "checkpoint_unary",
        "candidate_valid",
        "quality_max",
        "quality",
        "metadata",
    )
    missing = [name for name in required if name not in cache]
    if missing:
        raise ValueError(
            "large representation base cache is missing: " + ", ".join(missing)
        )
    total = int(args.development_images + args.holdout_images)
    for name in (
        "raw_descriptor",
        "relation_hidden",
        "checkpoint_unary",
        "candidate_valid",
        "quality_max",
    ):
        if int(cache[name].shape[0]) != total:
            raise ValueError(f"base cache {name} does not contain {total} images")
    if len(cache["quality"]) != total:
        raise ValueError("base cache quality records do not cover all images")
    sampled = cache["metadata"].get("sampled_dataset_indices")
    image_ids = cache["metadata"].get("image_ids")
    if not isinstance(sampled, list) or len(sampled) != total:
        raise ValueError("base cache lacks exact sampled dataset indices")
    if not isinstance(image_ids, list) or len(image_ids) != total:
        raise ValueError("base cache lacks exact image identities")
    return total


def fixed_rademacher_projection(
    input_dimension: int,
    output_dimension: int,
    seed: int,
) -> torch.Tensor:
    if input_dimension < 1 or output_dimension < 1:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (int(input_dimension), int(output_dimension)),
        generator=generator,
        dtype=torch.int64,
    )
    return (signs.float() * 2.0 - 1.0) / math.sqrt(float(output_dimension))


def build_row_sequence_features(
    outputs: dict[str, torch.Tensor],
    selector: nn.Module,
    row_projection: torch.Tensor,
    evidence_projection: torch.Tensor,
    row_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_tokens = outputs.get("structured_row_tokens")
    evidence = outputs.get("selection_curve_evidence")
    if not isinstance(row_tokens, torch.Tensor):
        raise ValueError("row-sequence audit requires structured_row_tokens")
    if not isinstance(evidence, torch.Tensor):
        raise ValueError("row-sequence audit requires selection_curve_evidence")
    row_logits = outputs["row_x_logits"].detach().float()
    pred_x = outputs["pred_x_rows"].detach().float()
    ranges = sort_range_norm(outputs["range_norm"].detach().float())
    rows = int(row_tokens.shape[2])
    if evidence.shape[:3] != row_tokens.shape[:3]:
        raise ValueError("row-token and curve-evidence shapes disagree")
    if row_ids.ndim != 1 or bool((row_ids >= rows).any()):
        raise ValueError("row sample ids are outside the detector row grid")

    sampled_rows = row_tokens.detach().float().index_select(2, row_ids)
    sampled_evidence = evidence.detach().float().index_select(2, row_ids)
    projected_rows = torch.matmul(
        sampled_rows,
        row_projection.to(device=row_tokens.device),
    )
    projected_evidence = torch.matmul(
        sampled_evidence,
        evidence_projection.to(device=row_tokens.device),
    )

    probability = torch.softmax(row_logits, dim=-1)
    confidence = probability.amax(dim=-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / math.log(float(max(int(row_logits.shape[-1]), 2)))
    confidence = confidence.index_select(2, row_ids)
    entropy = entropy.index_select(2, row_ids)
    sampled_x = pred_x.index_select(2, row_ids) / float(
        max(int(selector.input_w) - 1, 1)
    )

    y_all = selector._lane_row_grid(
        rows,
        device=row_tokens.device,
        dtype=torch.float32,
    )
    sampled_y = y_all.index_select(0, row_ids)
    visible = (
        (sampled_y.view(1, 1, -1) >= ranges[..., :1])
        & (sampled_y.view(1, 1, -1) <= ranges[..., 1:])
        & torch.isfinite(sampled_x)
    )
    y_feature = sampled_y.view(1, 1, -1).expand_as(sampled_x)
    reference = outputs.get("input_reference_x_rows")
    if isinstance(reference, torch.Tensor):
        reference_delta = (
            pred_x - reference.detach().float()
        ).abs().index_select(2, row_ids) / float(
            max(int(selector.input_w) - 1, 1)
        )
    else:
        reference_delta = torch.zeros_like(sampled_x)

    scalars = torch.stack(
        (
            sampled_x.nan_to_num(),
            confidence,
            entropy,
            visible.to(dtype=torch.float32),
            y_feature,
            reference_delta,
        ),
        dim=-1,
    )
    features = torch.cat(
        (projected_rows, projected_evidence, scalars),
        dim=-1,
    )
    return features, visible


@torch.no_grad()
def collect_row_cache(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model: nn.Module,
    selector: nn.Module,
    base_cache: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    cache_path = Path(args.row_cache).expanduser()
    checkpoint_stat = checkpoint_path.stat()
    signature = {
        "version": ROW_CACHE_VERSION,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "base_cache_signature": base_cache.get("signature"),
        "row_samples": int(args.row_samples),
        "channel_projection": int(args.channel_projection),
        "projection_seed": int(args.projection_seed),
    }
    if args.reuse_row_cache and cache_path.is_file():
        payload = _load_torch(cache_path)
        if payload.get("signature") != signature:
            raise ValueError(
                "row-sequence cache signature mismatch; use a new path or "
                "remove --reuse-row-cache"
            )
        return payload

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("dataset", {})["root"] = str(args.dataset_root)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.feature_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    total = int(args.development_images + args.holdout_images)
    loader = build_dataloader(cfg, split=args.split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy="uniform",
        max_batches=math.ceil(total / int(args.feature_batch_size)),
        num_workers=int(args.num_workers),
    )
    expected_indices = base_cache["metadata"]["sampled_dataset_indices"]
    if sampled_indices[:total] != expected_indices:
        raise ValueError("row cache and base cache sample identities differ")

    autocast_kwargs = _autocast_kwargs(device, args.amp_dtype)
    feature_rows: list[torch.Tensor] = []
    visible_rows: list[torch.Tensor] = []
    image_ids: list[str] = []
    row_ids_cpu: torch.Tensor | None = None
    row_projection_cpu: torch.Tensor | None = None
    evidence_projection_cpu: torch.Tensor | None = None
    collected = 0
    model.eval()
    for images, _targets, metas in tqdm(
        loader,
        desc="row-sequence frozen cache",
        ncols=92,
    ):
        if collected >= total:
            break
        images = images.to(device, non_blocking=True)
        with torch.autocast(**autocast_kwargs):
            outputs = model(images)
        row_tokens = outputs["structured_row_tokens"]
        evidence = outputs["selection_curve_evidence"]
        rows = int(row_tokens.shape[2])
        if row_ids_cpu is None:
            sample_count = min(int(args.row_samples), rows)
            row_ids_cpu = torch.linspace(
                0,
                rows - 1,
                sample_count,
            ).round().long().unique(sorted=True)
            row_projection_cpu = fixed_rademacher_projection(
                int(row_tokens.shape[-1]),
                int(args.channel_projection),
                int(args.projection_seed),
            )
            evidence_projection_cpu = fixed_rademacher_projection(
                int(evidence.shape[-1]),
                int(args.channel_projection),
                int(args.projection_seed) + 1,
            )
        assert row_projection_cpu is not None
        assert evidence_projection_cpu is not None
        row_features, row_visible = build_row_sequence_features(
            outputs,
            selector,
            row_projection_cpu,
            evidence_projection_cpu,
            row_ids_cpu.to(device),
        )
        keep = min(int(images.shape[0]), total - collected)
        feature_rows.append(
            row_features[:keep].detach().to(device="cpu", dtype=torch.float16)
        )
        visible_rows.append(row_visible[:keep].detach().cpu())
        for local_index in range(keep):
            image_id = str(metas[local_index].get("image_path", collected))
            expected_image = str(
                base_cache["metadata"]["image_ids"][collected]
            )
            if image_id != expected_image:
                raise ValueError(
                    "row cache image order differs from the base cache: "
                    f"{image_id!r} != {expected_image!r}"
                )
            image_ids.append(image_id)
            collected += 1
    if collected != total:
        raise RuntimeError(f"requested {total} row records but collected {collected}")
    assert row_ids_cpu is not None
    payload = {
        "signature": signature,
        "metadata": {
            "sampled_dataset_indices": sampled_indices[:total],
            "image_ids": image_ids,
            "augmentation_enabled": False,
            "projection_type": "fixed_rademacher",
            "row_ids": row_ids_cpu.tolist(),
            "row_feature_dimension": int(feature_rows[0].shape[-1]),
        },
        "row_features": torch.cat(feature_rows, dim=0),
        "row_visible": torch.cat(visible_rows, dim=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def row_feature_moments(
    features: torch.Tensor,
    visible: torch.Tensor,
    candidate_valid: torch.Tensor,
    indices: list[int],
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = int(features.shape[-1])
    total = torch.zeros(dimension, dtype=torch.float64)
    square = torch.zeros(dimension, dtype=torch.float64)
    count = 0
    for start in range(0, len(indices), int(chunk_size)):
        ids = indices[start : start + int(chunk_size)]
        values = features[ids].float()
        mask = visible[ids].bool() & candidate_valid[ids].bool().unsqueeze(-1)
        selected = values[mask]
        if selected.numel() == 0:
            continue
        total += selected.double().sum(dim=0)
        square += selected.double().square().sum(dim=0)
        count += int(selected.shape[0])
    if count < 2:
        raise ValueError("not enough visible row features for normalization")
    mean = total / float(count)
    variance = square / float(count) - mean.square()
    return mean.float(), variance.clamp_min(1e-6).sqrt().float()


class CandidateResidualProbe(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        nonlinear: bool,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-3))
        dimension = int(mean.numel())
        if nonlinear:
            self.residual = nn.Sequential(
                nn.Linear(dimension, int(hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden), 1),
            )
            nn.init.normal_(self.residual[-1].weight, std=1e-3)
            nn.init.zeros_(self.residual[-1].bias)
        else:
            self.residual = nn.Linear(dimension, 1)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        features: torch.Tensor,
        baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features.float() - self.mean) / self.std
        residual = self.residual(normalized).squeeze(-1)
        return baseline.float() + residual, residual


class RowSequenceResidualProbe(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        row_count: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden) % int(heads) != 0:
            raise ValueError("row hidden dimension must be divisible by heads")
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-3))
        self.input_projection = nn.Linear(int(mean.numel()), int(hidden))
        self.position = nn.Parameter(torch.zeros(1, int(row_count), int(hidden)))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden),
            nhead=int(heads),
            dim_feedforward=2 * int(hidden),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.pool_score = nn.Linear(int(hidden), 1)
        self.output = nn.Sequential(
            nn.LayerNorm(int(hidden)),
            nn.Linear(int(hidden), 1),
        )
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        visible: torch.Tensor,
        baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, rows, dimension = features.shape
        if rows != int(self.position.shape[1]) or dimension != int(self.mean.numel()):
            raise ValueError("row residual probe input shape mismatch")
        normalized = (features.float() - self.mean) / self.std
        hidden = self.input_projection(normalized) + self.position.unsqueeze(1)
        hidden = hidden.reshape(batch * candidates, rows, -1)
        keep = visible.bool().reshape(batch * candidates, rows)
        safe_keep = keep.clone()
        empty = ~safe_keep.any(dim=-1)
        if bool(empty.any()):
            safe_keep[empty, 0] = True
        encoded = self.encoder(hidden, src_key_padding_mask=~safe_keep)
        attention = self.pool_score(encoded).squeeze(-1)
        attention = attention.masked_fill(~safe_keep, -1e4)
        weight = torch.softmax(attention, dim=-1)
        pooled = (encoded * weight.unsqueeze(-1)).sum(dim=1)
        residual = self.output(pooled).reshape(batch, candidates)
        return baseline.float() + residual, residual


def prepare_full_cluster_targets(
    cache: dict[str, Any],
    *,
    representable_min: float,
    cluster_quality_min: float,
    target_temperature: float,
    pair_min_quality_gap: float,
) -> None:
    images = len(cache["quality"])
    candidates = int(cache["candidate_valid"].shape[1])
    max_clusters = max(
        (int(quality.shape[1]) for quality in cache["quality"]),
        default=0,
    )
    max_clusters = max(max_clusters, 1)
    cluster_mask = torch.zeros(
        (images, max_clusters, candidates),
        dtype=torch.bool,
    )
    cluster_target = torch.zeros(
        (images, max_clusters, candidates),
        dtype=torch.float16,
    )
    cluster_active = torch.zeros((images, max_clusters), dtype=torch.bool)
    all_pairs: list[torch.Tensor] = []
    all_pair_weights: list[torch.Tensor] = []
    for image_index, quality_value in enumerate(cache["quality"]):
        quality = quality_value.float()
        valid = cache["candidate_valid"][image_index].bool()
        local_pairs: list[torch.Tensor] = []
        local_weights: list[torch.Tensor] = []
        if int(quality.shape[1]) > 0:
            owner = quality.argmax(dim=-1)
            output_cluster = 0
            for gt_index in range(int(quality.shape[1])):
                q = quality[:, gt_index]
                cluster = valid & (owner == gt_index) & (
                    q >= float(cluster_quality_min)
                )
                if not bool(cluster.any()) or float(q[cluster].max()) < float(
                    representable_min
                ):
                    continue
                ids = torch.nonzero(cluster, as_tuple=False).flatten()
                cluster_mask[image_index, output_cluster, ids] = True
                cluster_target[image_index, output_cluster, ids] = torch.softmax(
                    q[ids] / max(float(target_temperature), 1e-6),
                    dim=0,
                ).half()
                cluster_active[image_index, output_cluster] = True
                output_cluster += 1
                if int(ids.numel()) < 2:
                    continue
                left, right = torch.triu_indices(
                    int(ids.numel()),
                    int(ids.numel()),
                    offset=1,
                )
                delta = q[ids[left]] - q[ids[right]]
                informative = delta.abs() >= float(pair_min_quality_gap)
                if not bool(informative.any()):
                    continue
                left = left[informative]
                right = right[informative]
                delta = delta[informative]
                better = torch.where(delta > 0.0, ids[left], ids[right])
                worse = torch.where(delta > 0.0, ids[right], ids[left])
                pairs = torch.stack((better, worse), dim=-1)
                weights = (delta.abs() / 0.10).clamp(0.20, 2.0)
                # Give every GT cluster equal total pair mass regardless of
                # how many duplicate candidates it happens to contain.
                weights = weights / weights.sum().clamp_min(1e-6)
                local_pairs.append(pairs)
                local_weights.append(weights)
        all_pairs.append(
            torch.cat(local_pairs, dim=0)
            if local_pairs
            else torch.empty((0, 2), dtype=torch.long)
        )
        all_pair_weights.append(
            torch.cat(local_weights, dim=0)
            if local_weights
            else torch.empty(0, dtype=torch.float32)
        )
    cache["_full_cluster_mask"] = cluster_mask
    cache["_full_cluster_target"] = cluster_target
    cache["_full_cluster_active"] = cluster_active
    cache["_full_cluster_pairs"] = all_pairs
    cache["_full_cluster_pair_weights"] = all_pair_weights
    cache["_full_cluster_contract"] = {
        "representable_min": float(representable_min),
        "cluster_quality_min": float(cluster_quality_min),
        "target_temperature": float(target_temperature),
        "pair_min_quality_gap": float(pair_min_quality_gap),
    }


def _cluster_losses(
    scores: torch.Tensor,
    batch_indices: list[int],
    cache: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = cache["_full_cluster_mask"][batch_indices].to(scores.device)
    target = cache["_full_cluster_target"][batch_indices].to(
        device=scores.device,
        dtype=scores.dtype,
    )
    active = cache["_full_cluster_active"][batch_indices].to(scores.device)
    cluster_logits = scores.unsqueeze(1).expand_as(target).masked_fill(~mask, -1e4)
    per_cluster = -(target * torch.log_softmax(cluster_logits, dim=-1)).sum(dim=-1)
    listwise = (per_cluster * active).sum() / active.sum().clamp_min(1)

    pair_rows: list[torch.Tensor] = []
    better_rows: list[torch.Tensor] = []
    worse_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    for row, record_index in enumerate(batch_indices):
        pairs = cache["_full_cluster_pairs"][record_index]
        if int(pairs.shape[0]) == 0:
            continue
        pair_rows.append(torch.full((int(pairs.shape[0]),), row, dtype=torch.long))
        better_rows.append(pairs[:, 0])
        worse_rows.append(pairs[:, 1])
        weight_rows.append(cache["_full_cluster_pair_weights"][record_index])
    if pair_rows:
        row_ids = torch.cat(pair_rows).to(scores.device)
        better = torch.cat(better_rows).to(scores.device)
        worse = torch.cat(worse_rows).to(scores.device)
        weights = torch.cat(weight_rows).to(device=scores.device, dtype=scores.dtype)
        margin = scores[row_ids, better] - scores[row_ids, worse]
        pairwise = (F.softplus(-margin) * weights).sum() / weights.sum().clamp_min(
            1e-6
        )
    else:
        pairwise = scores.sum() * 0.0
    return listwise, pairwise


def full_cluster_residual_loss(
    scores: torch.Tensor,
    residual: torch.Tensor,
    batch_indices: list[int],
    cache: dict[str, Any],
    *,
    representable_min: float,
    cluster_quality_min: float,
    target_temperature: float,
    pair_min_quality_gap: float,
    pairwise_weight: float,
    residual_l2_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected_contract = {
        "representable_min": float(representable_min),
        "cluster_quality_min": float(cluster_quality_min),
        "target_temperature": float(target_temperature),
        "pair_min_quality_gap": float(pair_min_quality_gap),
    }
    if cache.get("_full_cluster_contract") != expected_contract:
        raise ValueError("full-cluster targets were prepared with another contract")
    listwise, pairwise = _cluster_losses(
        scores,
        batch_indices,
        cache,
    )
    residual_l2 = residual.float().square().mean()
    total = (
        listwise
        + float(pairwise_weight) * pairwise
        + float(residual_l2_weight) * residual_l2
    )
    return total, {
        "listwise": listwise,
        "pairwise": pairwise,
        "residual_l2": residual_l2,
    }


def _forward_probe(
    probe: nn.Module,
    kind: str,
    source: str | None,
    ids: list[int],
    base_cache: dict[str, Any],
    row_cache: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = base_cache["checkpoint_unary"][ids].to(device).float()
    if kind == "candidate":
        if source is None:
            raise ValueError("candidate residual probe needs a feature source")
        features = base_cache[source][ids].to(device)
        return probe(features, baseline)
    if kind == "row":
        features = row_cache["row_features"][ids].to(device)
        visible = row_cache["row_visible"][ids].to(device)
        return probe(features, visible, baseline)
    raise ValueError(f"unknown residual probe kind: {kind}")


@torch.no_grad()
def _score_probe(
    probe: nn.Module,
    kind: str,
    source: str | None,
    indices: list[int],
    batch_size: int,
    base_cache: dict[str, Any],
    row_cache: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    probe.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(indices), int(batch_size)):
        ids = indices[start : start + int(batch_size)]
        score, _residual = _forward_probe(
            probe,
            kind,
            source,
            ids,
            base_cache,
            row_cache,
            device,
        )
        rows.append(score.detach().float().cpu())
    return torch.cat(rows, dim=0)


def train_residual_probe(
    base_cache: dict[str, Any],
    row_cache: dict[str, Any],
    splits: dict[str, list[int]],
    *,
    arm: str,
    kind: str,
    source: str | None,
    nonlinear: bool,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    candidate_moments: tuple[torch.Tensor, torch.Tensor] | None,
    row_moments: tuple[torch.Tensor, torch.Tensor] | None,
) -> dict[str, Any]:
    seed_everything(int(seed))
    if kind == "candidate":
        if candidate_moments is None:
            raise ValueError("candidate moments are missing")
        probe: nn.Module = CandidateResidualProbe(
            *candidate_moments,
            nonlinear=nonlinear,
            hidden=int(args.candidate_hidden),
            dropout=float(args.dropout),
        )
        learning_rate = float(args.candidate_lr)
        batch_size = int(args.candidate_batch_size)
    else:
        if row_moments is None:
            raise ValueError("row moments are missing")
        probe = RowSequenceResidualProbe(
            *row_moments,
            row_count=int(row_cache["row_features"].shape[2]),
            hidden=int(args.row_hidden),
            layers=int(args.row_layers),
            heads=int(args.row_heads),
            dropout=float(args.dropout),
        )
        learning_rate = float(args.row_lr)
        batch_size = int(args.row_batch_size)
    probe = probe.to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=float(args.weight_decay),
    )
    fit = splits["fit"]
    early = splits["early_stop"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 101)
    best_score = -float("inf")
    best_epoch = 0
    best_state = _state_to_cpu(probe)
    stale = 0
    trajectory: list[dict[str, Any]] = []

    for epoch in range(1, int(args.max_epochs) + 1):
        order = torch.randperm(len(fit), generator=generator).tolist()
        probe.train()
        totals = {"total": 0.0, "listwise": 0.0, "pairwise": 0.0, "l2": 0.0}
        batches = 0
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            ids = [fit[position] for position in local]
            score, residual = _forward_probe(
                probe,
                kind,
                source,
                ids,
                base_cache,
                row_cache,
                device,
            )
            loss, parts = full_cluster_residual_loss(
                score,
                residual,
                ids,
                base_cache,
                representable_min=float(args.representable_min),
                cluster_quality_min=float(args.cluster_quality_min),
                target_temperature=float(args.target_temperature),
                pair_min_quality_gap=float(args.pair_min_quality_gap),
                pairwise_weight=float(args.pairwise_weight),
                residual_l2_weight=float(args.residual_l2_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
            optimizer.step()
            totals["total"] += float(loss.detach().cpu())
            totals["listwise"] += float(parts["listwise"].detach().cpu())
            totals["pairwise"] += float(parts["pairwise"].detach().cpu())
            totals["l2"] += float(parts["residual_l2"].detach().cpu())
            batches += 1
        early_score = _score_probe(
            probe,
            kind,
            source,
            early,
            batch_size,
            base_cache,
            row_cache,
            device,
        )
        early_metrics = evaluate_scores(
            base_cache,
            early,
            early_score,
            float(args.representable_min),
        )
        selection_score = float(
            early_metrics["cluster_ranking"]["selection_score"]
        )
        trajectory.append(
            {
                "epoch": epoch,
                "train_loss": totals["total"] / max(batches, 1),
                "train_listwise_loss": totals["listwise"] / max(batches, 1),
                "train_pairwise_loss": totals["pairwise"] / max(batches, 1),
                "train_residual_l2": totals["l2"] / max(batches, 1),
                "early_stop": early_metrics,
            }
        )
        if selection_score > best_score + 1e-4:
            best_score = selection_score
            best_epoch = epoch
            best_state = _state_to_cpu(probe)
            stale = 0
        else:
            stale += 1
        if stale >= int(args.patience):
            break

    probe.load_state_dict(best_state)
    result: dict[str, Any] = {
        "arm": arm,
        "seed": int(seed),
        "kind": kind,
        "source": source,
        "architecture": (
            "row_transformer"
            if kind == "row"
            else ("nonlinear_residual" if nonlinear else "linear_residual")
        ),
        "best_epoch": int(best_epoch),
        "epochs_run": len(trajectory),
        "early_stop_best_selection_score": float(best_score),
        "trajectory": trajectory,
    }
    for split_name in ("fit", "early_stop", "holdout"):
        ids = splits[split_name]
        score = _score_probe(
            probe,
            kind,
            source,
            ids,
            batch_size,
            base_cache,
            row_cache,
            device,
        )
        result[split_name] = evaluate_scores(
            base_cache,
            ids,
            score,
            float(args.representable_min),
        )
    return result


def _mean(summary: dict[str, Any], split: str, metric: str) -> float:
    return float(summary[split][metric]["mean"])


def make_decision(
    baseline: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    thresholds = {
        "holdout_top1_rate": 0.40,
        "holdout_top2_rate": 0.70,
        "holdout_mean_regret": 0.10,
        "holdout_weighted_pairwise_accuracy": 0.70,
        "max_top1_generalization_gap": 0.15,
        "material_top1_gain": 0.05,
        "material_regret_reduction": 0.03,
    }
    baseline_top1 = float(baseline["holdout"]["cluster_ranking"]["top1_rate"])
    baseline_regret = float(
        baseline["holdout"]["cluster_ranking"]["mean_regret"]
    )

    def passes(summary: dict[str, Any]) -> bool:
        gap = _mean(summary, "fit", "top1_rate") - _mean(
            summary,
            "holdout",
            "top1_rate",
        )
        return (
            _mean(summary, "holdout", "top1_rate")
            >= thresholds["holdout_top1_rate"]
            and _mean(summary, "holdout", "top2_rate")
            >= thresholds["holdout_top2_rate"]
            and _mean(summary, "holdout", "mean_regret")
            <= thresholds["holdout_mean_regret"]
            and _mean(summary, "holdout", "weighted_pairwise_accuracy")
            >= thresholds["holdout_weighted_pairwise_accuracy"]
            and gap <= thresholds["max_top1_generalization_gap"]
        )

    row = summaries["row_sequence_residual_transformer"]
    hidden = summaries["relation_hidden_residual_nonlinear"]
    raw = summaries["raw_descriptor_residual_nonlinear"]
    row_pass = passes(row)
    hidden_pass = passes(hidden)
    raw_pass = passes(raw)
    row_top1 = _mean(row, "holdout", "top1_rate")
    row_regret = _mean(row, "holdout", "mean_regret")
    row_material = (
        row_top1 - baseline_top1 >= thresholds["material_top1_gain"]
        and baseline_regret - row_regret
        >= thresholds["material_regret_reduction"]
    )
    if row_pass:
        root = "pooled_selector_discards_representative_signal"
        action = (
            "Replace pooled candidate summaries with a row-aware ownership "
            "encoder; keep bounded-delta geometry frozen for the first gate."
        )
    elif row_material:
        root = "row_sequence_contains_partial_representative_signal"
        action = (
            "Run one full-validation row-aware selector gate. The row sequence "
            "materially improves observability but has not met the strict probe "
            "contract."
        )
    elif hidden_pass or raw_pass:
        root = "checkpoint_unary_requires_full_cluster_residual_reranking"
        action = (
            "A residual pooled selector is sufficient; do not alter geometry. "
            "Validate the winning residual head on official full validation."
        )
    else:
        root = "joint_query_ownership_representation_required"
        action = (
            "Neither full-cluster residual reranking nor unpooled row evidence "
            "recovers a stable winner. Co-train a dedicated one-to-one query "
            "ownership stream from the start while preserving stop-gradient "
            "into bounded-delta coordinate/reference paths."
        )
    return {
        "root_cause_classification": root,
        "recommended_next_experiment": action,
        "predeclared_pass_thresholds": thresholds,
        "signals": {
            "baseline_holdout_top1": baseline_top1,
            "baseline_holdout_mean_regret": baseline_regret,
            "raw_residual_pass": raw_pass,
            "hidden_residual_pass": hidden_pass,
            "row_sequence_pass": row_pass,
            "row_sequence_material_improvement": row_material,
            "row_sequence_holdout_top1_mean": row_top1,
            "row_sequence_holdout_mean_regret": row_regret,
            "row_sequence_top1_gain": row_top1 - baseline_top1,
            "row_sequence_regret_reduction": baseline_regret - row_regret,
        },
        "scope_warning": (
            "The row cache uses deterministic random channel projections and "
            "sampled rows. Failure localizes the signal within this bounded "
            "probe contract; it is not proof that every possible upstream "
            "feature lacks representative information."
        ),
    }


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("probe seeds must be unique")
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    base_cache_path = Path(args.base_cache).expanduser().resolve()
    for path in (config_path, checkpoint_path, base_cache_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing row observability artefact: {path}")
    base_cache = _load_torch(base_cache_path)
    total = _validate_base_cache(base_cache, args)
    base_signature = base_cache.get("signature", {})
    if base_signature.get("config_sha256") != _sha256(config_path):
        raise ValueError("base cache was produced with another resolved config")
    cached_checkpoint = Path(str(base_signature.get("checkpoint", ""))).expanduser()
    if not cached_checkpoint.exists() or cached_checkpoint.resolve() != checkpoint_path:
        raise ValueError("base cache was produced from another pointer checkpoint")
    if int(base_signature.get("sample_count", -1)) != total:
        raise ValueError("base cache sample-count provenance mismatch")
    splits = make_image_disjoint_splits(
        total,
        int(args.development_images),
        int(args.holdout_images),
        int(args.early_stop_images),
        int(args.split_seed),
    )
    prepare_full_cluster_targets(
        base_cache,
        representable_min=float(args.representable_min),
        cluster_quality_min=float(args.cluster_quality_min),
        target_temperature=float(args.target_temperature),
        pair_min_quality_gap=float(args.pair_min_quality_gap),
    )
    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = str(args.dataset_root)
    device = torch.device(args.device)
    seed_everything(int(args.split_seed))
    model = build_model(cfg).to(device)
    checkpoint_iteration = load_checkpoint(checkpoint_path, model, strict=False)
    selector = _root_selector(model)
    row_cache = collect_row_cache(
        args,
        cfg,
        model,
        selector,
        base_cache,
        config_path,
        checkpoint_path,
        device,
    )
    del model, selector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if row_cache["metadata"]["image_ids"] != base_cache["metadata"]["image_ids"]:
        raise ValueError("row and pooled caches are not image-aligned")
    baseline: dict[str, Any] = {}
    for split_name in ("fit", "early_stop", "holdout"):
        ids = splits[split_name]
        baseline[split_name] = evaluate_scores(
            base_cache,
            ids,
            base_cache["checkpoint_unary"][ids].float(),
            float(args.representable_min),
        )

    raw_moments = _feature_moments(
        base_cache["raw_descriptor"],
        base_cache["candidate_valid"],
        splits["fit"],
    )
    hidden_moments = _feature_moments(
        base_cache["relation_hidden"],
        base_cache["candidate_valid"],
        splits["fit"],
    )
    row_moments = row_feature_moments(
        row_cache["row_features"],
        row_cache["row_visible"],
        base_cache["candidate_valid"],
        splits["fit"],
    )
    arm_contracts = (
        (
            "raw_descriptor_residual_nonlinear",
            "candidate",
            "raw_descriptor",
            True,
            raw_moments,
            None,
        ),
        (
            "relation_hidden_residual_linear",
            "candidate",
            "relation_hidden",
            False,
            hidden_moments,
            None,
        ),
        (
            "relation_hidden_residual_nonlinear",
            "candidate",
            "relation_hidden",
            True,
            hidden_moments,
            None,
        ),
        (
            "row_sequence_residual_transformer",
            "row",
            None,
            True,
            None,
            row_moments,
        ),
    )
    runs: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm, kind, source, nonlinear, candidate_moments, local_row_moments in arm_contracts:
        print(f"observability arm: {arm}")
        arm_runs = [
            train_residual_probe(
                base_cache,
                row_cache,
                splits,
                arm=arm,
                kind=kind,
                source=source,
                nonlinear=nonlinear,
                seed=int(seed),
                args=args,
                device=device,
                candidate_moments=candidate_moments,
                row_moments=local_row_moments,
            )
            for seed in args.seeds
        ]
        runs[arm] = arm_runs
        summaries[arm] = summarize_runs(arm_runs)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "experiment": "V4 pooled-vs-row representative observability audit",
        "provenance": {
            "git_commit": _git_commit(),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_iteration": int(checkpoint_iteration),
            "base_cache": str(base_cache_path),
            "row_cache": str(Path(args.row_cache).expanduser().resolve()),
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "split": str(args.split),
            "augmentation_enabled": False,
            "geometry_gradient_enabled": False,
        },
        "contract": {
            "development_images": int(args.development_images),
            "fit_images": len(splits["fit"]),
            "early_stop_images": len(splits["early_stop"]),
            "holdout_images": len(splits["holdout"]),
            "image_disjoint": True,
            "probe_seeds": [int(value) for value in args.seeds],
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "row_sequence": {
                "sampled_rows": int(row_cache["row_features"].shape[2]),
                "channel_projection_per_stream": int(args.channel_projection),
                "projection_type": "fixed_rademacher",
                "projection_seed": int(args.projection_seed),
                "row_feature_dimension": int(row_cache["row_features"].shape[-1]),
                "hidden_dimension": int(args.row_hidden),
                "transformer_layers": int(args.row_layers),
                "attention_heads": int(args.row_heads),
            },
            "ranking_loss": {
                "representable_min": float(args.representable_min),
                "cluster_quality_min": float(args.cluster_quality_min),
                "target_temperature": float(args.target_temperature),
                "pair_min_quality_gap": float(args.pair_min_quality_gap),
                "pairwise_weight": float(args.pairwise_weight),
                "residual_l2_weight": float(args.residual_l2_weight),
                "score_contract": "checkpoint_unary + learned_residual",
            },
        },
        "checkpoint_unary_baseline": baseline,
        "runs": runs,
        "summaries": summaries,
    }
    result["decision"] = make_decision(baseline, summaries)
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
