from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    trace_postprocess,
)
from dynlaneseq_eg.tools.decompose_cluster_representative_selection import (
    nms_clusters_from_trace,
)
from dynlaneseq_eg.tools.probe_four_slot_coverage_selector import (
    _cpu_state,
    _load_cache,
    _load_source_selector,
    _set_seed,
    _validate_cache,
    source_scores,
)
from dynlaneseq_eg.tools.probe_official_set_selection import (
    official_unique_quality_targets,
    training_index_schedule,
)


def _safe_padding_mask(valid: torch.Tensor) -> torch.Tensor:
    """Keep one inert token visible when an entire sequence is padding."""

    safe_valid = valid.bool().clone()
    all_invalid = ~safe_valid.any(dim=-1)
    if bool(all_invalid.any()):
        safe_valid[all_invalid, 0] = True
    return ~safe_valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train frozen-feature cluster-existence and within-cluster "
            "representative-quality probes independently."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--reference-report", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--focal-beta", type=float, default=2.0)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--positive-iou", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--input-h", type=int, default=640)
    parser.add_argument("--input-w", type=int, default=1600)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--cluster-min-gain-050", type=float, default=5.0)
    parser.add_argument("--representative-min-gain-070", type=float, default=3.0)
    parser.add_argument("--hierarchical-min-gain-050", type=float, default=5.0)
    parser.add_argument("--hierarchical-min-gain-070", type=float, default=3.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probe", required=True)
    return parser.parse_args()


class ClusterExistenceProbe(nn.Module):
    """Score frozen NMS clusters after pooling their candidate descriptors."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.candidate_projection = nn.Linear(int(input_dim), int(hidden_dim))
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(
            candidate_layer,
            num_layers=max(1, int(num_layers)),
            enable_nested_tensor=False,
        )
        self.cluster_projection = nn.Linear(
            2 * int(hidden_dim) + 3,
            int(hidden_dim),
        )
        cluster_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cluster_encoder = nn.TransformerEncoder(
            cluster_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)

    def forward(
        self,
        features: torch.Tensor,
        candidate_valid: torch.Tensor,
        membership: torch.Tensor,
        cluster_valid: torch.Tensor,
        source_scores: torch.Tensor,
    ) -> torch.Tensor:
        candidate_padding_mask = _safe_padding_mask(candidate_valid)
        hidden = self.candidate_projection(self.input_norm(features))
        hidden = self.candidate_encoder(
            hidden,
            src_key_padding_mask=candidate_padding_mask,
        )
        member = membership.bool()
        member_float = member.to(hidden.dtype)
        count = member_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_state = torch.einsum("bcn,bnh->bch", member_float, hidden) / count
        expanded = hidden.unsqueeze(1).expand(-1, int(member.shape[1]), -1, -1)
        max_state = expanded.masked_fill(~member.unsqueeze(-1), -1e4).amax(dim=2)
        max_state = torch.where(
            cluster_valid.unsqueeze(-1),
            max_state,
            torch.zeros_like(max_state),
        )
        score_sum = torch.einsum(
            "bcn,bn->bc",
            member_float,
            source_scores.to(hidden.dtype),
        )
        score_mean = score_sum / count.squeeze(-1)
        score_max = source_scores.unsqueeze(1).expand(-1, int(member.shape[1]), -1)
        score_max = score_max.masked_fill(~member, -1e4).amax(dim=-1)
        score_max = torch.where(cluster_valid, score_max, torch.zeros_like(score_max))
        size_norm = count.squeeze(-1) / float(max(int(features.shape[1]), 1))
        scalars = torch.stack((size_norm, score_mean, score_max), dim=-1)
        cluster = self.cluster_projection(
            torch.cat((mean_state, max_state, scalars), dim=-1)
        )
        cluster_padding_mask = _safe_padding_mask(cluster_valid)
        cluster = self.cluster_encoder(
            cluster,
            src_key_padding_mask=cluster_padding_mask,
        )
        logits = self.output(self.output_norm(cluster)).squeeze(-1)
        return logits.float().masked_fill(~cluster_valid.bool(), -1e4)


class RepresentativeQualityProbe(nn.Module):
    """Predict continuous official IoU for every candidate, not ownership."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.input_projection = nn.Linear(int(input_dim), int(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=max(1, int(num_layers)),
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) + 1),
            nn.Linear(int(hidden_dim) + 1, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        candidate_valid: torch.Tensor,
        source_scores: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(self.input_norm(features))
        hidden = self.encoder(
            hidden,
            src_key_padding_mask=_safe_padding_mask(candidate_valid),
        )
        logits = self.output(
            torch.cat((hidden, source_scores.to(hidden.dtype).unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        return logits.float().masked_fill(~candidate_valid.bool(), -1e4)


def cluster_and_representative_targets(
    official_iou: torch.Tensor,
    clusters: dict[int, list[int]],
    *,
    positive_iou: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_count = int(official_iou.shape[1])
    representative = (
        official_iou.max(dim=0).values.float()
        if int(official_iou.shape[0]) > 0
        else official_iou.new_zeros(candidate_count, dtype=torch.float32)
    )
    if not clusters:
        return official_iou.new_zeros((0,), dtype=torch.float32), representative
    cluster_best = torch.stack(
        [
            official_iou[:, members].max(dim=1).values
            for members in clusters.values()
        ],
        dim=1,
    ) if int(official_iou.shape[0]) > 0 else official_iou.new_zeros(
        (0, len(clusters))
    )
    cluster_target = official_unique_quality_targets(cluster_best)
    cluster_target = torch.where(
        cluster_target >= float(positive_iou),
        cluster_target,
        torch.zeros_like(cluster_target),
    )
    return cluster_target.float(), representative


def build_hierarchical_supervision(
    cache: dict[str, Any],
    scores: torch.Tensor,
    *,
    input_h: int,
    input_w: int,
    min_valid_rows: int,
    row_visibility_thresh: float,
    nms_distance: float,
    nms_min_overlap_points: int,
    top_k: int,
    positive_iou: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    max_clusters = 0
    candidate_count = int(cache["features"].shape[1])
    for image_index, official_iou in enumerate(cache["official_iou"]):
        stage = {
            name: values[image_index]
            for name, values in cache["stage"].items()
        }
        score = scores[image_index]
        trace = trace_postprocess(
            stage,
            input_h=input_h,
            input_w=input_w,
            score_thresh=-1.0,
            quality_power=0.0,
            min_valid_rows=min_valid_rows,
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_min_overlap_points,
            top_k=top_k,
            row_visibility_thresh=row_visibility_thresh,
            score_override={
                candidate: float(score[candidate])
                for candidate in range(candidate_count)
            },
        )
        clusters = nms_clusters_from_trace(trace)
        keepers = list(clusters)
        cluster_target, representative_target = cluster_and_representative_targets(
            official_iou,
            clusters,
            positive_iou=positive_iou,
        )
        record = {
            "keepers": keepers,
            "members": [clusters[keeper] for keeper in keepers],
            "source_selected_cluster_ids": list(range(min(top_k, len(keepers)))),
            "cluster_targets": cluster_target,
            "representative_targets": representative_target,
        }
        records.append(record)
        max_clusters = max(max_clusters, len(keepers))
    if max_clusters < 1:
        raise ValueError("no valid NMS clusters were found")

    images = len(records)
    membership = torch.zeros(
        (images, max_clusters, candidate_count),
        dtype=torch.bool,
    )
    cluster_valid = torch.zeros((images, max_clusters), dtype=torch.bool)
    cluster_targets = torch.zeros((images, max_clusters), dtype=torch.float32)
    candidate_cluster = torch.full(
        (images, candidate_count),
        -1,
        dtype=torch.long,
    )
    representative_targets = torch.zeros(
        (images, candidate_count),
        dtype=torch.float32,
    )
    keeper_ids = torch.full((images, max_clusters), -1, dtype=torch.long)
    for image_index, record in enumerate(records):
        cluster_count = len(record["keepers"])
        cluster_valid[image_index, :cluster_count] = True
        cluster_targets[image_index, :cluster_count] = record["cluster_targets"]
        representative_targets[image_index] = record["representative_targets"]
        keeper_ids[image_index, :cluster_count] = torch.tensor(record["keepers"])
        for cluster_index, members in enumerate(record["members"]):
            membership[image_index, cluster_index, members] = True
            candidate_cluster[image_index, members] = int(cluster_index)
    return {
        "records": records,
        "membership": membership,
        "cluster_valid": cluster_valid,
        "cluster_targets": cluster_targets,
        "candidate_cluster": candidate_cluster,
        "representative_targets": representative_targets,
        "keeper_ids": keeper_ids,
    }


def masked_quality_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    group_ids: torch.Tensor | None,
    focal_beta: float,
    negative_weight: float,
    rank_weight: float,
    rank_margin: float,
    positive_iou: float,
) -> dict[str, torch.Tensor]:
    target = targets.to(logits.dtype)
    valid_float = valid.to(logits.dtype)
    probability = torch.sigmoid(logits)
    modulation = (target - probability).abs().pow(float(focal_beta))
    per_item = modulation * F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    item_weight = torch.where(
        target >= float(positive_iou),
        torch.ones_like(target),
        torch.full_like(target, float(negative_weight)),
    ) * valid_float
    quality = (per_item * item_weight).sum() / item_weight.sum().clamp_min(1.0)

    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    if group_ids is not None:
        group_valid = group_ids >= 0
        pair_valid = pair_valid & group_valid.unsqueeze(-1) & group_valid.unsqueeze(-2)
        pair_valid = pair_valid & (group_ids.unsqueeze(-1) == group_ids.unsqueeze(-2))
    target_delta = target.unsqueeze(-1) - target.unsqueeze(-2)
    pair_weight = (target_delta - float(rank_margin)).clamp_min(0.0)
    pair_weight = pair_weight * pair_valid.to(pair_weight.dtype)
    logit_delta = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    ranking = (
        F.softplus(-logit_delta) * pair_weight.detach()
    ).sum() / pair_weight.sum().clamp_min(1e-6)
    total = quality + float(rank_weight) * ranking
    return {"total": total, "quality": quality, "ranking": ranking}


def _selected_cluster_ids(
    logits: torch.Tensor,
    cluster_valid: torch.Tensor,
    *,
    top_k: int,
) -> list[int]:
    ids = torch.nonzero(cluster_valid.bool(), as_tuple=False).flatten().tolist()
    ids.sort(key=lambda index: float(logits[index]), reverse=True)
    return [int(value) for value in ids[: int(top_k)]]


def _representative_ids(
    cluster_ids: list[int],
    record: dict[str, Any],
    representative_logits: torch.Tensor,
    *,
    learned: bool,
) -> list[int]:
    selected: list[int] = []
    for cluster_index in cluster_ids:
        if learned:
            members = record["members"][int(cluster_index)]
            candidate = max(
                members,
                key=lambda index: float(representative_logits[int(index)]),
            )
        else:
            candidate = int(record["keepers"][int(cluster_index)])
        selected.append(int(candidate))
    return selected


def _new_counts() -> dict[str, int]:
    return {"gt": 0, "selected": 0, "hits_050": 0, "hits_070": 0}


def _update_counts(
    counts: dict[str, int],
    official_iou: torch.Tensor,
    selected_ids: list[int],
) -> None:
    counts["gt"] += int(official_iou.shape[0])
    counts["selected"] += len(selected_ids)
    counts["hits_050"] += evaluator_hungarian_assignment(
        official_iou,
        selected_ids,
        threshold=0.5,
    ).hit_count
    counts["hits_070"] += evaluator_hungarian_assignment(
        official_iou,
        selected_ids,
        threshold=0.7,
    ).hit_count


def _finish_counts(counts: dict[str, int]) -> dict[str, Any]:
    gt = int(counts["gt"])
    selected = int(counts["selected"])
    result: dict[str, Any] = {
        "gt_lanes": gt,
        "selected_predictions": selected,
    }
    for suffix in ("050", "070"):
        tp = int(counts[f"hits_{suffix}"])
        precision = tp / float(max(selected, 1))
        recall = tp / float(max(gt, 1))
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        result[f"tp_{suffix}"] = tp
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"f1_{suffix}"] = f1
    return result


@torch.no_grad()
def evaluate_hierarchy(
    cluster_probe: ClusterExistenceProbe,
    representative_probe: RepresentativeQualityProbe,
    cache: dict[str, Any],
    hierarchy: dict[str, Any],
    scores: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    cluster_probe.eval()
    representative_probe.eval()
    cluster_rows: list[torch.Tensor] = []
    representative_rows: list[torch.Tensor] = []
    for start in range(0, int(cache["features"].shape[0]), int(batch_size)):
        stop = start + int(batch_size)
        features = cache["features"][start:stop].to(device, dtype=torch.float32)
        candidate_valid = cache["candidate_valid"][start:stop].to(device)
        membership = hierarchy["membership"][start:stop].to(device)
        cluster_valid = hierarchy["cluster_valid"][start:stop].to(device)
        source = scores[start:stop].to(device)
        cluster_rows.append(
            cluster_probe(
                features,
                candidate_valid,
                membership,
                cluster_valid,
                source,
            ).cpu()
        )
        representative_rows.append(
            representative_probe(features, candidate_valid, source).cpu()
        )
    cluster_logits = torch.cat(cluster_rows)
    representative_logits = torch.cat(representative_rows)
    modes = {
        "source_nms": _new_counts(),
        "learned_cluster_source_representative": _new_counts(),
        "source_cluster_learned_representative": _new_counts(),
        "learned_hierarchical": _new_counts(),
    }
    oracle = {0.5: 0, 0.7: 0, "gt": 0}
    for image_index, official_iou in enumerate(cache["official_iou"]):
        record = hierarchy["records"][image_index]
        source_clusters = record["source_selected_cluster_ids"]
        learned_clusters = _selected_cluster_ids(
            cluster_logits[image_index],
            hierarchy["cluster_valid"][image_index],
            top_k=top_k,
        )
        selections = {
            "source_nms": _representative_ids(
                source_clusters,
                record,
                representative_logits[image_index],
                learned=False,
            ),
            "learned_cluster_source_representative": _representative_ids(
                learned_clusters,
                record,
                representative_logits[image_index],
                learned=False,
            ),
            "source_cluster_learned_representative": _representative_ids(
                source_clusters,
                record,
                representative_logits[image_index],
                learned=True,
            ),
            "learned_hierarchical": _representative_ids(
                learned_clusters,
                record,
                representative_logits[image_index],
                learned=True,
            ),
        }
        for name, selected in selections.items():
            _update_counts(modes[name], official_iou, selected)
        oracle["gt"] += int(official_iou.shape[0])
        for threshold in (0.5, 0.7):
            oracle[threshold] += cardinality_oracle_assignment(
                official_iou,
                threshold=threshold,
                top_k=top_k,
                candidate_valid=cache["candidate_valid"][image_index],
            ).hit_count
    return {
        "modes": {
            name: _finish_counts(counts)
            for name, counts in modes.items()
        },
        "oracle_top4": {
            f"{threshold:.2f}": {
                "tp": int(oracle[threshold]),
                "gt_lanes": int(oracle["gt"]),
                "recall": float(oracle[threshold]) / float(max(int(oracle["gt"]), 1)),
            }
            for threshold in (0.5, 0.7)
        },
    }


def verify_reference_metrics(
    evaluation: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    source = evaluation["modes"]["source_nms"]
    comparisons: dict[str, float] = {}
    for suffix in ("050", "070"):
        delta = float(source[f"recall_{suffix}"]) - float(
            reference["source"]["nms_top4"][f"recall_{suffix}"]
        )
        comparisons[f"recall_{suffix}_delta"] = delta
        if abs(delta) > 1e-12:
            raise ValueError(
                f"source NMS mismatch at {suffix}: delta={delta}"
            )
    return {"matched": True, "comparisons": comparisons}


def hierarchy_gate(
    evaluation: dict[str, Any],
    *,
    cluster_min_gain_050: float,
    representative_min_gain_070: float,
    hierarchical_min_gain_050: float,
    hierarchical_min_gain_070: float,
) -> dict[str, Any]:
    modes = evaluation["modes"]
    source = modes["source_nms"]

    def gains(name: str) -> dict[str, float]:
        row = modes[name]
        return {
            "gain_recall_050_points": 100.0
            * (float(row["recall_050"]) - float(source["recall_050"])),
            "gain_recall_070_points": 100.0
            * (float(row["recall_070"]) - float(source["recall_070"])),
        }

    cluster = gains("learned_cluster_source_representative")
    representative = gains("source_cluster_learned_representative")
    hierarchical = gains("learned_hierarchical")
    cluster_positive = bool(
        cluster["gain_recall_050_points"] >= float(cluster_min_gain_050)
    )
    representative_positive = bool(
        representative["gain_recall_070_points"]
        >= float(representative_min_gain_070)
    )
    hierarchical_positive = bool(
        hierarchical["gain_recall_050_points"]
        >= float(hierarchical_min_gain_050)
        and hierarchical["gain_recall_070_points"]
        >= float(hierarchical_min_gain_070)
    )
    if cluster_positive and representative_positive and hierarchical_positive:
        interpretation = "decoupled_hierarchical_supervision_is_supported"
    elif cluster_positive and not representative_positive:
        interpretation = "cluster_existence_is_learnable_but_quality_is_not"
    elif representative_positive and not cluster_positive:
        interpretation = "representative_quality_is_learnable_but_cluster_ranking_is_not"
    else:
        interpretation = "frozen_10k_descriptors_do_not_support_the_full_hierarchy"
    return {
        "cluster": {**cluster, "positive": cluster_positive},
        "representative": {**representative, "positive": representative_positive},
        "hierarchical": {**hierarchical, "positive": hierarchical_positive},
        "dual_head_positive": bool(
            cluster_positive and representative_positive and hierarchical_positive
        ),
        "interpretation": interpretation,
    }


def main() -> None:
    args = parse_args()
    for name in (
        "train_steps",
        "batch_size",
        "eval_interval",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "ff_dim",
        "top_k",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if int(args.hidden_dim) % int(args.num_heads) != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    _set_seed(args.seed)
    device = torch.device(args.device)
    train_cache = _load_cache(args.train_cache)
    val_cache = _load_cache(args.val_cache)
    _validate_cache(train_cache, "train")
    _validate_cache(val_cache, "validation")
    overlap = set(train_cache["metadata"]["image_paths"]) & set(
        val_cache["metadata"]["image_paths"]
    )
    if overlap:
        raise ValueError(f"train/validation cache overlap: {sorted(overlap)[0]}")
    selector, source_iteration = _load_source_selector(
        args.config,
        args.source_checkpoint,
    )
    feature_dim = int(train_cache["features"].shape[-1])
    if int(selector.input_norm.normalized_shape[0]) != feature_dim:
        raise ValueError("selector/cache feature dimension mismatch")
    train_scores = source_scores(
        selector,
        train_cache,
        device=device,
        batch_size=args.batch_size,
    )
    val_scores = source_scores(
        selector,
        val_cache,
        device=device,
        batch_size=args.batch_size,
    )
    build_kwargs = {
        "input_h": args.input_h,
        "input_w": args.input_w,
        "min_valid_rows": args.min_valid_rows,
        "row_visibility_thresh": args.row_visibility_thresh,
        "nms_distance": args.nms_distance,
        "nms_min_overlap_points": args.nms_min_overlap_points,
        "top_k": args.top_k,
        "positive_iou": args.positive_iou,
    }
    train_hierarchy = build_hierarchical_supervision(
        train_cache,
        train_scores,
        **build_kwargs,
    )
    val_hierarchy = build_hierarchical_supervision(
        val_cache,
        val_scores,
        **build_kwargs,
    )
    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
    }
    cluster_probe = ClusterExistenceProbe(feature_dim, **model_kwargs).to(device)
    representative_probe = RepresentativeQualityProbe(feature_dim, **model_kwargs).to(device)
    cluster_optimizer = torch.optim.AdamW(
        cluster_probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    representative_optimizer = torch.optim.AdamW(
        representative_probe.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    schedule = training_index_schedule(
        num_examples=int(train_cache["features"].shape[0]),
        batch_size=args.batch_size,
        steps=args.train_steps,
        seed=args.seed,
    )
    best_cluster = {
        "step": 0,
        "objective": float("-inf"),
        "state": _cpu_state(cluster_probe),
    }
    best_representative = {
        "step": 0,
        "objective": float("-inf"),
        "state": _cpu_state(representative_probe),
    }
    trajectory: list[dict[str, Any]] = []
    running = {
        "cluster": {"total": 0.0, "quality": 0.0, "ranking": 0.0},
        "representative": {"total": 0.0, "quality": 0.0, "ranking": 0.0},
    }
    for step in range(1, int(args.train_steps) + 1):
        indices = schedule[step - 1]
        features = train_cache["features"][indices].to(device, dtype=torch.float32)
        candidate_valid = train_cache["candidate_valid"][indices].to(device)
        source = train_scores[indices].to(device)

        cluster_probe.train()
        cluster_optimizer.zero_grad(set_to_none=True)
        cluster_logits = cluster_probe(
            features,
            candidate_valid,
            train_hierarchy["membership"][indices].to(device),
            train_hierarchy["cluster_valid"][indices].to(device),
            source,
        )
        cluster_losses = masked_quality_ranking_loss(
            cluster_logits,
            train_hierarchy["cluster_targets"][indices].to(device),
            train_hierarchy["cluster_valid"][indices].to(device),
            group_ids=None,
            focal_beta=args.focal_beta,
            negative_weight=args.negative_weight,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            positive_iou=args.positive_iou,
        )
        cluster_losses["total"].backward()
        cluster_optimizer.step()

        representative_probe.train()
        representative_optimizer.zero_grad(set_to_none=True)
        representative_logits = representative_probe(
            features,
            candidate_valid,
            source,
        )
        representative_losses = masked_quality_ranking_loss(
            representative_logits,
            train_hierarchy["representative_targets"][indices].to(device),
            candidate_valid,
            group_ids=train_hierarchy["candidate_cluster"][indices].to(device),
            focal_beta=args.focal_beta,
            negative_weight=args.negative_weight,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            positive_iou=args.positive_iou,
        )
        representative_losses["total"].backward()
        representative_optimizer.step()
        for loss_name in running["cluster"]:
            running["cluster"][loss_name] += float(cluster_losses[loss_name].detach())
            running["representative"][loss_name] += float(
                representative_losses[loss_name].detach()
            )

        should_evaluate = (
            step == 1
            or step == int(args.train_steps)
            or step % int(args.eval_interval) == 0
        )
        if should_evaluate:
            evaluation = evaluate_hierarchy(
                cluster_probe,
                representative_probe,
                val_cache,
                val_hierarchy,
                val_scores,
                device=device,
                batch_size=args.batch_size,
                top_k=args.top_k,
            )
            modes = evaluation["modes"]
            cluster_objective = float(
                modes["learned_cluster_source_representative"]["recall_050"]
            ) + float(
                modes["learned_cluster_source_representative"]["recall_070"]
            )
            representative_objective = float(
                modes["source_cluster_learned_representative"]["recall_050"]
            ) + float(
                modes["source_cluster_learned_representative"]["recall_070"]
            )
            if cluster_objective > float(best_cluster["objective"]):
                best_cluster = {
                    "step": int(step),
                    "objective": cluster_objective,
                    "state": _cpu_state(cluster_probe),
                }
            if representative_objective > float(best_representative["objective"]):
                best_representative = {
                    "step": int(step),
                    "objective": representative_objective,
                    "state": _cpu_state(representative_probe),
                }
            trajectory.append({"step": int(step), "evaluation": evaluation})
            cluster_probe.train()
            representative_probe.train()
        if step % int(args.log_interval) == 0 or step == int(args.train_steps):
            denominator = float(args.log_interval if step >= args.log_interval else step)
            print(
                f"hierarchical probe {step:05d}/{args.train_steps:05d} | "
                f"cluster={running['cluster']['total'] / denominator:.4f} | "
                f"representative={running['representative']['total'] / denominator:.4f}"
            )
            running = {
                "cluster": {"total": 0.0, "quality": 0.0, "ranking": 0.0},
                "representative": {"total": 0.0, "quality": 0.0, "ranking": 0.0},
            }

    cluster_probe.load_state_dict(best_cluster["state"])
    representative_probe.load_state_dict(best_representative["state"])
    final_evaluation = evaluate_hierarchy(
        cluster_probe,
        representative_probe,
        val_cache,
        val_hierarchy,
        val_scores,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    consistency = None
    if args.reference_report:
        reference_path = Path(args.reference_report)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        consistency = {
            "reference_report": str(reference_path),
            **verify_reference_metrics(final_evaluation, reference),
        }
    gate = hierarchy_gate(
        final_evaluation,
        cluster_min_gain_050=args.cluster_min_gain_050,
        representative_min_gain_070=args.representative_min_gain_070,
        hierarchical_min_gain_050=args.hierarchical_min_gain_050,
        hierarchical_min_gain_070=args.hierarchical_min_gain_070,
    )
    result = {
        "diagnostic_only": True,
        "warning": (
            "The detector, candidate curves, descriptors, and NMS partition "
            "are frozen at 10k. Positive results justify hierarchical "
            "supervision but are not official benchmark F1."
        ),
        "config": args.config,
        "source_checkpoint": args.source_checkpoint,
        "source_iteration": int(source_iteration),
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "train_images": int(train_cache["features"].shape[0]),
        "val_images": int(val_cache["features"].shape[0]),
        "best_steps": {
            "cluster": int(best_cluster["step"]),
            "representative": int(best_representative["step"]),
        },
        "consistency": consistency,
        "evaluation": final_evaluation,
        "gate": gate,
        "trajectory": trajectory,
        "probe_config": vars(args),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    probe_path = Path(args.save_probe)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cluster_model": best_cluster["state"],
            "representative_model": best_representative["state"],
            "best_steps": result["best_steps"],
            "config": vars(args),
        },
        probe_path,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "trajectory"}, indent=2))
    print(f"output_json: {output_path}")
    print(f"probe_checkpoint: {probe_path}")


if __name__ == "__main__":
    main()
