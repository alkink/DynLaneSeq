from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
)
from dynlaneseq_eg.tools.probe_four_slot_coverage_selector import (
    _cpu_state,
    _load_cache,
    _load_source_selector,
    _set_seed,
    _validate_cache,
    source_scores,
)
from dynlaneseq_eg.tools.probe_hierarchical_cluster_selector import (
    RepresentativeQualityProbe,
    _finish_counts,
    _new_counts,
    _representative_ids,
    _safe_padding_mask,
    _update_counts,
    build_hierarchical_supervision,
    hierarchy_gate,
)
from dynlaneseq_eg.tools.probe_official_set_selection import (
    training_index_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen hard-decision probe: autoregressive marginal-coverage "
            "cluster selection and within-cluster listwise representative ranking."
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
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--coverage-temperature", type=float, default=0.10)
    parser.add_argument("--coverage-weight-050", type=float, default=1.0)
    parser.add_argument("--coverage-weight-070", type=float, default=0.5)
    parser.add_argument("--coverage-weight-iou", type=float, default=0.1)
    parser.add_argument("--coverage-min-gain", type=float, default=1e-5)
    parser.add_argument("--representative-temperature", type=float, default=0.05)
    parser.add_argument("--representative-positive-iou", type=float, default=0.30)
    parser.add_argument("--representative-min-spread", type=float, default=0.01)
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


def cluster_iou_matrix(
    official_iou: torch.Tensor,
    record: dict[str, Any],
) -> torch.Tensor:
    if not record["members"]:
        return official_iou.new_zeros((int(official_iou.shape[0]), 0))
    if int(official_iou.shape[0]) == 0:
        return official_iou.new_zeros((0, len(record["members"])))
    return torch.stack(
        [
            official_iou[:, members].max(dim=1).values
            for members in record["members"]
        ],
        dim=1,
    ).float()


def coverage_set_utility(
    cluster_iou: torch.Tensor,
    selected: list[int],
    *,
    weight_050: float,
    weight_070: float,
    weight_iou: float,
) -> float:
    if int(cluster_iou.shape[0]) == 0 or not selected:
        return 0.0
    matrix = cluster_iou[:, selected].detach().cpu().numpy().astype(np.float64)
    reward = (
        float(weight_050) * (matrix >= 0.50)
        + float(weight_070) * (matrix >= 0.70)
        + float(weight_iou) * matrix
    )
    gt_ids, selected_ids = linear_sum_assignment(-reward)
    return float(reward[gt_ids, selected_ids].sum())


def marginal_coverage_teacher(
    cluster_iou: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    min_gain: float,
    weight_050: float,
    weight_070: float,
    weight_iou: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clusters = int(cluster_iou.shape[1])
    distribution = cluster_iou.new_zeros((int(top_k), clusters))
    sequence = torch.full((int(top_k),), -1, dtype=torch.long)
    active = torch.zeros((int(top_k),), dtype=torch.bool)
    selected: list[int] = []
    for step in range(min(int(top_k), clusters)):
        remaining = [index for index in range(clusters) if index not in selected]
        base = coverage_set_utility(
            cluster_iou,
            selected,
            weight_050=weight_050,
            weight_070=weight_070,
            weight_iou=weight_iou,
        )
        gains = torch.tensor(
            [
                coverage_set_utility(
                    cluster_iou,
                    [*selected, index],
                    weight_050=weight_050,
                    weight_070=weight_070,
                    weight_iou=weight_iou,
                )
                - base
                for index in remaining
            ],
            dtype=torch.float32,
        )
        useful = gains > float(min_gain)
        if not bool(useful.any()):
            break
        useful_ids = torch.nonzero(useful, as_tuple=False).flatten()
        useful_gains = gains[useful_ids]
        probabilities = torch.softmax(
            useful_gains / max(float(temperature), 1e-4),
            dim=0,
        )
        remaining_tensor = torch.tensor(remaining, dtype=torch.long)
        target_ids = remaining_tensor[useful_ids]
        distribution[step, target_ids] = probabilities.to(distribution.dtype)
        chosen = int(target_ids[int(useful_gains.argmax())])
        sequence[step] = chosen
        active[step] = True
        selected.append(chosen)
    return distribution, sequence, active


def build_hard_decision_targets(
    cache: dict[str, Any],
    hierarchy: dict[str, Any],
    *,
    top_k: int,
    temperature: float,
    min_gain: float,
    weight_050: float,
    weight_070: float,
    weight_iou: float,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    images, max_clusters = hierarchy["cluster_valid"].shape
    distributions = torch.zeros(
        (images, int(top_k), max_clusters),
        dtype=torch.float32,
    )
    sequences = torch.full((images, int(top_k)), -1, dtype=torch.long)
    active = torch.zeros((images, int(top_k)), dtype=torch.bool)
    for image_index, official_iou in enumerate(cache["official_iou"]):
        matrix = cluster_iou_matrix(
            official_iou,
            hierarchy["records"][image_index],
        )
        distribution, sequence, image_active = marginal_coverage_teacher(
            matrix,
            top_k=top_k,
            temperature=temperature,
            min_gain=min_gain,
            weight_050=weight_050,
            weight_070=weight_070,
            weight_iou=weight_iou,
        )
        clusters = int(matrix.shape[1])
        distributions[image_index, :, :clusters] = distribution
        sequences[image_index] = sequence
        active[image_index] = image_active
    return {
        "distribution": distributions,
        "sequence": sequences,
        "active": active,
        "statistics": {
            "images": int(images),
            "active_decisions": int(active.sum()),
            "mean_active_decisions_per_image": float(active.sum())
            / float(max(int(images), 1)),
            "active_fraction_by_step": active.float().mean(dim=0).tolist(),
        },
    }


class ClusterCoveragePointerProbe(nn.Module):
    """Select NMS clusters autoregressively without replacement."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        top_k: int,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.top_k = int(top_k)
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
        self.cluster_norm = nn.LayerNorm(int(hidden_dim))
        self.key = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.base_score = nn.Linear(int(hidden_dim), 1)
        self.context = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.step_embedding = nn.Embedding(int(top_k), int(hidden_dim))
        self.query = nn.Linear(2 * int(hidden_dim), int(hidden_dim))
        self.state_update = nn.GRUCell(int(hidden_dim), int(hidden_dim))
        self.scale = float(hidden_dim) ** -0.5

    def encode(
        self,
        features: torch.Tensor,
        candidate_valid: torch.Tensor,
        membership: torch.Tensor,
        cluster_valid: torch.Tensor,
        source_scores: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.candidate_projection(self.input_norm(features))
        hidden = self.candidate_encoder(
            hidden,
            src_key_padding_mask=_safe_padding_mask(candidate_valid),
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
        source = source_scores.to(hidden.dtype)
        score_sum = torch.einsum("bcn,bn->bc", member_float, source)
        score_mean = score_sum / count.squeeze(-1)
        score_max = source.unsqueeze(1).expand(-1, int(member.shape[1]), -1)
        score_max = score_max.masked_fill(~member, -1e4).amax(dim=-1)
        score_max = torch.where(cluster_valid, score_max, torch.zeros_like(score_max))
        size_norm = count.squeeze(-1) / float(max(int(features.shape[1]), 1))
        scalars = torch.stack((size_norm, score_mean, score_max), dim=-1)
        cluster = self.cluster_projection(
            torch.cat((mean_state, max_state, scalars), dim=-1)
        )
        cluster = self.cluster_encoder(
            cluster,
            src_key_padding_mask=_safe_padding_mask(cluster_valid),
        )
        return self.cluster_norm(cluster)

    def decode(
        self,
        cluster: torch.Tensor,
        cluster_valid: torch.Tensor,
        *,
        teacher_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, clusters, _hidden = cluster.shape
        valid_float = cluster_valid.to(cluster.dtype).unsqueeze(-1)
        context = (cluster * valid_float).sum(dim=1) / valid_float.sum(
            dim=1
        ).clamp_min(1.0)
        state = torch.tanh(self.context(context))
        key = self.key(cluster)
        base = self.base_score(cluster).squeeze(-1)
        available = cluster_valid.bool().clone()
        logits_rows: list[torch.Tensor] = []
        selection_rows: list[torch.Tensor] = []
        for step in range(self.top_k):
            step_state = self.step_embedding.weight[step].view(1, -1).expand(batch, -1)
            query = self.query(torch.cat((state, step_state), dim=-1))
            logits = torch.einsum("bh,bch->bc", query, key) * self.scale + base
            logits = logits.masked_fill(~available, -1e4)
            logits_rows.append(logits.float())
            if teacher_sequence is None:
                active = available.any(dim=-1)
                chosen = logits.argmax(dim=-1)
                chosen = torch.where(active, chosen, torch.full_like(chosen, -1))
            else:
                chosen = teacher_sequence[:, step]
                active = chosen >= 0
            selection_rows.append(chosen)
            safe_chosen = chosen.clamp_min(0)
            chosen_state = cluster[
                torch.arange(batch, device=cluster.device),
                safe_chosen,
            ]
            updated = self.state_update(chosen_state, state)
            state = torch.where(active.unsqueeze(-1), updated, state)
            if bool(active.any()):
                rows = torch.nonzero(active, as_tuple=False).flatten()
                available[rows, safe_chosen[rows]] = False
        return torch.stack(logits_rows, dim=1), torch.stack(selection_rows, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        candidate_valid: torch.Tensor,
        membership: torch.Tensor,
        cluster_valid: torch.Tensor,
        source_scores: torch.Tensor,
        *,
        teacher_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cluster = self.encode(
            features,
            candidate_valid,
            membership,
            cluster_valid,
            source_scores,
        )
        return self.decode(
            cluster,
            cluster_valid,
            teacher_sequence=teacher_sequence,
        )


def coverage_pointer_loss(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    log_probability = F.log_softmax(logits, dim=-1)
    per_step = -(target_distribution.to(log_probability.dtype) * log_probability).sum(
        dim=-1
    )
    weight = active.to(per_step.dtype)
    return (per_step * weight).sum() / weight.sum().clamp_min(1.0)


def representative_listwise_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    membership: torch.Tensor,
    cluster_valid: torch.Tensor,
    *,
    temperature: float,
    positive_iou: float,
    min_spread: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    member = membership.bool()
    count = member.sum(dim=-1)
    target_expanded = targets.unsqueeze(1).expand(-1, int(member.shape[1]), -1)
    best = target_expanded.masked_fill(~member, -1e4).amax(dim=-1)
    worst = target_expanded.masked_fill(~member, 1e4).amin(dim=-1)
    eligible = (
        cluster_valid.bool()
        & (count >= 2)
        & (best >= float(positive_iou))
        & ((best - worst) >= float(min_spread))
    )
    if not bool(eligible.any()):
        return logits.sum() * 0.0, {"eligible_clusters": 0}
    expanded_logits = logits.unsqueeze(1).expand(-1, int(member.shape[1]), -1)
    selected_logits = expanded_logits[eligible]
    selected_targets = target_expanded[eligible]
    selected_members = member[eligible]
    target_logits = selected_targets / max(float(temperature), 1e-4)
    target_logits = target_logits.masked_fill(~selected_members, -1e4)
    target_distribution = torch.softmax(target_logits, dim=-1)
    prediction_log_probability = F.log_softmax(
        selected_logits.masked_fill(~selected_members, -1e4),
        dim=-1,
    )
    loss = -(target_distribution * prediction_log_probability).sum(dim=-1).mean()
    return loss, {"eligible_clusters": int(eligible.sum())}


def representative_target_statistics(
    hierarchy: dict[str, Any],
    *,
    positive_iou: float,
    min_spread: float,
) -> dict[str, Any]:
    member = hierarchy["membership"].bool()
    cluster_valid = hierarchy["cluster_valid"].bool()
    targets = hierarchy["representative_targets"].float()
    count = member.sum(dim=-1)
    target_expanded = targets.unsqueeze(1).expand(-1, int(member.shape[1]), -1)
    best = target_expanded.masked_fill(~member, -1e4).amax(dim=-1)
    worst = target_expanded.masked_fill(~member, 1e4).amin(dim=-1)
    spread = best - worst
    multi_candidate = cluster_valid & (count >= 2)
    positive = multi_candidate & (best >= float(positive_iou))
    eligible = positive & (spread >= float(min_spread))
    eligible_spread = spread[eligible]
    return {
        "valid_clusters": int(cluster_valid.sum()),
        "multi_candidate_clusters": int(multi_candidate.sum()),
        "positive_multi_candidate_clusters": int(positive.sum()),
        "eligible_hard_clusters": int(eligible.sum()),
        "eligible_hard_clusters_per_image": float(eligible.sum())
        / float(max(int(member.shape[0]), 1)),
        "mean_eligible_target_spread": float(eligible_spread.mean())
        if int(eligible_spread.numel())
        else None,
    }


def _fill_cluster_ids(
    predicted: list[int],
    record: dict[str, Any],
    *,
    top_k: int,
) -> list[int]:
    output: list[int] = []
    for value in [*predicted, *record["source_selected_cluster_ids"], *range(len(record["keepers"]))]:
        value = int(value)
        if value not in output:
            output.append(value)
        if len(output) >= min(int(top_k), len(record["keepers"])):
            break
    return output


@torch.no_grad()
def evaluate_hard_decision(
    cluster_probe: ClusterCoveragePointerProbe,
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
    cluster_selections: list[list[int]] = []
    representative_rows: list[torch.Tensor] = []
    for start in range(0, int(cache["features"].shape[0]), int(batch_size)):
        stop = start + int(batch_size)
        features = cache["features"][start:stop].to(device, dtype=torch.float32)
        candidate_valid = cache["candidate_valid"][start:stop].to(device)
        membership = hierarchy["membership"][start:stop].to(device)
        cluster_valid = hierarchy["cluster_valid"][start:stop].to(device)
        source = scores[start:stop].to(device)
        _logits, selected = cluster_probe(
            features,
            candidate_valid,
            membership,
            cluster_valid,
            source,
        )
        for row in selected.cpu():
            cluster_selections.append(
                [int(value) for value in row.tolist() if int(value) >= 0]
            )
        representative_rows.append(
            representative_probe(features, candidate_valid, source).cpu()
        )
    representative_logits = torch.cat(representative_rows)
    modes = {
        "source_nms": _new_counts(),
        "hard_cluster_source_representative": _new_counts(),
        "source_cluster_hard_representative": _new_counts(),
        "hard_hierarchical": _new_counts(),
    }
    oracle = {0.5: 0, 0.7: 0, "gt": 0}
    for image_index, official_iou in enumerate(cache["official_iou"]):
        record = hierarchy["records"][image_index]
        source_clusters = record["source_selected_cluster_ids"]
        learned_clusters = _fill_cluster_ids(
            cluster_selections[image_index],
            record,
            top_k=top_k,
        )
        selections = {
            "source_nms": _representative_ids(
                source_clusters,
                record,
                representative_logits[image_index],
                learned=False,
            ),
            "hard_cluster_source_representative": _representative_ids(
                learned_clusters,
                record,
                representative_logits[image_index],
                learned=False,
            ),
            "source_cluster_hard_representative": _representative_ids(
                source_clusters,
                record,
                representative_logits[image_index],
                learned=True,
            ),
            "hard_hierarchical": _representative_ids(
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
        "modes": {name: _finish_counts(counts) for name, counts in modes.items()},
        "oracle_top4": {
            f"{threshold:.2f}": {
                "tp": int(oracle[threshold]),
                "gt_lanes": int(oracle["gt"]),
                "recall": float(oracle[threshold])
                / float(max(int(oracle["gt"]), 1)),
            }
            for threshold in (0.5, 0.7)
        },
    }


def _gate_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "modes": {
            "source_nms": evaluation["modes"]["source_nms"],
            "learned_cluster_source_representative": evaluation["modes"]
            ["hard_cluster_source_representative"],
            "source_cluster_learned_representative": evaluation["modes"]
            ["source_cluster_hard_representative"],
            "learned_hierarchical": evaluation["modes"]["hard_hierarchical"],
        }
    }


def verify_source_replay(
    evaluation: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    if "evaluation" in reference:
        expected = reference["evaluation"]["modes"]["source_nms"]
    elif "source" in reference:
        expected = reference["source"]["nms_top4"]
    else:
        raise ValueError("reference report has no source NMS metrics")
    actual = evaluation["modes"]["source_nms"]
    comparisons: dict[str, int] = {}
    for suffix in ("050", "070"):
        key = f"tp_{suffix}"
        delta = int(actual[key]) - int(expected[key])
        comparisons[f"{key}_delta"] = delta
        if delta != 0:
            raise ValueError(f"source NMS replay mismatch for {key}: {delta}")
    return {"matched": True, "comparisons": comparisons}


def evaluate_source_nms(
    cache: dict[str, Any],
    hierarchy: dict[str, Any],
) -> dict[str, Any]:
    counts = _new_counts()
    for image_index, official_iou in enumerate(cache["official_iou"]):
        record = hierarchy["records"][image_index]
        selected = [
            int(record["keepers"][cluster_index])
            for cluster_index in record["source_selected_cluster_ids"]
        ]
        _update_counts(counts, official_iou, selected)
    return {"modes": {"source_nms": _finish_counts(counts)}}


def _objective(evaluation: dict[str, Any], mode: str) -> float:
    row = evaluation["modes"][mode]
    return float(row["recall_050"]) + float(row["recall_070"])


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
    for name in ("coverage_temperature", "representative_temperature"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")
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
    hierarchy_kwargs = {
        "input_h": args.input_h,
        "input_w": args.input_w,
        "min_valid_rows": args.min_valid_rows,
        "row_visibility_thresh": args.row_visibility_thresh,
        "nms_distance": args.nms_distance,
        "nms_min_overlap_points": args.nms_min_overlap_points,
        "top_k": args.top_k,
        "positive_iou": args.representative_positive_iou,
    }
    train_hierarchy = build_hierarchical_supervision(
        train_cache,
        train_scores,
        **hierarchy_kwargs,
    )
    val_hierarchy = build_hierarchical_supervision(
        val_cache,
        val_scores,
        **hierarchy_kwargs,
    )
    preflight_consistency = None
    if args.reference_report:
        reference_path = Path(args.reference_report)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        preflight_consistency = {
            "reference_report": str(reference_path),
            **verify_source_replay(
                evaluate_source_nms(val_cache, val_hierarchy),
                reference,
            ),
        }
    target_kwargs = {
        "top_k": args.top_k,
        "temperature": args.coverage_temperature,
        "min_gain": args.coverage_min_gain,
        "weight_050": args.coverage_weight_050,
        "weight_070": args.coverage_weight_070,
        "weight_iou": args.coverage_weight_iou,
    }
    train_targets = build_hard_decision_targets(
        train_cache,
        train_hierarchy,
        **target_kwargs,
    )
    val_targets = build_hard_decision_targets(
        val_cache,
        val_hierarchy,
        **target_kwargs,
    )
    feature_dim = int(train_cache["features"].shape[-1])
    if int(val_cache["features"].shape[-1]) != feature_dim:
        raise ValueError("train/validation descriptor dimensions differ")
    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
    }
    cluster_probe = ClusterCoveragePointerProbe(
        feature_dim,
        top_k=args.top_k,
        **model_kwargs,
    ).to(device)
    representative_probe = RepresentativeQualityProbe(
        feature_dim,
        **model_kwargs,
    ).to(device)
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
    running_cluster = 0.0
    running_representative = 0.0
    running_steps = 0
    for step in range(1, int(args.train_steps) + 1):
        indices = schedule[step - 1]
        features = train_cache["features"][indices].to(device, dtype=torch.float32)
        candidate_valid = train_cache["candidate_valid"][indices].to(device)
        membership = train_hierarchy["membership"][indices].to(device)
        cluster_valid = train_hierarchy["cluster_valid"][indices].to(device)
        source = train_scores[indices].to(device)

        cluster_probe.train()
        cluster_optimizer.zero_grad(set_to_none=True)
        pointer_logits, _selections = cluster_probe(
            features,
            candidate_valid,
            membership,
            cluster_valid,
            source,
            teacher_sequence=train_targets["sequence"][indices].to(device),
        )
        cluster_loss = coverage_pointer_loss(
            pointer_logits,
            train_targets["distribution"][indices].to(device),
            train_targets["active"][indices].to(device),
        )
        cluster_loss.backward()
        cluster_optimizer.step()

        representative_probe.train()
        representative_optimizer.zero_grad(set_to_none=True)
        representative_logits = representative_probe(
            features,
            candidate_valid,
            source,
        )
        representative_loss, _loss_stats = representative_listwise_loss(
            representative_logits,
            train_hierarchy["representative_targets"][indices].to(device),
            membership,
            cluster_valid,
            temperature=args.representative_temperature,
            positive_iou=args.representative_positive_iou,
            min_spread=args.representative_min_spread,
        )
        representative_loss.backward()
        representative_optimizer.step()
        running_cluster += float(cluster_loss.detach())
        running_representative += float(representative_loss.detach())
        running_steps += 1

        should_evaluate = (
            step == 1
            or step == int(args.train_steps)
            or step % int(args.eval_interval) == 0
        )
        if should_evaluate:
            evaluation = evaluate_hard_decision(
                cluster_probe,
                representative_probe,
                val_cache,
                val_hierarchy,
                val_scores,
                device=device,
                batch_size=args.batch_size,
                top_k=args.top_k,
            )
            cluster_objective = _objective(
                evaluation,
                "hard_cluster_source_representative",
            )
            representative_objective = _objective(
                evaluation,
                "source_cluster_hard_representative",
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
        if step % int(args.log_interval) == 0 or step == int(args.train_steps):
            denominator = float(max(running_steps, 1))
            print(
                f"hard-decision {step:05d}/{args.train_steps:05d} | "
                f"coverage={running_cluster / denominator:.4f} | "
                f"representative={running_representative / denominator:.4f}"
            )
            running_cluster = 0.0
            running_representative = 0.0
            running_steps = 0

    cluster_probe.load_state_dict(best_cluster["state"])
    representative_probe.load_state_dict(best_representative["state"])
    evaluation_kwargs = {
        "device": device,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
    }
    train_evaluation = evaluate_hard_decision(
        cluster_probe,
        representative_probe,
        train_cache,
        train_hierarchy,
        train_scores,
        **evaluation_kwargs,
    )
    val_evaluation = evaluate_hard_decision(
        cluster_probe,
        representative_probe,
        val_cache,
        val_hierarchy,
        val_scores,
        **evaluation_kwargs,
    )
    consistency = preflight_consistency
    if args.reference_report:
        reference = json.loads(Path(args.reference_report).read_text(encoding="utf-8"))
        verify_source_replay(val_evaluation, reference)
    gate_kwargs = {
        "cluster_min_gain_050": args.cluster_min_gain_050,
        "representative_min_gain_070": args.representative_min_gain_070,
        "hierarchical_min_gain_050": args.hierarchical_min_gain_050,
        "hierarchical_min_gain_070": args.hierarchical_min_gain_070,
    }
    train_gate = hierarchy_gate(_gate_evaluation(train_evaluation), **gate_kwargs)
    val_gate = hierarchy_gate(_gate_evaluation(val_evaluation), **gate_kwargs)
    result = {
        "diagnostic_only": True,
        "warning": (
            "The detector, descriptors, candidate curves, and NMS partition "
            "are frozen at 10k. Only hard decision heads are optimized."
        ),
        "config": args.config,
        "source_checkpoint": args.source_checkpoint,
        "source_iteration": int(source_iteration),
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "train_images": int(train_cache["features"].shape[0]),
        "val_images": int(val_cache["features"].shape[0]),
        "target_statistics": {
            "train": {
                "coverage": train_targets["statistics"],
                "representative": representative_target_statistics(
                    train_hierarchy,
                    positive_iou=args.representative_positive_iou,
                    min_spread=args.representative_min_spread,
                ),
            },
            "validation": {
                "coverage": val_targets["statistics"],
                "representative": representative_target_statistics(
                    val_hierarchy,
                    positive_iou=args.representative_positive_iou,
                    min_spread=args.representative_min_spread,
                ),
            },
        },
        "best_steps": {
            "cluster": int(best_cluster["step"]),
            "representative": int(best_representative["step"]),
        },
        "consistency": consistency,
        "train": {"evaluation": train_evaluation, "gate": train_gate},
        "validation": {"evaluation": val_evaluation, "gate": val_gate},
        "decision": {
            "train_dual_head_positive": bool(train_gate["dual_head_positive"]),
            "validation_dual_head_positive": bool(
                val_gate["dual_head_positive"]
            ),
            "interpretation": (
                "hard_decision_loss_is_supported"
                if bool(val_gate["dual_head_positive"])
                else (
                    "hard_decision_fits_train_but_does_not_generalize"
                    if bool(train_gate["dual_head_positive"])
                    else "hard_decision_loss_does_not_unlock_frozen_descriptors"
                )
            ),
        },
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
    printable = {key: value for key, value in result.items() if key != "trajectory"}
    print(json.dumps(printable, indent=2))
    print(f"output_json: {output_path}")
    print(f"probe_checkpoint: {probe_path}")


if __name__ == "__main__":
    main()
