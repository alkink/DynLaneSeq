from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import fixed_indices
from .v23_ordered_slot_cost_volume import soft_viterbi_marginals
from .v25_image_mediated_lane_objects import (
    V25ImageMediatedLaneObjects,
    hard_viterbi_paths,
)


@dataclass(frozen=True)
class DiversePathResult:
    indices: torch.Tensor
    scores: torch.Tensor


def diverse_viterbi_paths(
    unary_logits: torch.Tensor,
    *,
    num_hypotheses: int,
    transition_radius_bins: int,
    transition_penalty: float,
    suppression_radius_bins: int = 5,
    suppression_penalty: float = 8.0,
) -> DiversePathResult:
    """Extract diverse coherent paths without averaging spatial modes.

    The first member is the exact MAP path.  Later members are decoded from
    the same immutable unary tensor after a local penalty is placed around all
    previously selected paths.  Scores always come from the immutable tensor,
    so suppression cannot make a weak alternative appear intrinsically good.
    """

    if unary_logits.ndim != 4:
        raise ValueError("diverse path decoder expects logits [B,S,R,X]")
    if num_hypotheses < 1:
        raise ValueError("num_hypotheses must be positive")
    if suppression_radius_bins < 0:
        raise ValueError("suppression_radius_bins must be non-negative")
    base = unary_logits.float()
    work = base.clone()
    log_probability = base.log_softmax(dim=-1)
    paths: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    bins = int(base.shape[-1])
    for _ in range(int(num_hypotheses)):
        path = hard_viterbi_paths(
            work,
            transition_radius_bins=transition_radius_bins,
            transition_penalty=transition_penalty,
        )
        score = log_probability.gather(-1, path.unsqueeze(-1)).squeeze(-1).mean(-1)
        paths.append(path)
        scores.append(score)
        if len(paths) == int(num_hypotheses):
            break
        penalty = torch.zeros_like(work)
        for offset in range(-suppression_radius_bins, suppression_radius_bins + 1):
            index = (path + offset).clamp(0, bins - 1)
            distance_weight = 1.0 - abs(offset) / float(suppression_radius_bins + 1)
            penalty.scatter_add_(
                -1,
                index.unsqueeze(-1),
                torch.full_like(index.unsqueeze(-1), suppression_penalty * distance_weight, dtype=work.dtype),
            )
        work = work - penalty
    return DiversePathResult(
        indices=torch.stack(paths, dim=2),
        scores=torch.stack(scores, dim=2),
    )


def dual_energy_log_mixture(
    image_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    mixture_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse image and proposal distributions while preserving both modes."""

    if image_logits.shape != proposal_logits.shape:
        raise ValueError("image/proposal energy shapes differ")
    if mixture_logits.shape != (*image_logits.shape[:2], 2):
        raise ValueError("mixture logits must be [B,S,2]")
    log_mix = mixture_logits.float().log_softmax(dim=-1)
    image = image_logits.float().log_softmax(dim=-1)
    proposal = proposal_logits.float().log_softmax(dim=-1)
    fused = torch.logaddexp(
        image + log_mix[..., 0, None, None],
        proposal + log_mix[..., 1, None, None],
    )
    return fused.to(dtype=image_logits.dtype), log_mix.exp().to(dtype=image_logits.dtype)


def exact_small_path_set_decode(
    hypotheses: torch.Tensor,
    path_scores: torch.Tensor,
    exist_logits: torch.Tensor,
    *,
    input_w: int,
    minimum_spacing_px: float = 12.0,
    order_penalty: float = 4.0,
    duplicate_penalty: float = 2.0,
    hole_penalty: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Exact hard decode over ``(K + dustbin)^4`` complete lane sets.

    This is deliberately deterministic.  It does not add another late neural
    reranker: the selected hypotheses are scored by the trained spatial energy,
    existence evidence and explicit set constraints.
    """

    if hypotheses.ndim != 4:
        raise ValueError("hypotheses must be [B,S,K,R]")
    batch, slots, count, rows = hypotheses.shape
    if slots != 4:
        raise ValueError("exact V25 set decode currently requires four slots")
    if path_scores.shape != (batch, slots, count):
        raise ValueError("path score shape mismatch")
    if exist_logits.shape != (batch, slots, 2):
        raise ValueError("existence logits must be [B,S,2]")
    choices = torch.tensor(
        list(product(range(count + 1), repeat=slots)),
        device=hypotheses.device,
        dtype=torch.long,
    )
    combinations = int(choices.shape[0])
    log_exist = exist_logits.float().log_softmax(dim=-1)
    relative_path_scores = path_scores.float() - path_scores.float().amax(
        dim=-1, keepdim=True
    )
    score = hypotheses.new_zeros((batch, combinations), dtype=torch.float32)
    gathered_paths: list[torch.Tensor] = []
    active_masks: list[torch.Tensor] = []
    for slot in range(slots):
        selected = choices[:, slot]
        active = selected < count
        safe = selected.clamp(max=count - 1)
        lane_paths = hypotheses[:, slot][:, safe]
        lane_score = relative_path_scores[:, slot][:, safe]
        score = score + torch.where(
            active.view(1, combinations),
            lane_score + log_exist[:, slot, 0].view(batch, 1),
            log_exist[:, slot, 1].view(batch, 1),
        )
        gathered_paths.append(lane_paths)
        active_masks.append(active)

    for left in range(slots - 1):
        right = left + 1
        both = (active_masks[left] & active_masks[right]).view(1, combinations)
        distance = gathered_paths[right] - gathered_paths[left]
        order = (float(minimum_spacing_px) - distance).clamp_min(0.0)
        score = score - float(order_penalty) * order.mean(dim=-1) / float(input_w) * both
        duplicate = torch.exp(-distance.abs() / max(float(minimum_spacing_px), 1.0)).mean(-1)
        score = score - float(duplicate_penalty) * duplicate * both

    # Canonical slots are packed from left to right.  An inactive slot followed
    # by an active one represents a semantic hole and receives a fixed cost.
    for left in range(slots - 1):
        hole = (~active_masks[left]) & active_masks[left + 1]
        score = score - float(hole_penalty) * hole.view(1, combinations)

    winner = score.argmax(dim=-1)
    selected_choice = choices[winner]
    selected_active = selected_choice < count
    selected_paths = hypotheses.new_zeros((batch, slots, rows))
    for slot in range(slots):
        safe = selected_choice[:, slot].clamp(max=count - 1)
        selected_paths[:, slot] = hypotheses[:, slot].gather(
            1, safe.view(batch, 1, 1).expand(batch, 1, rows)
        ).squeeze(1)
    return {
        "selected_path_indices": selected_choice,
        "selected_active": selected_active,
        "selected_paths": selected_paths,
        "set_scores": score,
        "winning_set_index": winner,
    }


class AuxiliaryProposalMemory(nn.Module):
    """One-to-many image proposals used only to form a spatial energy prior."""

    def __init__(
        self,
        *,
        dim: int,
        rows: int,
        bins: int,
        proposals: int = 32,
        groups: int = 4,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        anchor_weight: float = 0.08,
        anchor_sigma: float = 0.24,
    ) -> None:
        super().__init__()
        if proposals % groups:
            raise ValueError("proposal count must be divisible by groups")
        self.rows = int(rows)
        self.bins = int(bins)
        self.proposals = int(proposals)
        self.groups = int(groups)
        self.anchor_weight = float(anchor_weight)
        self.anchor_sigma = float(anchor_sigma)
        self.proposal_embedding = nn.Parameter(torch.empty(proposals, dim))
        self.row_embedding = nn.Parameter(torch.empty(rows, dim))
        self.anchor_projection = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.key_projection = nn.Conv2d(dim, dim, 1, bias=False)
        self.value_projection = nn.Conv2d(dim, dim, 1, bias=False)
        self.context_projection = nn.Linear(dim, dim)
        self.context_norm = nn.LayerNorm(dim)
        vertical = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical = nn.TransformerEncoder(
            vertical, num_layers=1, norm=nn.LayerNorm(dim)
        )
        per_group = proposals // groups
        anchors = torch.linspace(0.08, 0.92, per_group).repeat(groups)
        self.register_buffer("anchor_centres", anchors, persistent=True)
        nn.init.normal_(self.proposal_embedding, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, dim, rows, bins = features.shape
        if rows != self.rows or bins != self.bins:
            raise ValueError("proposal memory feature resolution mismatch")
        row_fraction = torch.linspace(
            0.0, 1.0, rows, device=features.device, dtype=torch.float32
        ).view(1, 1, rows)
        centres = self.anchor_centres.to(features.device).view(1, self.proposals, 1)
        geometry = torch.stack(
            (
                centres.expand(batch, self.proposals, rows),
                row_fraction.expand(batch, self.proposals, rows),
            ),
            dim=-1,
        )
        query = (
            self.proposal_embedding.view(1, self.proposals, 1, dim)
            + self.row_embedding.view(1, 1, rows, dim)
            + self.anchor_projection(geometry).to(dtype=features.dtype)
        )
        keys = self.key_projection(features)
        values = self.value_projection(features)
        scale = 1.0 / math.sqrt(float(dim))
        logits = torch.einsum("bnrc,bcrx->bnrx", query, keys) * scale
        x = (fixed_indices(bins, device=features.device, dtype=torch.float32) + 0.5) / float(bins)
        anchor = -0.5 * (
            (x.view(1, 1, 1, bins) - centres.unsqueeze(-1)) / self.anchor_sigma
        ).pow(2)
        logits = logits + self.anchor_weight * anchor.to(dtype=logits.dtype)
        probability = logits.float().softmax(dim=-1).to(dtype=features.dtype)
        context = torch.einsum("bnrx,bcrx->bnrc", probability, values)
        query = self.context_norm(query + self.context_projection(context))
        query = self.vertical(query.reshape(batch * self.proposals, rows, dim)).reshape(
            batch, self.proposals, rows, dim
        )
        logits = torch.einsum("bnrc,bcrx->bnrx", query, keys) * scale + self.anchor_weight * anchor.to(dtype=query.dtype)
        probability = logits.float().softmax(dim=-1)
        centres_px = (
            fixed_indices(bins, device=features.device, dtype=torch.float32) + 0.5
        ) * (1600.0 / float(bins))
        # The normalized path is resolution-independent; callers rescale it to
        # their actual input width.
        x_normalized = (
            probability
            * ((fixed_indices(bins, device=features.device, dtype=torch.float32) + 0.5) / float(bins)).view(1, 1, 1, bins)
        ).sum(-1)
        del centres_px
        return {
            "proposal_unary_logits": logits,
            "proposal_x_normalized": x_normalized,
            "proposal_state": query,
            "proposal_pooled_state": query.mean(dim=2),
        }


class V25DualEnergyMultiPath(V25ImageMediatedLaneObjects):
    """V25 final objects with proposal/image dual energy and diverse paths."""

    def __init__(
        self,
        *,
        num_path_hypotheses: int = 3,
        path_suppression_radius_bins: int = 5,
        path_suppression_penalty: float = 8.0,
        proposal_count: int = 32,
        proposal_groups: int = 4,
        proposal_dropout: float = 0.25,
        enable_proposal_fusion: bool = True,
        exact_set_selection: bool = True,
        minimum_spacing_px: float = 12.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        hidden_dim = int(self.slot_embedding.shape[-1])
        self.num_path_hypotheses = int(num_path_hypotheses)
        self.path_suppression_radius_bins = int(path_suppression_radius_bins)
        self.path_suppression_penalty = float(path_suppression_penalty)
        self.proposal_dropout = float(proposal_dropout)
        self.enable_proposal_fusion = bool(enable_proposal_fusion)
        self.exact_set_selection = bool(exact_set_selection)
        self.minimum_spacing_px = float(minimum_spacing_px)
        self.proposal_memory = AuxiliaryProposalMemory(
            dim=hidden_dim,
            rows=self.num_rows,
            bins=self.x_bins,
            proposals=int(proposal_count),
            groups=int(proposal_groups),
            num_heads=int(kwargs.get("num_heads", 4)),
            ff_dim=int(kwargs.get("ff_dim", 256)),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
        self.slot_proposal_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proposal_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.energy_mixture = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2)
        )
        self.row_visibility_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3),
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def _proposal_prior(
        self,
        lane_state: torch.Tensor,
        proposal: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_lane = lane_state.mean(dim=2)
        pooled_proposal = proposal["proposal_pooled_state"]
        dim = int(pooled_lane.shape[-1])
        affinity = torch.einsum(
            "bsc,bnc->bsn",
            self.slot_proposal_query(pooled_lane),
            self.proposal_key(pooled_proposal),
        ) / math.sqrt(float(dim))
        log_weight = affinity.float().log_softmax(dim=-1)
        proposal_energy = proposal["proposal_unary_logits"].float().log_softmax(
            dim=-1
        )
        prior: torch.Tensor | None = None
        chunk = 4
        for start in range(0, int(proposal_energy.shape[1]), chunk):
            stop = min(start + chunk, int(proposal_energy.shape[1]))
            # Fuse complete spatial proposal distributions.  No proposal x
            # expectation or Gaussian coordinate reconstruction is used, so
            # multiple proposal modes remain separate in energy space.
            component = (
                proposal_energy[:, None, start:stop]
                + log_weight[:, :, start:stop, None, None]
            )
            reduced = component.logsumexp(dim=2)
            prior = reduced if prior is None else torch.logaddexp(prior, reduced)
        if prior is None:
            raise RuntimeError("proposal prior has no components")
        return prior.to(dtype=lane_state.dtype), affinity

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(images)
        proposal = self.proposal_memory(output["image_features"])
        proposal_prior, proposal_affinity = self._proposal_prior(
            output["lane_object_state"], proposal
        )
        pooled = output["lane_object_state"].mean(dim=2)
        mixture_logits = self.energy_mixture(pooled)
        if self.training and self.proposal_dropout > 0.0:
            drop = torch.rand(
                (images.shape[0], 1), device=images.device
            ) < self.proposal_dropout
            mixture_logits = mixture_logits.clone()
            mixture_logits[..., 1] = mixture_logits[..., 1].masked_fill(drop, -30.0)
        if self.enable_proposal_fusion:
            fused, mixture = dual_energy_log_mixture(
                output["unary_logits"], proposal_prior, mixture_logits
            )
        else:
            fused = output["unary_logits"]
            mixture = torch.zeros_like(mixture_logits).softmax(dim=-1)
            mixture[..., 0] = 1.0
            mixture[..., 1] = 0.0
        path_logits = soft_viterbi_marginals(
            fused,
            transition_radius_bins=self.transition_radius_bins,
            transition_penalty=self.transition_penalty,
        )
        decoded = self._decode(fused, path_logits)
        posterior = decoded["path_posterior"].float()
        entropy = -(posterior * posterior.clamp_min(1.0e-9).log()).sum(-1).mean(-1)
        top2 = posterior.topk(k=2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).mean(-1)
        image_proposal_agreement = (
            output["unary_logits"].float().softmax(-1)
            * proposal_prior.float().softmax(-1)
        ).sum(-1).mean(-1)
        reliability_input = torch.cat(
            (
                pooled,
                entropy.unsqueeze(-1),
                margin.unsqueeze(-1),
                image_proposal_agreement.unsqueeze(-1),
            ),
            dim=-1,
        )
        quality = self.reliability_head(reliability_input)
        visibility = self.row_visibility_head(output["lane_object_state"]).squeeze(-1)

        output.update(proposal)
        output.update(
            {
                "image_energy_logits": output["unary_logits"],
                "proposal_energy_logits": proposal_prior,
                "proposal_affinity": proposal_affinity,
                "mixture_weights": mixture,
                "unary_logits": fused,
                "path_logits": path_logits,
                "soft_x_rows": decoded["soft_x_rows"],
                "row_argmax_x_rows": decoded["row_argmax_x_rows"],
                "hard_path_x_rows": decoded["hard_path_x_rows"],
                "hard_path_indices": decoded["hard_path_indices"],
                "path_posterior": posterior,
                "pred_x_rows": decoded["pred_x_rows"].clamp(0.0, float(self.input_w - 1)),
                "row_visibility_logits": visibility,
                "quality_logits": quality[..., 0],
                "quality50_logits": quality[..., 0],
                "quality75_logits": quality[..., 1],
                "path_entropy": entropy,
                "path_margin": margin,
                "image_proposal_agreement": image_proposal_agreement,
            }
        )
        if not self.training and self.num_path_hypotheses > 1:
            diverse = diverse_viterbi_paths(
                fused,
                num_hypotheses=self.num_path_hypotheses,
                transition_radius_bins=self.transition_radius_bins,
                transition_penalty=self.transition_penalty,
                suppression_radius_bins=self.path_suppression_radius_bins,
                suppression_penalty=self.path_suppression_penalty,
            )
            bin_width = float(self.input_w) / float(self.x_bins)
            hypotheses = (diverse.indices.float() + 0.5) * bin_width
            output["path_hypothesis_indices"] = diverse.indices
            output["path_hypotheses"] = hypotheses
            output["path_hypothesis_scores"] = diverse.scores
            if self.exact_set_selection:
                selected = exact_small_path_set_decode(
                    hypotheses,
                    diverse.scores,
                    output["exist_logits"],
                    input_w=self.input_w,
                    minimum_spacing_px=self.minimum_spacing_px,
                )
                output.update(selected)
                output["pred_x_rows"] = selected["selected_paths"].clamp(
                    0.0, float(self.input_w - 1)
                )
                active = selected["selected_active"]
                output["exist_logits"] = torch.stack(
                    (
                        torch.where(active, torch.full_like(active, 20.0, dtype=torch.float32), torch.full_like(active, -20.0, dtype=torch.float32)),
                        torch.where(active, torch.full_like(active, -20.0, dtype=torch.float32), torch.full_like(active, 20.0, dtype=torch.float32)),
                    ),
                    dim=-1,
                ).to(device=images.device, dtype=quality.dtype)
        return output
