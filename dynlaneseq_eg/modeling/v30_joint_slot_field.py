from __future__ import annotations

import math

import torch
from torch import nn


class FourSlotJointBeliefField(nn.Module):
    """Learn one row-wise spatial belief field for each final lane slot.

    The mature V7 proposal generator remains the support mechanism.  This
    module gives the final slots a second, explicitly spatial view of the same
    live P2 representation.  Proposal coordinates are used only as detached
    sampling locations: the new objective shapes the image/slot
    representation, rather than directly dragging proposal coordinates.

    ``route_gate`` is initialized to exactly zero.  Consequently enabling the
    module and loading a V7 checkpoint leaves every public V7 output bitwise
    unchanged before training, while the dense field loss can immediately
    back-propagate into P2/FPN/backbone.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        slot_dim: int,
        num_rows: int,
        input_w: int,
        hidden_dim: int = 64,
        route_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if int(feature_dim) < 1 or int(slot_dim) < 1:
            raise ValueError("joint field feature dimensions must be positive")
        if int(num_rows) < 1 or int(hidden_dim) < 1:
            raise ValueError("joint field rows/hidden_dim must be positive")
        if float(route_residual_scale) < 0.0:
            raise ValueError("joint field route residual scale must be non-negative")

        self.num_rows = int(num_rows)
        self.input_w = int(input_w)
        self.hidden_dim = int(hidden_dim)
        self.route_residual_scale = float(route_residual_scale)

        self.feature_norm = nn.LayerNorm(int(feature_dim))
        self.feature_key = nn.Linear(int(feature_dim), self.hidden_dim, bias=False)
        self.slot_norm = nn.LayerNorm(int(slot_dim))
        self.slot_query = nn.Linear(int(slot_dim), self.hidden_dim, bias=False)
        self.row_embedding = nn.Parameter(
            torch.empty(self.num_rows, self.hidden_dim)
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        # A zero gate is the exact V7 forward contract.  Unlike a zero output
        # projection, it does not block the field loss from reaching the live
        # image features on the first optimizer step.
        self.route_gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.row_embedding, std=0.02)

    def _sample_candidate_scores(
        self,
        field_logits: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, rows, x_bins = field_logits.shape
        if proposal_x_rows.ndim != 3:
            raise ValueError("proposal_x_rows must have shape [B,N,R]")
        if tuple(proposal_x_rows.shape[:1]) != (batch,) or int(
            proposal_x_rows.shape[-1]
        ) != rows:
            raise ValueError("proposal rows must align with the joint field")
        candidates = int(proposal_x_rows.shape[1])
        if tuple(proposal_range_norm.shape) != (batch, candidates, 2):
            raise ValueError("proposal_range_norm must have shape [B,N,2]")
        if tuple(candidate_valid.shape) != (batch, candidates):
            raise ValueError("candidate_valid must have shape [B,N]")

        # Coordinates are immutable evidence locations for this causal gate.
        # The field/image representation is live; proposal geometry still
        # follows V7's mature geometry losses.
        proposal_x = proposal_x_rows.detach().float().clamp(
            0.0, float(max(self.input_w - 1, 1))
        )
        feature_x = proposal_x * float(max(x_bins - 1, 0)) / float(
            max(self.input_w - 1, 1)
        )
        feature_x = feature_x.permute(0, 2, 1).contiguous()  # [B,R,N]
        left = feature_x.floor().long()
        right = (left + 1).clamp(max=max(x_bins - 1, 0))
        alpha = feature_x - left.to(dtype=feature_x.dtype)

        gather_shape = (batch, slots, rows, candidates)
        left_value = field_logits.gather(
            -1, left[:, None].expand(gather_shape)
        )
        right_value = field_logits.gather(
            -1, right[:, None].expand(gather_shape)
        )
        row_score = torch.lerp(
            left_value,
            right_value,
            alpha[:, None].to(dtype=field_logits.dtype),
        )

        row_fraction = torch.linspace(
            0.0,
            1.0,
            rows,
            device=field_logits.device,
            dtype=torch.float32,
        ).view(1, rows, 1)
        ranges = proposal_range_norm.detach().float()
        visible = (
            (row_fraction >= ranges[:, None, :, 0])
            & (row_fraction <= ranges[:, None, :, 1])
            & torch.isfinite(proposal_x_rows.detach().permute(0, 2, 1))
            & candidate_valid[:, None, :]
        )
        visible_slot = visible[:, None].to(dtype=row_score.dtype)
        count = visible_slot.sum(dim=2).clamp_min(1.0)
        candidate_score = (row_score * visible_slot).sum(dim=2) / count
        candidate_score = candidate_score.masked_fill(
            ~candidate_valid[:, None, :], 0.0
        )
        valid_float = candidate_valid[:, None, :].to(candidate_score.dtype)
        valid_count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        valid_mean = (
            candidate_score * valid_float
        ).sum(dim=-1, keepdim=True) / valid_count
        centered_score = (candidate_score - valid_mean).masked_fill(
            ~candidate_valid[:, None, :], 0.0
        )
        return candidate_score, centered_score

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        row_value_features: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if row_value_features.ndim != 4:
            raise ValueError("row_value_features must have shape [B,R,X,C]")
        batch, rows, _x_bins, _channels = row_value_features.shape
        if rows != self.num_rows:
            raise ValueError(
                f"joint field expected {self.num_rows} rows, received {rows}"
            )
        if slot_states.ndim != 3 or int(slot_states.shape[0]) != batch:
            raise ValueError("slot_states must have shape [B,S,D]")

        feature_key = self.feature_key(self.feature_norm(row_value_features))
        slot_query = self.slot_query(self.slot_norm(slot_states))
        row_query = slot_query[:, :, None, :] + self.row_embedding[
            None, None
        ].to(device=slot_states.device, dtype=slot_query.dtype)
        row_query = self.query_norm(row_query)
        field_logits = torch.einsum(
            "bsrh,brxh->bsrx", row_query, feature_key
        ) / math.sqrt(float(self.hidden_dim))

        candidate_score, centered_score = self._sample_candidate_scores(
            field_logits,
            proposal_x_rows,
            proposal_range_norm,
            candidate_valid,
        )
        gate = torch.tanh(self.route_gate.float())
        route_residual = (
            self.route_residual_scale
            * gate
            * torch.tanh(centered_score.float())
        )
        return {
            "field_logits": field_logits,
            "candidate_score": candidate_score,
            "route_residual": route_residual,
            "route_gate": gate,
        }
