from __future__ import annotations

import torch
from torch import nn


def reverse_complete_curve_relation(relation: torch.Tensor) -> torch.Tensor:
    """Reverse the directed fields in V20's 11-D complete-curve relation."""

    if relation.shape[-1] != 11:
        raise ValueError("V21A expects the V20 11-D curve relation")
    sign = relation.new_tensor(
        (-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    )
    return relation * sign


class _CurveProfileEncoder(nn.Module):
    """Encode one complete curve without mixing it with other proposals."""

    def __init__(
        self,
        *,
        profile_channels: int,
        rows: int,
        offsets: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("V21A hidden_dim must be divisible by num_heads")
        self.rows = int(rows)
        self.offsets = int(offsets)
        self.hidden_dim = int(hidden_dim)
        self.profile_norm = nn.LayerNorm(int(profile_channels))
        self.profile_projection = nn.Linear(int(profile_channels), self.hidden_dim)
        self.offset_embedding = nn.Parameter(
            torch.empty(self.offsets, self.hidden_dim)
        )
        self.local_mixer = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.offset_fusion = nn.Sequential(
            nn.LayerNorm(self.offsets * self.hidden_dim),
            nn.Linear(self.offsets * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.row_embedding = nn.Parameter(torch.empty(self.rows, self.hidden_dim))
        self.visibility_projection = nn.Linear(1, self.hidden_dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical = nn.TransformerEncoder(
            layer,
            num_layers=int(vertical_layers),
            enable_nested_tensor=False,
        )
        self.row_score = nn.Linear(self.hidden_dim, 1)
        self.output_norm = nn.LayerNorm(2 * self.hidden_dim)
        self.output = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        nn.init.normal_(self.offset_embedding, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)

    def forward(
        self,
        profile: torch.Tensor,
        row_weight: torch.Tensor,
    ) -> torch.Tensor:
        if profile.ndim < 4:
            raise ValueError("V21A profiles must end in [R,K,C]")
        if profile.shape[-3:-1] != (self.rows, self.offsets):
            raise ValueError("V21A profile rows/offsets do not match the contract")
        if row_weight.shape != profile.shape[:-2]:
            raise ValueError("V21A row weights do not align with profiles")
        prefix = profile.shape[:-3]
        flat = profile.detach().float().reshape(
            -1, self.rows, self.offsets, int(profile.shape[-1])
        )
        weight = row_weight.detach().float().reshape(-1, self.rows).clamp(0.0, 1.0)
        hidden = self.profile_projection(self.profile_norm(flat))
        hidden = hidden + self.offset_embedding.view(
            1, 1, self.offsets, self.hidden_dim
        )
        hidden = hidden + self.local_mixer(hidden)
        rows = self.offset_fusion(
            hidden.reshape(-1, self.rows, self.offsets * self.hidden_dim)
        )
        rows = rows + self.row_embedding.view(1, self.rows, self.hidden_dim)
        rows = rows + self.visibility_projection(weight.unsqueeze(-1))
        rows = self.vertical(rows)

        normalized = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        mean = (rows * normalized.unsqueeze(-1)).sum(dim=1)
        attention_logits = self.row_score(rows).squeeze(-1)
        attention_logits = attention_logits + weight.clamp_min(1.0e-6).log()
        attention = torch.softmax(attention_logits, dim=1)
        attended = (rows * attention.unsqueeze(-1)).sum(dim=1)
        encoded = self.output(self.output_norm(torch.cat((mean, attended), dim=-1)))
        return encoded.reshape(*prefix, self.hidden_dim)


class PairwiseVisualLaneVerifier(nn.Module):
    """Antisymmetric current-vs-candidate verifier for the V21A diagnostic.

    The module never scores a proposal ID in isolation.  It uses a shared curve
    encoder for both hypotheses and constructs the final superiority logit as
    ``0.5 * (h(candidate,current) - h(current,candidate))``.  Swapping the two
    hypotheses therefore negates the score by construction.
    """

    def __init__(
        self,
        *,
        profile_channels: int,
        state_dim: int,
        scalar_dim: int,
        relation_dim: int,
        rows: int,
        offsets: int,
        curve_dim: int = 96,
        state_hidden_dim: int = 64,
        scalar_hidden_dim: int = 32,
        relation_hidden_dim: int = 48,
        pair_hidden_dim: int = 192,
        num_heads: int = 4,
        ff_dim: int = 256,
        vertical_layers: int = 2,
    ) -> None:
        super().__init__()
        if int(relation_dim) != 11:
            raise ValueError("V21A relation_dim must be 11")
        self.curve_encoder = _CurveProfileEncoder(
            profile_channels=int(profile_channels),
            rows=int(rows),
            offsets=int(offsets),
            hidden_dim=int(curve_dim),
            num_heads=int(num_heads),
            ff_dim=int(ff_dim),
            vertical_layers=int(vertical_layers),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(int(state_dim)),
            nn.Linear(int(state_dim), int(state_hidden_dim)),
            nn.GELU(),
        )
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(int(scalar_dim)),
            nn.Linear(int(scalar_dim), int(scalar_hidden_dim)),
            nn.GELU(),
        )
        self.relation_projection = nn.Sequential(
            nn.LayerNorm(int(relation_dim)),
            nn.Linear(int(relation_dim), int(relation_hidden_dim)),
            nn.GELU(),
        )
        pair_input_dim = (
            4 * int(curve_dim)
            + 3 * int(state_hidden_dim)
            + 3 * int(scalar_hidden_dim)
            + int(relation_hidden_dim)
        )
        self.order_score = nn.Sequential(
            nn.LayerNorm(pair_input_dim),
            nn.Linear(pair_input_dim, int(pair_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(pair_hidden_dim), int(pair_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(pair_hidden_dim), 1),
        )

    def _ordered_score(
        self,
        first_curve: torch.Tensor,
        second_curve: torch.Tensor,
        first_state: torch.Tensor,
        second_state: torch.Tensor,
        first_scalar: torch.Tensor,
        second_scalar: torch.Tensor,
        first_to_second_relation: torch.Tensor,
    ) -> torch.Tensor:
        curve = torch.cat(
            (
                first_curve,
                second_curve,
                first_curve - second_curve,
                first_curve * second_curve,
            ),
            dim=-1,
        )
        state = torch.cat(
            (first_state, second_state, first_state - second_state), dim=-1
        )
        scalar = torch.cat(
            (first_scalar, second_scalar, first_scalar - second_scalar), dim=-1
        )
        relation = self.relation_projection(first_to_second_relation.float())
        return self.order_score(
            torch.cat((curve, state, scalar, relation), dim=-1)
        ).squeeze(-1)

    def _encoded_scores(
        self,
        *,
        source_curve: torch.Tensor,
        candidate_curve: torch.Tensor,
        source_state: torch.Tensor,
        candidate_state: torch.Tensor,
        source_scalar: torch.Tensor,
        candidate_scalar: torch.Tensor,
        candidate_to_source_relation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = int(candidate_curve.shape[1])
        source_curve = source_curve.unsqueeze(1).expand(-1, candidates, -1)
        source_state = source_state.unsqueeze(1).expand(-1, candidates, -1)
        source_scalar = source_scalar.unsqueeze(1).expand(-1, candidates, -1)
        forward = self._ordered_score(
            candidate_curve,
            source_curve,
            candidate_state,
            source_state,
            candidate_scalar,
            source_scalar,
            candidate_to_source_relation,
        )
        reverse = self._ordered_score(
            source_curve,
            candidate_curve,
            source_state,
            candidate_state,
            source_scalar,
            candidate_scalar,
            reverse_complete_curve_relation(candidate_to_source_relation),
        )
        return 0.5 * (forward - reverse), 0.5 * (reverse - forward)

    def forward(
        self,
        *,
        source_profile: torch.Tensor,
        candidate_profile: torch.Tensor,
        source_row_weight: torch.Tensor,
        candidate_row_weight: torch.Tensor,
        source_state: torch.Tensor,
        candidate_state: torch.Tensor,
        source_scalar: torch.Tensor,
        candidate_scalar: torch.Tensor,
        candidate_to_source_relation: torch.Tensor,
        candidate_valid: torch.Tensor,
        return_swapped: bool = False,
    ) -> dict[str, torch.Tensor]:
        if candidate_profile.ndim != 5 or source_profile.ndim != 4:
            raise ValueError("V21A profile tensors have invalid rank")
        batch, candidates = candidate_profile.shape[:2]
        if source_profile.shape[0] != batch:
            raise ValueError("V21A source/candidate batch mismatch")
        if candidate_valid.shape != (batch, candidates):
            raise ValueError("V21A candidate validity shape mismatch")
        source_curve = self.curve_encoder(source_profile, source_row_weight)
        candidate_curve = self.curve_encoder(
            candidate_profile, candidate_row_weight
        )
        source_state_encoded = self.state_projection(source_state.detach().float())
        candidate_state_encoded = self.state_projection(
            candidate_state.detach().float()
        )
        source_scalar_encoded = self.scalar_projection(
            source_scalar.detach().float()
        )
        candidate_scalar_encoded = self.scalar_projection(
            candidate_scalar.detach().float()
        )
        score, swapped = self._encoded_scores(
            source_curve=source_curve,
            candidate_curve=candidate_curve,
            source_state=source_state_encoded,
            candidate_state=candidate_state_encoded,
            source_scalar=source_scalar_encoded,
            candidate_scalar=candidate_scalar_encoded,
            candidate_to_source_relation=candidate_to_source_relation.detach().float(),
        )
        score = score.masked_fill(~candidate_valid.bool(), -1.0e4)
        output = {"score": score}
        if return_swapped:
            output["swapped_score"] = swapped.masked_fill(
                ~candidate_valid.bool(), 1.0e4
            )
        return output
