from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .position_encoding import SinePositionEncoding2D


class UnifiedLaneSetLayer(nn.Module):
    """Make one lane state jointly own set selection and ordered geometry.

    The previous persistent lane state was a read-only scoring sidecar: it
    consumed row states, but could not influence the curve represented by
    those rows and could not compare itself with the other lane candidates.
    This block closes both missing edges in the computation graph.

    A decoder block is deliberately split into explicit phases so the existing
    high-resolution row-reference decoder can remain the geometry operator::

        lane set self-attention
            -> broadcast the lane identity into all of its row states
            -> P2 row-reference geometry decoder
            -> collect the updated rows back into the same lane identity
            -> optional coarse-semantic decision view for the score head

    Consequently, localization losses reach the lane-set state through the
    lane-to-row edge, while foreground losses reach the visual row evidence
    through the row-to-lane edge.  Coarse context is a score-only *view* of
    that same identity: it is not persisted into the next geometry block.
    There is no detachable late selector and no scalar gate that the optimizer
    can collapse to bypass this path.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        *,
        semantic_context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads:
            raise ValueError("unified lane-set dim must be divisible by num_heads")

        self.set_attention = nn.MultiheadAttention(
            self.dim,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_set = nn.LayerNorm(self.dim)
        self.norm_set_ffn = nn.LayerNorm(self.dim)
        self.set_ffn = nn.Sequential(
            nn.Linear(self.dim, int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), self.dim),
        )

        # This fixed residual edge is intentional.  A learned scalar gate can
        # take the easy route of shrinking to zero, recreating the read-only
        # scoring sidecar that this layer replaces.
        self.norm_lane_to_rows = nn.LayerNorm(self.dim)
        self.lane_to_rows = nn.Linear(self.dim, self.dim)

        self.rows_to_lane = nn.MultiheadAttention(
            self.dim,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_lane_collect = nn.LayerNorm(self.dim)
        self.norm_rows_collect = nn.LayerNorm(self.dim)
        self.norm_collect_ffn = nn.LayerNorm(self.dim)
        self.collect_ffn = nn.Sequential(
            nn.Linear(self.dim, int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), self.dim),
        )
        self.drop = nn.Dropout(float(dropout))

        semantic_cfg = dict(semantic_context or {})
        self.semantic_enabled = bool(semantic_cfg.get("enabled", False))
        self.semantic_scales = tuple(
            str(scale) for scale in semantic_cfg.get("scales", ("p4", "p5"))
        )
        if self.semantic_enabled and not self.semantic_scales:
            raise ValueError("semantic_context.scales must not be empty")
        pool_size = semantic_cfg.get("pool_size", (10, 25))
        if not isinstance(pool_size, Sequence) or len(pool_size) != 2:
            raise ValueError("semantic_context.pool_size must contain [height, width]")
        self.semantic_pool_size = (int(pool_size[0]), int(pool_size[1]))
        if min(self.semantic_pool_size) < 1:
            raise ValueError("semantic_context.pool_size entries must be positive")

        self.semantic_position = (
            SinePositionEncoding2D(dim=self.dim)
            if self.semantic_enabled
            else None
        )
        self.semantic_attention = (
            nn.ModuleDict(
                {
                    scale: nn.MultiheadAttention(
                        self.dim,
                        self.num_heads,
                        dropout=float(dropout),
                        batch_first=True,
                    )
                    for scale in self.semantic_scales
                }
            )
            if self.semantic_enabled
            else nn.ModuleDict()
        )
        self.semantic_scale_embedding = (
            nn.Parameter(torch.zeros(len(self.semantic_scales), self.dim))
            if self.semantic_enabled
            else None
        )
        self.semantic_router = (
            nn.Linear(self.dim, len(self.semantic_scales))
            if self.semantic_enabled
            else None
        )
        self.norm_semantic_query = (
            nn.LayerNorm(self.dim) if self.semantic_enabled else None
        )
        self.norm_semantic_ffn = (
            nn.LayerNorm(self.dim) if self.semantic_enabled else None
        )
        self.semantic_ffn = (
            nn.Sequential(
                nn.Linear(self.dim, int(ff_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(ff_dim), self.dim),
            )
            if self.semantic_enabled
            else None
        )
        if self.semantic_router is not None:
            # Equal scale weighting is a neutral, deterministic start.
            nn.init.zeros_(self.semantic_router.weight)
            nn.init.zeros_(self.semantic_router.bias)

    @staticmethod
    def _validate_group_sizes(
        candidate_count: int,
        group_sizes: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if group_sizes is None:
            return None
        normalized = tuple(int(size) for size in group_sizes)
        if not normalized or any(size < 1 for size in normalized):
            raise ValueError("lane-set group sizes must be positive")
        if sum(normalized) != int(candidate_count):
            raise ValueError(
                "lane-set group sizes must sum to the candidate count: "
                f"{normalized} vs {candidate_count}"
            )
        return normalized

    def _set_attention(
        self,
        state: torch.Tensor,
        group_sizes: tuple[int, ...] | None,
    ) -> torch.Tensor:
        normalized = self._validate_group_sizes(int(state.shape[1]), group_sizes)
        if normalized is None or len(normalized) == 1:
            return self.set_attention(state, state, state, need_weights=False)[0]
        parts = torch.split(state, normalized, dim=1)
        return torch.cat(
            [
                self.set_attention(part, part, part, need_weights=False)[0]
                for part in parts
            ],
            dim=1,
        )

    def _semantic_context(
        self,
        lane_state: torch.Tensor,
        multi_scale_features: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.semantic_enabled:
            return lane_state.new_zeros(lane_state.shape)
        if multi_scale_features is None:
            raise ValueError(
                "unified lane semantic context is enabled but the encoder did "
                "not provide multi_scale_features"
            )
        assert self.semantic_position is not None
        assert self.semantic_scale_embedding is not None
        assert self.semantic_router is not None
        assert self.norm_semantic_query is not None

        query = self.norm_semantic_query(lane_state)
        contexts: list[torch.Tensor] = []
        for scale_index, scale in enumerate(self.semantic_scales):
            if scale not in multi_scale_features:
                raise KeyError(f"multi_scale_features is missing {scale!r}")
            feature = multi_scale_features[scale]
            if feature.ndim != 4 or int(feature.shape[1]) != self.dim:
                raise ValueError(
                    f"semantic feature {scale!r} must be [B,{self.dim},H,W], "
                    f"got {tuple(feature.shape)}"
                )
            pooled = F.adaptive_avg_pool2d(feature, self.semantic_pool_size)
            position = self.semantic_position(pooled).to(
                device=pooled.device,
                dtype=pooled.dtype,
            )
            scale_code = self.semantic_scale_embedding[scale_index].to(
                device=pooled.device,
                dtype=pooled.dtype,
            ).view(1, self.dim, 1, 1)
            key = (pooled + position + scale_code).flatten(2).transpose(1, 2)
            value = pooled.flatten(2).transpose(1, 2)
            contexts.append(
                self.semantic_attention[scale](
                    query,
                    key,
                    value,
                    need_weights=False,
                )[0]
            )
        stacked = torch.stack(contexts, dim=2)
        weights = torch.softmax(self.semantic_router(query).float(), dim=-1).to(
            dtype=lane_state.dtype
        )
        return (stacked * weights.unsqueeze(-1)).sum(dim=2)

    def prepare(
        self,
        lane_state: torch.Tensor,
        *,
        group_sizes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if lane_state.ndim != 3 or int(lane_state.shape[-1]) != self.dim:
            raise ValueError("lane_state must have shape [B,N,C]")
        normalized = self.norm_set(lane_state)
        lane_state = lane_state + self.drop(
            self._set_attention(normalized, group_sizes)
        )
        lane_state = lane_state + self.drop(
            self.set_ffn(self.norm_set_ffn(lane_state))
        )
        return lane_state

    def decision(
        self,
        lane_state: torch.Tensor,
        *,
        multi_scale_features: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return the score/ranking view without mutating geometry state."""

        if not self.semantic_enabled:
            return lane_state
        assert self.norm_semantic_ffn is not None
        assert self.semantic_ffn is not None
        decision_state = lane_state + self.drop(
            self._semantic_context(lane_state, multi_scale_features)
        )
        decision_state = decision_state + self.drop(
            self.semantic_ffn(self.norm_semantic_ffn(decision_state))
        )
        return decision_state

    def inject_rows(
        self,
        lane_state: torch.Tensor,
        row_states: torch.Tensor,
    ) -> torch.Tensor:
        if row_states.ndim != 4:
            raise ValueError("row_states must have shape [B,N,R,C]")
        if row_states.shape[:2] != lane_state.shape[:2]:
            raise ValueError("lane_state and row_states must share B/N axes")
        identity = self.lane_to_rows(self.norm_lane_to_rows(lane_state))
        return row_states + self.drop(identity).unsqueeze(2)

    def collect(
        self,
        lane_state: torch.Tensor,
        row_states: torch.Tensor,
    ) -> torch.Tensor:
        if row_states.ndim != 4:
            raise ValueError("row_states must have shape [B,N,R,C]")
        batch, candidates, rows, channels = row_states.shape
        if lane_state.shape != (batch, candidates, channels):
            raise ValueError("lane_state and row_states shapes are incompatible")
        query = self.norm_lane_collect(lane_state).reshape(
            batch * candidates,
            1,
            channels,
        )
        memory = self.norm_rows_collect(row_states).reshape(
            batch * candidates,
            rows,
            channels,
        )
        state = lane_state.reshape(batch * candidates, 1, channels)
        state = state + self.drop(
            self.rows_to_lane(query, memory, memory, need_weights=False)[0]
        )
        state = state + self.drop(self.collect_ffn(self.norm_collect_ffn(state)))
        return state.reshape(batch, candidates, channels).contiguous()
