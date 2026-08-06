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


class ProtectedOwnershipLayer(nn.Module):
    """Persistent query ownership without a differentiable geometry edge.

    ``UnifiedLaneSetLayer`` intentionally lets one state own both geometry and
    scoring.  That coupling is useful when it is stable, but it also lets a
    foreground loss change the coordinate operator.  This layer implements a
    stricter contract for the V5 experiment:

    * ownership has its own persistent state and query-identity embedding;
    * it compares all candidates as a set at every decoder layer;
    * it observes the current lane/row geometry and semantic pyramid only
      through stop-gradient inputs; and
    * it never writes back into the geometry state.

    Ownership can still affect which geometry query receives a target through
    the (non-differentiable) Hungarian assignment.  That assignment-mediated
    edge is deliberately outside this module.
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
            raise ValueError("protected ownership dim must be divisible by num_heads")

        self.set_attention = nn.MultiheadAttention(
            self.dim,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_set = nn.LayerNorm(self.dim)

        # Each query reads only the geometry rows carrying the same query id.
        # Inter-query competition is handled by ``set_attention`` above.
        self.geometry_attention = nn.MultiheadAttention(
            self.dim,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_geometry_query = nn.LayerNorm(self.dim)
        self.norm_geometry_lane = nn.LayerNorm(self.dim)
        self.norm_geometry_rows = nn.LayerNorm(self.dim)

        semantic_cfg = dict(semantic_context or {})
        self.semantic_enabled = bool(semantic_cfg.get("enabled", False))
        self.semantic_scales = tuple(
            str(scale) for scale in semantic_cfg.get("scales", ("p4", "p5"))
        )
        if self.semantic_enabled and not self.semantic_scales:
            raise ValueError("ownership semantic_context.scales must not be empty")
        pool_size = semantic_cfg.get("pool_size", (10, 25))
        if not isinstance(pool_size, Sequence) or len(pool_size) != 2:
            raise ValueError(
                "ownership semantic_context.pool_size must contain [height, width]"
            )
        self.semantic_pool_size = (int(pool_size[0]), int(pool_size[1]))
        if min(self.semantic_pool_size) < 1:
            raise ValueError("ownership semantic pool dimensions must be positive")

        self.semantic_position = (
            SinePositionEncoding2D(dim=self.dim)
            if self.semantic_enabled
            else None
        )
        # These adapters are ownership-only parameters.  Their inputs are
        # detached shared FPN features, so semantic ownership can learn without
        # moving the backbone/FPN representation used by bounded geometry.
        self.semantic_adapters = (
            nn.ModuleDict(
                {
                    scale: nn.Sequential(
                        nn.Conv2d(self.dim, self.dim, kernel_size=1),
                        nn.GroupNorm(8, self.dim),
                        nn.GELU(),
                    )
                    for scale in self.semantic_scales
                }
            )
            if self.semantic_enabled
            else nn.ModuleDict()
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
        if self.semantic_router is not None:
            nn.init.zeros_(self.semantic_router.weight)
            nn.init.zeros_(self.semantic_router.bias)

        self.norm_ffn = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), self.dim),
        )
        self.drop = nn.Dropout(float(dropout))

    @staticmethod
    def _validate_identity(
        state: torch.Tensor,
        query_identity: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError("ownership state must have shape [B,N,C]")
        if query_identity.ndim == 2:
            if query_identity.shape != state.shape[1:]:
                raise ValueError("ownership query identity must match [N,C]")
            return query_identity.unsqueeze(0).expand(state.shape[0], -1, -1)
        if query_identity.shape != state.shape:
            raise ValueError("ownership query identity must match [B,N,C]")
        return query_identity

    @staticmethod
    def _set_attention_by_group(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        value: torch.Tensor,
        group_sizes: tuple[int, ...] | None,
    ) -> torch.Tensor:
        if group_sizes is None or len(group_sizes) == 1:
            return attention(query, query, value, need_weights=False)[0]
        normalized = tuple(int(size) for size in group_sizes)
        if not normalized or any(size < 1 for size in normalized):
            raise ValueError("ownership group sizes must be positive")
        if sum(normalized) != int(query.shape[1]):
            raise ValueError("ownership group sizes must sum to candidate count")
        query_parts = torch.split(query, normalized, dim=1)
        value_parts = torch.split(value, normalized, dim=1)
        return torch.cat(
            [
                attention(q_part, q_part, v_part, need_weights=False)[0]
                for q_part, v_part in zip(query_parts, value_parts)
            ],
            dim=1,
        )

    def _semantic_context(
        self,
        state: torch.Tensor,
        identity: torch.Tensor,
        multi_scale_features: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.semantic_enabled:
            return state.new_zeros(state.shape)
        if multi_scale_features is None:
            raise ValueError(
                "protected ownership semantic context requires multi_scale_features"
            )
        assert self.semantic_position is not None
        assert self.semantic_scale_embedding is not None
        assert self.semantic_router is not None
        assert self.norm_semantic_query is not None

        query = self.norm_semantic_query(state) + identity
        contexts: list[torch.Tensor] = []
        for scale_index, scale in enumerate(self.semantic_scales):
            if scale not in multi_scale_features:
                raise KeyError(f"multi_scale_features is missing {scale!r}")
            raw_feature = multi_scale_features[scale]
            if raw_feature.ndim != 4 or int(raw_feature.shape[1]) != self.dim:
                raise ValueError(
                    f"ownership semantic feature {scale!r} must be "
                    f"[B,{self.dim},H,W], got {tuple(raw_feature.shape)}"
                )
            feature = self.semantic_adapters[scale](raw_feature.detach())
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
            dtype=state.dtype
        )
        return (stacked * weights.unsqueeze(-1)).sum(dim=2)

    def forward(
        self,
        state: torch.Tensor,
        query_identity: torch.Tensor,
        geometry_lane_state: torch.Tensor,
        row_states: torch.Tensor,
        *,
        multi_scale_features: Mapping[str, torch.Tensor] | None = None,
        group_sizes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if row_states.ndim != 4:
            raise ValueError("ownership row states must have shape [B,N,R,C]")
        batch, candidates, rows, channels = row_states.shape
        if state.shape != (batch, candidates, channels):
            raise ValueError("ownership and row-state shapes are incompatible")
        if geometry_lane_state.shape != state.shape:
            raise ValueError("ownership and geometry lane-state shapes are incompatible")
        if int(channels) != self.dim:
            raise ValueError("ownership state channel dimension is invalid")

        identity = self._validate_identity(state, query_identity).to(
            device=state.device,
            dtype=state.dtype,
        )
        normalized = self.norm_set(state)
        query_key = normalized + identity
        state = state + self.drop(
            self._set_attention_by_group(
                self.set_attention,
                query_key,
                normalized,
                group_sizes,
            )
        )

        # The detach calls live at the consumer boundary.  A future config
        # cannot accidentally reopen ownership -> geometry gradients merely by
        # passing a non-detached tensor from the caller.
        geometry_lane = self.norm_geometry_lane(geometry_lane_state.detach())
        geometry_rows = self.norm_geometry_rows(row_states.detach())
        memory = torch.cat((geometry_lane.unsqueeze(2), geometry_rows), dim=2)
        query = (self.norm_geometry_query(state) + identity).reshape(
            batch * candidates,
            1,
            channels,
        )
        memory = memory.reshape(batch * candidates, rows + 1, channels)
        state_flat = state.reshape(batch * candidates, 1, channels)
        state_flat = state_flat + self.drop(
            self.geometry_attention(
                query,
                memory,
                memory,
                need_weights=False,
            )[0]
        )
        state = state_flat.reshape(batch, candidates, channels)

        if self.semantic_enabled:
            state = state + self.drop(
                self._semantic_context(
                    state,
                    identity,
                    multi_scale_features,
                )
            )
        state = state + self.drop(self.ffn(self.norm_ffn(state)))
        return state.contiguous()
