from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    fixed_indices,
    fixed_linspace,
    fixed_row_fractions,
    fixed_sample_indices,
    soft_expected_x,
    sort_range_norm,
)
from .four_slot_selection import FourSlotLaneSelectionHead
from .unified_lane_set import ProtectedOwnershipLayer, UnifiedLaneSetLayer


class _ReuseFp32NchwFeatureMap(torch.autograd.Function):
    """Reuse one FP32 NCHW copy while preserving per-consumer BF16 gradients.

    The historical grid-sample path converted the same BHWC evidence tensor
    to FP32 NCHW independently in every decoder layer.  Reusing the ordinary
    conversion node would accumulate the four gradients in FP32 and cast only
    once, subtly changing AMP rounding.  This alias node instead casts each
    consumer's gradient back to the source dtype before normal autograd
    accumulation, exactly matching the old four-conversion graph.
    """

    @staticmethod
    def forward(
        ctx: Any,
        source_bhwc: torch.Tensor,
        cached_fp32_nchw: torch.Tensor,
    ) -> torch.Tensor:
        ctx.source_dtype = source_bhwc.dtype
        return cached_fp32_nchw

    @staticmethod
    def backward(
        ctx: Any,
        grad_nchw: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        grad_bhwc = grad_nchw.permute(0, 2, 3, 1).to(
            dtype=ctx.source_dtype
        )
        return grad_bhwc, None


def prepare_shared_grid_sample_feature_map(
    row_value_features: torch.Tensor,
) -> torch.Tensor:
    """Materialize the layer-invariant FP32 NCHW grid-sample input once."""

    with torch.autocast(
        device_type=row_value_features.device.type,
        enabled=False,
    ):
        return (
            row_value_features.detach()
            .float()
            .permute(0, 3, 1, 2)
            .contiguous()
        )


def reuse_shared_grid_sample_feature_map(
    row_value_features: torch.Tensor,
    cached_fp32_nchw: torch.Tensor,
) -> torch.Tensor:
    if row_value_features.requires_grad:
        return _ReuseFp32NchwFeatureMap.apply(
            row_value_features,
            cached_fp32_nchw,
        )
    return cached_fp32_nchw


def _heterogeneous_group_attention(
    attention: nn.MultiheadAttention,
    q: torch.Tensor,
    group_sizes: tuple[int, ...],
) -> torch.Tensor:
    """Apply self-attention independently to possibly unequal query groups.

    Groups with the same width are folded into the batch dimension so the
    32+8+8+8 hybrid layout needs only two attention calls, not four.  This
    preserves strict isolation between the inference group and train-only
    auxiliary groups without padding the small groups to 32 queries.
    """

    sizes = tuple(int(size) for size in group_sizes)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("group_sizes must contain positive integers")
    if sum(sizes) != int(q.shape[1]):
        raise ValueError(
            f"group_sizes sum to {sum(sizes)}, expected {int(q.shape[1])} queries"
        )
    if len(sizes) == 1:
        return attention(q, q, q, need_weights=False)[0]

    batch_rows = int(q.shape[0])
    chunks = list(torch.split(q, sizes, dim=1))
    outputs: list[torch.Tensor | None] = [None] * len(chunks)
    indices_by_size: dict[int, list[int]] = {}
    for index, size in enumerate(sizes):
        indices_by_size.setdefault(size, []).append(index)
    for size, indices in indices_by_size.items():
        folded = torch.cat([chunks[index] for index in indices], dim=0)
        folded_delta = attention(folded, folded, folded, need_weights=False)[0]
        for index, delta in zip(indices, folded_delta.split(batch_rows, dim=0)):
            if int(delta.shape[1]) != size:
                raise RuntimeError("grouped attention returned an unexpected width")
            outputs[index] = delta
    if any(output is None for output in outputs):
        raise RuntimeError("grouped attention failed to populate every group")
    return torch.cat([output for output in outputs if output is not None], dim=1)


class RowAwareCrossAttentionLayer(nn.Module):
    """Let lane-row tokens read row evidence and exchange structured context."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        num_groups: int = 1,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        if self.num_groups < 1:
            raise ValueError("structured_query.num_groups must be >= 1")
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.inter_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.intra_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_inter = nn.LayerNorm(dim)
        self.norm_intra = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def _grouped_inter_attention(
        self,
        q: torch.Tensor,
        batch_rows: int,
        num_instances: int,
        num_groups: int | None = None,
        group_sizes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if group_sizes is not None:
            if int(q.shape[0]) != int(batch_rows) or int(q.shape[1]) != int(num_instances):
                raise ValueError("grouped attention shape metadata does not match q")
            return _heterogeneous_group_attention(self.inter_attn, q, group_sizes)
        active_num_groups = self.num_groups if num_groups is None else int(num_groups)
        if active_num_groups < 1:
            raise ValueError("active num_groups must be >= 1")
        if active_num_groups == 1:
            return self.inter_attn(q, q, q, need_weights=False)[0]
        if num_instances % active_num_groups != 0:
            raise ValueError(
                f"num_instances={num_instances} must be divisible by active num_groups={active_num_groups}"
            )
        group_size = num_instances // active_num_groups
        grouped = q.view(batch_rows, active_num_groups, group_size, q.shape[-1])
        grouped = grouped.reshape(batch_rows * active_num_groups, group_size, q.shape[-1])
        delta = self.inter_attn(grouped, grouped, grouped, need_weights=False)[0]
        return delta.view(batch_rows, active_num_groups, group_size, q.shape[-1]).reshape(
            batch_rows, num_instances, q.shape[-1]
        )

    def forward(
        self,
        row_tokens: torch.Tensor,
        row_value_features: torch.Tensor,
        row_key_features: torch.Tensor,
        *,
        num_groups: int | None = None,
        group_sizes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        b, n, r, c = row_tokens.shape
        _, rv, x_bins, _ = row_value_features.shape
        _, rk, key_x_bins, _ = row_key_features.shape
        if rv != r or rk != r:
            raise ValueError(f"row_features has value/key rows {rv}/{rk}, expected {r}")
        if key_x_bins != x_bins:
            raise ValueError(f"row key/value x bins differ: {key_x_bins} vs {x_bins}")

        # Row-local cross-attention: each row sees only horizontal evidence from the same row.
        q = row_tokens.permute(0, 2, 1, 3).reshape(b * r, n, c)
        q_norm = self.norm_cross(q)
        key = row_key_features.reshape(b * r, x_bins, c)
        value = row_value_features.reshape(b * r, x_bins, c)
        q = q + self.drop(self.cross_attn(q_norm, key, value, need_weights=False)[0])

        # Group-isolated interaction avoids letting one-to-many training groups suppress each other.
        q_norm = self.norm_inter(q)
        q = q + self.drop(
            self._grouped_inter_attention(
                q_norm,
                batch_rows=b * r,
                num_instances=n,
                num_groups=num_groups,
                group_sizes=group_sizes,
            )
        )
        q = q.view(b, r, n, c).permute(0, 2, 1, 3).contiguous()

        # Vertical interaction lets rows of the same lane share continuity and curvature context.
        lane_rows = q.reshape(b * n, r, c)
        lane_rows_norm = self.norm_intra(lane_rows)
        lane_rows = lane_rows + self.drop(
            self.intra_attn(lane_rows_norm, lane_rows_norm, lane_rows_norm, need_weights=False)[0]
        )
        lane_rows_norm = self.norm_ffn(lane_rows)
        lane_rows = lane_rows + self.drop(self.ffn(lane_rows_norm))
        return lane_rows.view(b, n, r, c).contiguous()


class ReferenceGuidedRowLayer(nn.Module):
    """Refine lane-row states from P2 evidence sampled around an explicit curve.

    Unlike :class:`RowAwareCrossAttentionLayer`, this layer does not ask every
    row state to search the complete image row again.  It samples a small,
    differentiable horizontal profile around the current per-lane reference,
    lets the row state select evidence within that profile, and then applies
    the same inter-lane and intra-lane interactions as the original decoder.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        num_groups: int = 1,
        offsets_px: tuple[float, ...] = (-96.0, -48.0, -24.0, 0.0, 24.0, 48.0, 96.0),
        sampling_backend: str = "grid_sample",
        projection_backend: str = "separate",
        attention_backend: str = "materialized",
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.num_groups = int(num_groups)
        if self.num_groups < 1:
            raise ValueError("structured_query.num_groups must be >= 1")
        if self.dim % self.num_heads:
            raise ValueError("reference-guided attention requires dim divisible by num_heads")
        offsets = torch.tensor(tuple(float(value) for value in offsets_px), dtype=torch.float32)
        if offsets.ndim != 1 or int(offsets.numel()) < 3:
            raise ValueError("row_reference.offsets_px must contain at least three values")
        if not bool((offsets[1:] > offsets[:-1]).all()):
            raise ValueError("row_reference.offsets_px must be strictly increasing")
        if not bool((offsets < 0).any() and (offsets == 0).any() and (offsets > 0).any()):
            raise ValueError("row_reference.offsets_px must span negative, zero, and positive offsets")
        self.register_buffer("offsets_px", offsets, persistent=True)
        self.sampling_backend = str(sampling_backend).strip().lower()
        if self.sampling_backend not in {"grid_sample", "linear_gather"}:
            raise ValueError(
                "row_reference.sampling_backend must be grid_sample or "
                f"linear_gather, got {sampling_backend!r}"
            )
        self.projection_backend = str(projection_backend).strip().lower()
        if self.projection_backend not in {"separate", "fused"}:
            raise ValueError(
                "row_reference.projection_backend must be separate or "
                f"fused, got {projection_backend!r}"
            )
        self.attention_backend = str(attention_backend).strip().lower()
        if self.attention_backend not in {"materialized", "einsum"}:
            raise ValueError(
                "row_reference.attention_backend must be materialized or "
                f"einsum, got {attention_backend!r}"
            )

        self.local_query = nn.Linear(self.dim, self.dim, bias=False)
        self.local_key = nn.Linear(self.dim, self.dim, bias=False)
        self.local_value = nn.Linear(self.dim, self.dim, bias=False)
        self.local_out = nn.Linear(self.dim, self.dim)
        self.coordinate_proj = nn.Sequential(
            nn.Linear(2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.relative_offset_bias = nn.Parameter(
            torch.zeros(self.num_heads, int(offsets.numel()))
        )
        self.inter_attn = nn.MultiheadAttention(
            self.dim,
            self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.intra_attn = nn.MultiheadAttention(
            self.dim,
            self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, self.dim),
        )
        self.norm_cross = nn.LayerNorm(self.dim)
        self.norm_inter = nn.LayerNorm(self.dim)
        self.norm_intra = nn.LayerNorm(self.dim)
        self.norm_ffn = nn.LayerNorm(self.dim)
        self.drop = nn.Dropout(dropout)

    def _sample_local_profiles_grid(
        self,
        row_value_features: torch.Tensor,
        reference_x_rows: torch.Tensor,
        *,
        input_w: int,
        shared_feature_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorized bilinear sampling with an FP32 grid-sample island.

        CUDA ``grid_sample`` support for BF16 varies across the PyTorch
        versions used on the training servers.  Keeping only this operation in
        FP32 preserves gradients and avoids making the rest of the decoder
        leave autocast.
        """

        b, rows, _, channels = row_value_features.shape
        rb, instances, reference_rows = reference_x_rows.shape
        if rb != b or reference_rows != rows:
            raise ValueError(
                "reference_x_rows must match row evidence: "
                f"got {tuple(reference_x_rows.shape)} for "
                f"{tuple(row_value_features.shape)}"
            )
        device_type = row_value_features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            feature_map = (
                row_value_features.float()
                .permute(0, 3, 1, 2)
                .contiguous()
                if shared_feature_map is None
                else reuse_shared_grid_sample_feature_map(
                    row_value_features,
                    shared_feature_map,
                )
            )
            reference = reference_x_rows.float()
            offsets = self.offsets_px.to(device=reference.device, dtype=reference.dtype)
            sample_x = (reference.unsqueeze(-1) + offsets.view(1, 1, 1, -1)).clamp(
                min=0.0,
                max=float(max(int(input_w) - 1, 1)),
            )
            grid_x = 2.0 * sample_x / float(max(int(input_w) - 1, 1)) - 1.0
            grid_y = fixed_linspace(
                -1.0,
                1.0,
                rows,
                device=reference.device,
                dtype=reference.dtype,
            ).view(1, 1, rows, 1)
            grid_y = grid_y.expand(b, instances, rows, int(offsets.numel()))
            grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
                b,
                instances * rows,
                int(offsets.numel()),
                2,
            )
            sampled = F.grid_sample(
                feature_map,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled = sampled.permute(0, 2, 3, 1).reshape(
                b,
                instances,
                rows,
                int(offsets.numel()),
                channels,
            )
        return sampled.to(dtype=row_value_features.dtype)

    def _sample_local_profiles_linear(
        self,
        row_value_features: torch.Tensor,
        reference_x_rows: torch.Tensor,
        *,
        input_w: int,
    ) -> torch.Tensor:
        """Sample the same profiles without materializing an FP32 image map.

        ``row_value_features`` already has exactly the decoder's row count.
        The legacy grid therefore samples every y coordinate at an integer
        feature row and only interpolates horizontally.  Expressing that
        special case directly avoids, once per decoder block:

        * an FP32 copy of the complete P2 tensor;
        * an NCHW permutation/contiguous copy;
        * construction of a two-dimensional grid; and
        * a general two-dimensional ``grid_sample`` kernel.

        Coordinates are still computed in FP32.  Feature interpolation is
        performed in the AMP tensor dtype, matching the dtype returned by the
        legacy FP32 island while retaining the fast BF16 path.
        """

        b, rows, x_bins, channels = row_value_features.shape
        rb, instances, reference_rows = reference_x_rows.shape
        if rb != b or reference_rows != rows:
            raise ValueError(
                "reference_x_rows must match row evidence: "
                f"got {tuple(reference_x_rows.shape)} for "
                f"{tuple(row_value_features.shape)}"
            )
        device_type = row_value_features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            reference = reference_x_rows.float()
            offsets = self.offsets_px.to(device=reference.device, dtype=torch.float32)
            sample_x = (reference.unsqueeze(-1) + offsets.view(1, 1, 1, -1)).clamp(
                min=0.0,
                max=float(max(int(input_w) - 1, 1)),
            )
            feature_x = sample_x * float(max(x_bins - 1, 0)) / float(
                max(int(input_w) - 1, 1)
            )
            left = feature_x.floor().to(dtype=torch.long)
            right = (left + 1).clamp(max=max(x_bins - 1, 0))
            alpha = feature_x - left.to(dtype=feature_x.dtype)

        # Flatten batch and row together. Advanced indexing selects only the
        # requested N*K horizontal locations; unlike gather with an expanded
        # channel index, it does not materialize a large int64 index tensor.
        flat_features = row_value_features.reshape(b * rows, x_bins, channels)
        offsets_count = int(left.shape[-1])
        left = left.permute(0, 2, 1, 3).reshape(b * rows, instances * offsets_count)
        right = right.permute(0, 2, 1, 3).reshape(b * rows, instances * offsets_count)
        alpha = alpha.permute(0, 2, 1, 3).reshape(
            b * rows,
            instances * offsets_count,
            1,
        )
        row_index = fixed_indices(
            b * rows,
            device=row_value_features.device,
            dtype=torch.long,
        ).view(-1, 1)
        paired_index = torch.stack((left, right), dim=-1).reshape(
            b * rows,
            instances * offsets_count * 2,
        )
        paired_value = flat_features[row_index, paired_index].view(
            b * rows,
            instances * offsets_count,
            2,
            channels,
        )
        left_value = paired_value[:, :, 0]
        right_value = paired_value[:, :, 1]
        alpha = alpha.to(dtype=row_value_features.dtype)
        sampled = torch.lerp(left_value, right_value, alpha)
        return (
            sampled.view(b, rows, instances, offsets_count, channels)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def _sample_local_profiles(
        self,
        row_value_features: torch.Tensor,
        reference_x_rows: torch.Tensor,
        *,
        input_w: int,
        shared_feature_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sampling_backend == "linear_gather":
            return self._sample_local_profiles_linear(
                row_value_features,
                reference_x_rows,
                input_w=input_w,
            )
        return self._sample_local_profiles_grid(
            row_value_features,
            reference_x_rows,
            input_w=input_w,
            shared_feature_map=shared_feature_map,
        )

    def _grouped_inter_attention(
        self,
        q: torch.Tensor,
        batch_rows: int,
        num_instances: int,
        num_groups: int | None = None,
        group_sizes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if group_sizes is not None:
            if int(q.shape[0]) != int(batch_rows) or int(q.shape[1]) != int(num_instances):
                raise ValueError("grouped attention shape metadata does not match q")
            return _heterogeneous_group_attention(self.inter_attn, q, group_sizes)
        active_num_groups = self.num_groups if num_groups is None else int(num_groups)
        if active_num_groups < 1:
            raise ValueError("active num_groups must be >= 1")
        if active_num_groups == 1:
            return self.inter_attn(q, q, q, need_weights=False)[0]
        if num_instances % active_num_groups != 0:
            raise ValueError(
                f"num_instances={num_instances} must be divisible by "
                f"active num_groups={active_num_groups}"
            )
        group_size = num_instances // active_num_groups
        grouped = q.view(batch_rows, active_num_groups, group_size, q.shape[-1])
        grouped = grouped.reshape(batch_rows * active_num_groups, group_size, q.shape[-1])
        delta = self.inter_attn(grouped, grouped, grouped, need_weights=False)[0]
        return delta.view(batch_rows, active_num_groups, group_size, q.shape[-1]).reshape(
            batch_rows,
            num_instances,
            q.shape[-1],
        )

    def forward(
        self,
        row_tokens: torch.Tensor,
        row_value_features: torch.Tensor,
        reference_x_rows: torch.Tensor,
        *,
        input_w: int,
        num_groups: int | None = None,
        group_sizes: tuple[int, ...] | None = None,
        shared_grid_sample_feature_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, r, c = row_tokens.shape
        if c != self.dim:
            raise ValueError(f"row token dim {c} does not match layer dim {self.dim}")
        profiles = self._sample_local_profiles(
            row_value_features,
            reference_x_rows,
            input_w=int(input_w),
            shared_feature_map=shared_grid_sample_feature_map,
        )
        offsets = int(profiles.shape[3])
        head_dim = self.dim // self.num_heads

        query = self.local_query(self.norm_cross(row_tokens)).view(
            b,
            n,
            r,
            self.num_heads,
            head_dim,
        )
        if self.projection_backend == "fused":
            # Optional kernel experiment: retain original parameter names and
            # checkpoint layout while issuing one concatenated GEMM.
            key_value = F.linear(
                profiles,
                torch.cat((self.local_key.weight, self.local_value.weight), dim=0),
            )
            key, value = key_value.split(self.dim, dim=-1)
        else:
            # Numerically identical path used by the validated 10k gate.
            key = self.local_key(profiles)
            value = self.local_value(profiles)
        key = key.view(
            b,
            n,
            r,
            offsets,
            self.num_heads,
            head_dim,
        ).permute(0, 1, 2, 4, 3, 5)
        value = value.view(
            b,
            n,
            r,
            offsets,
            self.num_heads,
            head_dim,
        ).permute(0, 1, 2, 4, 3, 5)
        if self.attention_backend == "einsum":
            attention = torch.einsum(
                "bnrhd,bnrhkd->bnrhk",
                query,
                key,
            ) / math.sqrt(float(head_dim))
        else:
            attention = (query.unsqueeze(-2) * key).sum(dim=-1) / math.sqrt(
                float(head_dim)
            )
        attention = attention + self.relative_offset_bias.view(
            1,
            1,
            1,
            self.num_heads,
            offsets,
        )
        attention = torch.softmax(attention, dim=-1)
        if self.attention_backend == "einsum":
            context = torch.einsum(
                "bnrhk,bnrhkd->bnrhd",
                attention,
                value,
            ).reshape(b, n, r, c)
        else:
            context = (attention.unsqueeze(-1) * value).sum(dim=-2).reshape(
                b,
                n,
                r,
                c,
            )

        x_norm = 2.0 * reference_x_rows.to(dtype=row_tokens.dtype) / float(
            max(int(input_w) - 1, 1)
        ) - 1.0
        y_norm = fixed_linspace(
            -1.0,
            1.0,
            r,
            device=row_tokens.device,
            dtype=row_tokens.dtype,
        ).view(1, 1, r).expand(b, n, r)
        coordinate_code = self.coordinate_proj(torch.stack((x_norm, y_norm), dim=-1))
        row_tokens = row_tokens + self.drop(self.local_out(context) + coordinate_code)

        # Same-row competition and same-lane vertical continuity match the
        # original structured decoder; only visual acquisition changes.
        q = row_tokens.permute(0, 2, 1, 3).reshape(b * r, n, c)
        q_norm = self.norm_inter(q)
        q = q + self.drop(
            self._grouped_inter_attention(
                q_norm,
                batch_rows=b * r,
                num_instances=n,
                num_groups=num_groups,
                group_sizes=group_sizes,
            )
        )
        q = q.view(b, r, n, c).permute(0, 2, 1, 3).contiguous()
        lane_rows = q.reshape(b * n, r, c)
        lane_rows_norm = self.norm_intra(lane_rows)
        lane_rows = lane_rows + self.drop(
            self.intra_attn(
                lane_rows_norm,
                lane_rows_norm,
                lane_rows_norm,
                need_weights=False,
            )[0]
        )
        lane_rows_norm = self.norm_ffn(lane_rows)
        lane_rows = lane_rows + self.drop(self.ffn(lane_rows_norm))
        return lane_rows.view(b, n, r, c).contiguous()


class PersistentLaneStateLayer(nn.Module):
    """Update one lane-level state from that query's explicit row states.

    The row decoder remains responsible for ordered geometry.  This module
    gives existence and visible-range prediction a persistent state with the
    *same query identity* instead of rebuilding a lane descriptor by pooling
    rows independently at every decoder block.  Each lane state can only read
    the rows belonging to that lane; inter-lane reasoning stays in the row
    decoder where the assignment groups are already enforced.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.norm_lane = nn.LayerNorm(dim)
        self.norm_rows = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        lane_state: torch.Tensor,
        row_states: torch.Tensor,
    ) -> torch.Tensor:
        if lane_state.ndim != 3 or row_states.ndim != 4:
            raise ValueError(
                "lane_state/row_states must have shapes [B,N,C]/[B,N,R,C]"
            )
        b, n, c = lane_state.shape
        if row_states.shape[:2] != (b, n) or int(row_states.shape[-1]) != c:
            raise ValueError("lane and row states must share batch/query/channel axes")

        query = self.norm_lane(lane_state).reshape(b * n, 1, c)
        rows = self.norm_rows(row_states).reshape(
            b * n,
            int(row_states.shape[2]),
            c,
        )
        state = lane_state.reshape(b * n, 1, c)
        state = state + self.drop(
            self.cross_attn(query, rows, rows, need_weights=False)[0]
        )
        state = state + self.drop(self.ffn(self.norm_ffn(state)))
        return state.reshape(b, n, c).contiguous()


class RelationAwareCandidateBlock(nn.Module):
    """Candidate self-attention with an explicit per-pair curve bias.

    A generic set transformer must infer geometric duplication indirectly from
    compressed unary descriptors.  This block instead receives a detached
    ``[B, N, N, C_rel]`` relation tensor and maps it to one additive bias per
    attention head.  Values remain candidate states, so the relation tensor
    informs set comparison without becoming a second deployment score.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        relation_dim: int,
        relation_hidden_dim: int,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("relation attention hidden_dim must divide num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.norm_attention = nn.LayerNorm(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.relation_bias = nn.Sequential(
            nn.LayerNorm(int(relation_dim)),
            nn.Linear(int(relation_dim), int(relation_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(relation_hidden_dim), self.num_heads),
        )
        self.attention_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_dropout = nn.Dropout(float(dropout))
        self.norm_ffn = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )
        self.ffn_dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        hidden: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or relations.ndim != 4:
            raise ValueError(
                "relation attention expects [B,N,C] states and [B,N,N,C_rel] relations"
            )
        batch, candidates, channels = hidden.shape
        if int(channels) != self.hidden_dim:
            raise ValueError("candidate hidden dimension does not match relation block")
        if relations.shape[:3] != (batch, candidates, candidates):
            raise ValueError("pairwise relation axes do not match candidate states")

        normalized = self.norm_attention(hidden)
        qkv = self.qkv(normalized).view(
            batch,
            candidates,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        content_logits = torch.matmul(
            query.float(),
            key.float().transpose(-2, -1),
        ) / math.sqrt(float(self.head_dim))
        relation_logits = self.relation_bias(relations.float()).permute(
            0, 3, 1, 2
        )
        attention = torch.softmax(content_logits + relation_logits, dim=-1).to(
            dtype=value.dtype
        )
        attended = torch.matmul(attention, value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch,
            candidates,
            self.hidden_dim,
        )
        hidden = hidden + self.attention_dropout(
            self.attention_output(attended)
        )
        hidden = hidden + self.ffn_dropout(self.ffn(self.norm_ffn(hidden)))
        return hidden


class RelationAwareCandidateEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        relation_dim: int,
        relation_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelationAwareCandidateBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    relation_dim=relation_dim,
                    relation_hidden_dim=relation_hidden_dim,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, relations)
        return hidden


class SetAwareLaneSelectionHead(nn.Module):
    """Permutation-equivariant scorer over the complete lane proposal set.

    Legacy mode remains a zero-initialized residual on top of the deployed
    existence-quality score for checkpoint-compatible diagnostics. Unified
    mode is a standalone score: it sees the exact final curve distribution,
    visible row states, curve-aligned P2 evidence, and the other candidates.
    It never consumes or multiplies the legacy existence/quality scores.
    """

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        curve_samples: int = 20,
        base_quality_power: float = 0.5,
        range_temperature: float = 0.02,
        unified_score: bool = False,
        prior_prob: float = 0.05,
        detach_geometry_features: bool = True,
        use_curve_evidence: bool = False,
        use_semantic_decision: bool = False,
        candidate_interaction: str = "transformer",
        relation_sigma_px: float = 20.0,
        relation_hidden_dim: int = 32,
        pointer_max_selections: int = 4,
        pointer_min_valid_rows: int = 5,
        pointer_similarity_prior: float = 0.5,
        pointer_teacher_mode: str = "fixed_sequence",
        row_grid_mode: str = "legacy_linspace",
        pointer_quality_policy_mode: str = "shared",
        pointer_quality_prior_max_scale: float = 2.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.input_w = int(input_w)
        self.curve_samples = int(curve_samples)
        self.base_quality_power = float(base_quality_power)
        self.range_temperature = float(range_temperature)
        self.unified_score = bool(unified_score)
        self.prior_prob = float(prior_prob)
        self.detach_geometry_features = bool(detach_geometry_features)
        self.use_curve_evidence = bool(use_curve_evidence)
        self.use_semantic_decision = bool(use_semantic_decision)
        self.candidate_interaction = str(candidate_interaction).strip().lower()
        self.relation_sigma_px = float(relation_sigma_px)
        self.relation_dim = 6
        self.pointer_max_selections = int(pointer_max_selections)
        self.pointer_min_valid_rows = int(pointer_min_valid_rows)
        self.pointer_similarity_prior = float(pointer_similarity_prior)
        self.pointer_teacher_mode = str(pointer_teacher_mode).strip().lower()
        self.row_grid_mode = str(row_grid_mode).strip().lower()
        self.pointer_quality_policy_mode = str(
            pointer_quality_policy_mode
        ).strip().lower()
        self.pointer_quality_prior_max_scale = float(
            pointer_quality_prior_max_scale
        )
        # Diagnostics may opt in to the compact tensors needed to rerun only
        # the lightweight pointer decoder while retaining the normal
        # ``inference_only`` detector path.  This is runtime state, not a
        # checkpointed parameter or a deployment output by default.
        self.retain_pointer_diagnostic_tensors = False
        if self.curve_samples < 1:
            raise ValueError("set_selection.curve_samples must be positive")
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("set_selection.hidden_dim must be divisible by num_heads")
        if self.base_quality_power < 0.0:
            raise ValueError("set_selection.base_quality_power must be non-negative")
        if self.range_temperature <= 0.0:
            raise ValueError("set_selection.range_temperature must be positive")
        if not 0.0 < self.prior_prob < 1.0:
            raise ValueError("set_selection.prior_prob must be between zero and one")
        if self.relation_sigma_px <= 0.0:
            raise ValueError("set_selection.relation_sigma_px must be positive")
        if self.pointer_max_selections < 1:
            raise ValueError("set_selection.pointer_max_selections must be positive")
        if self.pointer_min_valid_rows < 1:
            raise ValueError("set_selection.pointer_min_valid_rows must be positive")
        if self.pointer_similarity_prior < 0.0:
            raise ValueError("set_selection.pointer_similarity_prior must be non-negative")
        if self.pointer_teacher_mode not in {
            "fixed_sequence",
            "permutation_invariant_set",
            "cluster_soft_randomized",
            "cluster_soft_remaining_mixture",
        }:
            raise ValueError(
                "set_selection.pointer_teacher_mode must be fixed_sequence "
                "or permutation_invariant_set, cluster_soft_randomized, "
                "or cluster_soft_remaining_mixture"
            )
        if self.row_grid_mode not in {"legacy_linspace", "fixed_rows"}:
            raise ValueError(
                "set_selection.row_grid_mode must be legacy_linspace or fixed_rows"
            )
        if self.pointer_quality_policy_mode not in {"shared", "decoupled"}:
            raise ValueError(
                "set_selection.pointer_quality_policy_mode must be shared "
                "or decoupled"
            )
        if self.pointer_quality_prior_max_scale <= 0.0:
            raise ValueError(
                "set_selection.pointer_quality_prior_max_scale must be positive"
            )
        if self.candidate_interaction not in {
            "independent",
            "transformer",
            "relation_transformer",
            "sequential_pointer",
        }:
            raise ValueError(
                "set_selection.candidate_interaction must be independent, "
                "transformer, relation_transformer, or sequential_pointer"
            )
        if (
            self.pointer_quality_policy_mode == "decoupled"
            and self.candidate_interaction != "sequential_pointer"
        ):
            raise ValueError(
                "decoupled pointer quality/policy requires "
                "candidate_interaction=sequential_pointer"
            )

        # Unified mode uses two range-masked row-state summaries and optional
        # exact final-curve P2 evidence. Legacy mode preserves the historical
        # lane-query + visible-row input and includes its base score scalar.
        state_streams = (
            2
            + int(self.unified_score and self.use_curve_evidence)
            + int(self.unified_score and self.use_semantic_decision)
        )
        scalar_count = 10 + int(not self.unified_score)
        input_dim = (
            state_streams * self.dim
            + scalar_count
            + 2 * self.curve_samples
        )
        self.input_norm = nn.LayerNorm(input_dim)
        # Preserve the historical projection keys so old selector checkpoints
        # remain loadable.  Only the independent diagnostic arm adds a
        # candidate-local FFN after this shared projection.
        self.input_projection = nn.Linear(input_dim, int(hidden_dim))
        if self.candidate_interaction == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(hidden_dim),
                nhead=int(num_heads),
                dim_feedforward=int(ff_dim),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(num_layers),
                enable_nested_tensor=False,
            )
            self.independent_ffn = nn.Identity()
        elif self.candidate_interaction in {
            "relation_transformer",
            "sequential_pointer",
        }:
            self.encoder = RelationAwareCandidateEncoder(
                int(hidden_dim),
                num_layers=int(num_layers),
                num_heads=int(num_heads),
                ff_dim=int(ff_dim),
                dropout=float(dropout),
                relation_dim=self.relation_dim,
                relation_hidden_dim=int(relation_hidden_dim),
            )
            self.independent_ffn = nn.Identity()
        else:
            # The independent arm deliberately shares the complete descriptor
            # and output contract with the set arm, but has no candidate-axis
            # communication.  This makes the 2x2 score gate a clean test of
            # trainable set comparison rather than a feature ablation.
            self.encoder = nn.Identity()
            self.independent_ffn = nn.Sequential(
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(ff_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(ff_dim), int(hidden_dim)),
            )
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)
        if self.candidate_interaction == "sequential_pointer":
            pointer_dim = int(hidden_dim)
            if self.pointer_quality_policy_mode == "decoupled":
                # V4.5.1 separates candidate-local localization quality from
                # the selected-set policy.  The residual quality adapter is
                # initialized as an exact identity so a V4.5 checkpoint can
                # enter this contract without changing its deployed logits.
                self.pointer_quality_adapter = nn.Sequential(
                    nn.LayerNorm(pointer_dim),
                    nn.Linear(pointer_dim, pointer_dim),
                    nn.GELU(),
                    nn.Linear(pointer_dim, pointer_dim),
                )
                self.pointer_policy_output_norm = nn.LayerNorm(pointer_dim)
                self.pointer_policy_output = nn.Linear(pointer_dim, 1)
                self.pointer_quality_scale_raw = nn.Parameter(
                    torch.full(
                        (self.pointer_max_selections,),
                        self._quality_scale_raw(1.0),
                    )
                )
                nn.init.zeros_(self.pointer_quality_adapter[-1].weight)
                nn.init.zeros_(self.pointer_quality_adapter[-1].bias)
                nn.init.zeros_(self.pointer_policy_output.weight)
                nn.init.zeros_(self.pointer_policy_output.bias)
            else:
                self.pointer_quality_adapter = None
                self.pointer_policy_output_norm = None
                self.pointer_policy_output = None
                self.register_parameter("pointer_quality_scale_raw", None)
            self.pointer_key = nn.Linear(pointer_dim, pointer_dim, bias=False)
            self.pointer_context = nn.Linear(pointer_dim, pointer_dim)
            self.pointer_step_embedding = nn.Embedding(
                self.pointer_max_selections,
                pointer_dim,
            )
            self.pointer_query = nn.Linear(2 * pointer_dim, pointer_dim)
            self.pointer_state_update = nn.GRUCell(pointer_dim, pointer_dim)
            self.pointer_stop_embedding = nn.Parameter(
                torch.randn(pointer_dim) * 0.02
            )
            self.pointer_stop = nn.Sequential(
                nn.LayerNorm(3 * pointer_dim),
                nn.Linear(3 * pointer_dim, pointer_dim),
                nn.GELU(),
                nn.Linear(pointer_dim, 1),
            )
            self.pointer_relation_bias = nn.Sequential(
                nn.LayerNorm(self.relation_dim),
                nn.Linear(self.relation_dim, int(relation_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(relation_hidden_dim), 1),
            )
            self.pointer_similarity_scale_raw = nn.Parameter(
                torch.log(
                    torch.expm1(
                        torch.tensor(max(self.pointer_similarity_prior, 1e-4))
                    )
                )
            )
            self.pointer_scale = float(pointer_dim) ** -0.5
            nn.init.zeros_(self.pointer_relation_bias[-1].weight)
            nn.init.zeros_(self.pointer_relation_bias[-1].bias)
            nn.init.zeros_(self.pointer_stop[-1].weight)
            # A new selector should acquire lanes before learning to stop.
            nn.init.constant_(self.pointer_stop[-1].bias, -2.0)
        else:
            self.pointer_quality_adapter = None
            self.pointer_policy_output_norm = None
            self.pointer_policy_output = None
            self.register_parameter("pointer_quality_scale_raw", None)
            self.pointer_key = None
            self.pointer_context = None
            self.pointer_step_embedding = None
            self.pointer_query = None
            self.pointer_state_update = None
            self.register_parameter("pointer_stop_embedding", None)
            self.pointer_stop = None
            self.pointer_relation_bias = None
            self.register_parameter("pointer_similarity_scale_raw", None)
            self.pointer_scale = 1.0
        if self.unified_score:
            nn.init.normal_(self.output.weight, std=0.01)
            nn.init.constant_(
                self.output.bias,
                math.log(self.prior_prob / (1.0 - self.prior_prob)),
            )
        else:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def _quality_scale_raw(self, scale: float) -> float:
        """Map a bounded positive quality scale to its unconstrained value."""

        ratio = float(scale) / self.pointer_quality_prior_max_scale
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        return math.log(ratio / (1.0 - ratio))

    def pointer_quality_scale(self) -> torch.Tensor:
        """Return the bounded per-step localization-quality prior scale."""

        if self.pointer_quality_scale_raw is None:
            return self.output.weight.new_ones((self.pointer_max_selections,))
        return self.pointer_quality_prior_max_scale * torch.sigmoid(
            self.pointer_quality_scale_raw
        )

    def _base_probability(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        exist = torch.softmax(outputs["exist_logits"].float(), dim=-1)[..., 0]
        quality_logits = outputs.get("quality_logits")
        if self.base_quality_power > 0.0 and quality_logits is not None:
            quality = torch.sigmoid(quality_logits.float()).clamp_min(1e-6)
            exist = exist * quality.pow(self.base_quality_power)
        return exist.clamp(1e-6, 1.0 - 1e-6)

    def _lane_row_grid(
        self,
        rows: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the normalized lane-row coordinates used by range masks."""

        if self.row_grid_mode == "fixed_rows":
            return fixed_row_fractions(rows, device=device, dtype=dtype)
        return fixed_linspace(0.0, 1.0, rows, device=device, dtype=dtype)

    def build_selection_features(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build the frozen proposal descriptors consumed by the set scorer.

        Keeping feature construction separate from scoring lets diagnostics
        freeze the detector (including curve-aligned evidence extraction) and
        train only the selection transformer.  The normal forward path still
        calls this method, so the refactor does not change deployed scores.
        """
        observe = (
            (lambda value: value.detach())
            if self.detach_geometry_features
            else (lambda value: value)
        )
        row_tokens = observe(outputs["structured_row_tokens"])
        lane_query = observe(outputs["queries"])
        ranges = sort_range_norm(observe(outputs["range_norm"]).float())
        pred_x = observe(outputs["pred_x_rows"]).float()
        row_logits = observe(outputs["row_x_logits"]).float()
        batch, candidates, rows, _channels = row_tokens.shape

        y_norm = self._lane_row_grid(
            rows,
            device=row_tokens.device,
            dtype=torch.float32,
        ).view(1, 1, rows)
        temperature = max(self.range_temperature, 1e-4)
        row_weight = torch.sigmoid((y_norm - ranges[..., :1]) / temperature)
        row_weight = row_weight * torch.sigmoid(
            (ranges[..., 1:] - y_norm) / temperature
        )
        denominator = row_weight.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        row_weight_state = row_weight.to(dtype=row_tokens.dtype)
        denominator_state = denominator.to(dtype=row_tokens.dtype)
        visible_row_state = (
            row_tokens * row_weight_state.unsqueeze(-1)
        ).sum(dim=2) / denominator_state
        centered_state = row_tokens - visible_row_state.unsqueeze(2)
        state_variance = (
            centered_state.float().square().mean(dim=-1) * row_weight
        ).sum(dim=-1, keepdim=True) / denominator

        log_max_probability = row_logits.amax(dim=-1) - torch.logsumexp(
            row_logits,
            dim=-1,
        )
        row_confidence = log_max_probability.exp()
        confidence_row_weight = row_weight * row_confidence
        confidence_denominator = confidence_row_weight.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-4)
        confidence_weighted_state = (
            row_tokens
            * confidence_row_weight.to(dtype=row_tokens.dtype).unsqueeze(-1)
        ).sum(dim=2) / confidence_denominator.to(dtype=row_tokens.dtype)
        confidence_mean = (
            row_confidence * row_weight
        ).sum(dim=-1, keepdim=True) / denominator
        confidence_max = row_confidence.amax(dim=-1, keepdim=True)

        pred_x_norm = pred_x / float(max(self.input_w - 1, 1))
        first_difference = (
            pred_x_norm[..., 1:] - pred_x_norm[..., :-1]
        ).abs()
        first_weight = row_weight[..., 1:] * row_weight[..., :-1]
        slope = (
            first_difference * first_weight
        ).sum(dim=-1, keepdim=True) / first_weight.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-4)
        second_difference = (
            pred_x_norm[..., 2:]
            - 2.0 * pred_x_norm[..., 1:-1]
            + pred_x_norm[..., :-2]
        ).abs()
        second_weight = (
            row_weight[..., 2:]
            * row_weight[..., 1:-1]
            * row_weight[..., :-2]
        )
        curvature = (
            second_difference * second_weight
        ).sum(dim=-1, keepdim=True) / second_weight.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-4)

        reference = outputs.get("input_reference_x_rows")
        if reference is None:
            reference_mean = pred_x.new_zeros((batch, candidates, 1))
            reference_max = pred_x.new_zeros((batch, candidates, 1))
        else:
            reference_delta = (
                pred_x - observe(reference).float()
            ).abs() / float(max(self.input_w - 1, 1))
            reference_mean = (
                reference_delta * row_weight
            ).sum(dim=-1, keepdim=True) / denominator
            reference_max = reference_delta.amax(dim=-1, keepdim=True)

        sample_count = min(self.curve_samples, rows)
        sample_ids = fixed_sample_indices(
            rows,
            sample_count,
            device=pred_x.device,
        )
        sampled_x = pred_x_norm.index_select(-1, sample_ids)
        sampled_confidence = row_confidence.index_select(-1, sample_ids)
        if sample_count < self.curve_samples:
            padding = self.curve_samples - sample_count
            sampled_x = F.pad(sampled_x, (0, padding))
            sampled_confidence = F.pad(sampled_confidence, (0, padding))

        scalar_parts = [
            ranges,
            ranges[..., 1:] - ranges[..., :1],
            confidence_mean,
            confidence_max,
            slope,
            curvature,
            reference_mean,
            reference_max,
            state_variance,
        ]
        if self.unified_score:
            state_parts = [visible_row_state, confidence_weighted_state]
            if self.use_semantic_decision:
                decision_query = outputs.get("decision_queries")
                if not isinstance(decision_query, torch.Tensor):
                    raise ValueError(
                        "unified set selection with semantic decision requires "
                        "decision_queries"
                    )
                # ``decision_queries`` is produced by score-only semantic
                # adapters whose lane/FPN inputs are detached by V4.  Keeping
                # this stream live lets score supervision train those adapters
                # without reopening a path into geometry.
                state_parts.append(decision_query)
            if self.use_curve_evidence:
                curve_evidence = outputs.get("selection_curve_evidence")
                if not isinstance(curve_evidence, torch.Tensor):
                    raise ValueError(
                        "unified set selection with curve evidence requires "
                        "selection_curve_evidence"
                    )
                curve_state = (
                    observe(curve_evidence)
                    * row_weight.to(dtype=curve_evidence.dtype).unsqueeze(-1)
                ).sum(dim=2) / denominator.to(dtype=curve_evidence.dtype)
                state_parts.append(curve_state)
            scalar_parts.extend((sampled_x, sampled_confidence))
            scalar_features = torch.cat(scalar_parts, dim=-1).to(
                dtype=row_tokens.dtype
            )
            features = torch.cat((*state_parts, scalar_features), dim=-1)
        else:
            base_probability = self._base_probability(outputs).detach()
            scalar_parts.extend(
                (base_probability.unsqueeze(-1), sampled_x, sampled_confidence)
            )
            scalar_features = torch.cat(scalar_parts, dim=-1).to(
                dtype=lane_query.dtype
            )
            features = torch.cat(
                (lane_query, visible_row_state, scalar_features),
                dim=-1,
            )
        return features

    def build_pairwise_relations(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build detached, symmetric curve relations for every candidate pair.

        Channels are mean/top/bottom normalized curve distance, common visible
        row overlap, range IoU, and a soft strip-similarity term.  These are
        precisely the geometric facts used by the successful cached MMR
        counterfactual, but remain soft and learnable inside attention.
        """

        pred_x = outputs["pred_x_rows"].detach().float()
        ranges = sort_range_norm(outputs["range_norm"].detach().float())
        if pred_x.ndim != 3 or ranges.shape != pred_x.shape[:2] + (2,):
            raise ValueError("selection curve/range tensors have incompatible shapes")
        _batch, _candidates, rows = pred_x.shape
        y_norm = self._lane_row_grid(
            rows,
            device=pred_x.device,
            dtype=pred_x.dtype,
        ).view(1, 1, rows)
        temperature = max(self.range_temperature, 1e-4)
        visible = torch.sigmoid((y_norm - ranges[..., :1]) / temperature)
        visible = visible * torch.sigmoid(
            (ranges[..., 1:] - y_norm) / temperature
        )

        visible_i = visible.unsqueeze(2)
        visible_k = visible.unsqueeze(1)
        common = visible_i * visible_k
        common_count = common.sum(dim=-1).clamp_min(1e-4)
        union = visible_i + visible_k - common
        visible_overlap = common.sum(dim=-1) / union.sum(dim=-1).clamp_min(1e-4)

        distance_px = (
            pred_x.unsqueeze(2) - pred_x.unsqueeze(1)
        ).abs()
        distance_norm = distance_px / float(max(self.input_w - 1, 1))
        mean_distance = (distance_norm * common).sum(dim=-1) / common_count
        top_weight = common * (1.0 - y_norm.view(1, 1, 1, rows))
        bottom_weight = common * y_norm.view(1, 1, 1, rows)
        top_distance = (distance_norm * top_weight).sum(dim=-1) / top_weight.sum(
            dim=-1
        ).clamp_min(1e-4)
        bottom_distance = (
            distance_norm * bottom_weight
        ).sum(dim=-1) / bottom_weight.sum(dim=-1).clamp_min(1e-4)

        range_start = torch.maximum(
            ranges[..., 0].unsqueeze(2),
            ranges[..., 0].unsqueeze(1),
        )
        range_end = torch.minimum(
            ranges[..., 1].unsqueeze(2),
            ranges[..., 1].unsqueeze(1),
        )
        range_intersection = (range_end - range_start).clamp_min(0.0)
        range_union = torch.maximum(
            ranges[..., 1].unsqueeze(2),
            ranges[..., 1].unsqueeze(1),
        ) - torch.minimum(
            ranges[..., 0].unsqueeze(2),
            ranges[..., 0].unsqueeze(1),
        )
        range_iou = range_intersection / range_union.clamp_min(1e-4)
        strip_similarity = (
            torch.exp(-distance_px / self.relation_sigma_px) * common
        ).sum(dim=-1) / common_count
        return torch.stack(
            (
                mean_distance,
                top_distance,
                bottom_distance,
                visible_overlap,
                range_iou,
                strip_similarity,
            ),
            dim=-1,
        ).detach()

    def encode_selection_features(
        self,
        features: torch.Tensor,
        relations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode detached proposal descriptors before deployment decisions."""

        hidden = self.input_projection(self.input_norm(features))
        if self.candidate_interaction == "independent":
            hidden = hidden + self.independent_ffn(hidden)
        if self.candidate_interaction in {
            "relation_transformer",
            "sequential_pointer",
        }:
            if relations is None:
                raise ValueError(
                    f"{self.candidate_interaction} requires pairwise relations"
                )
            hidden = self.encoder(hidden, relations)
        else:
            hidden = self.encoder(hidden)
        return hidden

    def score_selection_features(
        self,
        features: torch.Tensor,
        relations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score descriptors returned by :meth:`build_selection_features`."""

        hidden = self.encode_selection_features(features, relations)
        if self.pointer_quality_policy_mode == "decoupled":
            if self.pointer_quality_adapter is None:
                raise RuntimeError("decoupled quality adapter was not initialized")
            quality_input = hidden.detach()
            hidden = quality_input + self.pointer_quality_adapter(quality_input)
        return self.output(self.output_norm(hidden)).squeeze(-1).float()

    def build_pointer_candidate_valid(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"].detach()
        ranges = sort_range_norm(outputs["range_norm"].detach().float())
        rows = int(pred_x.shape[-1])
        # Keep this grid bit-for-bit consistent with
        # pairwise_range_aware_row_strip_iou.  ``fixed_y_rows`` places row r
        # at r * input_h / rows, hence its normalized coordinate is r / rows
        # (the last row is *not* 1.0).  ``linspace(0, 1, rows)`` previously
        # disagreed at range boundaries and could make a Hungarian target
        # valid for the target builder but invalid for the pointer decoder.
        y_norm = fixed_row_fractions(
            rows,
            device=pred_x.device,
            dtype=ranges.dtype,
        ).view(1, 1, rows)
        visible = (
            (y_norm >= ranges[..., :1])
            & (y_norm <= ranges[..., 1:])
            & torch.isfinite(pred_x)
        )
        return visible.sum(dim=-1) >= self.pointer_min_valid_rows

    def decode_pointer(
        self,
        candidate_hidden: torch.Tensor,
        relations: torch.Tensor,
        unary_logits: torch.Tensor,
        candidate_valid: torch.Tensor,
        *,
        policy_logits: torch.Tensor | None = None,
        teacher_indices: torch.Tensor | None = None,
        teacher_candidate_mask: torch.Tensor | None = None,
        forced_actions: torch.Tensor | None = None,
        force_candidate_after_stop: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Autoregressively select lanes without replacement, or emit STOP.

        Candidate geometry and relations have already crossed a stop-gradient
        boundary.  Each decision observes the candidates selected by previous
        steps, which gives the learned selector the conditional suppression
        behavior that parallel scalar Top-K cannot represent.
        """

        if self.candidate_interaction != "sequential_pointer":
            raise RuntimeError("pointer decoding requires sequential_pointer mode")
        if any(
            module is None
            for module in (
                self.pointer_key,
                self.pointer_context,
                self.pointer_step_embedding,
                self.pointer_query,
                self.pointer_state_update,
                self.pointer_stop,
                self.pointer_relation_bias,
            )
        ):
            raise RuntimeError("pointer modules were not initialized")
        batch, candidates, hidden_dim = candidate_hidden.shape
        if relations.shape[:3] != (batch, candidates, candidates):
            raise ValueError("pointer relation axes do not match candidates")
        if unary_logits.shape != (batch, candidates):
            raise ValueError("pointer unary-logit shape mismatch")
        if policy_logits is not None and policy_logits.shape != (
            batch,
            candidates,
        ):
            raise ValueError("pointer policy-logit shape mismatch")
        if (
            self.pointer_quality_policy_mode == "decoupled"
            and policy_logits is None
        ):
            raise ValueError("decoupled pointer requires policy logits")
        if candidate_valid.shape != (batch, candidates):
            raise ValueError("pointer validity-mask shape mismatch")
        if teacher_indices is not None and teacher_indices.shape != (
            batch,
            self.pointer_max_selections,
        ):
            raise ValueError("pointer teacher target shape mismatch")
        if teacher_candidate_mask is not None and teacher_candidate_mask.shape != (
            batch,
            candidates,
        ):
            raise ValueError("pointer teacher candidate-mask shape mismatch")
        if teacher_indices is not None and teacher_candidate_mask is not None:
            raise ValueError("pointer accepts only one teacher contract at a time")
        if forced_actions is not None and forced_actions.shape != (
            batch,
            self.pointer_max_selections,
        ):
            raise ValueError("pointer forced-action shape mismatch")
        if forced_actions is not None and (
            teacher_indices is not None or teacher_candidate_mask is not None
        ):
            raise ValueError("pointer forced actions are inference-only")
        if force_candidate_after_stop and (
            teacher_indices is not None or teacher_candidate_mask is not None
        ):
            raise ValueError("forced STOP continuation is inference-only")
        if forced_actions is not None and bool(
            ((forced_actions < -1) | (forced_actions > candidates)).any()
        ):
            raise ValueError("pointer forced action is outside the class range")
        if teacher_candidate_mask is not None and bool(
            (teacher_candidate_mask.bool() & ~candidate_valid.bool()).any()
        ):
            raise ValueError("pointer set teacher contains an invalid candidate")

        valid_float = candidate_valid.to(candidate_hidden.dtype).unsqueeze(-1)
        context = (candidate_hidden * valid_float).sum(dim=1)
        context = context / valid_float.sum(dim=1).clamp_min(1.0)
        state = torch.tanh(self.pointer_context(context))
        keys = self.pointer_key(candidate_hidden)
        available = candidate_valid.bool().clone()
        selected_mask = torch.zeros_like(candidate_valid, dtype=torch.bool)
        stopped = torch.zeros(batch, dtype=torch.bool, device=candidate_hidden.device)
        remaining_teacher = (
            teacher_candidate_mask.to(
                device=candidate_hidden.device,
                dtype=torch.bool,
            ).clone()
            if teacher_candidate_mask is not None
            else None
        )
        teacher_stopped = torch.zeros_like(stopped)
        batch_ids = torch.arange(batch, device=candidate_hidden.device)
        logits_by_step: list[torch.Tensor] = []
        selected_by_step: list[torch.Tensor] = []
        selected_probability_by_step: list[torch.Tensor] = []
        relation_bias_by_step: list[torch.Tensor] = []
        unary_component_by_step: list[torch.Tensor] = []
        policy_component_by_step: list[torch.Tensor] = []
        content_by_step: list[torch.Tensor] = []
        stop_logit_by_step: list[torch.Tensor] = []
        stop_would_win_by_step: list[torch.Tensor] = []
        stop_margin_by_step: list[torch.Tensor] = []
        teacher_class_masks_by_step: list[torch.Tensor] = []
        teacher_active_by_step: list[torch.Tensor] = []

        for step in range(self.pointer_max_selections):
            step_token = self.pointer_step_embedding.weight[step].view(
                1,
                hidden_dim,
            ).expand(batch, -1)
            query = self.pointer_query(torch.cat((state, step_token), dim=-1))
            content = torch.einsum("bh,bnh->bn", query, keys)
            content = content.float() * self.pointer_scale

            selected_float = selected_mask.to(relations.dtype)
            selected_count = selected_float.sum(dim=-1, keepdim=True)
            relation_summary = torch.einsum(
                "bikc,bk->bic",
                relations,
                selected_float,
            ) / selected_count.clamp_min(1.0).unsqueeze(-1)
            learned_relation_bias = self.pointer_relation_bias(
                relation_summary
            ).squeeze(-1).float()
            # The sixth relation channel is the same soft strip similarity
            # that made the frozen MMR counterfactual succeed.  Its positive,
            # learnable scale supplies a structural duplicate penalty from
            # the first update rather than asking a dot-product attention
            # layer to rediscover curve distance from scratch.
            selected_similarity = torch.einsum(
                "bik,bk->bi",
                relations[..., 5],
                selected_float,
            ) / selected_count.clamp_min(1.0)
            similarity_scale = F.softplus(self.pointer_similarity_scale_raw)
            relation_bias = learned_relation_bias - (
                similarity_scale.float() * selected_similarity.float()
            )
            relation_bias = torch.where(
                selected_count > 0,
                relation_bias,
                torch.zeros_like(relation_bias),
            )
            if self.pointer_quality_policy_mode == "decoupled":
                quality_scale = self.pointer_quality_scale()[step].float()
                unary_component = quality_scale * unary_logits.detach().float()
                policy_component = policy_logits.float()
            else:
                unary_component = unary_logits.float()
                policy_component = torch.zeros_like(unary_component)
            candidate_logits = (
                unary_component + policy_component + content + relation_bias
            )
            candidate_logits = candidate_logits.masked_fill(~available, -1e4)
            stop_input = torch.cat((state, context, step_token), dim=-1)
            stop_logit = self.pointer_stop(stop_input).squeeze(-1).float()
            step_logits = torch.cat(
                (candidate_logits, stop_logit.unsqueeze(-1)),
                dim=-1,
            )
            logits_by_step.append(step_logits)
            relation_bias_by_step.append(relation_bias)
            unary_component_by_step.append(unary_component)
            policy_component_by_step.append(policy_component)
            content_by_step.append(content)
            stop_logit_by_step.append(stop_logit)
            best_candidate_logit = candidate_logits.max(dim=-1).values
            # STOP is the final class, so torch.argmax resolves an exact tie
            # in favor of the lower-index candidate class.
            stop_would_win = stop_logit > best_candidate_logit
            stop_would_win_by_step.append(stop_would_win)
            stop_margin_by_step.append(stop_logit - best_candidate_logit)

            if remaining_teacher is not None:
                teacher_active = ~teacher_stopped
                has_remaining = remaining_teacher.any(dim=-1)
                teacher_class_mask = torch.zeros(
                    (batch, candidates + 1),
                    dtype=torch.bool,
                    device=step_logits.device,
                )
                teacher_class_mask[:, :candidates] = (
                    remaining_teacher & teacher_active.unsqueeze(-1)
                )
                teacher_class_mask[:, candidates] = teacher_active & ~has_remaining
                teacher_choice_logits = candidate_logits.masked_fill(
                    ~remaining_teacher,
                    -1e4,
                )
                chosen_candidate = teacher_choice_logits.argmax(dim=-1)
                chosen = torch.where(
                    has_remaining,
                    chosen_candidate,
                    torch.full_like(chosen_candidate, candidates),
                )
                chosen = torch.where(
                    teacher_active,
                    chosen,
                    torch.full_like(chosen, candidates),
                )
                teacher_class_masks_by_step.append(teacher_class_mask)
                teacher_active_by_step.append(teacher_active)
                teacher_stopped = teacher_stopped | (teacher_active & ~has_remaining)
            elif teacher_indices is None:
                chosen = step_logits.argmax(dim=-1)
                if force_candidate_after_stop:
                    has_available = available.any(dim=-1)
                    best_candidate = candidate_logits.argmax(dim=-1)
                    chosen = torch.where(
                        (chosen == candidates) & has_available,
                        best_candidate,
                        chosen,
                    )
                if forced_actions is not None:
                    forced = forced_actions[:, step].to(
                        device=chosen.device,
                        dtype=torch.long,
                    )
                    force_mask = forced >= 0
                    forced_candidate = force_mask & (forced < candidates)
                    if bool(
                        (
                            forced_candidate
                            & ~available.gather(
                                1,
                                forced.clamp(min=0, max=candidates - 1).unsqueeze(-1),
                            ).squeeze(-1)
                        ).any()
                    ):
                        raise ValueError(
                            "pointer forced an unavailable or invalid candidate"
                        )
                    chosen = torch.where(force_mask, forced, chosen)
                chosen = torch.where(
                    stopped,
                    torch.full_like(chosen, candidates),
                    chosen,
                )
            else:
                target = teacher_indices[:, step]
                # Ignored positions occur only after the supervised STOP.
                chosen = torch.where(
                    target >= 0,
                    target,
                    torch.full_like(target, candidates),
                )
            chose_stop = chosen == candidates
            chose_candidate = (chosen >= 0) & (chosen < candidates) & ~stopped
            safe_candidate = chosen.clamp(min=0, max=max(candidates - 1, 0))
            chosen_state = candidate_hidden[batch_ids, safe_candidate]
            update_input = torch.where(
                chose_candidate.unsqueeze(-1),
                chosen_state,
                self.pointer_stop_embedding.view(1, -1),
            )
            updated_state = self.pointer_state_update(update_input, state)
            state = torch.where(stopped.unsqueeze(-1), state, updated_state)

            chosen_probability = torch.softmax(step_logits, dim=-1).gather(
                1,
                chosen.clamp(min=0, max=candidates).unsqueeze(-1),
            ).squeeze(-1)
            emitted = torch.where(
                chose_candidate,
                chosen,
                torch.full_like(chosen, -1),
            )
            selected_by_step.append(emitted)
            selected_probability_by_step.append(
                torch.where(
                    chose_candidate,
                    chosen_probability,
                    torch.zeros_like(chosen_probability),
                )
            )
            if bool(chose_candidate.any()):
                rows = torch.nonzero(chose_candidate, as_tuple=False).flatten()
                ids = safe_candidate[rows]
                selected_mask[rows, ids] = True
                available[rows, ids] = False
                if remaining_teacher is not None:
                    remaining_teacher[rows, ids] = False
            stopped = stopped | chose_stop

        result = {
            "selection_pointer_logits": torch.stack(logits_by_step, dim=1),
            "selection_pointer_indices": torch.stack(selected_by_step, dim=1),
            "selection_pointer_scores": torch.stack(
                selected_probability_by_step,
                dim=1,
            ),
            "selection_pointer_relation_bias": torch.stack(
                relation_bias_by_step,
                dim=1,
            ),
            "selection_pointer_unary_component": torch.stack(
                unary_component_by_step,
                dim=1,
            ),
            "selection_pointer_policy_component": torch.stack(
                policy_component_by_step,
                dim=1,
            ),
            "selection_pointer_content_component": torch.stack(
                content_by_step,
                dim=1,
            ),
            "selection_pointer_stop_component": torch.stack(
                stop_logit_by_step,
                dim=1,
            ),
            "selection_pointer_stop_would_win": torch.stack(
                stop_would_win_by_step,
                dim=1,
            ),
            "selection_pointer_stop_margin": torch.stack(
                stop_margin_by_step,
                dim=1,
            ),
            "selection_pointer_quality_scale": self.pointer_quality_scale(),
        }
        if teacher_class_masks_by_step:
            result["selection_pointer_teacher_class_mask"] = torch.stack(
                teacher_class_masks_by_step,
                dim=1,
            )
            result["selection_pointer_teacher_active"] = torch.stack(
                teacher_active_by_step,
                dim=1,
            )
        return result

    def reroll_pointer_with_teacher(
        self,
        outputs: dict[str, torch.Tensor],
        teacher_indices: torch.Tensor,
    ) -> None:
        """Replace training pointer logits with a teacher-forced rollout."""

        required = (
            "_selection_pointer_hidden",
            "_selection_pointer_relations",
            "_selection_pointer_unary_logits",
            "_selection_pointer_candidate_valid",
        )
        if self.pointer_quality_policy_mode == "decoupled":
            required = (*required, "_selection_pointer_policy_logits")
        missing = [name for name in required if name not in outputs]
        if missing:
            raise ValueError(
                "pointer teacher forcing is missing forward tensors: "
                + ", ".join(missing)
            )
        teacher = teacher_indices.to(
            device=outputs["_selection_pointer_hidden"].device,
            dtype=torch.long,
        )
        candidates = int(outputs["_selection_pointer_hidden"].shape[1])
        if self.pointer_teacher_mode == "permutation_invariant_set":
            target_mask = torch.zeros(
                (teacher.shape[0], candidates),
                dtype=torch.bool,
                device=teacher.device,
            )
            valid = (teacher >= 0) & (teacher < candidates)
            rows, steps = torch.nonzero(valid, as_tuple=True)
            if rows.numel() > 0:
                target_mask[rows, teacher[rows, steps]] = True
            rollout = self.decode_pointer(
                outputs["_selection_pointer_hidden"],
                outputs["_selection_pointer_relations"],
                outputs["_selection_pointer_unary_logits"],
                outputs["_selection_pointer_candidate_valid"],
                policy_logits=outputs.get("_selection_pointer_policy_logits"),
                teacher_candidate_mask=target_mask,
            )
            outputs["selection_pointer_unique_target_mask"] = target_mask
            outputs["selection_pointer_teacher_class_mask"] = rollout[
                "selection_pointer_teacher_class_mask"
            ]
            outputs["selection_pointer_teacher_active"] = rollout[
                "selection_pointer_teacher_active"
            ]
        else:
            rollout = self.decode_pointer(
                outputs["_selection_pointer_hidden"],
                outputs["_selection_pointer_relations"],
                outputs["_selection_pointer_unary_logits"],
                outputs["_selection_pointer_candidate_valid"],
                policy_logits=outputs.get("_selection_pointer_policy_logits"),
                teacher_indices=teacher,
            )
        # Keep greedy indices/scores from the original forward for diagnostics;
        # only the differentiable step logits must follow the training target.
        outputs["selection_pointer_logits"] = rollout[
            "selection_pointer_logits"
        ]
        outputs["selection_pointer_teacher_indices"] = teacher
        outputs["selection_pointer_teacher_relation_bias"] = rollout[
            "selection_pointer_relation_bias"
        ]
        for name in (
            "selection_pointer_unary_component",
            "selection_pointer_policy_component",
            "selection_pointer_content_component",
            "selection_pointer_stop_component",
            "selection_pointer_quality_scale",
        ):
            outputs[name] = rollout[name]

    def reroll_pointer_with_cluster_teacher(
        self,
        outputs: dict[str, torch.Tensor],
        teacher: dict[str, torch.Tensor],
    ) -> None:
        """Run a soft-cluster policy on sampled, GT-valid teacher prefixes."""

        if self.pointer_teacher_mode not in {
            "cluster_soft_randomized",
            "cluster_soft_remaining_mixture",
        }:
            raise ValueError(
                "cluster-soft teacher requires pointer_teacher_mode="
                "cluster_soft_randomized or cluster_soft_remaining_mixture"
            )
        indices = teacher.get("indices")
        probabilities = teacher.get("probabilities")
        active = teacher.get("active")
        if not isinstance(indices, torch.Tensor):
            raise ValueError("cluster teacher indices are missing")
        if not isinstance(probabilities, torch.Tensor):
            raise ValueError("cluster teacher probabilities are missing")
        if not isinstance(active, torch.Tensor):
            raise ValueError("cluster teacher active mask is missing")
        candidates = int(outputs["_selection_pointer_hidden"].shape[1])
        expected = (
            indices.shape[0],
            self.pointer_max_selections,
            candidates + 1,
        )
        if probabilities.shape != expected:
            raise ValueError("cluster teacher probability shape mismatch")
        if active.shape != indices.shape:
            raise ValueError("cluster teacher active-mask shape mismatch")
        candidate_valid = outputs["_selection_pointer_candidate_valid"].bool()
        for row in range(int(indices.shape[0])):
            emitted = indices[row][
                (indices[row] >= 0) & (indices[row] < candidates)
            ]
            if emitted.numel() != emitted.unique().numel():
                raise ValueError("cluster teacher repeats a candidate")
            if emitted.numel() > 0 and not bool(
                candidate_valid[row, emitted].all()
            ):
                raise ValueError("cluster teacher contains an invalid candidate")

        self.reroll_pointer_with_teacher(outputs, indices)
        outputs["selection_pointer_teacher_probabilities"] = probabilities.to(
            device=outputs["selection_pointer_logits"].device,
            dtype=outputs["selection_pointer_logits"].dtype,
        )
        outputs["selection_pointer_teacher_active"] = active.to(
            device=outputs["selection_pointer_logits"].device,
            dtype=torch.bool,
        )
        for source_name, output_name in (
            ("candidate_steps", "selection_pointer_teacher_candidate_steps"),
            ("support_sizes", "selection_pointer_teacher_support_sizes"),
            ("target_entropy", "selection_pointer_teacher_target_entropy"),
            ("target_quality", "selection_pointer_teacher_target_quality"),
            (
                "remaining_cluster_count",
                "selection_pointer_teacher_remaining_cluster_count",
            ),
            (
                "representable_count",
                "selection_pointer_teacher_representable_count",
            ),
            ("fallback_count", "selection_pointer_teacher_fallback_count"),
            (
                "reservation_exclusion_count",
                "selection_pointer_teacher_reservation_exclusion_count",
            ),
        ):
            value = teacher.get(source_name)
            if isinstance(value, torch.Tensor):
                outputs[output_name] = value.to(
                    device=outputs["selection_pointer_logits"].device
                )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        features = self.build_selection_features(outputs)
        relations = (
            self.build_pairwise_relations(outputs)
            if self.candidate_interaction in {
                "relation_transformer",
                "sequential_pointer",
            }
            else None
        )
        hidden = self.encode_selection_features(features, relations)
        if self.pointer_quality_policy_mode == "decoupled":
            if (
                self.pointer_quality_adapter is None
                or self.pointer_policy_output_norm is None
                or self.pointer_policy_output is None
            ):
                raise RuntimeError("decoupled pointer branches were not initialized")
            quality_input = hidden.detach()
            quality_hidden = quality_input + self.pointer_quality_adapter(
                quality_input
            )
            raw_logits = self.output(
                self.output_norm(quality_hidden)
            ).squeeze(-1).float()
            policy_logits = self.pointer_policy_output(
                self.pointer_policy_output_norm(hidden)
            ).squeeze(-1).float()
        else:
            raw_logits = self.output(self.output_norm(hidden)).squeeze(-1).float()
            policy_logits = None
        if self.candidate_interaction == "sequential_pointer":
            if relations is None:
                raise RuntimeError("sequential pointer has no relation tensor")
            candidate_valid = self.build_pointer_candidate_valid(outputs)
            # Training targets depend on the frozen final curves.  Defer the
            # lightweight rollout until forward_with_matches has constructed
            # those targets; otherwise we would retain an unused greedy graph
            # and then build a second teacher-forced graph for every batch.
            pointer = (
                {}
                if self.training
                else self.decode_pointer(
                    hidden,
                    relations,
                    raw_logits,
                    candidate_valid,
                    policy_logits=policy_logits,
                )
            )
            return {
                "selection_logits": raw_logits,
                "selection_delta_logits": torch.zeros_like(raw_logits),
                **pointer,
                # Private differentiable transport used only to rerun the
                # lightweight pointer decoder after geometry-only targets are
                # known.  These tensors are removed from inference outputs.
                "_selection_pointer_hidden": hidden,
                "_selection_pointer_relations": relations,
                "_selection_pointer_unary_logits": raw_logits,
                **(
                    {"_selection_pointer_policy_logits": policy_logits}
                    if policy_logits is not None
                    else {}
                ),
                "_selection_pointer_candidate_valid": candidate_valid,
            }
        if self.unified_score:
            # Keep the legacy diagnostic key in the output contract, but make
            # its value explicit: unified mode has no residual/delta path.
            return raw_logits, torch.zeros_like(raw_logits)
        base_probability = self._base_probability(outputs).detach()
        base_logits = torch.logit(base_probability)
        return base_logits + raw_logits, raw_logits


class StructuredLaneQueryHead(nn.Module):
    """Instance-geometry S0 head with row-wise image evidence.

    The head keeps the DynLaneSeq output contract but replaces a single slot
    vector with a lane instance token plus per-row geometry tokens.
    """

    def __init__(
        self,
        dim: int = 256,
        num_instances: int = 64,
        num_rows: int = 72,
        x_bins: int = 200,
        input_w: int = 800,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        use_x_pos: bool = True,
        evidence_x_bins: int | None = None,
        num_groups: int = 1,
        exist_prior_prob: float | None = None,
        intermediate_supervision: bool = False,
        inference_group_index: int | None = None,
        training_auxiliary_group_sizes: list[int] | tuple[int, ...] | None = None,
        row_reference: dict[str, Any] | None = None,
        lane_state: dict[str, Any] | None = None,
        ownership: dict[str, Any] | None = None,
        set_selection: dict[str, Any] | None = None,
        lane_pooling: str = "mean_max",
    ):
        super().__init__()
        self.dim = int(dim)
        self.primary_num_instances = int(num_instances)
        self.training_auxiliary_group_sizes = tuple(
            int(size) for size in (training_auxiliary_group_sizes or ())
        )
        if any(size < 1 for size in self.training_auxiliary_group_sizes):
            raise ValueError(
                "structured_query.training_auxiliary_group_sizes must contain positive integers"
            )
        self.num_instances = self.primary_num_instances + sum(
            self.training_auxiliary_group_sizes
        )
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.evidence_x_bins = int(evidence_x_bins) if evidence_x_bins is not None else self.x_bins
        self.input_w = int(input_w)
        self.use_x_pos = bool(use_x_pos)
        legacy_num_groups = int(num_groups)
        if self.training_auxiliary_group_sizes:
            if legacy_num_groups != 1:
                raise ValueError(
                    "training_auxiliary_group_sizes requires num_groups=1 for the primary group"
                )
            if inference_group_index is not None:
                raise ValueError(
                    "training_auxiliary_group_sizes already fixes inference to the primary group"
                )
            self.interaction_group_sizes = (
                self.primary_num_instances,
                *self.training_auxiliary_group_sizes,
            )
            self.num_groups = len(self.interaction_group_sizes)
        else:
            self.num_groups = legacy_num_groups
            if self.num_groups < 1:
                raise ValueError("structured_query.num_groups must be >= 1")
            if self.num_instances % self.num_groups != 0:
                raise ValueError(
                    f"structured_query.num_instances={self.num_instances} must be divisible by num_groups={self.num_groups}"
                )
            group_size = self.num_instances // self.num_groups
            self.interaction_group_sizes = tuple(
                group_size for _ in range(self.num_groups)
            )
        self.exist_prior_prob = None if exist_prior_prob is None else float(exist_prior_prob)
        self.intermediate_supervision = bool(intermediate_supervision)
        self.inference_group_index = None if inference_group_index is None else int(inference_group_index)
        self.row_reference_cfg = dict(row_reference or {})
        self.row_reference_enabled = bool(self.row_reference_cfg.get("enabled", False))
        self.row_reference_prediction_mode = str(
            self.row_reference_cfg.get("prediction_mode", "absolute")
        ).strip().lower()
        if self.row_reference_prediction_mode not in {"absolute", "bounded_delta"}:
            raise ValueError(
                "structured_query.row_reference.prediction_mode must be "
                "absolute or bounded_delta"
            )
        if (
            self.row_reference_prediction_mode == "bounded_delta"
            and not self.row_reference_enabled
        ):
            raise ValueError("bounded_delta prediction requires row_reference.enabled=true")
        self.detach_reference_between_layers = bool(
            self.row_reference_cfg.get("detach_between_layers", False)
        )
        self.lane_state_cfg = dict(lane_state or {})
        self.lane_state_enabled = bool(self.lane_state_cfg.get("enabled", False))
        self.lane_state_mode = str(
            self.lane_state_cfg.get("mode", "read_only")
        ).strip().lower()
        if self.lane_state_mode not in {"read_only", "causal_set"}:
            raise ValueError(
                "structured_query.lane_state.mode must be read_only or causal_set"
            )
        if self.lane_state_mode == "causal_set" and not self.lane_state_enabled:
            raise ValueError("causal_set lane state requires lane_state.enabled=true")
        if self.lane_state_mode == "causal_set" and (
            self.num_groups != 1 or self.training_auxiliary_group_sizes
        ):
            raise ValueError(
                "causal_set lane state requires one deployable query set and "
                "does not permit train-only query groups"
            )
        self.single_logit_score = bool(
            self.lane_state_cfg.get("single_logit_score", False)
        )
        self.detach_score_geometry = bool(
            self.lane_state_cfg.get("detach_score_geometry", False)
        )
        self.ownership_cfg = dict(ownership or {})
        self.ownership_enabled = bool(self.ownership_cfg.get("enabled", False))
        self.ownership_retain_diagnostic_tensors = bool(
            self.ownership_cfg.get("retain_diagnostic_tensors", False)
        )
        if self.ownership_enabled and not self.lane_state_enabled:
            raise ValueError(
                "protected ownership requires a persistent geometry lane state"
            )
        if self.ownership_enabled and not self.detach_score_geometry:
            raise ValueError(
                "protected ownership requires lane_state.detach_score_geometry=true"
            )
        if self.ownership_enabled and not bool(
            self.ownership_cfg.get("detach_geometry_inputs", True)
        ):
            raise ValueError(
                "protected ownership does not permit differentiable geometry inputs"
            )
        if self.ownership_enabled and (
            self.num_groups != 1 or self.training_auxiliary_group_sizes
        ):
            raise ValueError(
                "protected ownership requires one deployable query set and "
                "does not permit train-only query groups"
            )
        self.set_selection_cfg = dict(set_selection or {})
        self.set_selection_enabled = bool(self.set_selection_cfg.get("enabled", False))
        self.lane_pooling = str(lane_pooling).strip().lower()
        if self.lane_pooling not in {"mean_max", "mean"}:
            raise ValueError(
                "structured_query.lane_pooling must be mean_max or mean"
            )
        if self.evidence_x_bins < 1:
            raise ValueError("structured_query.evidence_x_bins must be >= 1")
        if self.inference_group_index is not None and not 0 <= self.inference_group_index < self.num_groups:
            raise ValueError(
                "structured_query.inference_group_index must be in "
                f"[0, {self.num_groups - 1}], got {self.inference_group_index}"
            )
        if self.exist_prior_prob is not None and not 0.0 < self.exist_prior_prob < 1.0:
            raise ValueError("structured_query.exist_prior_prob must be between 0 and 1")

        # Keep the deployable embedding identical to the 32-query control.
        # Train-only embeddings are initialized after every shared module so
        # their extra RNG draws cannot perturb the control parameter seed.
        self.instance_tokens = nn.Embedding(self.primary_num_instances, self.dim)
        self.row_tokens = nn.Embedding(self.num_rows, self.dim)
        self.x_tokens = nn.Embedding(self.evidence_x_bins, self.dim) if self.use_x_pos else None
        nn.init.normal_(self.instance_tokens.weight, std=0.02)
        nn.init.normal_(self.row_tokens.weight, std=0.02)
        if self.x_tokens is not None:
            nn.init.normal_(self.x_tokens.weight, std=0.02)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.GroupNorm(8, self.dim),
            nn.GELU(),
        )
        if self.row_reference_enabled:
            offsets_px = tuple(
                float(value)
                for value in self.row_reference_cfg.get(
                    "offsets_px",
                    [-96.0, -48.0, -24.0, 0.0, 24.0, 48.0, 96.0],
                )
            )
            self.layers = nn.ModuleList(
                [
                    ReferenceGuidedRowLayer(
                        dim=self.dim,
                        num_heads=int(num_heads),
                        ff_dim=int(ff_dim),
                        dropout=float(dropout),
                        num_groups=self.num_groups,
                        offsets_px=offsets_px,
                        sampling_backend=str(
                            self.row_reference_cfg.get(
                                "sampling_backend",
                                "grid_sample",
                            )
                        ),
                        projection_backend=str(
                            self.row_reference_cfg.get(
                                "projection_backend",
                                "separate",
                            )
                        ),
                        attention_backend=str(
                            self.row_reference_cfg.get(
                                "attention_backend",
                                "materialized",
                            )
                        ),
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.reference_query_norm = nn.LayerNorm(self.dim)
            self.reference_query = nn.Linear(self.dim, self.dim, bias=False)
            self.reference_key = nn.Linear(self.dim, self.dim, bias=False)
            self.reference_context = nn.Linear(self.dim, self.dim)
            self.reference_coordinate = nn.Sequential(
                nn.Linear(2, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )
            visual_scale = float(self.row_reference_cfg.get("visual_logit_scale", 8.0))
            if visual_scale <= 0.0:
                raise ValueError("row_reference.visual_logit_scale must be positive")
            self.reference_logit_scale = nn.Parameter(
                torch.tensor(math.log(visual_scale), dtype=torch.float32)
            )
            self.initial_prior_sigma_px = float(
                self.row_reference_cfg.get("initial_prior_sigma_px", 240.0)
            )
            self.initial_prior_strength = float(
                self.row_reference_cfg.get("initial_prior_strength", 1.0)
            )
            self.output_prior_sigma_px = float(
                self.row_reference_cfg.get("output_prior_sigma_px", 96.0)
            )
            self.output_prior_strength = float(
                self.row_reference_cfg.get("output_prior_strength", 1.5)
            )
            if self.initial_prior_sigma_px <= 0.0 or self.output_prior_sigma_px <= 0.0:
                raise ValueError("row-reference prior sigmas must be positive")
            if self.initial_prior_strength < 0.0 or self.output_prior_strength < 0.0:
                raise ValueError("row-reference prior strengths must be non-negative")
            bottom = torch.linspace(
                0.08,
                0.92,
                self.primary_num_instances,
                dtype=torch.float32,
            )
            row_fraction = torch.linspace(0.0, 1.0, self.num_rows, dtype=torch.float32)
            centers = 0.5 + (bottom[:, None] - 0.5) * (
                0.35 + 0.65 * row_fraction[None, :]
            )
            self.reference_anchor_logits = nn.Parameter(
                torch.logit(centers.clamp(1e-4, 1.0 - 1e-4))
            )
            if self.training_auxiliary_group_sizes:
                auxiliary_bottom = torch.cat(
                    [
                        torch.linspace(
                            0.08,
                            0.92,
                            group_size,
                            dtype=torch.float32,
                        )
                        for group_size in self.training_auxiliary_group_sizes
                    ],
                    dim=0,
                )
                auxiliary_centers = 0.5 + (auxiliary_bottom[:, None] - 0.5) * (
                    0.35 + 0.65 * row_fraction[None, :]
                )
                self.training_auxiliary_reference_anchor_logits = nn.Parameter(
                    torch.logit(
                        auxiliary_centers.clamp(1e-4, 1.0 - 1e-4)
                    )
                )
            else:
                self.training_auxiliary_reference_anchor_logits = None
        else:
            self.layers = nn.ModuleList(
                [
                    RowAwareCrossAttentionLayer(
                        dim=self.dim,
                        num_heads=int(num_heads),
                        ff_dim=int(ff_dim),
                        dropout=float(dropout),
                        num_groups=self.num_groups,
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.reference_query_norm = None
            self.reference_query = None
            self.reference_key = None
            self.reference_context = None
            self.reference_coordinate = None
            self.reference_logit_scale = None
            self.reference_anchor_logits = None
            self.training_auxiliary_reference_anchor_logits = None
        self.lane_state_layers = nn.ModuleList(
            [
                (
                    UnifiedLaneSetLayer(
                        dim=self.dim,
                        num_heads=int(
                            self.lane_state_cfg.get("num_heads", num_heads)
                        ),
                        ff_dim=int(self.lane_state_cfg.get("ff_dim", ff_dim)),
                        dropout=float(
                            self.lane_state_cfg.get("dropout", dropout)
                        ),
                        semantic_context=self.lane_state_cfg.get(
                            "semantic_context"
                        ),
                    )
                    if self.lane_state_mode == "causal_set"
                    else PersistentLaneStateLayer(
                        dim=self.dim,
                        num_heads=int(
                            self.lane_state_cfg.get("num_heads", num_heads)
                        ),
                        ff_dim=int(self.lane_state_cfg.get("ff_dim", ff_dim)),
                        dropout=float(
                            self.lane_state_cfg.get("dropout", dropout)
                        ),
                    )
                )
                for _ in range(int(num_layers))
            ]
            if self.lane_state_enabled
            else []
        )
        if self.row_reference_prediction_mode == "bounded_delta":
            raw_delta_offsets = self.row_reference_cfg.get(
                "delta_offsets_px",
                self.row_reference_cfg.get(
                    "offsets_px",
                    [-96.0, -48.0, -24.0, 0.0, 24.0, 48.0, 96.0],
                ),
            )
            delta_offsets = tuple(float(value) for value in raw_delta_offsets)
            if len(delta_offsets) < 2:
                raise ValueError("bounded_delta requires at least two delta offsets")
            if any(
                right <= left
                for left, right in zip(delta_offsets[:-1], delta_offsets[1:])
            ):
                raise ValueError("bounded_delta offsets must be strictly increasing")
            if delta_offsets[0] >= 0.0 or delta_offsets[-1] <= 0.0:
                raise ValueError("bounded_delta offsets must span negative and positive motion")
            if abs(sum(delta_offsets) / float(len(delta_offsets))) > 1e-6:
                raise ValueError(
                    "bounded_delta offsets must have zero mean for identity initialization"
                )
            self.row_delta_norms = nn.ModuleList(
                [
                    nn.LayerNorm(self.dim, elementwise_affine=False)
                    for _ in range(int(num_layers))
                ]
            )
            self.row_delta_heads = nn.ModuleList(
                [
                    nn.Linear(self.dim, len(delta_offsets), bias=False)
                    for _ in range(int(num_layers))
                ]
            )
            for head in self.row_delta_heads:
                # The image-grounded full-row acquisition is a safe initial
                # curve.  Zero logits make every local block start as an exact
                # identity update while preserving non-zero DFL gradients for
                # learning the required direction.
                nn.init.zeros_(head.weight)
            self.register_buffer(
                "row_delta_offsets_px",
                torch.tensor(delta_offsets, dtype=torch.float32),
            )
            self.row_delta_min_px = float(delta_offsets[0])
            self.row_delta_max_px = float(delta_offsets[-1])
            self.row_norm = None
            self.row_x = None
        else:
            self.row_delta_norms = nn.ModuleList()
            self.row_delta_heads = nn.ModuleList()
            self.register_buffer(
                "row_delta_offsets_px",
                torch.empty(0, dtype=torch.float32),
            )
            self.row_delta_min_px = 0.0
            self.row_delta_max_px = 0.0
            self.row_norm = nn.LayerNorm(self.dim)
            self.row_x = nn.Linear(self.dim, self.x_bins)
        self.lane_norm = nn.LayerNorm(self.dim)
        self.decision_norm = (
            nn.LayerNorm(self.dim) if self.detach_score_geometry else None
        )
        self.exist = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 1 if self.single_logit_score else 2),
        )
        self.range = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 2))
        self.quality = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 1))
        selection_interaction = str(
            self.set_selection_cfg.get("candidate_interaction", "transformer")
        ).strip().lower()
        if self.set_selection_enabled and selection_interaction == "four_slot":
            self.set_selection_head = FourSlotLaneSelectionHead(
                self.dim,
                input_w=self.input_w,
                hidden_dim=int(self.set_selection_cfg.get("hidden_dim", self.dim)),
                num_slots=int(
                    self.set_selection_cfg.get("four_slot_num_slots", 4)
                ),
                proposal_layers=int(
                    self.set_selection_cfg.get("num_layers", 2)
                ),
                slot_layers=int(
                    self.set_selection_cfg.get("four_slot_num_layers", 2)
                ),
                num_heads=int(
                    self.set_selection_cfg.get("num_heads", num_heads)
                ),
                ff_dim=int(
                    self.set_selection_cfg.get("ff_dim", 2 * self.dim)
                ),
                dropout=float(
                    self.set_selection_cfg.get("dropout", dropout)
                ),
                curve_samples=int(
                    self.set_selection_cfg.get("curve_samples", 20)
                ),
                range_temperature=float(
                    self.set_selection_cfg.get("range_temperature", 0.02)
                ),
                min_valid_rows=int(
                    self.set_selection_cfg.get("four_slot_min_valid_rows", 5)
                ),
                refinement_enabled=bool(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_enabled",
                        False,
                    )
                ),
                refinement_hidden_dim=int(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_hidden_dim",
                        self.set_selection_cfg.get("hidden_dim", self.dim),
                    )
                ),
                refinement_delta_offsets_px=tuple(
                    float(value)
                    for value in self.set_selection_cfg.get(
                        "four_slot_refinement_delta_offsets_px",
                        (-24.0, -12.0, -6.0, 0.0, 6.0, 12.0, 24.0),
                    )
                ),
                refinement_straight_through_routing=bool(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_straight_through_routing",
                        False,
                    )
                ),
                refinement_detach_slot_states=bool(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_detach_slot_states",
                        True,
                    )
                ),
                refinement_route_temperature=float(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_route_temperature",
                        1.0,
                    )
                ),
                factorized_routing=bool(
                    self.set_selection_cfg.get(
                        "four_slot_factorized_routing",
                        False,
                    )
                ),
                active_prior_prob=float(
                    self.set_selection_cfg.get(
                        "four_slot_active_prior_prob",
                        0.80,
                    )
                ),
                geometry_detach_router_states=bool(
                    self.set_selection_cfg.get(
                        "four_slot_geometry_detach_router_states",
                        True,
                    )
                ),
                geometry_router_state_gradient_scale=float(
                    self.set_selection_cfg.get(
                        "four_slot_geometry_router_state_gradient_scale",
                        1.0,
                    )
                ),
                refinement_structured_unique_routing=bool(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_structured_unique_routing",
                        False,
                    )
                ),
                refinement_route_gradient_scale=float(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_route_gradient_scale",
                        1.0,
                    )
                ),
                refinement_reference_mode=str(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_reference_mode",
                        "hard_st",
                    )
                ),
                refinement_neighborhood_max_candidates=int(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_neighborhood_max_candidates",
                        4,
                    )
                ),
                refinement_neighborhood_max_mean_distance_px=float(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_neighborhood_max_mean_distance_px",
                        48.0,
                    )
                ),
                refinement_neighborhood_min_common_fraction=float(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_neighborhood_min_common_fraction",
                        0.50,
                    )
                ),
                refinement_neighborhood_distance_temperature_px=float(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_neighborhood_distance_temperature_px",
                        24.0,
                    )
                ),
                refinement_neighborhood_gradient_scale=float(
                    self.set_selection_cfg.get(
                        "four_slot_refinement_neighborhood_gradient_scale",
                        0.10,
                    )
                ),
                range_refinement_enabled=bool(
                    self.set_selection_cfg.get(
                        "four_slot_range_refinement_enabled",
                        False,
                    )
                ),
                range_delta_offsets_norm=tuple(
                    float(value)
                    for value in self.set_selection_cfg.get(
                        "four_slot_range_delta_offsets_norm",
                        (-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10),
                    )
                ),
                slot_owned_geometry_enabled=bool(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_enabled",
                        False,
                    )
                ),
                slot_owned_geometry_hidden_dim=int(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_hidden_dim",
                        self.set_selection_cfg.get("hidden_dim", self.dim),
                    )
                ),
                slot_owned_geometry_delta_offsets_px=tuple(
                    float(value)
                    for value in self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_delta_offsets_px",
                        (
                            -160.0,
                            -96.0,
                            -48.0,
                            -24.0,
                            0.0,
                            24.0,
                            48.0,
                            96.0,
                            160.0,
                        ),
                    )
                ),
                slot_owned_geometry_evidence_offsets_px=tuple(
                    float(value)
                    for value in self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_evidence_offsets_px",
                        (-48.0, -24.0, -12.0, 0.0, 12.0, 24.0, 48.0),
                    )
                ),
                slot_owned_geometry_route_temperature=float(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_route_temperature",
                        1.0,
                    )
                ),
                slot_owned_geometry_route_gradient_scale=float(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_route_gradient_scale",
                        1.0,
                    )
                ),
                slot_owned_geometry_structured_unique_routing=bool(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_structured_unique_routing",
                        True,
                    )
                ),
                slot_owned_geometry_vertical_layers=int(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_vertical_layers",
                        2,
                    )
                ),
                slot_owned_geometry_vertical_num_heads=int(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_vertical_num_heads",
                        self.set_selection_cfg.get("num_heads", num_heads),
                    )
                ),
                slot_owned_geometry_vertical_ff_dim=int(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_vertical_ff_dim",
                        2 * self.set_selection_cfg.get("hidden_dim", self.dim),
                    )
                ),
                slot_owned_geometry_vertical_dropout=float(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_vertical_dropout",
                        0.0,
                    )
                ),
                slot_owned_geometry_zero_init_delta_heads=bool(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_zero_init_delta_heads",
                        False,
                    )
                ),
                slot_owned_geometry_delta_head_init_std=float(
                    self.set_selection_cfg.get(
                        "four_slot_slot_owned_geometry_delta_head_init_std",
                        1.0e-3,
                    )
                ),
            )
        elif self.set_selection_enabled:
            self.set_selection_head = SetAwareLaneSelectionHead(
                self.dim,
                input_w=self.input_w,
                hidden_dim=int(self.set_selection_cfg.get("hidden_dim", self.dim)),
                num_layers=int(self.set_selection_cfg.get("num_layers", 2)),
                num_heads=int(self.set_selection_cfg.get("num_heads", num_heads)),
                ff_dim=int(self.set_selection_cfg.get("ff_dim", 2 * self.dim)),
                dropout=float(self.set_selection_cfg.get("dropout", dropout)),
                curve_samples=int(self.set_selection_cfg.get("curve_samples", 20)),
                base_quality_power=float(
                    self.set_selection_cfg.get("base_quality_power", 0.5)
                ),
                range_temperature=float(
                    self.set_selection_cfg.get("range_temperature", 0.02)
                ),
                unified_score=bool(
                    self.set_selection_cfg.get("unified_score", False)
                ),
                prior_prob=float(
                    self.set_selection_cfg.get("prior_prob", 0.05)
                ),
                detach_geometry_features=bool(
                    self.set_selection_cfg.get(
                        "detach_geometry_features",
                        True,
                    )
                ),
                use_curve_evidence=bool(
                    self.set_selection_cfg.get("use_curve_evidence", False)
                ),
                use_semantic_decision=bool(
                    self.set_selection_cfg.get("use_semantic_decision", False)
                ),
                candidate_interaction=str(
                    self.set_selection_cfg.get(
                        "candidate_interaction",
                        "transformer",
                    )
                ),
                relation_sigma_px=float(
                    self.set_selection_cfg.get("relation_sigma_px", 20.0)
                ),
                relation_hidden_dim=int(
                    self.set_selection_cfg.get("relation_hidden_dim", 32)
                ),
                pointer_max_selections=int(
                    self.set_selection_cfg.get("pointer_max_selections", 4)
                ),
                pointer_min_valid_rows=int(
                    self.set_selection_cfg.get("pointer_min_valid_rows", 5)
                ),
                pointer_similarity_prior=float(
                    self.set_selection_cfg.get(
                        "pointer_similarity_prior",
                        0.5,
                    )
                ),
                pointer_teacher_mode=str(
                    self.set_selection_cfg.get(
                        "pointer_teacher_mode",
                        "fixed_sequence",
                    )
                ),
                row_grid_mode=str(
                    self.set_selection_cfg.get(
                        "row_grid_mode",
                        "legacy_linspace",
                    )
                ),
                pointer_quality_policy_mode=str(
                    self.set_selection_cfg.get(
                        "pointer_quality_policy_mode",
                        "shared",
                    )
                ),
                pointer_quality_prior_max_scale=float(
                    self.set_selection_cfg.get(
                        "pointer_quality_prior_max_scale",
                        2.0,
                    )
                ),
            )
        else:
            self.set_selection_head = None
        if (
            self.set_selection_head is not None
            and self.set_selection_head.use_semantic_decision
            and not self.detach_score_geometry
        ):
            raise ValueError(
                "set_selection.use_semantic_decision requires "
                "lane_state.detach_score_geometry=true"
            )
        nn.init.constant_(self.range[-1].weight, 0.0)
        with torch.no_grad():
            self.range[-1].bias.copy_(torch.tensor([-2.0, 2.0]))
            if self.exist_prior_prob is not None:
                lane_logit = math.log(
                    self.exist_prior_prob / (1.0 - self.exist_prior_prob)
                )
                if self.single_logit_score:
                    self.exist[-1].bias.copy_(
                        self.exist[-1].bias.new_tensor([lane_logit])
                    )
                else:
                    self.exist[-1].bias.copy_(
                        self.exist[-1].bias.new_tensor(
                            [0.5 * lane_logit, -0.5 * lane_logit]
                        )
                    )
        self.training_auxiliary_instance_tokens = (
            nn.Embedding(sum(self.training_auxiliary_group_sizes), self.dim)
            if self.training_auxiliary_group_sizes
            else None
        )
        if self.training_auxiliary_instance_tokens is not None:
            nn.init.normal_(self.training_auxiliary_instance_tokens.weight, std=0.02)

        # Ownership modules are intentionally initialized *after* every V4
        # geometry/score module.  With the same global seed, enabling V5 does
        # not consume RNG before the bounded-delta geometry is initialized;
        # V4 and V5 therefore start from an identical geometry parameter draw.
        self.ownership_tokens = (
            nn.Embedding(self.primary_num_instances, self.dim)
            if self.ownership_enabled
            else None
        )
        self.ownership_layers = nn.ModuleList(
            [
                ProtectedOwnershipLayer(
                    dim=self.dim,
                    num_heads=int(
                        self.ownership_cfg.get("num_heads", num_heads)
                    ),
                    ff_dim=int(self.ownership_cfg.get("ff_dim", ff_dim)),
                    dropout=float(
                        self.ownership_cfg.get("dropout", dropout)
                    ),
                    semantic_context=self.ownership_cfg.get(
                        "semantic_context"
                    ),
                )
                for _ in range(int(num_layers))
            ]
            if self.ownership_enabled
            else []
        )
        if self.ownership_tokens is not None:
            nn.init.normal_(self.ownership_tokens.weight, std=0.02)

    def _row_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature_proj(features)
        if feat.shape[-2:] != (self.num_rows, self.evidence_x_bins):
            feat = F.interpolate(feat, size=(self.num_rows, self.evidence_x_bins), mode="bilinear", align_corners=False)
        b, c, r, x = feat.shape
        feat_value = feat.permute(0, 2, 3, 1).contiguous()
        feat_key = feat_value
        if self.x_tokens is not None:
            x_pos = self.x_tokens.weight.to(device=features.device, dtype=features.dtype).view(1, 1, x, c)
            feat_key = feat_key + x_pos
        return feat_value, feat_key

    def _sample_final_curve_evidence(
        self,
        row_value_features: torch.Tensor,
        pred_x_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Linearly sample P2 at the exact final predicted curve.

        The row dimension is already aligned, so only horizontal interpolation
        is required. The interpolation weight remains differentiable with
        respect to final x while avoiding a general 2-D grid-sample kernel.
        """

        batch, rows, x_bins, channels = row_value_features.shape
        if pred_x_rows.ndim != 3 or pred_x_rows.shape[0] != batch:
            raise ValueError("pred_x_rows must have shape [batch, lanes, rows]")
        if int(pred_x_rows.shape[-1]) != rows:
            raise ValueError("pred_x_rows must share the P2 row count")
        candidates = int(pred_x_rows.shape[1])
        with torch.autocast(
            device_type=row_value_features.device.type,
            enabled=False,
        ):
            feature_x = pred_x_rows.float().clamp(
                0.0,
                float(max(self.input_w - 1, 1)),
            )
            feature_x = feature_x * float(max(x_bins - 1, 0)) / float(
                max(self.input_w - 1, 1)
            )
            left = feature_x.floor().long()
            right = (left + 1).clamp(max=max(x_bins - 1, 0))
            alpha = feature_x - left.to(dtype=feature_x.dtype)

        flat = row_value_features.reshape(batch * rows, x_bins, channels)
        left = left.permute(0, 2, 1).reshape(batch * rows, candidates)
        right = right.permute(0, 2, 1).reshape(batch * rows, candidates)
        alpha = alpha.permute(0, 2, 1).reshape(
            batch * rows,
            candidates,
            1,
        )
        row_index = fixed_indices(
            batch * rows,
            device=row_value_features.device,
            dtype=torch.long,
        ).view(-1, 1)
        paired_index = torch.stack((left, right), dim=-1).reshape(
            batch * rows,
            candidates * 2,
        )
        values = flat[row_index, paired_index].view(
            batch * rows,
            candidates,
            2,
            channels,
        )
        sampled = torch.lerp(
            values[:, :, 0],
            values[:, :, 1],
            alpha.to(dtype=row_value_features.dtype),
        )
        return sampled.view(batch, rows, candidates, channels).permute(
            0,
            2,
            1,
            3,
        ).contiguous()

    def _reference_prior_logits(
        self,
        reference_x_rows: torch.Tensor,
        *,
        x_bins: int,
        sigma_px: float,
        strength: float,
    ) -> torch.Tensor:
        positions = fixed_linspace(
            0.0,
            float(max(self.input_w - 1, 1)),
            int(x_bins),
            device=reference_x_rows.device,
            dtype=reference_x_rows.dtype,
        )
        distance = (positions.view(1, 1, 1, -1) - reference_x_rows.unsqueeze(-1)) / float(
            sigma_px
        )
        return -0.5 * float(strength) * distance.square()

    def _resize_reference_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if int(logits.shape[-1]) == self.x_bins:
            return logits
        b, n, r, _ = logits.shape
        resized = F.interpolate(
            logits.reshape(b * n * r, 1, int(logits.shape[-1])),
            size=self.x_bins,
            mode="linear",
            align_corners=True,
        )
        return resized.reshape(b, n, r, self.x_bins)

    def _initialize_image_reference(
        self,
        row_tokens: torch.Tensor,
        row_value_features: torch.Tensor,
        row_key_features: torch.Tensor,
        anchor_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.reference_query_norm is None
            or self.reference_query is None
            or self.reference_key is None
            or self.reference_context is None
            or self.reference_coordinate is None
            or self.reference_logit_scale is None
        ):
            raise RuntimeError("row-reference modules were not initialized")
        query = F.normalize(
            self.reference_query(self.reference_query_norm(row_tokens)),
            dim=-1,
            eps=1e-6,
        )
        key = F.normalize(self.reference_key(row_key_features), dim=-1, eps=1e-6)
        visual_logits = torch.einsum("bnrc,brxc->bnrx", query, key)
        visual_logits = visual_logits * self.reference_logit_scale.exp().clamp(
            min=1.0,
            max=100.0,
        )

        anchor_x = torch.sigmoid(anchor_logits).to(dtype=row_tokens.dtype)
        anchor_x = anchor_x * float(max(self.input_w - 1, 1))
        prior = self._reference_prior_logits(
            anchor_x.unsqueeze(0),
            x_bins=self.evidence_x_bins,
            sigma_px=self.initial_prior_sigma_px,
            strength=self.initial_prior_strength,
        )
        reference_logits = visual_logits + prior
        probability = torch.softmax(reference_logits, dim=-1)
        context = torch.einsum("bnrx,brxc->bnrc", probability, row_value_features)
        full_logits = self._resize_reference_logits(reference_logits)
        reference_x = soft_expected_x(
            full_logits,
            input_w=self.input_w,
            x_bins=self.x_bins,
        )

        x_norm = 2.0 * reference_x / float(max(self.input_w - 1, 1)) - 1.0
        y_norm = fixed_linspace(
            -1.0,
            1.0,
            self.num_rows,
            device=row_tokens.device,
            dtype=row_tokens.dtype,
        ).view(1, 1, self.num_rows).expand_as(x_norm)
        coordinate = self.reference_coordinate(torch.stack((x_norm, y_norm), dim=-1))
        row_tokens = row_tokens + self.reference_context(context) + coordinate
        return row_tokens, reference_x

    def _score_only_decision_inputs(
        self,
        lane_state: torch.Tensor,
        multi_scale_features: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Detach the score view from every geometry-producing tensor path.

        Semantic attention and the foreground head remain trainable.  Only
        their inputs are detached, so foreground supervision cannot move the
        lane state, the P2/P4/P5 feature hierarchy, or the coordinate heads.
        """

        if not self.detach_score_geometry:
            return lane_state, multi_scale_features
        detached_features = (
            None
            if multi_scale_features is None
            else {
                name: value.detach()
                for name, value in multi_scale_features.items()
            }
        )
        return lane_state.detach(), detached_features

    def forward(
        self,
        features: torch.Tensor,
        multi_scale_features: dict[str, torch.Tensor] | None = None,
        inference_only: bool = False,
    ) -> dict[str, Any]:
        b = int(features.shape[0])
        dtype = features.dtype
        device = features.device
        instance = self.instance_tokens.weight.to(device=device, dtype=dtype)
        if (
            self.training_auxiliary_instance_tokens is not None
            and not inference_only
        ):
            instance = torch.cat(
                (
                    instance,
                    self.training_auxiliary_instance_tokens.weight.to(
                        device=device,
                        dtype=dtype,
                    ),
                ),
                dim=0,
            )
        instance_start = 0
        instance_end = int(instance.shape[0])
        active_num_groups = self.num_groups
        active_group_sizes: tuple[int, ...] | None = None
        if inference_only and self.training_auxiliary_group_sizes:
            instance_end = self.primary_num_instances
            instance = instance[:instance_end]
            active_num_groups = 1
            active_group_sizes = (self.primary_num_instances,)
        elif inference_only and self.inference_group_index is not None:
            group_size = self.num_instances // self.num_groups
            group_start = self.inference_group_index * group_size
            instance_start = group_start
            instance_end = group_start + group_size
            instance = instance[instance_start:instance_end]
            # The retained group must interact as one complete candidate set.
            # This is mathematically the same group-isolated attention it saw
            # during training, without evaluating the three train-only groups.
            active_num_groups = 1
        elif self.training_auxiliary_group_sizes:
            active_group_sizes = self.interaction_group_sizes
        row = self.row_tokens.weight.to(device=device, dtype=dtype)
        row_tokens = instance[:, None, :] + row[None, :, :]
        row_tokens = row_tokens.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        lane_state = (
            instance.unsqueeze(0).expand(b, -1, -1).contiguous()
            if self.lane_state_enabled
            else None
        )
        ownership_identity = None
        ownership_state = None
        if self.ownership_tokens is not None:
            ownership_identity = self.ownership_tokens.weight[
                instance_start:instance_end
            ].to(device=device, dtype=dtype)
            ownership_state = ownership_identity.unsqueeze(0).expand(
                b,
                -1,
                -1,
            ).contiguous()
        decision_lane_state = (
            ownership_state if ownership_state is not None else lane_state
        )
        row_value_features, row_key_features = self._row_features(features)
        shared_grid_sample_feature_map = None
        if self.row_reference_enabled and any(
            isinstance(layer, ReferenceGuidedRowLayer)
            and layer.sampling_backend == "grid_sample"
            for layer in self.layers
        ):
            shared_grid_sample_feature_map = (
                prepare_shared_grid_sample_feature_map(row_value_features)
            )

        intermediate_outputs: list[dict[str, torch.Tensor]] = []
        bounded_delta_max_abs_by_layer: list[torch.Tensor] = []
        bounded_delta_mean_abs_by_layer: list[torch.Tensor] = []
        if self.row_reference_enabled:
            if self.reference_anchor_logits is None:
                raise RuntimeError("row-reference anchors were not initialized")
            all_anchor_logits = self.reference_anchor_logits
            if (
                self.training_auxiliary_reference_anchor_logits is not None
                and not inference_only
            ):
                all_anchor_logits = torch.cat(
                    (
                        all_anchor_logits,
                        self.training_auxiliary_reference_anchor_logits,
                    ),
                    dim=0,
                )
            anchor_logits = all_anchor_logits[instance_start:instance_end]
            row_tokens, reference_x = self._initialize_image_reference(
                row_tokens,
                row_value_features,
                row_key_features,
                anchor_logits,
            )
            outputs: dict[str, torch.Tensor] | None = None
            for layer_index, layer in enumerate(self.layers):
                if not isinstance(layer, ReferenceGuidedRowLayer):
                    raise TypeError("row-reference mode requires ReferenceGuidedRowLayer")
                lane_layer = (
                    self.lane_state_layers[layer_index]
                    if lane_state is not None
                    else None
                )
                if isinstance(lane_layer, UnifiedLaneSetLayer):
                    lane_state = lane_layer.prepare(
                        lane_state,
                        group_sizes=active_group_sizes,
                    )
                    row_tokens = lane_layer.inject_rows(lane_state, row_tokens)
                row_tokens = layer(
                    row_tokens,
                    row_value_features,
                    reference_x,
                    input_w=self.input_w,
                    num_groups=active_num_groups,
                    group_sizes=active_group_sizes,
                    shared_grid_sample_feature_map=(
                        shared_grid_sample_feature_map
                        if layer.sampling_backend == "grid_sample"
                        else None
                    ),
                )
                if isinstance(lane_layer, UnifiedLaneSetLayer):
                    lane_state = lane_layer.collect(lane_state, row_tokens)
                elif lane_layer is not None:
                    lane_state = lane_layer(
                        lane_state,
                        row_tokens,
                    )
                if ownership_state is not None:
                    if lane_state is None or ownership_identity is None:
                        raise RuntimeError(
                            "protected ownership lost its geometry identity"
                        )
                    ownership_state = self.ownership_layers[layer_index](
                        ownership_state,
                        ownership_identity,
                        lane_state,
                        row_tokens,
                        multi_scale_features=multi_scale_features,
                        group_sizes=active_group_sizes,
                    )
                    decision_lane_state = ownership_state
                elif isinstance(lane_layer, UnifiedLaneSetLayer):
                    score_lane_state, score_features = (
                        self._score_only_decision_inputs(
                            lane_state,
                            multi_scale_features,
                        )
                    )
                    decision_lane_state = lane_layer.decision(
                        score_lane_state,
                        multi_scale_features=score_features,
                    )
                elif lane_layer is not None:
                    decision_lane_state = (
                        lane_state.detach()
                        if self.detach_score_geometry
                        else lane_state
                    )
                else:
                    decision_lane_state = None
                row_logit_bias = (
                    self._reference_prior_logits(
                        reference_x,
                        x_bins=self.x_bins,
                        sigma_px=self.output_prior_sigma_px,
                        strength=self.output_prior_strength,
                    )
                    if self.row_reference_prediction_mode == "absolute"
                    else None
                )
                row_delta_norm = (
                    self.row_delta_norms[layer_index]
                    if self.row_reference_prediction_mode == "bounded_delta"
                    else None
                )
                row_delta_head = (
                    self.row_delta_heads[layer_index]
                    if self.row_reference_prediction_mode == "bounded_delta"
                    else None
                )
                layer_outputs = self._predict_from_row_tokens(
                    row_tokens,
                    instance,
                    include_quality=layer_index == len(self.layers) - 1,
                    row_x_logit_bias=row_logit_bias,
                    input_reference_x_rows=reference_x,
                    lane_state=lane_state,
                    decision_lane_state=decision_lane_state,
                    row_delta_norm=row_delta_norm,
                    row_delta_head=row_delta_head,
                )
                # These reductions are audit-only.  Keeping them out of the
                # training graph avoids eight small CUDA reductions per
                # microbatch without changing a model output consumed by any
                # loss.  Contract/evaluation tools use inference_only=True.
                if (
                    self.row_reference_prediction_mode == "bounded_delta"
                    and inference_only
                ):
                    delta_abs = layer_outputs["pred_delta_x_rows"].detach().abs()
                    bounded_delta_max_abs_by_layer.append(
                        delta_abs.amax(dim=(1, 2))
                    )
                    bounded_delta_mean_abs_by_layer.append(
                        delta_abs.mean(dim=(1, 2))
                    )
                predicted_reference = layer_outputs["pred_x_rows"]
                reference_x = (
                    predicted_reference.detach()
                    if self.detach_reference_between_layers
                    and layer_index < len(self.layers) - 1
                    else predicted_reference
                )
                if (
                    self.intermediate_supervision
                    and not inference_only
                    and layer_index < len(self.layers) - 1
                ):
                    intermediate_outputs.append(layer_outputs)
                outputs = layer_outputs
            if outputs is None:
                raise ValueError("row-reference decoder requires at least one decoder layer")
            if bounded_delta_max_abs_by_layer:
                outputs["bounded_delta_max_abs_by_layer"] = torch.stack(
                    bounded_delta_max_abs_by_layer,
                    dim=-1,
                )
                outputs["bounded_delta_mean_abs_by_layer"] = torch.stack(
                    bounded_delta_mean_abs_by_layer,
                    dim=-1,
                )
        else:
            intermediate_states: list[
                tuple[
                    torch.Tensor,
                    torch.Tensor | None,
                    torch.Tensor | None,
                ]
            ] = []
            for layer_index, layer in enumerate(self.layers):
                lane_layer = (
                    self.lane_state_layers[layer_index]
                    if lane_state is not None
                    else None
                )
                if isinstance(lane_layer, UnifiedLaneSetLayer):
                    lane_state = lane_layer.prepare(
                        lane_state,
                        group_sizes=active_group_sizes,
                    )
                    row_tokens = lane_layer.inject_rows(lane_state, row_tokens)
                row_tokens = layer(
                    row_tokens,
                    row_value_features,
                    row_key_features,
                    num_groups=active_num_groups,
                    group_sizes=active_group_sizes,
                )
                if isinstance(lane_layer, UnifiedLaneSetLayer):
                    lane_state = lane_layer.collect(lane_state, row_tokens)
                elif lane_layer is not None:
                    lane_state = lane_layer(
                        lane_state,
                        row_tokens,
                    )
                if ownership_state is not None:
                    if lane_state is None or ownership_identity is None:
                        raise RuntimeError(
                            "protected ownership lost its geometry identity"
                        )
                    ownership_state = self.ownership_layers[layer_index](
                        ownership_state,
                        ownership_identity,
                        lane_state,
                        row_tokens,
                        multi_scale_features=multi_scale_features,
                        group_sizes=active_group_sizes,
                    )
                    decision_lane_state = ownership_state
                elif isinstance(lane_layer, UnifiedLaneSetLayer):
                    score_lane_state, score_features = (
                        self._score_only_decision_inputs(
                            lane_state,
                            multi_scale_features,
                        )
                    )
                    decision_lane_state = lane_layer.decision(
                        score_lane_state,
                        multi_scale_features=score_features,
                    )
                elif lane_layer is not None:
                    decision_lane_state = (
                        lane_state.detach()
                        if self.detach_score_geometry
                        else lane_state
                    )
                else:
                    decision_lane_state = None
                if (
                    self.intermediate_supervision
                    and not inference_only
                    and layer_index < len(self.layers) - 1
                ):
                    intermediate_states.append(
                        (row_tokens, lane_state, decision_lane_state)
                    )
            outputs = self._predict_from_row_tokens(
                row_tokens,
                instance,
                include_quality=True,
                lane_state=lane_state,
                decision_lane_state=decision_lane_state,
            )
            if self.intermediate_supervision and not inference_only:
                intermediate_outputs = [
                    self._predict_from_row_tokens(
                        tokens,
                        instance,
                        include_quality=False,
                        lane_state=intermediate_lane_state,
                        decision_lane_state=intermediate_decision_state,
                    )
                    for (
                        tokens,
                        intermediate_lane_state,
                        intermediate_decision_state,
                    ) in intermediate_states
                ]
        if self.training_auxiliary_group_sizes and not inference_only:
            full_count = self.num_instances
            primary_end = self.primary_num_instances
            auxiliary_outputs = self._slice_prediction_outputs(
                outputs,
                primary_end,
                full_count,
                full_count,
            )
            outputs = self._slice_prediction_outputs(
                outputs,
                0,
                primary_end,
                full_count,
            )
            primary_intermediate_outputs = []
            auxiliary_intermediate_outputs = []
            for layer_outputs in intermediate_outputs:
                primary_intermediate_outputs.append(
                    self._slice_prediction_outputs(
                        layer_outputs,
                        0,
                        primary_end,
                        full_count,
                    )
                )
                auxiliary_intermediate_outputs.append(
                    self._slice_prediction_outputs(
                        layer_outputs,
                        primary_end,
                        full_count,
                        full_count,
                    )
                )
            intermediate_outputs = primary_intermediate_outputs
            outputs["_training_auxiliary_outputs"] = auxiliary_outputs
            outputs["_training_auxiliary_aux_outputs"] = auxiliary_intermediate_outputs
            outputs["_training_auxiliary_group_sizes"] = self.training_auxiliary_group_sizes
        if self.set_selection_head is not None:
            if self.set_selection_head.use_curve_evidence:
                curve_x = outputs["pred_x_rows"]
                curve_features = row_value_features
                if self.set_selection_head.detach_geometry_features:
                    # The score branch observes both the curve and its P2
                    # evidence.  Neither coordinate nor feature-tower gradient
                    # may cross back into the geometry detector.
                    curve_x = curve_x.detach()
                    curve_features = curve_features.detach()
                outputs["selection_curve_evidence"] = (
                    self._sample_final_curve_evidence(
                        curve_features,
                        curve_x,
                    )
                )
            if bool(
                getattr(
                    self.set_selection_head,
                    "requires_row_value_features",
                    False,
                )
            ):
                selection_result = self.set_selection_head(
                    outputs,
                    row_value_features=row_value_features.detach(),
                )
            else:
                selection_result = self.set_selection_head(outputs)
            if isinstance(selection_result, dict):
                outputs.update(selection_result)
            else:
                selection_logits, selection_delta_logits = selection_result
                outputs["selection_logits"] = selection_logits
                outputs["selection_delta_logits"] = selection_delta_logits
        row_tokens = outputs["structured_row_tokens"]
        if not isinstance(row_tokens, torch.Tensor):
            raise TypeError("structured_row_tokens must be a tensor")
        if self.intermediate_supervision and not inference_only:
            outputs["aux_outputs"] = intermediate_outputs
        if inference_only:
            inference_outputs = {
                "exist_logits": outputs["exist_logits"],
                "pred_x_rows": outputs["pred_x_rows"],
                "range_norm": outputs["range_norm"],
                "quality_logits": outputs["quality_logits"],
            }
            if "ownership_logits" in outputs:
                inference_outputs["ownership_logits"] = outputs[
                    "ownership_logits"
                ]
            if (
                self.ownership_retain_diagnostic_tensors
                and "ownership_state" in outputs
            ):
                inference_outputs["ownership_state"] = outputs[
                    "ownership_state"
                ]
            if "selection_logits" in outputs:
                inference_outputs["selection_logits"] = outputs["selection_logits"]
            for name in (
                "selection_pointer_logits",
                "selection_pointer_indices",
                "selection_pointer_scores",
                "selection_pointer_relation_bias",
                "selection_slot_logits",
                "selection_slot_active_logits",
                "selection_slot_real_route_logits",
                "selection_slot_candidate_valid",
                "selection_slot_raw_indices",
                "selection_slot_raw_collision_count",
                "selection_slot_route_entropy",
                "selection_slot_indices",
                "selection_slot_scores",
                "selection_slot_global_repair_count",
                "selection_slot_pred_x_rows",
                "selection_slot_range_norm",
                "selection_slot_active",
                "selection_slot_geometry_valid",
                "selection_slot_geometry_route_indices",
                "selection_slot_input_reference_x_rows",
                "selection_slot_input_range_norm",
                "selection_slot_row_delta_logits",
                "selection_slot_row_delta_offsets_px",
                "selection_slot_range_delta",
                "selection_slot_range_delta_logits",
                "selection_slot_range_delta_offsets_norm",
                "selection_slot_range_delta_boundary_mass",
                "selection_slot_delta_mean_abs",
                "selection_slot_delta_max_abs",
                "selection_slot_delta_boundary_mass",
                "selection_slot_neighborhood_support",
                "selection_slot_neighborhood_mean_support",
                "selection_slot_neighborhood_alternative_fraction",
                "selection_slot_neighborhood_entropy",
                "selection_slot_neighborhood_top1_mass",
                "selection_slot_neighborhood_mix",
                "selection_slot_neighborhood_reference_shift_px",
                "selection_slot_owned_weight",
                "selection_slot_owned_entropy",
                "selection_slot_owned_top1_mass",
                "selection_slot_owned_reference_shift_px",
            ):
                if name in outputs:
                    inference_outputs[name] = outputs[name]
            for name in (
                "bounded_delta_max_abs_by_layer",
                "bounded_delta_mean_abs_by_layer",
            ):
                if name in outputs:
                    inference_outputs[name] = outputs[name]
            if (
                self.set_selection_head is not None
                and self.set_selection_head.retain_pointer_diagnostic_tensors
            ):
                for name in (
                    "_selection_pointer_hidden",
                    "_selection_pointer_relations",
                    "_selection_pointer_unary_logits",
                    "_selection_pointer_policy_logits",
                    "_selection_pointer_candidate_valid",
                ):
                    if name in outputs:
                        inference_outputs[name] = outputs[name]
            return inference_outputs
        # Keep the public debug container without running reductions that are
        # not consumed by training, evaluation, or model outputs.  In
        # particular, abs() on the full row-evidence tensor otherwise
        # materializes a roughly 500 MiB temporary at the paper batch size.
        outputs["structured_debug"] = {}
        return outputs

    @staticmethod
    def _slice_prediction_outputs(
        outputs: dict[str, torch.Tensor],
        start: int,
        end: int,
        candidate_count: int,
    ) -> dict[str, torch.Tensor]:
        """Slice the candidate axis while preserving scalar/debug tensors."""

        sliced: dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and int(value.shape[1]) == int(candidate_count)
            ):
                sliced[key] = value[:, int(start) : int(end)]
            else:
                sliced[key] = value
        return sliced

    def _predict_from_row_tokens(
        self,
        row_tokens: torch.Tensor,
        instance: torch.Tensor,
        *,
        include_quality: bool,
        row_x_logit_bias: torch.Tensor | None = None,
        input_reference_x_rows: torch.Tensor | None = None,
        lane_state: torch.Tensor | None = None,
        decision_lane_state: torch.Tensor | None = None,
        row_delta_norm: nn.Module | None = None,
        row_delta_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply lane heads while respecting the configured coordinate frame."""
        b = int(row_tokens.shape[0])
        if self.row_reference_prediction_mode == "bounded_delta":
            if (
                input_reference_x_rows is None
                or row_delta_norm is None
                or row_delta_head is None
            ):
                raise ValueError(
                    "bounded_delta prediction requires an input reference and "
                    "layer-local readout modules"
                )
            # The caller resolves the layer-local modules while its decoder
            # index is a Python loop constant.  Passing the modules explicitly keeps
            # PyTorch 2.1 Dynamo from trying to index a ModuleList with a
            # symbolic integer inside this helper; eager semantics are
            # unchanged.
            normalized_rows = row_delta_norm(row_tokens)
        else:
            if self.row_norm is None:
                raise RuntimeError("absolute row normalization was not initialized")
            normalized_rows = self.row_norm(row_tokens)
        if lane_state is not None:
            if lane_state.shape != (b, int(normalized_rows.shape[1]), self.dim):
                raise ValueError("lane_state shape does not match row tokens")
            lane_query = self.lane_norm(lane_state)
            if decision_lane_state is None:
                decision_query = lane_query
            else:
                if decision_lane_state.shape != lane_state.shape:
                    raise ValueError(
                        "decision_lane_state shape does not match lane state"
                    )
                decision_query = (
                    self.decision_norm(decision_lane_state)
                    if self.decision_norm is not None
                    else self.lane_norm(decision_lane_state)
                )
        else:
            instance_residual = instance.unsqueeze(0).expand(b, -1, -1)
            lane_summary = normalized_rows.mean(dim=2)
            if self.lane_pooling == "mean_max":
                lane_summary = lane_summary + normalized_rows.amax(dim=2)
            lane_query = self.lane_norm(lane_summary + instance_residual)
            if self.detach_score_geometry:
                assert self.decision_norm is not None
                decision_query = self.decision_norm(
                    (lane_summary + instance_residual).detach()
                )
            else:
                decision_query = lane_query
        if self.row_reference_prediction_mode == "bounded_delta":
            assert row_delta_head is not None
            row_x_logits = row_delta_head(normalized_rows)
            offsets = self.row_delta_offsets_px.to(
                device=row_x_logits.device,
                dtype=row_x_logits.dtype,
            )
            probability = torch.softmax(row_x_logits, dim=-1)
            delta_x_rows = (probability * offsets).sum(dim=-1)
            delta_x_rows = delta_x_rows.clamp(
                min=self.row_delta_min_px,
                max=self.row_delta_max_px,
            )
            pred_x_rows = (
                input_reference_x_rows.to(dtype=delta_x_rows.dtype)
                + delta_x_rows
            ).clamp(0.0, float(max(self.input_w - 1, 0)))
        else:
            if self.row_x is None:
                raise RuntimeError("absolute row projection was not initialized")
            row_x_logits = self.row_x(normalized_rows)
            if row_x_logit_bias is not None:
                if row_x_logit_bias.shape != row_x_logits.shape:
                    raise ValueError(
                        "row_x_logit_bias shape must match row logits: "
                        f"{tuple(row_x_logit_bias.shape)} vs {tuple(row_x_logits.shape)}"
                    )
                row_x_logits = row_x_logits + row_x_logit_bias.to(
                    dtype=row_x_logits.dtype
                )
            pred_x_rows = soft_expected_x(
                row_x_logits,
                input_w=self.input_w,
                x_bins=self.x_bins,
            )
        range_raw = self.range(lane_query)
        range_norm = sort_range_norm(torch.sigmoid(range_raw))
        raw_exist = self.exist(decision_query)
        if self.single_logit_score:
            foreground_logit = raw_exist.squeeze(-1)
            exist_logits = torch.stack(
                (foreground_logit, torch.zeros_like(foreground_logit)),
                dim=-1,
            )
        else:
            exist_logits = raw_exist
        outputs = {
            "exist_logits": exist_logits,
            "pred_x_rows": pred_x_rows,
            "range_norm": range_norm,
            "row_x_logits": row_x_logits,
            "range_raw": range_raw,
            "queries": lane_query,
            "decision_queries": decision_query,
            "structured_row_tokens": normalized_rows,
        }
        if self.ownership_enabled:
            if decision_lane_state is None:
                raise RuntimeError("protected ownership requires a decision state")
            outputs["ownership_logits"] = exist_logits
            outputs["ownership_state"] = decision_lane_state
        if self.row_reference_prediction_mode == "bounded_delta":
            outputs["row_x_offsets_px"] = self.row_delta_offsets_px
            outputs["pred_delta_x_rows"] = delta_x_rows
        if self.single_logit_score:
            # One scalar is now used by matching, foreground supervision and
            # deployment.  The legacy two-logit tensor above is only an exact
            # compatibility view: softmax(...)[0] == sigmoid(score_logit).
            outputs["score_logits"] = foreground_logit
        if input_reference_x_rows is not None:
            outputs["input_reference_x_rows"] = input_reference_x_rows
        if include_quality:
            outputs["quality_logits"] = self.quality(decision_query).squeeze(-1)
        return outputs


def build_structured_query_head(model_cfg: dict[str, Any]) -> StructuredLaneQueryHead | None:
    structured_cfg = model_cfg.get("structured_query", {})
    if not bool(structured_cfg.get("enabled", False)):
        return None
    return StructuredLaneQueryHead(
        dim=int(model_cfg.get("dim", 256)),
        num_instances=int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 20))),
        num_rows=int(model_cfg.get("num_rows", 72)),
        x_bins=int(model_cfg.get("x_bins", 200)),
        input_w=int(model_cfg.get("input_w", 800)),
        num_heads=int(structured_cfg.get("num_heads", model_cfg.get("num_heads", 8))),
        num_layers=int(structured_cfg.get("num_layers", 2)),
        ff_dim=int(structured_cfg.get("ff_dim", model_cfg.get("decoder_ff_dim", 1024))),
        dropout=float(structured_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        use_x_pos=bool(structured_cfg.get("use_x_pos", True)),
        evidence_x_bins=int(structured_cfg.get("evidence_x_bins", structured_cfg.get("attn_x_bins", model_cfg.get("x_bins", 200)))),
        num_groups=int(structured_cfg.get("num_groups", 1)),
        exist_prior_prob=structured_cfg.get("exist_prior_prob"),
        intermediate_supervision=bool(structured_cfg.get("intermediate_supervision", False)),
        inference_group_index=structured_cfg.get("inference_group_index"),
        training_auxiliary_group_sizes=structured_cfg.get(
            "training_auxiliary_group_sizes"
        ),
        row_reference=structured_cfg.get("row_reference"),
        lane_state=structured_cfg.get("lane_state"),
        ownership=structured_cfg.get("ownership"),
        set_selection=structured_cfg.get("set_selection"),
        lane_pooling=str(structured_cfg.get("lane_pooling", "mean_max")),
    )
