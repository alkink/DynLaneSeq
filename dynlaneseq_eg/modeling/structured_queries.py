from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import soft_expected_x, sort_range_norm


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
            feature_map = row_value_features.float().permute(0, 3, 1, 2).contiguous()
            reference = reference_x_rows.float()
            offsets = self.offsets_px.to(device=reference.device, dtype=reference.dtype)
            sample_x = (reference.unsqueeze(-1) + offsets.view(1, 1, 1, -1)).clamp(
                min=0.0,
                max=float(max(int(input_w) - 1, 1)),
            )
            grid_x = 2.0 * sample_x / float(max(int(input_w) - 1, 1)) - 1.0
            grid_y = torch.linspace(
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
        row_index = torch.arange(
            b * rows,
            device=row_value_features.device,
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
    ) -> torch.Tensor:
        b, n, r, c = row_tokens.shape
        if c != self.dim:
            raise ValueError(f"row token dim {c} does not match layer dim {self.dim}")
        profiles = self._sample_local_profiles(
            row_value_features,
            reference_x_rows,
            input_w=int(input_w),
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
        y_norm = torch.linspace(
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

        # Unified mode uses two range-masked row-state summaries and optional
        # exact final-curve P2 evidence. Legacy mode preserves the historical
        # lane-query + visible-row input and includes its base score scalar.
        state_streams = 2 + int(self.unified_score and self.use_curve_evidence)
        scalar_count = 10 + int(not self.unified_score)
        input_dim = (
            state_streams * self.dim
            + scalar_count
            + 2 * self.curve_samples
        )
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, int(hidden_dim))
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
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)
        if self.unified_score:
            nn.init.normal_(self.output.weight, std=0.01)
            nn.init.constant_(
                self.output.bias,
                math.log(self.prior_prob / (1.0 - self.prior_prob)),
            )
        else:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def _base_probability(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        exist = torch.softmax(outputs["exist_logits"].float(), dim=-1)[..., 0]
        quality_logits = outputs.get("quality_logits")
        if self.base_quality_power > 0.0 and quality_logits is not None:
            quality = torch.sigmoid(quality_logits.float()).clamp_min(1e-6)
            exist = exist * quality.pow(self.base_quality_power)
        return exist.clamp(1e-6, 1.0 - 1e-6)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_tokens = outputs["structured_row_tokens"]
        lane_query = outputs["queries"]
        observe = (
            (lambda value: value.detach())
            if self.detach_geometry_features
            else (lambda value: value)
        )
        ranges = sort_range_norm(observe(outputs["range_norm"]).float())
        pred_x = observe(outputs["pred_x_rows"]).float()
        row_logits = observe(outputs["row_x_logits"]).float()
        batch, candidates, rows, _channels = row_tokens.shape

        y_norm = torch.linspace(
            0.0,
            1.0,
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
        sample_ids = torch.linspace(
            0,
            rows - 1,
            sample_count,
            device=pred_x.device,
        ).round().long()
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
            if self.use_curve_evidence:
                curve_evidence = outputs.get("selection_curve_evidence")
                if not isinstance(curve_evidence, torch.Tensor):
                    raise ValueError(
                        "unified set selection with curve evidence requires "
                        "selection_curve_evidence"
                    )
                curve_state = (
                    curve_evidence
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
        hidden = self.input_projection(self.input_norm(features))
        hidden = self.encoder(hidden)
        raw_logits = self.output(self.output_norm(hidden)).squeeze(-1).float()
        if self.unified_score:
            # Keep the legacy diagnostic key in the output contract, but make
            # its value explicit: unified mode has no residual/delta path.
            return raw_logits, torch.zeros_like(raw_logits)
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
        self.row_norm = nn.LayerNorm(self.dim)
        self.lane_norm = nn.LayerNorm(self.dim)
        self.row_x = nn.Linear(self.dim, self.x_bins)
        self.exist = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 2))
        self.range = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 2))
        self.quality = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 1))
        self.set_selection_head = (
            SetAwareLaneSelectionHead(
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
            )
            if self.set_selection_enabled
            else None
        )
        nn.init.constant_(self.range[-1].weight, 0.0)
        with torch.no_grad():
            self.range[-1].bias.copy_(torch.tensor([-2.0, 2.0]))
            if self.exist_prior_prob is not None:
                lane_logit = 0.5 * math.log(self.exist_prior_prob / (1.0 - self.exist_prior_prob))
                self.exist[-1].bias.copy_(self.exist[-1].bias.new_tensor([lane_logit, -lane_logit]))
        self.training_auxiliary_instance_tokens = (
            nn.Embedding(sum(self.training_auxiliary_group_sizes), self.dim)
            if self.training_auxiliary_group_sizes
            else None
        )
        if self.training_auxiliary_instance_tokens is not None:
            nn.init.normal_(self.training_auxiliary_instance_tokens.weight, std=0.02)

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
        row_index = torch.arange(
            batch * rows,
            device=row_value_features.device,
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
        positions = torch.linspace(
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
        y_norm = torch.linspace(
            -1.0,
            1.0,
            self.num_rows,
            device=row_tokens.device,
            dtype=row_tokens.dtype,
        ).view(1, 1, self.num_rows).expand_as(x_norm)
        coordinate = self.reference_coordinate(torch.stack((x_norm, y_norm), dim=-1))
        row_tokens = row_tokens + self.reference_context(context) + coordinate
        return row_tokens, reference_x

    def forward(
        self,
        features: torch.Tensor,
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
        row_value_features, row_key_features = self._row_features(features)

        intermediate_outputs: list[dict[str, torch.Tensor]] = []
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
                row_tokens = layer(
                    row_tokens,
                    row_value_features,
                    reference_x,
                    input_w=self.input_w,
                    num_groups=active_num_groups,
                    group_sizes=active_group_sizes,
                )
                row_logit_bias = self._reference_prior_logits(
                    reference_x,
                    x_bins=self.x_bins,
                    sigma_px=self.output_prior_sigma_px,
                    strength=self.output_prior_strength,
                )
                layer_outputs = self._predict_from_row_tokens(
                    row_tokens,
                    instance,
                    include_quality=layer_index == len(self.layers) - 1,
                    row_x_logit_bias=row_logit_bias,
                    input_reference_x_rows=reference_x,
                )
                reference_x = layer_outputs["pred_x_rows"]
                if (
                    self.intermediate_supervision
                    and not inference_only
                    and layer_index < len(self.layers) - 1
                ):
                    intermediate_outputs.append(layer_outputs)
                outputs = layer_outputs
            if outputs is None:
                raise ValueError("row-reference decoder requires at least one decoder layer")
        else:
            intermediate_row_tokens = []
            for layer_index, layer in enumerate(self.layers):
                row_tokens = layer(
                    row_tokens,
                    row_value_features,
                    row_key_features,
                    num_groups=active_num_groups,
                    group_sizes=active_group_sizes,
                )
                if (
                    self.intermediate_supervision
                    and not inference_only
                    and layer_index < len(self.layers) - 1
                ):
                    intermediate_row_tokens.append(row_tokens)
            outputs = self._predict_from_row_tokens(row_tokens, instance, include_quality=True)
            if self.intermediate_supervision and not inference_only:
                intermediate_outputs = [
                    self._predict_from_row_tokens(tokens, instance, include_quality=False)
                    for tokens in intermediate_row_tokens
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
                if self.set_selection_head.detach_geometry_features:
                    # Score learning may update the visual evidence tower, but
                    # it must not move the curve merely to make the sampled
                    # descriptor easier to classify.
                    curve_x = curve_x.detach()
                outputs["selection_curve_evidence"] = (
                    self._sample_final_curve_evidence(
                        row_value_features,
                        curve_x,
                    )
                )
            selection_logits, selection_delta_logits = self.set_selection_head(outputs)
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
            if "selection_logits" in outputs:
                inference_outputs["selection_logits"] = outputs["selection_logits"]
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
    ) -> dict[str, torch.Tensor]:
        """Apply the shared lane heads to one decoder-layer state.

        Sharing these heads across layers isolates deep supervision from an
        increase in prediction-head capacity.  Auxiliary layers intentionally
        omit quality calibration; quality remains a final-layer decision.
        """
        b = int(row_tokens.shape[0])
        row_tokens = self.row_norm(row_tokens)
        instance_residual = instance.unsqueeze(0).expand(b, -1, -1)
        lane_summary = row_tokens.mean(dim=2)
        if self.lane_pooling == "mean_max":
            lane_summary = lane_summary + row_tokens.amax(dim=2)
        lane_query = self.lane_norm(lane_summary + instance_residual)
        row_x_logits = self.row_x(row_tokens)
        if row_x_logit_bias is not None:
            if row_x_logit_bias.shape != row_x_logits.shape:
                raise ValueError(
                    "row_x_logit_bias shape must match row logits: "
                    f"{tuple(row_x_logit_bias.shape)} vs {tuple(row_x_logits.shape)}"
                )
            row_x_logits = row_x_logits + row_x_logit_bias.to(dtype=row_x_logits.dtype)
        pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
        range_raw = self.range(lane_query)
        range_norm = sort_range_norm(torch.sigmoid(range_raw))
        outputs = {
            "exist_logits": self.exist(lane_query),
            "pred_x_rows": pred_x_rows,
            "range_norm": range_norm,
            "row_x_logits": row_x_logits,
            "range_raw": range_raw,
            "queries": lane_query,
            "structured_row_tokens": row_tokens,
        }
        if input_reference_x_rows is not None:
            outputs["input_reference_x_rows"] = input_reference_x_rows
        if include_quality:
            outputs["quality_logits"] = self.quality(lane_query).squeeze(-1)
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
        set_selection=structured_cfg.get("set_selection"),
        lane_pooling=str(structured_cfg.get("lane_pooling", "mean_max")),
    )
