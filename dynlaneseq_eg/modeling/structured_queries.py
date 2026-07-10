from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import soft_expected_x, sort_range_norm


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

    def _grouped_inter_attention(self, q: torch.Tensor, batch_rows: int, num_instances: int) -> torch.Tensor:
        if self.num_groups == 1:
            return self.inter_attn(q, q, q, need_weights=False)[0]
        if num_instances % self.num_groups != 0:
            raise ValueError(
                f"num_instances={num_instances} must be divisible by structured_query.num_groups={self.num_groups}"
            )
        group_size = num_instances // self.num_groups
        grouped = q.view(batch_rows, self.num_groups, group_size, q.shape[-1])
        grouped = grouped.reshape(batch_rows * self.num_groups, group_size, q.shape[-1])
        delta = self.inter_attn(grouped, grouped, grouped, need_weights=False)[0]
        return delta.view(batch_rows, self.num_groups, group_size, q.shape[-1]).reshape(
            batch_rows, num_instances, q.shape[-1]
        )

    def forward(self, row_tokens: torch.Tensor, row_value_features: torch.Tensor, row_key_features: torch.Tensor) -> torch.Tensor:
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
        q = q + self.drop(self._grouped_inter_attention(q_norm, batch_rows=b * r, num_instances=n))
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
        multi_scale_enabled: bool = False,
        multi_scale_scales: list[str] | tuple[str, ...] | None = None,
        multi_scale_fusion: str = "static_softmax",
        multi_scale_layer_scales: list[str] | tuple[str, ...] | None = None,
        multi_scale_pos_encoding: str = "learned",
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_instances = int(num_instances)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.evidence_x_bins = int(evidence_x_bins) if evidence_x_bins is not None else self.x_bins
        self.input_w = int(input_w)
        self.use_x_pos = bool(use_x_pos)
        self.num_groups = int(num_groups)
        self.exist_prior_prob = None if exist_prior_prob is None else float(exist_prior_prob)
        self.multi_scale_enabled = bool(multi_scale_enabled)
        self.multi_scale_scales = tuple(multi_scale_scales or ("p2", "p3", "p4"))
        self.multi_scale_fusion = str(multi_scale_fusion).lower()
        self.num_layers = int(num_layers)
        self.multi_scale_layer_scales = tuple(multi_scale_layer_scales or ())
        self.multi_scale_pos_encoding = str(multi_scale_pos_encoding).lower()
        if self.num_groups < 1:
            raise ValueError("structured_query.num_groups must be >= 1")
        if self.evidence_x_bins < 1:
            raise ValueError("structured_query.evidence_x_bins must be >= 1")
        if self.num_instances % self.num_groups != 0:
            raise ValueError(
                f"structured_query.num_instances={self.num_instances} must be divisible by num_groups={self.num_groups}"
            )
        if self.exist_prior_prob is not None and not 0.0 < self.exist_prior_prob < 1.0:
            raise ValueError("structured_query.exist_prior_prob must be between 0 and 1")
        if self.multi_scale_enabled:
            if len(self.multi_scale_scales) < 1:
                raise ValueError("structured_query.multi_scale.scales must not be empty")
            if self.multi_scale_fusion not in {"static_softmax", "layerwise_native"}:
                raise ValueError(
                    "structured_query.multi_scale.fusion must be 'static_softmax' or 'layerwise_native'"
                )
            if self.multi_scale_fusion == "layerwise_native":
                if len(self.multi_scale_layer_scales) != self.num_layers:
                    raise ValueError(
                        "structured_query.multi_scale.layer_scales must contain exactly one scale per decoder layer "
                        f"({self.num_layers}), got {len(self.multi_scale_layer_scales)}"
                    )
                unknown_scales = sorted(set(self.multi_scale_layer_scales) - set(self.multi_scale_scales))
                if unknown_scales:
                    raise ValueError(
                        "structured_query.multi_scale.layer_scales contains scales not exposed by "
                        f"model.multi_scale_evidence.scales: {unknown_scales}"
                    )
                if self.use_x_pos and self.multi_scale_pos_encoding != "normalized_sine":
                    raise ValueError(
                        "layerwise_native fusion requires multi_scale.pos_encoding='normalized_sine' when x position is enabled"
                    )
        if self.multi_scale_pos_encoding not in {"learned", "normalized_sine"}:
            raise ValueError("structured_query.multi_scale.pos_encoding must be 'learned' or 'normalized_sine'")

        self.instance_tokens = nn.Embedding(self.num_instances, self.dim)
        self.row_tokens = nn.Embedding(self.num_rows, self.dim)
        self.x_tokens = (
            nn.Embedding(self.evidence_x_bins, self.dim)
            if self.use_x_pos and self.multi_scale_pos_encoding == "learned"
            else None
        )
        nn.init.normal_(self.instance_tokens.weight, std=0.02)
        nn.init.normal_(self.row_tokens.weight, std=0.02)
        if self.x_tokens is not None:
            nn.init.normal_(self.x_tokens.weight, std=0.02)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.GroupNorm(8, self.dim),
            nn.GELU(),
        )
        if self.multi_scale_enabled and self.multi_scale_fusion == "static_softmax":
            init_scale_logits = torch.zeros(len(self.multi_scale_scales))
            if "p2" in self.multi_scale_scales:
                init_scale_logits[self.multi_scale_scales.index("p2")] = 2.0
            self.scale_logits = nn.Parameter(init_scale_logits)
        else:
            self.scale_logits = None
        self.layers = nn.ModuleList(
            [
                RowAwareCrossAttentionLayer(
                    dim=self.dim,
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                    num_groups=self.num_groups,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.row_norm = nn.LayerNorm(self.dim)
        self.lane_norm = nn.LayerNorm(self.dim)
        self.row_x = nn.Linear(self.dim, self.x_bins)
        self.exist = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 2))
        self.range = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 2))
        self.quality = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 1))
        nn.init.constant_(self.range[-1].weight, 0.0)
        with torch.no_grad():
            self.range[-1].bias.copy_(torch.tensor([-2.0, 2.0]))
            if self.exist_prior_prob is not None:
                lane_logit = 0.5 * math.log(self.exist_prior_prob / (1.0 - self.exist_prior_prob))
                self.exist[-1].bias.copy_(self.exist[-1].bias.new_tensor([lane_logit, -lane_logit]))

    def _project_row_features(self, features: torch.Tensor, preserve_width: bool = False) -> torch.Tensor:
        feat = self.feature_proj(features)
        target_width = int(feat.shape[-1]) if preserve_width else self.evidence_x_bins
        if feat.shape[-2:] != (self.num_rows, target_width):
            feat = F.interpolate(feat, size=(self.num_rows, target_width), mode="bilinear", align_corners=False)
        return feat.permute(0, 2, 3, 1).contiguous()

    def _normalized_sine_x_position(self, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Continuous x encoding shared by pyramid levels with different widths."""
        compute_dtype = torch.float32
        x = (torch.arange(width, device=device, dtype=compute_dtype) + 0.5) / float(width)
        x = x * (2.0 * math.pi)
        half_dim = (self.dim + 1) // 2
        if half_dim == 1:
            frequency = torch.ones(1, device=device, dtype=compute_dtype)
        else:
            frequency = torch.exp(
                -math.log(10000.0)
                * torch.arange(half_dim, device=device, dtype=compute_dtype)
                / float(half_dim - 1)
            )
        angles = x[:, None] * frequency[None, :]
        pos = torch.cat((angles.sin(), angles.cos()), dim=-1)[:, : self.dim]
        return pos.to(dtype=dtype).view(1, 1, width, self.dim)

    def _add_x_position(self, feat_value: torch.Tensor) -> torch.Tensor:
        if not self.use_x_pos:
            return feat_value
        _, _, width, channels = feat_value.shape
        if self.x_tokens is not None:
            if width != self.evidence_x_bins:
                raise ValueError(
                    f"learned x position has {self.evidence_x_bins} bins but row evidence has width {width}"
                )
            x_pos = self.x_tokens.weight.to(device=feat_value.device, dtype=feat_value.dtype).view(
                1, 1, width, channels
            )
        else:
            x_pos = self._normalized_sine_x_position(
                width=width,
                device=feat_value.device,
                dtype=feat_value.dtype,
            )
        return feat_value + x_pos

    def _row_features(
        self,
        features: torch.Tensor,
        multi_scale_features: dict[str, torch.Tensor] | None = None,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, torch.Tensor]]:
        debug: dict[str, torch.Tensor] = {}
        if self.multi_scale_enabled:
            if multi_scale_features is None:
                raise ValueError("structured_query.multi_scale is enabled but encoder did not provide multi_scale_features")
            if self.multi_scale_fusion == "layerwise_native":
                row_banks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                for scale in dict.fromkeys(self.multi_scale_layer_scales):
                    if scale == "p2":
                        scale_features = multi_scale_features.get("p2", features)
                    else:
                        if scale not in multi_scale_features:
                            raise KeyError(f"multi_scale_features is missing required scale {scale!r}")
                        scale_features = multi_scale_features[scale]
                    feat_value = self._project_row_features(scale_features, preserve_width=True)
                    row_banks[scale] = (feat_value, self._add_x_position(feat_value))
                    debug[f"structured_ms_feature_{scale}_abs"] = feat_value.detach().abs().mean()

                layer_banks = [row_banks[scale] for scale in self.multi_scale_layer_scales]
                for layer_index, (scale, (feat_value, _)) in enumerate(
                    zip(self.multi_scale_layer_scales, layer_banks)
                ):
                    debug[f"structured_ms_layer_{layer_index}_scale_index"] = feat_value.new_tensor(
                        float(self.multi_scale_scales.index(scale))
                    )
                    debug[f"structured_ms_layer_{layer_index}_width"] = feat_value.new_tensor(
                        float(feat_value.shape[2])
                    )
                return layer_banks, debug

            row_values = []
            used_scales = []
            for scale in self.multi_scale_scales:
                if scale == "p2":
                    scale_features = multi_scale_features.get("p2", features)
                else:
                    if scale not in multi_scale_features:
                        raise KeyError(f"multi_scale_features is missing required scale {scale!r}")
                    scale_features = multi_scale_features[scale]
                row_values.append(self._project_row_features(scale_features))
                used_scales.append(scale)
            stacked = torch.stack(row_values, dim=0)
            assert self.scale_logits is not None
            scale_weights = torch.softmax(self.scale_logits[: len(row_values)], dim=0).to(
                device=features.device, dtype=features.dtype
            )
            feat_value = (stacked * scale_weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
            for scale, weight in zip(used_scales, scale_weights.detach()):
                debug[f"structured_ms_weight_{scale}"] = weight
        else:
            feat_value = self._project_row_features(features)
        feat_key = self._add_x_position(feat_value)
        return [(feat_value, feat_key)] * self.num_layers, debug

    def forward(
        self,
        features: torch.Tensor,
        multi_scale_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        b = int(features.shape[0])
        dtype = features.dtype
        device = features.device
        instance = self.instance_tokens.weight.to(device=device, dtype=dtype)
        row = self.row_tokens.weight.to(device=device, dtype=dtype)
        row_tokens = instance[:, None, :] + row[None, :, :]
        row_tokens = row_tokens.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        layer_row_features, row_feature_debug = self._row_features(features, multi_scale_features)

        for layer, (row_value_features, row_key_features) in zip(self.layers, layer_row_features):
            row_tokens = layer(row_tokens, row_value_features, row_key_features)

        row_tokens = self.row_norm(row_tokens)
        instance_residual = instance.unsqueeze(0).expand(b, -1, -1)
        lane_query = self.lane_norm(row_tokens.mean(dim=2) + row_tokens.amax(dim=2) + instance_residual)
        row_x_logits = self.row_x(row_tokens)
        pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
        range_raw = self.range(lane_query)
        range_norm = sort_range_norm(torch.sigmoid(range_raw))
        quality_logits = self.quality(lane_query).squeeze(-1)
        return {
            "exist_logits": self.exist(lane_query),
            "row_x_logits": row_x_logits,
            "pred_x_rows": pred_x_rows,
            "range_raw": range_raw,
            "range_norm": range_norm,
            "quality_logits": quality_logits,
            "queries": lane_query,
            "structured_row_tokens": row_tokens,
            "structured_debug": {
                "structured_row_abs": row_tokens.detach().abs().mean(),
                "structured_feature_abs": torch.stack(
                    [value.detach().abs().mean() for value, _ in layer_row_features]
                ).mean(),
                **row_feature_debug,
            },
        }


def build_structured_query_head(model_cfg: dict[str, Any]) -> StructuredLaneQueryHead | None:
    structured_cfg = model_cfg.get("structured_query", {})
    if not bool(structured_cfg.get("enabled", False)):
        return None
    multi_scale_cfg = structured_cfg.get("multi_scale", {})
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
        multi_scale_enabled=bool(multi_scale_cfg.get("enabled", False)),
        multi_scale_scales=list(multi_scale_cfg.get("scales", ["p2", "p3", "p4"])),
        multi_scale_fusion=str(multi_scale_cfg.get("fusion", "static_softmax")),
        multi_scale_layer_scales=list(multi_scale_cfg.get("layer_scales", [])),
        multi_scale_pos_encoding=str(multi_scale_cfg.get("pos_encoding", "learned")),
    )
