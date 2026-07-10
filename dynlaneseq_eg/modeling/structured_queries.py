from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import soft_expected_x, sort_range_norm
from .position_encoding import SinePositionEncoding2D


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


class SemanticInstanceContext(nn.Module):
    """Verify P2-informed lane instances against coarse semantic memories.

    Each pyramid level is attended independently before a query-conditioned
    fusion.  This avoids the sequence-length prior that would make a flattened
    P4 memory dominate P5 simply because P4 contains more spatial tokens.
    The returned decision residual is zero at initialization, so enabling this
    branch does not perturb the baseline existence/quality path at step zero.
    """

    def __init__(
        self,
        dim: int,
        scales: list[str] | tuple[str, ...],
        num_heads: int,
        ff_dim: int,
        dropout: float,
        exist_prior_prob: float | None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.scales = tuple(str(scale) for scale in scales)
        if not self.scales:
            raise ValueError("structured_query.semantic_instance.scales must not be empty")

        self.position = SinePositionEncoding2D(dim=self.dim)
        self.query_norm = nn.LayerNorm(self.dim)
        self.cross_attn = nn.MultiheadAttention(
            self.dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.scale_router = nn.Linear(self.dim, len(self.scales))
        self.context_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), self.dim),
        )
        self.drop = nn.Dropout(float(dropout))
        self.decision_adapter = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.dim, self.dim),
        )
        self.aux_exist = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 2),
        )

        # Equal routing is a neutral starting point; the query has already read
        # P2, so learned routing is image- and instance-conditioned from layer 1.
        nn.init.zeros_(self.scale_router.weight)
        nn.init.zeros_(self.scale_router.bias)
        # Preserve the original score path exactly at initialization.
        nn.init.zeros_(self.decision_adapter[-1].weight)
        nn.init.zeros_(self.decision_adapter[-1].bias)
        if exist_prior_prob is not None:
            lane_logit = 0.5 * math.log(exist_prior_prob / (1.0 - exist_prior_prob))
            with torch.no_grad():
                self.aux_exist[-1].bias.copy_(
                    self.aux_exist[-1].bias.new_tensor([lane_logit, -lane_logit])
                )

    def forward(
        self,
        instance_query: torch.Tensor,
        multi_scale_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        query_norm = self.query_norm(instance_query)
        scale_contexts = []
        for scale in self.scales:
            if scale not in multi_scale_features:
                raise KeyError(f"multi_scale_features is missing semantic scale {scale!r}")
            feature = multi_scale_features[scale]
            if feature.ndim != 4 or int(feature.shape[1]) != self.dim:
                raise ValueError(
                    f"semantic scale {scale!r} must have shape [B, {self.dim}, H, W], "
                    f"got {tuple(feature.shape)}"
                )
            position = self.position(feature).to(device=feature.device, dtype=feature.dtype)
            key = (feature + position).flatten(2).transpose(1, 2).contiguous()
            value = feature.flatten(2).transpose(1, 2).contiguous()
            scale_contexts.append(
                self.cross_attn(query_norm, key, value, need_weights=False)[0]
            )

        stacked_context = torch.stack(scale_contexts, dim=2)
        scale_weights = torch.softmax(self.scale_router(query_norm).float(), dim=-1).to(
            dtype=instance_query.dtype
        )
        context = (stacked_context * scale_weights.unsqueeze(-1)).sum(dim=2)
        semantic_query = instance_query + self.drop(context)
        semantic_query = semantic_query + self.drop(self.ffn(self.context_norm(semantic_query)))
        decision_residual = self.decision_adapter(semantic_query)
        aux_exist_logits = self.aux_exist(semantic_query)
        debug = {
            "structured_semantic_query_abs": semantic_query.detach().abs().mean(),
            "structured_semantic_decision_delta_abs": decision_residual.detach().abs().mean(),
            "structured_semantic_aux_lane_prob": torch.sigmoid(
                aux_exist_logits[..., 0] - aux_exist_logits[..., 1]
            ).detach().mean(),
        }
        for scale_index, scale in enumerate(self.scales):
            debug[f"structured_semantic_weight_{scale}"] = (
                scale_weights[..., scale_index].detach().mean()
            )
        return semantic_query, decision_residual, aux_exist_logits, debug


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
        semantic_instance_enabled: bool = False,
        semantic_instance_scales: list[str] | tuple[str, ...] | None = None,
        semantic_instance_after_layer: int = 1,
        semantic_instance_num_heads: int | None = None,
        semantic_instance_ff_dim: int | None = None,
        semantic_instance_dropout: float | None = None,
        semantic_instance_detach_query: bool = True,
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
        self.num_layers = int(num_layers)
        self.semantic_instance_enabled = bool(semantic_instance_enabled)
        self.semantic_instance_scales = tuple(semantic_instance_scales or ("p4", "p5"))
        self.semantic_instance_after_layer = int(semantic_instance_after_layer)
        self.semantic_instance_detach_query = bool(semantic_instance_detach_query)
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
        if self.semantic_instance_enabled and not 1 <= self.semantic_instance_after_layer <= self.num_layers:
            raise ValueError(
                "structured_query.semantic_instance.after_layer must be between 1 and "
                f"num_layers={self.num_layers}"
            )

        self.instance_tokens = nn.Embedding(self.num_instances, self.dim)
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
        # Keep construction of all baseline modules above unchanged.  Appending
        # the optional branch here also preserves baseline RNG initialization.
        self.semantic_query_norm = (
            nn.LayerNorm(self.dim) if self.semantic_instance_enabled else None
        )
        self.semantic_instance = (
            SemanticInstanceContext(
                dim=self.dim,
                scales=self.semantic_instance_scales,
                num_heads=int(semantic_instance_num_heads or num_heads),
                ff_dim=int(semantic_instance_ff_dim or ff_dim),
                dropout=float(dropout if semantic_instance_dropout is None else semantic_instance_dropout),
                exist_prior_prob=self.exist_prior_prob,
            )
            if self.semantic_instance_enabled
            else None
        )
        nn.init.constant_(self.range[-1].weight, 0.0)
        with torch.no_grad():
            self.range[-1].bias.copy_(torch.tensor([-2.0, 2.0]))
            if self.exist_prior_prob is not None:
                lane_logit = 0.5 * math.log(self.exist_prior_prob / (1.0 - self.exist_prior_prob))
                self.exist[-1].bias.copy_(self.exist[-1].bias.new_tensor([lane_logit, -lane_logit]))

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
        row_value_features, row_key_features = self._row_features(features)

        semantic_query = None
        semantic_decision_residual = None
        semantic_aux_exist_logits = None
        semantic_debug: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.layers, start=1):
            row_tokens = layer(row_tokens, row_value_features, row_key_features)
            if self.semantic_instance is not None and layer_index == self.semantic_instance_after_layer:
                if multi_scale_features is None:
                    raise ValueError(
                        "structured_query.semantic_instance is enabled but encoder did not provide multi_scale_features"
                    )
                assert self.semantic_query_norm is not None
                semantic_seed = row_tokens.mean(dim=2) + row_tokens.amax(dim=2) + instance.unsqueeze(0)
                if self.semantic_instance_detach_query:
                    semantic_seed = semantic_seed.detach()
                semantic_seed = self.semantic_query_norm(semantic_seed)
                (
                    semantic_query,
                    semantic_decision_residual,
                    semantic_aux_exist_logits,
                    semantic_debug,
                ) = self.semantic_instance(semantic_seed, multi_scale_features)

        row_tokens = self.row_norm(row_tokens)
        instance_residual = instance.unsqueeze(0).expand(b, -1, -1)
        lane_query = self.lane_norm(row_tokens.mean(dim=2) + row_tokens.amax(dim=2) + instance_residual)
        decision_query = (
            lane_query + semantic_decision_residual
            if semantic_decision_residual is not None
            else lane_query
        )
        row_x_logits = self.row_x(row_tokens)
        pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
        range_raw = self.range(lane_query)
        range_norm = sort_range_norm(torch.sigmoid(range_raw))
        quality_logits = self.quality(decision_query).squeeze(-1)
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "exist_logits": self.exist(decision_query),
            "row_x_logits": row_x_logits,
            "pred_x_rows": pred_x_rows,
            "range_raw": range_raw,
            "range_norm": range_norm,
            "quality_logits": quality_logits,
            "queries": lane_query,
            "decision_queries": decision_query,
            "structured_row_tokens": row_tokens,
            "structured_debug": {
                "structured_row_abs": row_tokens.detach().abs().mean(),
                "structured_feature_abs": row_value_features.detach().abs().mean(),
                **semantic_debug,
            },
        }
        if semantic_query is not None and semantic_aux_exist_logits is not None:
            output["semantic_instance_tokens"] = semantic_query
            output["semantic_aux_exist_logits"] = semantic_aux_exist_logits
        return output


def build_structured_query_head(model_cfg: dict[str, Any]) -> StructuredLaneQueryHead | None:
    structured_cfg = model_cfg.get("structured_query", {})
    if not bool(structured_cfg.get("enabled", False)):
        return None
    semantic_cfg = structured_cfg.get("semantic_instance", {})
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
        semantic_instance_enabled=bool(semantic_cfg.get("enabled", False)),
        semantic_instance_scales=list(semantic_cfg.get("scales", ["p4", "p5"])),
        semantic_instance_after_layer=int(semantic_cfg.get("after_layer", 1)),
        semantic_instance_num_heads=int(semantic_cfg.get("num_heads", structured_cfg.get("num_heads", model_cfg.get("num_heads", 8)))),
        semantic_instance_ff_dim=int(semantic_cfg.get("ff_dim", structured_cfg.get("ff_dim", model_cfg.get("decoder_ff_dim", 1024)))),
        semantic_instance_dropout=float(semantic_cfg.get("dropout", structured_cfg.get("dropout", model_cfg.get("dropout", 0.1)))),
        semantic_instance_detach_query=bool(semantic_cfg.get("detach_query", True)),
    )
