from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backbone_resnet import ResNet34Backbone
from .common import input_to_grid, sort_range_norm
from .cross_attention_decoder import LaneCrossAttentionDecoder
from .evidence import CurveAlignedSampler
from .fpn import SimpleFPN
from .heads_s0 import CenterlineAuxHead, S0Heads, SegAuxHead
from .lane_queries import LaneQueries
from .position_encoding import SinePositionEncoding2D
from .structured_queries import build_structured_query_head


class SlotDynamicEvidence(nn.Module):
    """Inject image evidence into decoded slot queries without changing slot order."""

    def __init__(
        self,
        dim: int = 256,
        num_slots: int = 20,
        input_w: int = 800,
        input_h: int = 288,
        num_points: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        x_delta_scale: float = 0.35,
        y_delta_scale: float = 0.05,
        base_x_min: float = 0.05,
        base_x_max: float = 0.95,
        y_positions: list[float] | None = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.num_points = int(num_points)
        self.x_delta_scale = float(x_delta_scale)
        self.y_delta_scale = float(y_delta_scale)
        if self.num_points < 1:
            raise ValueError("dynamic_evidence.num_points must be >= 1")
        if y_positions is None:
            y = torch.linspace(0.20, 0.92, self.num_points)
        else:
            if len(y_positions) != self.num_points:
                raise ValueError("dynamic_evidence.y_positions must match num_points")
            y = torch.tensor(y_positions, dtype=torch.float32)
        self.register_buffer("base_x", torch.linspace(float(base_x_min), float(base_x_max), self.num_slots))
        self.register_buffer("base_y", y.float().clamp(0.0, 1.0))
        self.coord_mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_points * 2),
        )
        self.point_embedding = nn.Parameter(torch.zeros(1, 1, self.num_points, self.dim))
        nn.init.normal_(self.point_embedding, std=0.02)
        self.point_score = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.dim),
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, n, c = queries.shape
        if n != self.num_slots:
            raise ValueError(f"Expected {self.num_slots} query slots, got {n}")
        raw_delta = self.coord_mlp(queries).view(b, n, self.num_points, 2)
        base_x = self.base_x.to(device=queries.device, dtype=queries.dtype).view(1, n, 1)
        base_y = self.base_y.to(device=queries.device, dtype=queries.dtype).view(1, 1, self.num_points)
        x_norm = (base_x + self.x_delta_scale * torch.tanh(raw_delta[..., 0])).clamp(0.0, 1.0)
        y_norm = (base_y + self.y_delta_scale * torch.tanh(raw_delta[..., 1])).clamp(0.0, 1.0)
        x = x_norm * float(self.input_w - 1)
        y = y_norm * float(self.input_h - 1)
        grid = input_to_grid(x, y, self.input_w, self.input_h).view(b, n * self.num_points, 1, 2)
        sampled = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous().view(b, n, self.num_points, c)
        point_token = sampled + queries.unsqueeze(2) + self.point_embedding.to(device=queries.device, dtype=queries.dtype)
        point_logits = self.point_score(point_token).squeeze(-1)
        point_weights = torch.softmax(point_logits.float(), dim=-1).to(dtype=sampled.dtype)
        pooled = (sampled * point_weights.unsqueeze(-1)).sum(dim=2)
        delta = self.adapter(pooled)
        debug = {
            "dynamic_evidence_refs": torch.stack([x_norm, y_norm], dim=-1).detach(),
            "dynamic_evidence_delta_abs": delta.detach().abs().mean(),
            "dynamic_evidence_x_mean": x.detach().mean(),
            "dynamic_evidence_weight_entropy": (
                -(point_weights.float() * point_weights.float().clamp_min(1e-6).log()).sum(dim=-1).mean().detach()
            ),
        }
        return queries + delta, debug


class GeometryGuidedQueryRefiner(nn.Module):
    """Refine slot queries by sampling evidence along each slot's draft lane geometry."""

    def __init__(
        self,
        dim: int = 256,
        input_w: int = 800,
        input_h: int = 288,
        num_rows: int = 72,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        pooling: str = "mean",
        local_window_enabled: bool = False,
        offsets_px: list[float] | None = None,
        local_reduce: str = "max",
    ):
        super().__init__()
        self.pooling = str(pooling).lower()
        if self.pooling not in {"mean", "mean_max", "attention"}:
            raise ValueError("s0_geometry_evidence.pooling must be 'mean', 'mean_max', or 'attention'")
        self.local_reduce = str(local_reduce).lower()
        if self.local_reduce not in {"mean", "max"}:
            raise ValueError("s0_geometry_evidence.local_reduce must be 'mean' or 'max'")
        self.local_window_enabled = bool(local_window_enabled)
        self.sampler = CurveAlignedSampler(
            input_w=input_w,
            input_h=input_h,
            num_rows=num_rows,
            local_window_enabled=self.local_window_enabled,
            offsets_px=offsets_px,
        )
        self.row_embedding = nn.Parameter(torch.zeros(1, 1, num_rows, dim))
        nn.init.normal_(self.row_embedding, std=0.02)
        self.row_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        adapter_dim = dim * 2 if self.pooling == "mean_max" else dim
        self.adapter = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        queries: torch.Tensor,
        features: torch.Tensor,
        draft_x_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        local_window_abs = features.new_tensor(0.0)
        if self.local_window_enabled:
            local_evidence = self.sampler.sample_local_window(features, draft_x_rows)
            local_window_abs = local_evidence.detach().abs().mean()
            if self.local_reduce == "mean":
                evidence = local_evidence.mean(dim=3)
            else:
                evidence = local_evidence.amax(dim=3)
        else:
            evidence = self.sampler(features, draft_x_rows)
        if self.pooling == "attention":
            token = evidence + queries.unsqueeze(2) + self.row_embedding.to(device=evidence.device, dtype=evidence.dtype)
            row_logits = self.row_score(token).squeeze(-1)
            row_weights = torch.softmax(row_logits.float(), dim=-1).to(dtype=evidence.dtype)
            pooled = (evidence * row_weights.unsqueeze(-1)).sum(dim=2)
            entropy = -(row_weights.float() * row_weights.float().clamp_min(1e-6).log()).sum(dim=-1).mean().detach()
        elif self.pooling == "mean_max":
            pooled = torch.cat([evidence.mean(dim=2), evidence.amax(dim=2)], dim=-1)
            entropy = evidence.new_tensor(0.0)
        else:
            pooled = evidence.mean(dim=2)
            entropy = evidence.new_tensor(0.0)
        delta = self.adapter(pooled)
        debug = {
            "s0_geometry_evidence_abs": evidence.detach().abs().mean(),
            "s0_geometry_local_window_abs": local_window_abs,
            "s0_geometry_delta_abs": delta.detach().abs().mean(),
            "s0_geometry_sample_x_mean": draft_x_rows.detach().mean(),
            "s0_geometry_row_entropy": entropy,
        }
        return queries + delta, debug


class ImageGroundedAffineRefiner(nn.Module):
    """Geometry-only affine correction from curve-aligned image evidence.

    The refiner deliberately predicts only two numbers per lane instance:
    a horizontal shift and a lane-level tilt.  It does not change existence,
    quality, range, or the number/order of proposals.
    """

    def __init__(
        self,
        dim: int = 256,
        input_w: int = 800,
        input_h: int = 288,
        num_rows: int = 72,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        offsets_px: list[float] | None = None,
        max_shift_px: float = 32.0,
        max_tilt_px: float = 24.0,
        local_reduce: str = "mean_max",
        row_pooling: str = "mean_max",
        detach_sample_x: bool = True,
        detach_base_x: bool = True,
        y_norm_mode: str = "centered",
        use_query: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.num_rows = int(num_rows)
        self.max_shift_px = float(max_shift_px)
        self.max_tilt_px = float(max_tilt_px)
        self.detach_sample_x = bool(detach_sample_x)
        self.detach_base_x = bool(detach_base_x)
        self.use_query = bool(use_query)
        self.local_reduce = str(local_reduce).lower()
        if self.local_reduce not in {"mean", "max", "mean_max"}:
            raise ValueError("igar.local_reduce must be 'mean', 'max', or 'mean_max'")
        self.row_pooling = str(row_pooling).lower()
        if self.row_pooling not in {"mean", "max", "mean_max"}:
            raise ValueError("igar.row_pooling must be 'mean', 'max', or 'mean_max'")
        self.y_norm_mode = str(y_norm_mode).lower()
        if self.y_norm_mode not in {"centered", "zero_one"}:
            raise ValueError("igar.y_norm_mode must be 'centered' or 'zero_one'")
        self.sampler = CurveAlignedSampler(
            input_w=input_w,
            input_h=input_h,
            num_rows=num_rows,
            local_window_enabled=True,
            offsets_px=offsets_px or [-16.0, -8.0, 0.0, 8.0, 16.0],
        )
        local_dim = self.dim * (2 if self.local_reduce == "mean_max" else 1)
        pooled_dim = local_dim * (2 if self.row_pooling == "mean_max" else 1)
        if self.use_query:
            pooled_dim += self.dim
        self.net = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _reduce_offsets(self, samples: torch.Tensor) -> torch.Tensor:
        if self.local_reduce == "mean":
            return samples.mean(dim=3)
        if self.local_reduce == "max":
            return samples.amax(dim=3)
        return torch.cat([samples.mean(dim=3), samples.amax(dim=3)], dim=-1)

    def _pool_rows(self, row_features: torch.Tensor) -> torch.Tensor:
        if self.row_pooling == "mean":
            return row_features.mean(dim=2)
        if self.row_pooling == "max":
            return row_features.amax(dim=2)
        return torch.cat([row_features.mean(dim=2), row_features.amax(dim=2)], dim=-1)

    def _y_norm(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.y_norm_mode == "zero_one":
            return torch.linspace(0.0, 1.0, self.num_rows, device=device, dtype=dtype)
        return torch.linspace(-1.0, 1.0, self.num_rows, device=device, dtype=dtype)

    def forward(
        self,
        coarse_x_rows: torch.Tensor,
        features: torch.Tensor,
        queries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample_x = coarse_x_rows.detach() if self.detach_sample_x else coarse_x_rows
        local_samples = self.sampler.sample_local_window(features, sample_x)
        row_features = self._reduce_offsets(local_samples)
        pooled = self._pool_rows(row_features)
        if self.use_query:
            if queries is None:
                raise RuntimeError("IGAR configured with use_query=True but queries were not provided")
            pooled = torch.cat([pooled, queries.to(dtype=pooled.dtype)], dim=-1)
        raw = self.net(pooled)
        shift = torch.tanh(raw[..., 0]) * self.max_shift_px
        tilt = torch.tanh(raw[..., 1]) * self.max_tilt_px
        y_norm = self._y_norm(coarse_x_rows.device, coarse_x_rows.dtype).view(1, 1, self.num_rows)
        delta = shift.unsqueeze(-1).to(dtype=coarse_x_rows.dtype) + tilt.unsqueeze(-1).to(dtype=coarse_x_rows.dtype) * y_norm
        base = coarse_x_rows.detach() if self.detach_base_x else coarse_x_rows
        final_x = (base + delta).clamp(0.0, float(self.input_w - 1))
        debug = {
            "igar_shift_abs": shift.detach().abs().mean(),
            "igar_tilt_abs": tilt.detach().abs().mean(),
            "igar_delta_abs": delta.detach().abs().mean(),
            "igar_evidence_abs": local_samples.detach().abs().mean(),
            "igar_sample_x_mean": sample_x.detach().mean(),
        }
        return final_x, debug


class DynamicProposalGenerator(nn.Module):
    """Dense, image-conditioned lane proposal head kept separate from static slots."""

    def __init__(
        self,
        dim: int = 256,
        input_w: int = 800,
        input_h: int = 288,
        num_rows: int = 72,
        top_k: int = 16,
        hidden_dim: int = 256,
        heatmap_bias_init: float = -4.6,
    ):
        super().__init__()
        self.dim = int(dim)
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.num_rows = int(num_rows)
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("dynamic_proposal.top_k must be >= 1")
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(dim, int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), 1, kernel_size=1),
        )
        self.x_rows_head = nn.Sequential(
            nn.Conv2d(dim, int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), self.num_rows, kernel_size=1),
        )
        self.range_head = nn.Sequential(
            nn.Conv2d(dim, int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), 2, kernel_size=1),
        )
        self.geom_encoder = nn.Sequential(
            nn.Linear(self.num_rows + 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.type_embedding = nn.Embedding(2, dim)
        nn.init.constant_(self.heatmap_head[-1].bias, float(heatmap_bias_init))

    @staticmethod
    def gather_feat(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        dim = feat.shape[-1]
        return feat.gather(1, indices.unsqueeze(-1).expand(-1, -1, dim))

    def forward(self, features: torch.Tensor, pos_embed: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        b, c, h, w = features.shape
        heatmap_logits = self.heatmap_head(features)
        dense_x_rows = torch.sigmoid(self.x_rows_head(features)) * float(self.input_w - 1)
        dense_range = sort_range_norm(torch.sigmoid(self.range_head(features)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        with torch.no_grad():
            heatmap_prob = torch.sigmoid(heatmap_logits)
            pooled = F.max_pool2d(heatmap_prob, kernel_size=3, stride=1, padding=1)
            keep = heatmap_prob == pooled
            scores = heatmap_prob.masked_fill(~keep, 0.0).flatten(1)
            k = min(self.top_k, int(scores.shape[1]))
            _, topk_indices = torch.topk(scores, k=k, dim=1)

        feature_flat = features.flatten(2).transpose(1, 2).contiguous()
        pos_flat = pos_embed.expand(b, -1, -1, -1).flatten(2).transpose(1, 2).contiguous()
        x_flat = dense_x_rows.flatten(2).transpose(1, 2).contiguous()
        range_flat = dense_range.flatten(2).transpose(1, 2).contiguous()
        heatmap_flat = heatmap_logits.flatten(1)

        feat_gathered = self.gather_feat(feature_flat, topk_indices)
        pos_gathered = self.gather_feat(pos_flat, topk_indices)
        x_gathered = self.gather_feat(x_flat, topk_indices)
        range_gathered = self.gather_feat(range_flat, topk_indices)
        score_logits = heatmap_flat.gather(1, topk_indices)
        geom_input = torch.cat([x_gathered / float(self.input_w), range_gathered], dim=-1)
        q_dyn = (
            feat_gathered
            + pos_gathered
            + self.geom_encoder(geom_input)
            + self.type_embedding.weight[1].to(device=features.device, dtype=features.dtype).view(1, 1, c)
        )
        stage = {
            "pred_x_rows": x_gathered,
            "range_norm": range_gathered,
            "exist_logits": torch.stack([score_logits, -score_logits], dim=-1),
            "quality_logits": score_logits,
            "quality_pred_x_rows": x_gathered,
            "proposal_indices": topk_indices,
            "proposal_scores": torch.sigmoid(score_logits),
        }
        dense = {
            "heatmap_logits": heatmap_logits,
            "x_rows": dense_x_rows,
            "range_norm": dense_range,
        }
        return {
            "stage": stage,
            "dense": dense,
            "queries": q_dyn,
            "topk_indices": topk_indices,
            "debug": {
                "dynamic_proposal_score_mean": torch.sigmoid(score_logits).detach().mean(),
                "dynamic_proposal_x_mean": x_gathered.detach().mean(),
            },
        }


class DynLaneSeqEncoder(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.get("model", cfg)
        dim = int(model_cfg.get("dim", 256))
        fpn_channels = int(model_cfg.get("fpn_channels", 128))
        num_slots = int(model_cfg.get("num_slots", 20))
        input_h = int(model_cfg.get("input_h", 288))
        input_w = int(model_cfg.get("input_w", 800))
        self.query_content_init = str(model_cfg.get("query_content_init", "learned")).lower()
        self.memory_value_with_pos = bool(model_cfg.get("memory_value_with_pos", False))
        self.freeze_backbone_bn = bool(model_cfg.get("freeze_backbone_bn", False))
        self.freeze_all_bn = bool(model_cfg.get("freeze_all_bn", False))
        seg_aux_cfg = model_cfg.get("seg_aux", {})
        centerline_aux_cfg = model_cfg.get("centerline_aux", {})
        dynamic_evidence_cfg = model_cfg.get("dynamic_evidence", {})
        dynamic_proposal_cfg = model_cfg.get("dynamic_proposal", {})
        ms_cfg = model_cfg.get("multi_scale_evidence", {})
        self.multi_scale_enabled = bool(ms_cfg.get("enabled", False))
        self.multi_scale_names = list(ms_cfg.get("scales", ["p2", "p3", "p4"]))
        self.seg_aux_enabled = bool(seg_aux_cfg.get("enabled", False))
        self.centerline_aux_enabled = bool(centerline_aux_cfg.get("enabled", False))
        self.dynamic_evidence_enabled = bool(dynamic_evidence_cfg.get("enabled", False))
        self.dynamic_proposal_enabled = bool(dynamic_proposal_cfg.get("enabled", False))
        self.seg_aux_extra_scales = list(seg_aux_cfg.get("extra_scales", []))
        self.backbone = ResNet34Backbone(pretrained=bool(model_cfg.get("pretrained_backbone", True)))
        self.fpn = SimpleFPN(out_channels=fpn_channels)
        self.proj = nn.Conv2d(fpn_channels, dim, 1)
        self.ms_proj = nn.ModuleDict(
            {
                name: nn.Conv2d(fpn_channels, dim, 1)
                for name in self.multi_scale_names
                if name != "p2"
            }
        )
        self.seg_aux_head = (
            SegAuxHead(
                dim=dim,
                input_h=input_h,
                input_w=input_w,
                dropout=float(seg_aux_cfg.get("dropout", 0.1)),
            )
            if self.seg_aux_enabled
            else None
        )
        self.seg_aux_extra_heads = nn.ModuleDict(
            {
                name: SegAuxHead(
                    dim=dim,
                    input_h=input_h,
                    input_w=input_w,
                    dropout=float(seg_aux_cfg.get("dropout", 0.1)),
                )
                for name in self.seg_aux_extra_scales
            }
        )
        self.centerline_aux_head = (
            CenterlineAuxHead(
                dim=dim,
                num_rows=int(model_cfg.get("num_rows", 72)),
                x_bins=int(model_cfg.get("x_bins", 200)),
                dropout=float(centerline_aux_cfg.get("dropout", seg_aux_cfg.get("dropout", 0.1))),
            )
            if self.centerline_aux_enabled
            else None
        )
        self.dynamic_evidence = (
            SlotDynamicEvidence(
                dim=dim,
                num_slots=num_slots,
                input_h=input_h,
                input_w=input_w,
                num_points=int(dynamic_evidence_cfg.get("num_points", 5)),
                hidden_dim=int(dynamic_evidence_cfg.get("hidden_dim", dim)),
                dropout=float(dynamic_evidence_cfg.get("dropout", 0.0)),
                x_delta_scale=float(dynamic_evidence_cfg.get("x_delta_scale", 0.35)),
                y_delta_scale=float(dynamic_evidence_cfg.get("y_delta_scale", 0.05)),
                base_x_min=float(dynamic_evidence_cfg.get("base_x_min", 0.05)),
                base_x_max=float(dynamic_evidence_cfg.get("base_x_max", 0.95)),
                y_positions=dynamic_evidence_cfg.get("y_positions"),
            )
            if self.dynamic_evidence_enabled
            else None
        )
        self.dynamic_proposal = (
            DynamicProposalGenerator(
                dim=dim,
                input_h=input_h,
                input_w=input_w,
                num_rows=int(model_cfg.get("num_rows", 72)),
                top_k=int(dynamic_proposal_cfg.get("top_k", 16)),
                hidden_dim=int(dynamic_proposal_cfg.get("hidden_dim", dim)),
                heatmap_bias_init=float(dynamic_proposal_cfg.get("heatmap_bias_init", -4.6)),
            )
            if self.dynamic_proposal_enabled
            else None
        )
        self.pos = SinePositionEncoding2D(dim=dim)
        self.queries = LaneQueries(num_slots=num_slots, dim=dim)
        self.decoder = LaneCrossAttentionDecoder(
            num_layers=int(model_cfg.get("decoder_layers", 2)),
            dim=dim,
            num_heads=int(model_cfg.get("num_heads", 8)),
            ff_dim=int(model_cfg.get("decoder_ff_dim", 1024)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        if self.freeze_all_bn:
            self._set_all_bn_eval()
        elif self.freeze_backbone_bn:
            self._set_backbone_bn_eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.freeze_all_bn:
                self._set_all_bn_eval()
            elif self.freeze_backbone_bn:
                self._set_backbone_bn_eval()
        return self

    def _set_backbone_bn_eval(self) -> None:
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

    def _set_all_bn_eval(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad_(False)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(images)
        if self.multi_scale_enabled:
            pyramid = self.fpn(feats, return_pyramid=True)
            f_proj = self.proj(pyramid["p2"])
            multi_scale_features = {"p2": f_proj}
            for name in self.multi_scale_names:
                if name == "p2":
                    continue
                multi_scale_features[name] = self.ms_proj[name](pyramid[name])
        else:
            fpn = self.fpn(feats)
            f_proj = self.proj(fpn)
            multi_scale_features = None
        pos_embed = self.pos(f_proj)
        f_pos = f_proj + pos_embed
        mem_key = f_pos.flatten(2).transpose(1, 2).contiguous()
        mem_value_src = f_pos if self.memory_value_with_pos else f_proj
        mem_value = mem_value_src.flatten(2).transpose(1, 2).contiguous()
        q_seed = self.queries(images.shape[0])
        if self.query_content_init == "zero":
            q_content = torch.zeros_like(q_seed)
            q_pos = q_seed
        elif self.query_content_init == "learned":
            q_content = q_seed
            q_pos = torch.zeros_like(q_seed)
        elif self.query_content_init == "learned_pos":
            q_content = q_seed
            q_pos = q_seed
        else:
            raise ValueError(f"Unsupported query_content_init: {self.query_content_init}")
        q1 = self.decoder(q_content, q_pos, mem_key, mem_value)
        q1_pre_dynamic = q1
        dynamic_debug = None
        if self.dynamic_evidence is not None:
            q1, dynamic_debug = self.dynamic_evidence(q1, f_proj)
        out = {"features": f_proj, "memory": mem_value, "memory_key": mem_key, "queries": q1, "q0": q_seed}
        if self.dynamic_proposal is not None:
            out["dynamic_proposals"] = self.dynamic_proposal(f_proj, pos_embed)
        if dynamic_debug is not None:
            out["queries_pre_dynamic"] = q1_pre_dynamic
            out["dynamic_evidence"] = dynamic_debug
        if multi_scale_features is not None:
            out["multi_scale_features"] = multi_scale_features
        if self.seg_aux_head is not None:
            out["seg_logits"] = self.seg_aux_head(f_proj)
        if self.centerline_aux_head is not None:
            out["centerline_logits"] = self.centerline_aux_head(f_proj)
        if multi_scale_features is not None:
            for name, head in self.seg_aux_extra_heads.items():
                if name not in multi_scale_features:
                    raise KeyError(f"seg_aux extra scale {name!r} is not in multi_scale_features")
                out[f"seg_logits_{name}"] = head(multi_scale_features[name])
        return out


class DynLaneSeqS0(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        geometry_cfg = model_cfg.get("s0_geometry_evidence", {})
        igar_cfg = model_cfg.get("igar", {})
        structured_cfg = model_cfg.get("structured_query", {})
        dynamic_row_evidence_cfg = structured_cfg.get("dynamic_row_evidence", {})
        orthogonal_grounder_cfg = structured_cfg.get("orthogonal_grounder", {})
        orthogonal_verifier_cfg = structured_cfg.get("orthogonal_verifier", {})
        self.encoder = DynLaneSeqEncoder(cfg)
        self.heads = S0Heads(
            dim=int(model_cfg.get("dim", 256)),
            num_rows=int(model_cfg.get("num_rows", 72)),
            x_bins=int(model_cfg.get("x_bins", 200)),
            input_w=int(model_cfg.get("input_w", 800)),
        )
        self.structured_query_head = build_structured_query_head(model_cfg)
        if bool(geometry_cfg.get("enabled", False)) and bool(model_cfg.get("dynamic_evidence", {}).get("enabled", False)):
            raise ValueError("Use either dynamic_evidence v1 or s0_geometry_evidence v2, not both")
        if self.structured_query_head is not None and (
            bool(geometry_cfg.get("enabled", False))
            or bool(model_cfg.get("dynamic_evidence", {}).get("enabled", False))
            or bool(model_cfg.get("dynamic_proposal", {}).get("enabled", False))
        ):
            raise ValueError("structured_query must be isolated from dynamic_evidence, dynamic_proposal, and s0_geometry_evidence")
        self.s0_geometry_detach_draft = bool(geometry_cfg.get("detach_draft_x", True))
        self.s0_geometry_refiner = (
            GeometryGuidedQueryRefiner(
                dim=int(model_cfg.get("dim", 256)),
                input_h=int(model_cfg.get("input_h", 288)),
                input_w=int(model_cfg.get("input_w", 800)),
                num_rows=int(model_cfg.get("num_rows", 72)),
                hidden_dim=int(geometry_cfg.get("hidden_dim", model_cfg.get("dim", 256))),
                dropout=float(geometry_cfg.get("dropout", 0.0)),
                pooling=str(geometry_cfg.get("pooling", "mean")),
                local_window_enabled=bool(geometry_cfg.get("local_window_enabled", False)),
                offsets_px=geometry_cfg.get("offsets_px"),
                local_reduce=str(geometry_cfg.get("local_reduce", "max")),
            )
            if bool(geometry_cfg.get("enabled", False))
            else None
        )
        self.igar_refiner = (
            ImageGroundedAffineRefiner(
                dim=int(model_cfg.get("dim", 256)),
                input_h=int(model_cfg.get("input_h", 288)),
                input_w=int(model_cfg.get("input_w", 800)),
                num_rows=int(model_cfg.get("num_rows", 72)),
                hidden_dim=int(igar_cfg.get("hidden_dim", 128)),
                dropout=float(igar_cfg.get("dropout", 0.0)),
                offsets_px=igar_cfg.get("offsets_px"),
                max_shift_px=float(igar_cfg.get("max_shift_px", 32.0)),
                max_tilt_px=float(igar_cfg.get("max_tilt_px", 24.0)),
                local_reduce=str(igar_cfg.get("local_reduce", "mean_max")),
                row_pooling=str(igar_cfg.get("row_pooling", "mean_max")),
                detach_sample_x=bool(igar_cfg.get("detach_sample_x", True)),
                detach_base_x=bool(igar_cfg.get("detach_base_x", True)),
                y_norm_mode=str(igar_cfg.get("y_norm_mode", "centered")),
                use_query=bool(igar_cfg.get("use_query", False)),
            )
            if bool(igar_cfg.get("enabled", False))
            else None
        )
        if bool(igar_cfg.get("freeze_base", False)):
            if self.igar_refiner is None:
                raise ValueError("igar.freeze_base requires igar.enabled=true")
            self.freeze_non_igar_parameters()
        if bool(dynamic_row_evidence_cfg.get("freeze_base", False)):
            if self.structured_query_head is None or self.structured_query_head.dynamic_row_evidence_head is None:
                raise ValueError(
                    "structured_query.dynamic_row_evidence.freeze_base requires dynamic_row_evidence.enabled=true"
                )
            self.freeze_non_dynamic_row_evidence_parameters()
        if bool(orthogonal_grounder_cfg.get("freeze_base", False)):
            if self.structured_query_head is None or self.structured_query_head.orthogonal_grounder is None:
                raise ValueError(
                    "structured_query.orthogonal_grounder.freeze_base requires orthogonal_grounder.enabled=true"
                )
            self.freeze_non_orthogonal_grounder_parameters()
        if bool(orthogonal_verifier_cfg.get("freeze_base", False)):
            if self.structured_query_head is None or self.structured_query_head.orthogonal_verifier is None:
                raise ValueError(
                    "structured_query.orthogonal_verifier.freeze_base requires orthogonal_verifier.enabled=true"
                )
            self.freeze_non_orthogonal_verifier_parameters()

    def freeze_non_igar_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("igar_refiner."))

    def freeze_non_dynamic_row_evidence_parameters(self) -> None:
        prefix = "structured_query_head.dynamic_row_evidence_head."
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))

    def freeze_non_orthogonal_grounder_parameters(self) -> None:
        prefix = "structured_query_head.orthogonal_grounder."
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))

    def freeze_non_orthogonal_verifier_parameters(self) -> None:
        prefix = "structured_query_head.orthogonal_verifier."
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))

    def apply_igar(self, coarse: dict[str, torch.Tensor], features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.igar_refiner is None:
            return coarse
        final_x, igar_debug = self.igar_refiner(
            coarse["pred_x_rows"],
            features,
            queries=coarse.get("queries"),
        )
        final = dict(coarse)
        final["pred_x_rows"] = final_x
        final["quality_pred_x_rows"] = final_x
        out = dict(final)
        out["coarse"] = coarse
        out["final"] = final
        out["igar"] = igar_debug
        return out

    def forward(self, images: torch.Tensor, targets=None, return_features: bool = False) -> dict[str, torch.Tensor]:
        enc = self.encoder.forward_features(images)
        if self.structured_query_head is not None:
            out = self.structured_query_head(enc["features"])
            out["memory"] = enc["memory"]
            out["memory_key"] = enc["memory_key"]
            out["q0"] = enc["q0"]
        else:
            q = enc["queries"]
            if self.s0_geometry_refiner is not None:
                draft = self.heads(q)
                sample_x = draft["pred_x_rows"].detach() if self.s0_geometry_detach_draft else draft["pred_x_rows"]
                q_final, geometry_debug = self.s0_geometry_refiner(q, enc["features"], sample_x)
                final = self.heads(q_final)
                out = dict(final)
                out["coarse"] = draft
                out["final"] = final
                out["queries_pre_geometry"] = q
                out["queries"] = q_final
                out["geometry_evidence"] = geometry_debug
            else:
                out = self.heads(q)
                out["queries"] = q
        out = self.apply_igar(out, enc["features"])
        if "seg_logits" in enc:
            out["seg_logits"] = enc["seg_logits"]
        if "centerline_logits" in enc:
            out["centerline_logits"] = enc["centerline_logits"]
        if "dynamic_evidence" in enc:
            out["dynamic_evidence"] = enc["dynamic_evidence"]
        if "dynamic_proposals" in enc:
            out["dynamic_proposals"] = enc["dynamic_proposals"]
        if return_features:
            out["features"] = enc["features"]
        return out
