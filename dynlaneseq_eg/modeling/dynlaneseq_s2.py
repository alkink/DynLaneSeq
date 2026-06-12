from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import soft_expected_x
from .dynlaneseq_s0 import DynLaneSeqEncoder, GeometryGuidedQueryRefiner
from .evidence import CurveAlignedSampler, DynamicOffsetFusion, EvidenceAdapter, MultiScaleCurveAlignedSampler, SamplerCurriculum
from .heads_s0 import ExistenceHead, RangeHead, S0Heads
from .row_token_decoder import RowTokenDecoder
from .structured_queries import build_structured_query_head


class ActiveCorridorSearch(nn.Module):
    """Supervised soft-argmax search over lateral evidence around coarse lanes."""

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        offsets_px: list[float] | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        zero_init: bool = True,
        center_init_bias: float = 2.0,
    ):
        super().__init__()
        offsets = torch.tensor(offsets_px or [-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0])
        if offsets.ndim != 1 or offsets.numel() < 3:
            raise ValueError("ActiveCorridorSearch expects at least three lateral offsets")
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.num_offsets = int(offsets.numel())
        self.register_buffer("offsets_px", offsets.float())
        self.offset_embedding = nn.Parameter(torch.zeros(1, 1, 1, self.num_offsets, self.dim))
        nn.init.normal_(self.offset_embedding, std=0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        bias = torch.zeros(self.num_offsets)
        center_idx = int((offsets.abs()).argmin().item())
        bias[center_idx] = float(center_init_bias)
        self.offset_bias = nn.Parameter(bias)

    def forward(
        self,
        offset_samples: torch.Tensor,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, n, p, o, c = offset_samples.shape
        if o != self.num_offsets:
            raise ValueError(f"Expected {self.num_offsets} offset samples, got {o}")
        if p != self.num_rows:
            raise ValueError(f"Expected {self.num_rows} rows, got {p}")
        query = queries.unsqueeze(2).unsqueeze(3).expand(b, n, p, o, c)
        row = row_embedding.view(1, 1, p, 1, c).expand(b, n, p, o, c)
        token = offset_samples + query + row + self.offset_embedding.to(dtype=offset_samples.dtype)
        logits = self.net(token).squeeze(-1) + self.offset_bias.to(device=token.device, dtype=token.dtype)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=offset_samples.dtype)
        evidence = (offset_samples * weights.unsqueeze(-1)).sum(dim=3)
        offsets = self.offsets_px.to(device=offset_samples.device, dtype=offset_samples.dtype)
        pred_delta = (weights * offsets.view(1, 1, 1, o)).sum(dim=-1)
        entropy = -(weights.float() * weights.float().clamp_min(1e-6).log()).sum(dim=-1)
        center_idx = int((self.offsets_px.abs()).argmin().item())
        debug = {
            "active_offset_entropy": entropy.detach().mean(),
            "active_offset_max_prob": weights.detach().float().max(dim=-1).values.mean(),
            "active_offset_center_prob": weights.detach().float()[..., center_idx].mean(),
            "active_pred_delta_abs": pred_delta.detach().abs().mean(),
        }
        return evidence, pred_delta, logits, debug


class DynLaneSeqS2(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        evidence_cfg = model_cfg.get("evidence_sampler", {})
        multi_scale_cfg = model_cfg.get("multi_scale_evidence", {})
        active_cfg = model_cfg.get("active_corridor", {})
        geometry_cfg = model_cfg.get("s0_geometry_evidence", {})
        oracle_cfg = model_cfg.get("oracle_coarse", {})
        structured_cfg = model_cfg.get("structured_query", {})
        self.input_w = int(model_cfg.get("input_w", 800))
        self.input_h = int(model_cfg.get("input_h", 288))
        self.num_rows = int(model_cfg.get("num_rows", 72))
        self.x_bins = int(model_cfg.get("x_bins", 200))
        dim = int(model_cfg.get("dim", 256))
        self.structured_query_head = build_structured_query_head(model_cfg)
        self.active_corridor_enabled = bool(active_cfg.get("enabled", False))
        self.oracle_coarse_enabled = bool(oracle_cfg.get("enabled", False))
        self.oracle_score_logit = float(oracle_cfg.get("score_logit", 8.0))
        self.oracle_bg_logit = float(oracle_cfg.get("background_logit", 8.0))
        self.active_corridor_detach_center = bool(active_cfg.get("detach_center", True))
        self.active_corridor_detach_refined_x = bool(active_cfg.get("detach_refined_x_for_decoder", False))
        local_window_cfg = evidence_cfg.get("local_window", {})
        self.dynamic_offset_enabled = bool(local_window_cfg.get("enabled", False)) and str(
            local_window_cfg.get("aggregation", "mean")
        ).lower() in {"dynamic", "learned", "token"}
        self.multi_scale_enabled = bool(multi_scale_cfg.get("enabled", False))
        self.multi_scale_return_separate = bool(multi_scale_cfg.get("return_separate", False))
        self.multi_scale_scales = list(multi_scale_cfg.get("scales", ["p2", "p3", "p4"]))
        if self.multi_scale_enabled and self.dynamic_offset_enabled:
            raise ValueError("multi_scale_evidence and dynamic local-window aggregation should be ablated separately")
        self.s2_mode = str(model_cfg.get("s2_mode", "direct")).lower()
        if self.s2_mode not in {"direct", "residual"}:
            raise ValueError(f"Unsupported S2 mode: {self.s2_mode}")
        if self.structured_query_head is not None and self.s2_mode != "residual":
            raise ValueError("structured_query currently requires residual S2/S3 mode")
        if bool(geometry_cfg.get("enabled", False)) and bool(model_cfg.get("dynamic_evidence", {}).get("enabled", False)):
            raise ValueError("Use either dynamic_evidence v1 or s0_geometry_evidence v2, not both")
        if bool(geometry_cfg.get("enabled", False)) and self.s2_mode != "residual":
            raise ValueError("s0_geometry_evidence currently requires residual S2/S3 mode")
        if self.structured_query_head is not None and (
            bool(geometry_cfg.get("enabled", False))
            or bool(model_cfg.get("dynamic_evidence", {}).get("enabled", False))
            or bool(model_cfg.get("dynamic_proposal", {}).get("enabled", False))
            or self.oracle_coarse_enabled
        ):
            raise ValueError(
                "structured_query must be isolated from dynamic_evidence, dynamic_proposal, s0_geometry_evidence, and oracle_coarse"
            )
        if self.structured_query_head is not None and int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 20))) != int(
            model_cfg.get("num_slots", 20)
        ):
            raise ValueError("structured_query.num_instances must match model.num_slots for S1/S2/S3 stages")
        self.encoder = DynLaneSeqEncoder(cfg)
        if self.s2_mode == "residual":
            self.heads = S0Heads(
                dim=dim,
                num_rows=self.num_rows,
                x_bins=self.x_bins,
                input_w=self.input_w,
            )
            self.coarse_x_embed = nn.Linear(1, dim)
            self.residual_logit_scale = float(model_cfg.get("residual_logit_scale", 1.0))
            self.detach_coarse_x = bool(model_cfg.get("detach_coarse_x", False))
        else:
            self.exist_head = ExistenceHead(dim)
            self.range_head = RangeHead(dim)
        self.sampler = CurveAlignedSampler(
            input_w=self.input_w,
            input_h=self.input_h,
            num_rows=self.num_rows,
            local_window_enabled=bool(local_window_cfg.get("enabled", False)) and not self.dynamic_offset_enabled,
            offsets_px=local_window_cfg.get("offsets_px", [-8, -4, 0, 4, 8]),
        )
        self.active_corridor_sampler = (
            CurveAlignedSampler(
                input_w=self.input_w,
                input_h=self.input_h,
                num_rows=self.num_rows,
                local_window_enabled=False,
                offsets_px=active_cfg.get("offsets_px", [-32, -24, -16, -8, 0, 8, 16, 24, 32]),
            )
            if self.active_corridor_enabled
            else None
        )
        self.active_corridor = (
            ActiveCorridorSearch(
                dim=dim,
                num_rows=self.num_rows,
                offsets_px=active_cfg.get("offsets_px", [-32, -24, -16, -8, 0, 8, 16, 24, 32]),
                hidden_dim=int(active_cfg.get("hidden_dim", dim)),
                dropout=float(active_cfg.get("dropout", 0.0)),
                zero_init=bool(active_cfg.get("zero_init", True)),
                center_init_bias=float(active_cfg.get("center_init_bias", 2.0)),
            )
            if self.active_corridor_enabled
            else None
        )
        self.offset_fusion = (
            DynamicOffsetFusion(
                dim=dim,
                num_offsets=len(self.sampler.offsets_px),
                hidden_dim=int(local_window_cfg.get("hidden_dim", dim)),
                dropout=float(local_window_cfg.get("dropout", 0.0)),
                zero_init=bool(local_window_cfg.get("zero_init", True)),
            )
            if self.dynamic_offset_enabled
            else None
        )
        self.multi_scale_sampler = (
            MultiScaleCurveAlignedSampler(
                input_w=self.input_w,
                input_h=self.input_h,
                num_rows=self.num_rows,
                dim=dim,
                scales=self.multi_scale_scales,
                gate_hidden_dim=int(multi_scale_cfg.get("gate_hidden_dim", dim)),
                dropout=float(multi_scale_cfg.get("dropout", 0.0)),
                zero_init_gate=bool(multi_scale_cfg.get("zero_init_gate", True)),
                fusion_mode=str(multi_scale_cfg.get("fusion_mode", "weighted_sum")),
                base_scale=str(multi_scale_cfg.get("base_scale", "p2")),
                residual_scale_init=float(multi_scale_cfg.get("residual_scale_init", 0.0)),
                initial_gate_bias=multi_scale_cfg.get("initial_gate_bias"),
            )
            if self.multi_scale_enabled and not self.multi_scale_return_separate
            else None
        )
        self.curriculum = SamplerCurriculum(
            noise_std=float(evidence_cfg.get("noise_std", 3.0)),
            detach_sample_coords=bool(evidence_cfg.get("detach_sample_coords", True)),
            input_w=self.input_w,
        )
        self.s0_geometry_detach_draft = bool(geometry_cfg.get("detach_draft_x", True))
        self.s0_geometry_refiner = (
            GeometryGuidedQueryRefiner(
                dim=dim,
                input_h=self.input_h,
                input_w=self.input_w,
                num_rows=self.num_rows,
                hidden_dim=int(geometry_cfg.get("hidden_dim", dim)),
                dropout=float(geometry_cfg.get("dropout", 0.0)),
                pooling=str(geometry_cfg.get("pooling", "mean")),
                local_window_enabled=bool(geometry_cfg.get("local_window_enabled", False)),
                offsets_px=geometry_cfg.get("offsets_px"),
                local_reduce=str(geometry_cfg.get("local_reduce", "max")),
            )
            if bool(geometry_cfg.get("enabled", False))
            else None
        )
        self.adapter = EvidenceAdapter(dim=dim, gamma_init=float(model_cfg.get("evidence_gamma_init", 0.1)))
        self.row_embedding = nn.Embedding(self.num_rows, dim)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        self.row_decoder = RowTokenDecoder(
            num_rows=self.num_rows,
            dim=dim,
            x_bins=self.x_bins,
            num_layers=int(model_cfg.get("row_decoder_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ff_dim=int(model_cfg.get("row_decoder_ff_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            zero_init_head=bool(model_cfg.get("zero_init_residual_head", self.s2_mode == "residual")),
            local_attn_window=int(model_cfg.get("row_local_attn_window", 0)),
            visibility_head=bool(model_cfg.get("row_visibility", {}).get("enabled", False)),
        )
        if self.active_corridor_enabled and (self.multi_scale_enabled or self.dynamic_offset_enabled):
            raise ValueError("active_corridor should be tested separately from multi-scale and dynamic offset fusion")

    def build_coarse_tokens(
        self,
        queries: torch.Tensor,
        base_row_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_row_tokens is not None:
            return base_row_tokens
        b, n, d = queries.shape
        row_emb = self.row_embedding.weight.view(1, 1, self.num_rows, d)
        return queries.unsqueeze(2) + row_emb

    def build_oracle_coarse(
        self,
        queries: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        b, n, _ = queries.shape
        device = queries.device
        dtype = queries.dtype
        p = self.num_rows
        x_bins = self.x_bins
        lane_logit = float(self.oracle_score_logit)
        bg_logit = float(self.oracle_bg_logit)
        exist_logits = queries.new_empty((b, n, 2))
        exist_logits[..., 0] = -bg_logit
        exist_logits[..., 1] = bg_logit
        pred_x_rows = queries.new_zeros((b, n, p))
        range_norm = queries.new_zeros((b, n, 2))
        row_x_logits = queries.new_full((b, n, p, x_bins), -lane_logit)
        quality_logits = queries.new_full((b, n), -bg_logit)
        for bi, target in enumerate(targets):
            x_rows = target["x_rows"].to(device=device, dtype=dtype)
            valid = target["valid_mask"].to(device=device).bool()
            x_bin_targets = target["x_bins"].to(device=device).long()
            range_y = target["range_y"].to(device=device, dtype=dtype)
            lanes = min(int(x_rows.shape[0]), n)
            if lanes <= 0:
                continue
            exist_logits[bi, :lanes, 0] = lane_logit
            exist_logits[bi, :lanes, 1] = -lane_logit
            pred_x_rows[bi, :lanes] = x_rows[:lanes].clamp(0, self.input_w - 1)
            range_norm[bi, :lanes] = (range_y[:lanes] / float(self.input_h)).clamp(0.0, 1.0)
            quality_logits[bi, :lanes] = lane_logit
            bins = x_bin_targets[:lanes].clamp(0, x_bins - 1)
            row_x_logits[bi, :lanes].scatter_(-1, bins.unsqueeze(-1), lane_logit)
            row_x_logits[bi, :lanes] = row_x_logits[bi, :lanes].masked_fill(~valid[:lanes].unsqueeze(-1), 0.0)
        return {
            "exist_logits": exist_logits,
            "row_x_logits": row_x_logits,
            "pred_x_rows": pred_x_rows,
            "range_raw": range_norm,
            "range_norm": range_norm,
            "quality_logits": quality_logits,
            "quality_pred_x_rows": pred_x_rows,
        }

    def bridge_evidence(self, evidence: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return evidence, {}

    def sample_evidence(
        self,
        features: torch.Tensor | dict[str, torch.Tensor],
        sample_x: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.multi_scale_enabled and self.multi_scale_return_separate:
            if not isinstance(features, dict):
                raise TypeError("multi_scale_evidence.return_separate requires encoder multi_scale_features")
            evidence = {}
            debug = {}
            for scale_name in self.multi_scale_scales:
                if scale_name not in features:
                    raise KeyError(f"Missing multi-scale feature: {scale_name}")
                evidence[scale_name] = self.sampler(features[scale_name], sample_x)
                debug[f"ms_raw_{scale_name}_abs"] = evidence[scale_name].abs().mean().detach()
            return evidence, debug
        if self.multi_scale_sampler is not None:
            if not isinstance(features, dict):
                raise TypeError("multi_scale_evidence requires encoder multi_scale_features")
            return self.multi_scale_sampler(features, sample_x, queries, self.row_embedding.weight)
        if self.offset_fusion is None:
            if not isinstance(features, torch.Tensor):
                raise TypeError("single-scale evidence expects a feature tensor")
            return self.sampler(features, sample_x), {}
        if not isinstance(features, torch.Tensor):
            raise TypeError("dynamic offset fusion expects a single-scale feature tensor")
        offset_samples = self.sampler.sample_local_window(features, sample_x)
        evidence, offset_debug = self.offset_fusion(offset_samples, queries, self.row_embedding.weight)
        return evidence, offset_debug

    def sample_active_corridor(
        self,
        features: torch.Tensor,
        coarse_x: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.active_corridor is None or self.active_corridor_sampler is None:
            raise RuntimeError("active_corridor is not enabled")
        center_x = coarse_x.detach() if self.active_corridor_detach_center else coarse_x
        offset_samples = self.active_corridor_sampler.sample_local_window(features, center_x)
        evidence, pred_delta, logits, debug = self.active_corridor(offset_samples, queries, self.row_embedding.weight)
        refined_x = center_x + pred_delta
        offsets = self.active_corridor.offsets_px.to(device=features.device, dtype=features.dtype)
        debug = {
            **debug,
            "active_center_x_rows": center_x,
            "active_refined_x_rows": refined_x,
            "active_pred_delta_x_rows": pred_delta,
            "active_offset_logits": logits,
            "active_offsets_px": offsets,
            "active_refined_x_mean": refined_x.detach().mean(),
        }
        return evidence, refined_x, debug

    def build_final_tokens(
        self,
        queries: torch.Tensor,
        evidence: torch.Tensor,
        stage_extra: torch.Tensor | None = None,
        base_row_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _, d = evidence.shape
        if base_row_tokens is None:
            row_emb = self.row_embedding.weight.view(1, 1, self.num_rows, d)
            tokens = queries.unsqueeze(2) + row_emb
        else:
            tokens = base_row_tokens
        tokens = tokens + self.adapter(evidence)
        if stage_extra is not None:
            tokens = tokens + stage_extra
        return tokens

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
        matches: list[dict[str, torch.Tensor]] | None = None,
        sampler_alpha: float = 0.0,
        return_features: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        enc = self.encoder.forward_features(images)
        structured = self.structured_query_head(enc["features"]) if self.structured_query_head is not None else None
        q = structured["queries"] if structured is not None else enc["queries"]
        structured_row_tokens = structured["structured_row_tokens"] if structured is not None else None
        geometry_debug = None
        q_pre_geometry = None
        geometry_draft = None
        if self.s2_mode == "residual":
            if self.s0_geometry_refiner is not None:
                q_pre_geometry = q
                geometry_draft = self.heads(q)
                geometry_x = (
                    geometry_draft["pred_x_rows"].detach()
                    if self.s0_geometry_detach_draft
                    else geometry_draft["pred_x_rows"]
                )
                q, geometry_debug = self.s0_geometry_refiner(q, enc["features"], geometry_x)
            if self.oracle_coarse_enabled:
                if targets is None:
                    raise ValueError("oracle_coarse.enabled requires targets in forward")
                coarse = self.build_oracle_coarse(q, targets)
            else:
                coarse = structured if structured is not None else self.heads(q)
            coarse_x = coarse["pred_x_rows"]
            sample_x = self.curriculum.build_sample_x(coarse_x, targets, matches, alpha=float(sampler_alpha))
            features_for_sampling = enc["multi_scale_features"] if self.multi_scale_enabled else enc["features"]
            if self.active_corridor_enabled:
                if not isinstance(features_for_sampling, torch.Tensor):
                    raise TypeError("active_corridor currently expects a single feature tensor")
                evidence, refined_x, offset_debug = self.sample_active_corridor(features_for_sampling, coarse_x, q)
                sample_x_for_log = refined_x.detach()
                stage_x = refined_x.detach() if self.active_corridor_detach_refined_x else refined_x
            else:
                evidence, offset_debug = self.sample_evidence(features_for_sampling, sample_x, q)
                sample_x_for_log = sample_x
                stage_x = coarse_x.detach() if self.detach_coarse_x else coarse_x
            evidence, bridge_debug = self.bridge_evidence(evidence, q)
            stage_x_norm = (stage_x / float(self.input_w)).unsqueeze(-1)
            row = self.row_decoder(
                self.build_final_tokens(
                    q,
                    evidence,
                    stage_extra=self.coarse_x_embed(stage_x_norm),
                    base_row_tokens=structured_row_tokens,
                ),
                input_w=self.input_w,
            )
            base_logits = coarse["row_x_logits"].detach() if self.detach_coarse_x else coarse["row_x_logits"]
            row_x_logits = base_logits + self.residual_logit_scale * row["row_x_logits"]
            pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
            out = {
                "coarse": {
                    **coarse,
                    "row_hidden": row["row_hidden"],
                },
                "final": {
                    "exist_logits": coarse["exist_logits"],
                    "row_x_logits": row_x_logits,
                    "pred_x_rows": pred_x_rows,
                    "range_raw": coarse["range_raw"],
                    "range_norm": coarse["range_norm"],
                    "quality_logits": coarse["quality_logits"],
                    "quality_pred_x_rows": coarse["pred_x_rows"],
                    "row_hidden": row["row_hidden"],
                },
                "evidence": {
                    "sample_x_rows": sample_x_for_log,
                    "E_seq": evidence,
                    "evidence_scale": self.adapter.gamma,
                    **offset_debug,
                    **bridge_debug,
                    **(geometry_debug or {}),
                },
                "queries": q,
                "row_delta_logits": row["row_x_logits"],
            }
            if structured is not None:
                out["structured_row_tokens"] = structured_row_tokens
                out["structured_debug"] = structured.get("structured_debug", {})
            if geometry_debug is not None:
                out["queries_pre_geometry"] = q_pre_geometry
                out["geometry_evidence"] = geometry_debug
                out["s0_geometry_draft"] = geometry_draft
            if "row_visibility_logits" in row:
                out["final"]["row_visibility_logits"] = row["row_visibility_logits"]
        else:
            exist_logits = self.exist_head(q)
            range_raw, range_norm = self.range_head(q)
            coarse_row = self.row_decoder(self.build_coarse_tokens(q, base_row_tokens=structured_row_tokens), input_w=self.input_w)
            coarse_logits = coarse_row["row_x_logits"]
            coarse_x = coarse_row["pred_x_rows"]
            sample_x = self.curriculum.build_sample_x(coarse_x, targets, matches, alpha=float(sampler_alpha))
            features_for_sampling = enc["multi_scale_features"] if self.multi_scale_enabled else enc["features"]
            evidence, offset_debug = self.sample_evidence(features_for_sampling, sample_x, q)
            evidence, bridge_debug = self.bridge_evidence(evidence, q)
            row = self.row_decoder(
                self.build_final_tokens(q, evidence, base_row_tokens=structured_row_tokens),
                input_w=self.input_w,
            )
            out = {
                "coarse": {
                    "exist_logits": exist_logits,
                    "row_x_logits": coarse_logits,
                    "pred_x_rows": coarse_x,
                    "range_raw": range_raw,
                    "range_norm": range_norm,
                    "row_hidden": coarse_row["row_hidden"],
                },
                "final": {
                    "exist_logits": exist_logits,
                    "row_x_logits": row["row_x_logits"],
                    "pred_x_rows": row["pred_x_rows"],
                    "range_raw": range_raw,
                    "range_norm": range_norm,
                    "row_hidden": row["row_hidden"],
                },
                "evidence": {
                    "sample_x_rows": sample_x,
                    "E_seq": evidence,
                    "evidence_scale": self.adapter.gamma,
                    **offset_debug,
                    **bridge_debug,
                },
                "queries": q,
            }
            if "row_visibility_logits" in row:
                out["final"]["row_visibility_logits"] = row["row_visibility_logits"]
        for key, value in enc.items():
            if key.startswith("seg_logits") or key == "centerline_logits":
                out[key] = value
        if "dynamic_evidence" in enc:
            out["dynamic_evidence"] = enc["dynamic_evidence"]
        if "dynamic_proposals" in enc:
            out["dynamic_proposals"] = enc["dynamic_proposals"]
        if return_features:
            out["features"] = enc["features"]
            if "multi_scale_features" in enc:
                out["multi_scale_features"] = enc["multi_scale_features"]
        return out
