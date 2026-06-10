from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import fixed_y_rows, sort_range_norm
from .dynlaneseq_s2 import DynLaneSeqS2
from .evidence import AsymmetricContextModulationBridge, DynamicDepthwiseBridge, FiLMBridge, SequenceLowRankBridge


class DynLaneSeqS3(DynLaneSeqS2):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        model_cfg = cfg.get("model", cfg)
        bridge_cfg = model_cfg.get("bridge", {})
        decision_cfg = model_cfg.get("final_decision", {})
        quality_calib_cfg = model_cfg.get("quality_calibrator", {})
        dim = int(model_cfg.get("dim", 256))
        self.final_decision_enabled = bool(decision_cfg.get("enabled", False))
        self.final_decision_pooling = str(decision_cfg.get("pooling", "range_mean")).lower()
        self.final_decision_detach_base = bool(decision_cfg.get("detach_base", True))
        if self.final_decision_enabled:
            self.delta_exist_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
            nn.init.zeros_(self.delta_exist_head[-1].weight)
            nn.init.zeros_(self.delta_exist_head[-1].bias)
        self.quality_calibrator_enabled = bool(quality_calib_cfg.get("enabled", False))
        self.quality_calibrator_detach_base = bool(quality_calib_cfg.get("detach_base", True))
        self.quality_calibrator_detach_row_hidden = bool(quality_calib_cfg.get("detach_row_hidden", True))
        self.quality_calibrator_calibrate_exist = bool(quality_calib_cfg.get("calibrate_exist", True))
        self.quality_calibrator_exist_delta_scale = float(quality_calib_cfg.get("exist_delta_scale", 1.0))
        self.quality_calibrator_quality_base = str(quality_calib_cfg.get("quality_base", "none")).lower()
        self.quality_calibrator_range_padding_px = float(quality_calib_cfg.get("range_padding_px", 0.0))
        self.quality_calibrator_pooling = str(quality_calib_cfg.get("pooling", "range_mean")).lower()
        self.quality_calibrator_curvature_feature = bool(quality_calib_cfg.get("curvature_feature", False))
        self.quality_calibrator_stats_dim = 12 if self.quality_calibrator_curvature_feature else 10
        quality_pool_dim = dim * 2 if self.quality_calibrator_pooling == "range_mean_max" else dim
        if self.quality_calibrator_enabled:
            hidden_dim = int(quality_calib_cfg.get("hidden_dim", dim))
            dropout = float(quality_calib_cfg.get("dropout", 0.1))
            self.quality_calibrator = nn.Sequential(
                nn.Linear(quality_pool_dim + self.quality_calibrator_stats_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
            )
            nn.init.zeros_(self.quality_calibrator[-1].weight)
            nn.init.zeros_(self.quality_calibrator[-1].bias)
        bridge_type = bridge_cfg.get("type", "low_rank_sequence")
        if bridge_type == "film":
            self.bridge = FiLMBridge(dim=dim)
        elif bridge_type == "low_rank_sequence":
            self.bridge = SequenceLowRankBridge(
                dim=dim,
                rank=int(bridge_cfg.get("rank", 16)),
                kernel_size=int(bridge_cfg.get("kernel_size", 3)),
                bridge_scale_init=float(bridge_cfg.get("bridge_scale_init", 0.1)),
                force_fp32=bool(bridge_cfg.get("force_fp32", True)),
                param_clip=float(bridge_cfg["param_clip"]) if "param_clip" in bridge_cfg else None,
                delta_clip=float(bridge_cfg["delta_clip"]) if "delta_clip" in bridge_cfg else None,
            )
        elif bridge_type == "dynamic_depthwise_sequence":
            self.bridge = DynamicDepthwiseBridge(
                dim=dim,
                kernel_size=int(bridge_cfg.get("kernel_size", 3)),
                bridge_scale_init=float(bridge_cfg.get("bridge_scale_init", 0.0)),
                force_fp32=bool(bridge_cfg.get("force_fp32", True)),
                param_clip=float(bridge_cfg["param_clip"]) if "param_clip" in bridge_cfg else None,
                delta_clip=float(bridge_cfg["delta_clip"]) if "delta_clip" in bridge_cfg else None,
                generator_init_std=float(bridge_cfg.get("generator_init_std", 1e-3)),
            )
        elif bridge_type == "asymmetric_context_modulation":
            self.bridge = AsymmetricContextModulationBridge(
                dim=dim,
                kernel_size=int(bridge_cfg.get("kernel_size", 3)),
                base_scale=str(bridge_cfg.get("base_scale", "p2")),
                context_scale=str(bridge_cfg.get("context_scale", "p3")),
                bridge_scale_init=float(bridge_cfg.get("bridge_scale_init", 0.0)),
                acm_scale_init=float(bridge_cfg.get("acm_scale_init", 0.01)),
                use_acm_scale=bool(bridge_cfg.get("use_acm_scale", True)),
                context_hidden_dim=int(bridge_cfg.get("context_hidden_dim", 512)),
                context_dropout=float(bridge_cfg.get("context_dropout", 0.0)),
                force_fp32=bool(bridge_cfg.get("force_fp32", True)),
                param_clip=float(bridge_cfg["param_clip"]) if "param_clip" in bridge_cfg else None,
                delta_clip=float(bridge_cfg["delta_clip"]) if "delta_clip" in bridge_cfg else None,
                generator_init_std=float(bridge_cfg.get("generator_init_std", 1e-3)),
            )
        else:
            raise ValueError(f"Unsupported S3 bridge type: {bridge_type}")

    def bridge_evidence(self, evidence: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(self.bridge, AsymmetricContextModulationBridge):
            return self.bridge(evidence, queries, self.row_embedding.weight)
        return self.bridge(evidence, queries)

    def pool_lane_evidence(self, row_hidden: torch.Tensor, range_norm: torch.Tensor) -> torch.Tensor:
        if self.final_decision_pooling == "mean":
            return row_hidden.mean(dim=2)
        if self.final_decision_pooling == "max":
            return row_hidden.amax(dim=2)
        if self.final_decision_pooling != "range_mean":
            raise ValueError(f"Unsupported final_decision.pooling: {self.final_decision_pooling}")
        _, _, p, _ = row_hidden.shape
        y_norm = fixed_y_rows(p, self.input_h, device=row_hidden.device, dtype=row_hidden.dtype) / float(self.input_h)
        ranges = sort_range_norm(range_norm.to(device=row_hidden.device, dtype=row_hidden.dtype))
        mask = (y_norm.view(1, 1, p) >= ranges[..., :1]) & (y_norm.view(1, 1, p) <= ranges[..., 1:])
        mask_f = mask.to(dtype=row_hidden.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        return (row_hidden * mask_f).sum(dim=2) / denom

    def range_row_mask(
        self,
        range_norm: torch.Tensor,
        num_rows: int,
        padding_px: float = 0.0,
    ) -> torch.Tensor:
        y_norm = fixed_y_rows(num_rows, self.input_h, device=range_norm.device, dtype=range_norm.dtype) / float(self.input_h)
        ranges = sort_range_norm(range_norm)
        pad = float(padding_px) / float(self.input_h)
        return (y_norm.view(1, 1, num_rows) >= (ranges[..., :1] - pad)) & (
            y_norm.view(1, 1, num_rows) <= (ranges[..., 1:] + pad)
        )

    @staticmethod
    def masked_mean_rows(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=values.dtype)
        if values.ndim == 4:
            mask_f = mask_f.unsqueeze(-1)
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        return (values * mask_f).sum(dim=2) / denom

    @staticmethod
    def masked_max_rows(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fill_value = torch.finfo(values.dtype).min
        if values.ndim == 4:
            masked = values.masked_fill(~mask.unsqueeze(-1), fill_value)
            max_values = masked.amax(dim=2)
            any_valid = mask.any(dim=2).unsqueeze(-1)
        else:
            masked = values.masked_fill(~mask, fill_value)
            max_values = masked.amax(dim=2)
            any_valid = mask.any(dim=2)
        return torch.where(any_valid, max_values, torch.zeros_like(max_values))

    def pool_for_quality_calibrator(self, row_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.quality_calibrator_pooling == "range_max":
            return self.masked_max_rows(row_hidden, mask)
        if self.quality_calibrator_pooling == "range_mean_max":
            return torch.cat([self.masked_mean_rows(row_hidden, mask), self.masked_max_rows(row_hidden, mask)], dim=-1)
        if self.quality_calibrator_pooling != "range_mean":
            raise ValueError(f"Unsupported quality_calibrator.pooling: {self.quality_calibrator_pooling}")
        return self.masked_mean_rows(row_hidden, mask)

    def quality_scalar_stats(self, outputs: dict[str, Any], mask: torch.Tensor) -> torch.Tensor:
        final = outputs["final"]
        coarse = outputs["coarse"]
        evidence = outputs.get("evidence", {})
        row_logits = final["row_x_logits"].float()
        row_probs = torch.softmax(row_logits, dim=-1)
        row_conf = row_probs.max(dim=-1).values
        offset_logits = evidence.get("active_offset_logits")
        pred_delta = evidence.get("active_pred_delta_x_rows")
        if offset_logits is None:
            offset_entropy = row_conf.new_zeros(row_conf.shape)
            offset_max_prob = row_conf.new_zeros(row_conf.shape)
        else:
            offset_probs = torch.softmax(offset_logits.float(), dim=-1)
            offset_entropy = -(offset_probs * offset_probs.clamp_min(1e-6).log()).sum(dim=-1)
            offset_max_prob = offset_probs.max(dim=-1).values
        if pred_delta is None:
            abs_delta = row_conf.new_zeros(row_conf.shape)
        else:
            abs_delta = pred_delta.float().abs()
        final_coarse_diff = (final["pred_x_rows"].float() - coarse["pred_x_rows"].float()).abs()
        stats_per_row = [
            offset_entropy.detach(),
            offset_max_prob.detach(),
            abs_delta.detach() / float(self.input_w),
            final_coarse_diff.detach() / float(self.input_w),
            row_conf.detach(),
        ]
        if self.quality_calibrator_curvature_feature:
            pred_x = final["pred_x_rows"].float()
            d2 = (pred_x[:, :, 2:] - 2.0 * pred_x[:, :, 1:-1] + pred_x[:, :, :-2]).abs()
            curvature = torch.nn.functional.pad(d2, (1, 1), mode="constant", value=0.0)
            stats_per_row.append(curvature.detach() / float(self.input_w))
        pooled = []
        for values in stats_per_row:
            pooled.append(self.masked_mean_rows(values, mask))
            pooled.append(self.masked_max_rows(values, mask))
        return torch.stack(pooled, dim=-1)

    def apply_final_decision(self, outputs: dict[str, Any]) -> dict[str, Any]:
        if not self.final_decision_enabled:
            return outputs
        final = outputs["final"]
        coarse = outputs["coarse"]
        pooled = self.pool_lane_evidence(final["row_hidden"], coarse["range_norm"])
        delta_exist = self.delta_exist_head(pooled)
        base_exist = coarse["exist_logits"].detach() if self.final_decision_detach_base else coarse["exist_logits"]
        final["exist_logits"] = base_exist + delta_exist
        evidence = outputs.setdefault("evidence", {})
        evidence["final_delta_exist_abs"] = delta_exist.detach().abs().mean()
        return outputs

    def apply_quality_calibrator(self, outputs: dict[str, Any]) -> dict[str, Any]:
        if not self.quality_calibrator_enabled:
            return outputs
        final = outputs["final"]
        coarse = outputs["coarse"]
        mask = self.range_row_mask(
            coarse["range_norm"].to(device=final["row_hidden"].device, dtype=final["row_hidden"].dtype),
            final["row_hidden"].shape[2],
            padding_px=self.quality_calibrator_range_padding_px,
        )
        row_hidden = final["row_hidden"].detach() if self.quality_calibrator_detach_row_hidden else final["row_hidden"]
        pooled_hidden = self.pool_for_quality_calibrator(row_hidden, mask)
        scalar_stats = self.quality_scalar_stats(outputs, mask).to(device=pooled_hidden.device, dtype=pooled_hidden.dtype)
        calib = self.quality_calibrator(torch.cat([pooled_hidden, scalar_stats], dim=-1))
        delta_exist = calib[..., :2]
        delta_quality = calib[..., 2]
        if self.quality_calibrator_calibrate_exist:
            base_exist = coarse["exist_logits"].detach() if self.quality_calibrator_detach_base else final["exist_logits"]
            final["exist_logits"] = base_exist + self.quality_calibrator_exist_delta_scale * delta_exist
        if self.quality_calibrator_quality_base == "coarse":
            base_quality = coarse["quality_logits"].detach() if self.quality_calibrator_detach_base else coarse["quality_logits"]
            final["quality_logits"] = base_quality + delta_quality
        elif self.quality_calibrator_quality_base in {"", "none", "zero"}:
            final["quality_logits"] = delta_quality
        else:
            raise ValueError(f"Unsupported quality_calibrator.quality_base: {self.quality_calibrator_quality_base}")
        final["quality_pred_x_rows"] = final["pred_x_rows"]
        evidence = outputs.setdefault("evidence", {})
        evidence["quality_calib_delta_exist_abs"] = delta_exist.detach().abs().mean()
        evidence["quality_calib_quality_abs"] = delta_quality.detach().abs().mean()
        evidence["quality_calib_stats_abs"] = scalar_stats.detach().abs().mean()
        return outputs

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs = self.apply_final_decision(outputs)
        return self.apply_quality_calibrator(outputs)
