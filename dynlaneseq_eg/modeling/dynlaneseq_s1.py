from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import soft_expected_x
from .dynlaneseq_s0 import DynLaneSeqEncoder, GeometryGuidedQueryRefiner
from .heads_s0 import ExistenceHead, RangeHead, S0Heads
from .row_token_decoder import RowTokenDecoder


class DynLaneSeqS1(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        geometry_cfg = model_cfg.get("s0_geometry_evidence", {})
        self.input_w = int(model_cfg.get("input_w", 800))
        self.input_h = int(model_cfg.get("input_h", 288))
        self.num_rows = int(model_cfg.get("num_rows", 72))
        self.x_bins = int(model_cfg.get("x_bins", 200))
        dim = int(model_cfg.get("dim", 256))
        self.s1_mode = str(model_cfg.get("s1_mode", "direct")).lower()
        if self.s1_mode not in {"direct", "residual"}:
            raise ValueError(f"Unsupported S1 mode: {self.s1_mode}")
        self.encoder = DynLaneSeqEncoder(cfg)
        if bool(geometry_cfg.get("enabled", False)) and bool(model_cfg.get("dynamic_evidence", {}).get("enabled", False)):
            raise ValueError("Use either dynamic_evidence v1 or s0_geometry_evidence v2, not both")
        if bool(geometry_cfg.get("enabled", False)) and self.s1_mode != "residual":
            raise ValueError("s0_geometry_evidence currently requires residual S1 mode")
        if self.s1_mode == "residual":
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
            zero_init_head=bool(model_cfg.get("zero_init_residual_head", self.s1_mode == "residual")),
            local_attn_window=int(model_cfg.get("row_local_attn_window", 0)),
            visibility_head=bool(model_cfg.get("row_visibility", {}).get("enabled", False)),
        )

    def build_row_tokens(self, queries: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = queries.shape
        row_emb = self.row_embedding.weight.view(1, 1, self.num_rows, d)
        tokens = queries.unsqueeze(2) + row_emb
        if extra is not None:
            tokens = tokens + extra
        return tokens

    def forward(self, images: torch.Tensor, targets=None, return_features: bool = False) -> dict[str, torch.Tensor]:
        enc = self.encoder.forward_features(images)
        q = enc["queries"]
        geometry_debug = None
        q_pre_geometry = None
        geometry_draft = None
        if self.s1_mode == "residual":
            if self.s0_geometry_refiner is not None:
                q_pre_geometry = q
                geometry_draft = self.heads(q)
                geometry_x = (
                    geometry_draft["pred_x_rows"].detach()
                    if self.s0_geometry_detach_draft
                    else geometry_draft["pred_x_rows"]
                )
                q, geometry_debug = self.s0_geometry_refiner(q, enc["features"], geometry_x)
            coarse = self.heads(q)
            coarse_x = coarse["pred_x_rows"].detach() if self.detach_coarse_x else coarse["pred_x_rows"]
            coarse_x_norm = (coarse_x / float(self.input_w)).unsqueeze(-1)
            row = self.row_decoder(
                self.build_row_tokens(q, extra=self.coarse_x_embed(coarse_x_norm)),
                input_w=self.input_w,
            )
            base_logits = coarse["row_x_logits"].detach() if self.detach_coarse_x else coarse["row_x_logits"]
            row_x_logits = base_logits + self.residual_logit_scale * row["row_x_logits"]
            pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
            out = {
                "exist_logits": coarse["exist_logits"],
                "row_x_logits": row_x_logits,
                "pred_x_rows": pred_x_rows,
                "range_raw": coarse["range_raw"],
                "range_norm": coarse["range_norm"],
                "quality_logits": coarse["quality_logits"],
                "quality_pred_x_rows": coarse["pred_x_rows"],
                "row_hidden": row["row_hidden"],
                "queries": q,
                "coarse": coarse,
                "row_delta_logits": row["row_x_logits"],
            }
            if geometry_debug is not None:
                out["queries_pre_geometry"] = q_pre_geometry
                out["geometry_evidence"] = geometry_debug
                out["s0_geometry_draft"] = geometry_draft
            if "row_visibility_logits" in row:
                out["row_visibility_logits"] = row["row_visibility_logits"]
        else:
            range_raw, range_norm = self.range_head(q)
            row = self.row_decoder(self.build_row_tokens(q), input_w=self.input_w)
            out = {
                "exist_logits": self.exist_head(q),
                "row_x_logits": row["row_x_logits"],
                "pred_x_rows": row["pred_x_rows"],
                "range_raw": range_raw,
                "range_norm": range_norm,
                "row_hidden": row["row_hidden"],
                "queries": q,
            }
            if "row_visibility_logits" in row:
                out["row_visibility_logits"] = row["row_visibility_logits"]
        if "seg_logits" in enc:
            out["seg_logits"] = enc["seg_logits"]
        if "centerline_logits" in enc:
            out["centerline_logits"] = enc["centerline_logits"]
        if "dynamic_evidence" in enc:
            out["dynamic_evidence"] = enc["dynamic_evidence"]
        if "dynamic_proposals" in enc:
            out["dynamic_proposals"] = enc["dynamic_proposals"]
        if geometry_debug is not None:
            out.setdefault("evidence", {})
            out["evidence"].update(geometry_debug)
        if return_features:
            out["features"] = enc["features"]
        return out
