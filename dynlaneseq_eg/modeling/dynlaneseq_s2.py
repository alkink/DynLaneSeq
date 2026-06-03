from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import soft_expected_x
from .dynlaneseq_s0 import DynLaneSeqEncoder
from .evidence import CurveAlignedSampler, DynamicOffsetFusion, EvidenceAdapter, SamplerCurriculum
from .heads_s0 import ExistenceHead, RangeHead, S0Heads
from .row_token_decoder import RowTokenDecoder


class DynLaneSeqS2(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        evidence_cfg = model_cfg.get("evidence_sampler", {})
        self.input_w = int(model_cfg.get("input_w", 800))
        self.input_h = int(model_cfg.get("input_h", 288))
        self.num_rows = int(model_cfg.get("num_rows", 72))
        self.x_bins = int(model_cfg.get("x_bins", 200))
        dim = int(model_cfg.get("dim", 256))
        local_window_cfg = evidence_cfg.get("local_window", {})
        self.dynamic_offset_enabled = bool(local_window_cfg.get("enabled", False)) and str(
            local_window_cfg.get("aggregation", "mean")
        ).lower() in {"dynamic", "learned", "token"}
        self.s2_mode = str(model_cfg.get("s2_mode", "direct")).lower()
        if self.s2_mode not in {"direct", "residual"}:
            raise ValueError(f"Unsupported S2 mode: {self.s2_mode}")
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
        self.curriculum = SamplerCurriculum(
            noise_std=float(evidence_cfg.get("noise_std", 3.0)),
            detach_sample_coords=bool(evidence_cfg.get("detach_sample_coords", True)),
            input_w=self.input_w,
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

    def build_coarse_tokens(self, queries: torch.Tensor) -> torch.Tensor:
        b, n, d = queries.shape
        row_emb = self.row_embedding.weight.view(1, 1, self.num_rows, d)
        return queries.unsqueeze(2) + row_emb

    def bridge_evidence(self, evidence: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return evidence, {}

    def sample_evidence(
        self,
        features: torch.Tensor,
        sample_x: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.offset_fusion is None:
            return self.sampler(features, sample_x), {}
        offset_samples = self.sampler.sample_local_window(features, sample_x)
        evidence, offset_debug = self.offset_fusion(offset_samples, queries, self.row_embedding.weight)
        return evidence, offset_debug

    def build_final_tokens(self, queries: torch.Tensor, evidence: torch.Tensor, stage_extra: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _, d = evidence.shape
        row_emb = self.row_embedding.weight.view(1, 1, self.num_rows, d)
        tokens = queries.unsqueeze(2) + row_emb + self.adapter(evidence)
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
        q = enc["queries"]
        if self.s2_mode == "residual":
            coarse = self.heads(q)
            coarse_x = coarse["pred_x_rows"]
            sample_x = self.curriculum.build_sample_x(coarse_x, targets, matches, alpha=float(sampler_alpha))
            evidence, offset_debug = self.sample_evidence(enc["features"], sample_x, q)
            evidence, bridge_debug = self.bridge_evidence(evidence, q)
            coarse_x_embed_src = coarse_x.detach() if self.detach_coarse_x else coarse_x
            coarse_x_norm = (coarse_x_embed_src / float(self.input_w)).unsqueeze(-1)
            row = self.row_decoder(
                self.build_final_tokens(
                    q,
                    evidence,
                    stage_extra=self.coarse_x_embed(coarse_x_norm),
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
                    "sample_x_rows": sample_x,
                    "E_seq": evidence,
                    "evidence_scale": self.adapter.gamma,
                    **offset_debug,
                    **bridge_debug,
                },
                "queries": q,
                "row_delta_logits": row["row_x_logits"],
            }
            if "row_visibility_logits" in row:
                out["final"]["row_visibility_logits"] = row["row_visibility_logits"]
        else:
            exist_logits = self.exist_head(q)
            range_raw, range_norm = self.range_head(q)
            coarse_row = self.row_decoder(self.build_coarse_tokens(q), input_w=self.input_w)
            coarse_logits = coarse_row["row_x_logits"]
            coarse_x = coarse_row["pred_x_rows"]
            sample_x = self.curriculum.build_sample_x(coarse_x, targets, matches, alpha=float(sampler_alpha))
            evidence, offset_debug = self.sample_evidence(enc["features"], sample_x, q)
            evidence, bridge_debug = self.bridge_evidence(evidence, q)
            row = self.row_decoder(self.build_final_tokens(q, evidence), input_w=self.input_w)
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
        if "seg_logits" in enc:
            out["seg_logits"] = enc["seg_logits"]
        if return_features:
            out["features"] = enc["features"]
        return out
