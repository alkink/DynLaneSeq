from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .dynlaneseq_s3 import DynLaneSeqS3


class DynLaneSeqS4(DynLaneSeqS3):
    """Optional one-step zoom-in refinement on top of S3."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        model_cfg = cfg.get("model", cfg)
        zoom_cfg = model_cfg.get("zoom_refine", {})
        self.hidden_scale = nn.Parameter(torch.tensor(float(zoom_cfg.get("hidden_scale_init", 0.1))))
        self.detach_stage1_hidden = bool(zoom_cfg.get("detach_stage1_hidden", True))

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
        matches: list[dict[str, torch.Tensor]] | None = None,
        sampler_alpha: float = 0.0,
        sampler_beta: float = 0.0,
        return_features: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        stage1_out = super().forward(
            images,
            targets=targets,
            matches=matches,
            sampler_alpha=sampler_alpha,
            return_features=True,
        )
        enc_features = stage1_out["features"]
        q = stage1_out["queries"]
        exist_logits = stage1_out["coarse"]["exist_logits"]
        range_raw = stage1_out["coarse"]["range_raw"]
        range_norm = stage1_out["coarse"]["range_norm"]
        stage1_x = stage1_out["final"]["pred_x_rows"]
        sample_x2 = self.curriculum.build_sample_x(stage1_x, targets, matches, alpha=float(sampler_beta))
        evidence2 = self.sampler(enc_features, sample_x2)
        evidence2, bridge_debug = self.bridge_evidence(evidence2, q)
        h1 = stage1_out["final"]["row_hidden"]
        if self.detach_stage1_hidden:
            h1 = h1.detach()
        row2 = self.row_decoder(
            self.build_final_tokens(q, evidence2, stage_extra=self.hidden_scale * h1),
            input_w=self.input_w,
        )
        out = {
            "coarse": stage1_out["coarse"],
            "stage1": {
                "row_x_logits": stage1_out["final"]["row_x_logits"],
                "pred_x_rows": stage1_x,
                "row_hidden": stage1_out["final"]["row_hidden"],
            },
            "stage2": {
                "exist_logits": exist_logits,
                "row_x_logits": row2["row_x_logits"],
                "pred_x_rows": row2["pred_x_rows"],
                "range_raw": range_raw,
                "range_norm": range_norm,
                "row_hidden": row2["row_hidden"],
            },
            "evidence": {
                "sample_x_stage1": stage1_out["evidence"]["sample_x_rows"],
                "sample_x_stage2": sample_x2,
                "E_seq_stage1": stage1_out["evidence"]["E_seq"],
                "E_seq_stage2": evidence2,
                "hidden_scale": self.hidden_scale,
                **bridge_debug,
            },
            "queries": q,
        }
        if "seg_logits" in stage1_out:
            out["seg_logits"] = stage1_out["seg_logits"]
        if return_features:
            out["features"] = enc_features
        return out
