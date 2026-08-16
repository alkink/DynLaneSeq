from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .v25_image_mediated_lane_objects import V25ImageMediatedLaneObjects


class DynLaneSeqV25(nn.Module):
    """Standalone image-mediated four-lane object detector."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.get("model", cfg)
        v25 = cfg.get("v25", {})
        self.detector = V25ImageMediatedLaneObjects(
            input_h=int(model_cfg.get("input_h", 640)),
            input_w=int(model_cfg.get("input_w", 1600)),
            num_rows=int(model_cfg.get("num_rows", 160)),
            x_bins=int(model_cfg.get("x_bins", 800)),
            fpn_channels=int(model_cfg.get("fpn_channels", 256)),
            hidden_dim=int(v25.get("hidden_dim", 96)),
            decoder_layers=int(v25.get("decoder_layers", 3)),
            num_heads=int(v25.get("num_heads", 4)),
            ff_dim=int(v25.get("ff_dim", 256)),
            dropout=float(v25.get("dropout", 0.1)),
            transition_radius_bins=int(v25.get("transition_radius_bins", 8)),
            transition_penalty=float(v25.get("transition_penalty", 0.15)),
            enable_competition=bool(v25.get("enable_competition", False)),
            enable_slot_interaction=bool(
                v25.get("enable_slot_interaction", False)
            ),
            competition_weight=float(v25.get("competition_weight", 2.0)),
            anchor_prior_weight=float(v25.get("anchor_prior_weight", 0.10)),
            anchor_prior_sigma=float(v25.get("anchor_prior_sigma", 0.28)),
            decode_mode=str(v25.get("decode_mode", "hard_path")),
            pretrained_backbone=bool(
                model_cfg.get("pretrained_backbone", True)
            ),
            require_pretrained_backbone=bool(
                model_cfg.get("require_pretrained_backbone", True)
            ),
            pretrained_weights_path=str(
                model_cfg.get("pretrained_backbone_path", "")
            ),
            freeze_batch_norm_stats=bool(
                v25.get("freeze_batch_norm_stats", True)
            ),
        )
        self.supports_inference_only = True

    def prepare_for_inference(self) -> None:
        return None

    def forward(
        self,
        images: torch.Tensor,
        targets=None,
        return_features: bool = False,
        inference_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        del targets, return_features
        output = self.detector(images)
        if not inference_only:
            return output
        keep = {"exist_logits", "pred_x_rows", "range_norm", "quality_logits"}
        return {name: value for name, value in output.items() if name in keep}
