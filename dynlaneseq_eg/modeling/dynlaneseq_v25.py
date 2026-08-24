from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .v25_dual_energy_multi_path import V25DualEnergyMultiPath
from .v25_image_mediated_lane_objects import V25ImageMediatedLaneObjects


class DynLaneSeqV25(nn.Module):
    """Standalone image-mediated four-lane object detector."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.get("model", cfg)
        v25 = cfg.get("v25", {})
        detector_type = (
            V25DualEnergyMultiPath
            if bool(v25.get("enable_dual_energy_multi_path", False))
            else V25ImageMediatedLaneObjects
        )
        common_kwargs = dict(
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
            enable_pattern_query_initializer=bool(
                v25.get("pattern_query_initializer", {}).get("enabled", False)
            ),
            pattern_query_count=int(
                v25.get("pattern_query_initializer", {}).get("pattern_count", 16)
            ),
            pattern_query_pooled_rows=int(
                v25.get("pattern_query_initializer", {}).get("pooled_rows", 10)
            ),
            pattern_query_pooled_columns=int(
                v25.get("pattern_query_initializer", {}).get("pooled_columns", 25)
            ),
            pattern_query_projection_channels=int(
                v25.get("pattern_query_initializer", {}).get(
                    "projection_channels", 16
                )
            ),
            pattern_query_hidden_dim=int(
                v25.get("pattern_query_initializer", {}).get("hidden_dim", 128)
            ),
        )
        if detector_type is V25DualEnergyMultiPath:
            common_kwargs.update(
                num_path_hypotheses=int(v25.get("num_path_hypotheses", 3)),
                path_suppression_radius_bins=int(
                    v25.get("path_suppression_radius_bins", 5)
                ),
                path_suppression_penalty=float(
                    v25.get("path_suppression_penalty", 8.0)
                ),
                proposal_count=int(v25.get("proposal_count", 32)),
                proposal_groups=int(v25.get("proposal_groups", 4)),
                proposal_dropout=float(v25.get("proposal_dropout", 0.25)),
                enable_proposal_fusion=bool(
                    v25.get("enable_proposal_fusion", True)
                ),
                exact_set_selection=bool(v25.get("exact_set_selection", True)),
                minimum_spacing_px=float(
                    v25.get("loss", {}).get("minimum_spacing_px", 12.0)
                ),
            )
        self.detector = detector_type(**common_kwargs)
        self.supports_inference_only = True

    def prepare_for_inference(self) -> None:
        return None

    def forward(
        self,
        images: torch.Tensor,
        targets=None,
        return_features: bool = False,
        inference_only: bool = False,
        query_anchor_x_rows: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del targets, return_features
        output = self.detector(
            images,
            query_anchor_x_rows=query_anchor_x_rows,
        )
        if not inference_only:
            return output
        keep = {"exist_logits", "pred_x_rows", "range_norm", "quality_logits"}
        return {name: value for name, value in output.items() if name in keep}
