from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .dynlaneseq_s0 import DynLaneSeqS0
from .v23_ordered_slot_cost_volume import V23OrderedSlotCostVolume


class DynLaneSeqV23(nn.Module):
    """Frozen V7 teacher plus an independent ordered cost-volume student."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.get("model", cfg)
        v23_cfg = cfg.get("v23", {})
        # DynLaneSeqS0 reads the structured-query configuration but does not
        # dispatch on model.name, so the same complete V7 config is safe here.
        self.teacher = DynLaneSeqS0(cfg)
        self.student = V23OrderedSlotCostVolume(
            input_h=int(model_cfg.get("input_h", 640)),
            input_w=int(model_cfg.get("input_w", 1600)),
            num_rows=int(model_cfg.get("num_rows", 160)),
            x_bins=int(model_cfg.get("x_bins", 800)),
            fpn_channels=int(model_cfg.get("fpn_channels", 256)),
            hidden_dim=int(v23_cfg.get("hidden_dim", 96)),
            query_dim=int(v23_cfg.get("query_dim", 96)),
            vertical_layers=int(v23_cfg.get("vertical_layers", 2)),
            num_heads=int(v23_cfg.get("num_heads", 4)),
            ff_dim=int(v23_cfg.get("ff_dim", 256)),
            teacher_prior_sigma_px=float(
                v23_cfg.get("teacher_prior_sigma_px", 18.0)
            ),
            teacher_prior_weight=float(v23_cfg.get("teacher_prior_weight", 1.0)),
            transition_radius_bins=int(
                v23_cfg.get("transition_radius_bins", 8)
            ),
            transition_penalty=float(v23_cfg.get("transition_penalty", 0.15)),
            freeze_batch_norm_stats=bool(
                v23_cfg.get("freeze_batch_norm_stats", True)
            ),
        )
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        self.supports_inference_only = True

    def train(self, mode: bool = True):
        super().train(mode)
        # A parent .train() call must never update V7 BN buffers or activate
        # its dropout. This was the principal V18 representation-drift bug.
        self.teacher.eval()
        return self

    def prepare_for_inference(self) -> None:
        self.teacher.prepare_for_inference()

    def forward(
        self,
        images: torch.Tensor,
        targets=None,
        return_features: bool = False,
        inference_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        del targets, return_features
        # V7 remains the exact FP32 source even when the student is trained
        # under BF16 autocast. Outer autocast otherwise changes route ties and
        # would silently violate the frozen-teacher contract.
        with torch.no_grad(), torch.autocast(
            device_type=images.device.type,
            enabled=False,
        ):
            teacher_outputs = self.teacher(images.float(), inference_only=True)
        output = self.student(images, teacher_outputs)
        if inference_only:
            keep = {
                "exist_logits",
                "pred_x_rows",
                "range_norm",
                "quality_logits",
            }
            return {name: value for name, value in output.items() if name in keep}
        return output
