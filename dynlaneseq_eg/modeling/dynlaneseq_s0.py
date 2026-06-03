from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbone_resnet import ResNet34Backbone
from .cross_attention_decoder import LaneCrossAttentionDecoder
from .fpn import SimpleFPN
from .heads_s0 import S0Heads, SegAuxHead
from .lane_queries import LaneQueries
from .position_encoding import SinePositionEncoding2D


class DynLaneSeqEncoder(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.get("model", cfg)
        dim = int(model_cfg.get("dim", 256))
        fpn_channels = int(model_cfg.get("fpn_channels", 128))
        self.query_content_init = str(model_cfg.get("query_content_init", "learned")).lower()
        self.memory_value_with_pos = bool(model_cfg.get("memory_value_with_pos", False))
        self.freeze_backbone_bn = bool(model_cfg.get("freeze_backbone_bn", False))
        seg_aux_cfg = model_cfg.get("seg_aux", {})
        self.seg_aux_enabled = bool(seg_aux_cfg.get("enabled", False))
        self.backbone = ResNet34Backbone(pretrained=bool(model_cfg.get("pretrained_backbone", True)))
        self.fpn = SimpleFPN(out_channels=fpn_channels)
        self.proj = nn.Conv2d(fpn_channels, dim, 1)
        self.seg_aux_head = (
            SegAuxHead(
                dim=dim,
                input_h=int(model_cfg.get("input_h", 288)),
                input_w=int(model_cfg.get("input_w", 800)),
                dropout=float(seg_aux_cfg.get("dropout", 0.1)),
            )
            if self.seg_aux_enabled
            else None
        )
        self.pos = SinePositionEncoding2D(dim=dim)
        self.queries = LaneQueries(num_slots=int(model_cfg.get("num_slots", 20)), dim=dim)
        self.decoder = LaneCrossAttentionDecoder(
            num_layers=int(model_cfg.get("decoder_layers", 2)),
            dim=dim,
            num_heads=int(model_cfg.get("num_heads", 8)),
            ff_dim=int(model_cfg.get("decoder_ff_dim", 1024)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        if self.freeze_backbone_bn:
            self._set_backbone_bn_eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            self._set_backbone_bn_eval()
        return self

    def _set_backbone_bn_eval(self) -> None:
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(images)
        fpn = self.fpn(feats)
        f_proj = self.proj(fpn)
        f_pos = f_proj + self.pos(f_proj)
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
        out = {"features": f_proj, "memory": mem_value, "memory_key": mem_key, "queries": q1, "q0": q_seed}
        if self.seg_aux_head is not None:
            out["seg_logits"] = self.seg_aux_head(f_proj)
        return out


class DynLaneSeqS0(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        self.encoder = DynLaneSeqEncoder(cfg)
        self.heads = S0Heads(
            dim=int(model_cfg.get("dim", 256)),
            num_rows=int(model_cfg.get("num_rows", 72)),
            x_bins=int(model_cfg.get("x_bins", 200)),
            input_w=int(model_cfg.get("input_w", 800)),
        )

    def forward(self, images: torch.Tensor, targets=None, return_features: bool = False) -> dict[str, torch.Tensor]:
        enc = self.encoder.forward_features(images)
        out = self.heads(enc["queries"])
        out["queries"] = enc["queries"]
        if "seg_logits" in enc:
            out["seg_logits"] = enc["seg_logits"]
        if return_features:
            out["features"] = enc["features"]
        return out
