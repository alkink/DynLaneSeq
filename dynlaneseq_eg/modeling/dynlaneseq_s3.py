from __future__ import annotations

from typing import Any

import torch

from .dynlaneseq_s2 import DynLaneSeqS2
from .evidence import AsymmetricContextModulationBridge, DynamicDepthwiseBridge, FiLMBridge, SequenceLowRankBridge


class DynLaneSeqS3(DynLaneSeqS2):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        model_cfg = cfg.get("model", cfg)
        bridge_cfg = model_cfg.get("bridge", {})
        dim = int(model_cfg.get("dim", 256))
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
