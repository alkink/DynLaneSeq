from __future__ import annotations

from typing import Any

import torch

from .dynlaneseq_s2 import DynLaneSeqS2
from .evidence import DynamicDepthwiseBridge, FiLMBridge, SequenceLowRankBridge


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
        else:
            raise ValueError(f"Unsupported S3 bridge type: {bridge_type}")

    def bridge_evidence(self, evidence: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.bridge(evidence, queries)
