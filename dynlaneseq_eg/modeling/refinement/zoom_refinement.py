from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoomRefinementConfig:
    enabled: bool = False
    share_decoder: bool = True
    share_adapter: bool = True
    share_bridge: bool = True
    hidden_scale_init: float = 0.1
    detach_sample_coords: bool = True
    detach_stage1_hidden: bool = True

