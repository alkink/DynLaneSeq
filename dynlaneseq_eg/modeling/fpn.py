from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SimpleFPN(nn.Module):
    def __init__(self, in_channels: dict[str, int] | None = None, out_channels: int = 128):
        super().__init__()
        in_channels = in_channels or {"c2": 64, "c3": 128, "c4": 256, "c5": 512}
        self.lateral = nn.ModuleDict({k: nn.Conv2d(v, out_channels, 1) for k, v in in_channels.items()})
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pyramid_outputs = nn.ModuleDict(
            {
                "p3": self._make_output(out_channels),
                "p4": self._make_output(out_channels),
                "p5": self._make_output(out_channels),
            }
        )

    @staticmethod
    def _make_output(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: dict[str, torch.Tensor], return_pyramid: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        p5 = self.lateral["c5"](feats["c5"])
        p4 = self.lateral["c4"](feats["c4"]) + F.interpolate(p5, size=feats["c4"].shape[-2:], mode="nearest")
        p3 = self.lateral["c3"](feats["c3"]) + F.interpolate(p4, size=feats["c3"].shape[-2:], mode="nearest")
        p2 = self.lateral["c2"](feats["c2"]) + F.interpolate(p3, size=feats["c2"].shape[-2:], mode="nearest")
        p2 = self.output(p2)
        if not return_pyramid:
            return p2
        return {
            "p2": p2,
            "p3": self.pyramid_outputs["p3"](p3),
            "p4": self.pyramid_outputs["p4"](p4),
            "p5": self.pyramid_outputs["p5"](p5),
        }
