from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PositiveWeightedFusion(nn.Module):
    """Fuse two feature maps with normalized, strictly positive weights."""

    def __init__(self, first_init: float = 1.0, second_init: float = 1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([float(first_init), float(second_init)]))

    def normalized_weights(self) -> torch.Tensor:
        weights = F.softplus(self.logits)
        return weights / weights.sum().clamp_min(1e-4)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError(f"fusion inputs must have identical shapes, got {first.shape} and {second.shape}")
        weights = self.normalized_weights().to(device=first.device, dtype=first.dtype)
        return weights[0] * first + weights[1] * second


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


class BalancedDetailFPN(nn.Module):
    """Detail-preserving top-down pyramid for row-wise lane localization.

    Every backbone level is normalized before fusion.  P2 keeps a direct C2
    detail path and receives the semantic P3 path through a separately learned,
    positive normalized fusion.  This prevents the magnitude of an upsampled
    deep feature from silently overwhelming the high-resolution lane evidence.
    """

    def __init__(
        self,
        in_channels: dict[str, int] | None = None,
        out_channels: int = 128,
        num_groups: int = 8,
        upsample_mode: str = "bilinear",
        detail_init: float = 2.0,
        context_init: float = 0.0,
    ):
        super().__init__()
        in_channels = in_channels or {"c2": 64, "c3": 128, "c4": 256, "c5": 512}
        if out_channels % int(num_groups) != 0:
            raise ValueError(
                f"BalancedDetailFPN out_channels={out_channels} must be divisible by num_groups={num_groups}"
            )
        if upsample_mode not in {"nearest", "bilinear"}:
            raise ValueError(f"Unsupported BalancedDetailFPN upsample_mode: {upsample_mode!r}")
        self.upsample_mode = str(upsample_mode)
        self.lateral = nn.ModuleDict(
            {
                name: self._make_lateral(channels, out_channels, int(num_groups))
                for name, channels in in_channels.items()
            }
        )
        self.refine = nn.ModuleDict(
            {
                name: self._make_output(out_channels, int(num_groups))
                for name in ("p2", "p3", "p4", "p5")
            }
        )
        self.fuse_p4 = PositiveWeightedFusion(1.0, 1.0)
        self.fuse_p3 = PositiveWeightedFusion(1.0, 1.0)
        self.fuse_p2 = PositiveWeightedFusion(float(detail_init), float(context_init))

    @staticmethod
    def _make_lateral(in_channels: int, out_channels: int, num_groups: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_output(channels: int, num_groups: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
        )

    def _upsample(self, feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self.upsample_mode == "bilinear":
            kwargs["align_corners"] = False
        return F.interpolate(
            feature,
            size=target.shape[-2:],
            mode=self.upsample_mode,
            **kwargs,
        )

    def fusion_weights(self) -> dict[str, torch.Tensor]:
        """Return detached normalized weights for diagnostics and logging."""

        return {
            "p4": self.fuse_p4.normalized_weights().detach(),
            "p3": self.fuse_p3.normalized_weights().detach(),
            "p2": self.fuse_p2.normalized_weights().detach(),
        }

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        return_pyramid: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        lateral = {name: block(feats[name]) for name, block in self.lateral.items()}

        p5 = self.refine["p5"](lateral["c5"])
        p4 = self.refine["p4"](
            self.fuse_p4(lateral["c4"], self._upsample(p5, lateral["c4"]))
        )
        p3 = self.refine["p3"](
            self.fuse_p3(lateral["c3"], self._upsample(p4, lateral["c3"]))
        )
        # The first input deliberately remains the direct, normalized C2 detail
        # branch.  The second input is semantic context and starts at ~25%.
        p2 = self.refine["p2"](
            self.fuse_p2(lateral["c2"], self._upsample(p3, lateral["c2"]))
        )
        if not return_pyramid:
            return p2
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
