from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


ARM_NAMES = ("G", "S", "T")


def matched_arm_input(ribbons: torch.Tensor, arm: str) -> torch.Tensor:
    """Return the matched 18-channel input for a V35 arm.

    Channel layout is ``previous[6], target[6], following[6]``.  Geometry-only
    keeps the exact tensor shape but removes RGB information.  Single-frame
    repeats target channels so all arms use the same scorer parameterization.
    """

    if ribbons.ndim != 4 or int(ribbons.shape[1]) != 18:
        raise ValueError("V35 ribbons must have shape [B,18,H,W]")
    arm = str(arm).upper()
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown V35 arm: {arm!r}")
    if arm == "G":
        return torch.zeros_like(ribbons)
    if arm == "S":
        target = ribbons[:, 6:12]
        return torch.cat((target, target, target), dim=1)
    return ribbons


class Residual2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.block(inputs), inplace=True)


class Residual1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.block(inputs), inplace=True)


class CandidateRibbonScorer(nn.Module):
    """Shared candidate scorer used by all matched V35 arms."""

    def __init__(self, image_channels: int = 18, geometry_channels: int = 4) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 5, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            Residual2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            Residual2d(64),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            Residual2d(96),
            nn.Conv2d(96, 128, 3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Conv1d(geometry_channels, 32, 5, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            Residual1d(32),
            nn.Conv1d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            Residual1d(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        ribbons: torch.Tensor,
        geometry: torch.Tensor,
        *,
        arm: str,
    ) -> torch.Tensor:
        image_inputs = matched_arm_input(ribbons, arm)
        if geometry.ndim != 3 or int(geometry.shape[1]) != 4:
            raise ValueError("V35 geometry must have shape [B,4,H]")
        image_features = self.image_encoder(image_inputs)
        geometry_features = self.geometry_encoder(geometry)
        return self.head(torch.cat((image_features, geometry_features), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class V35LossWeights:
    ranking: float = 1.0
    quality: float = 0.25


def v35_pair_loss(
    good_score: torch.Tensor,
    wrong_score: torch.Tensor,
    good_quality: torch.Tensor,
    wrong_quality: torch.Tensor,
    *,
    weights: V35LossWeights = V35LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if good_score.shape != wrong_score.shape:
        raise ValueError("good/wrong V35 score shapes must match")
    ranking = F.softplus(-(good_score - wrong_score)).mean()
    quality = 0.5 * (
        F.binary_cross_entropy_with_logits(
            good_score, good_quality.to(dtype=good_score.dtype).clamp(0.0, 1.0)
        )
        + F.binary_cross_entropy_with_logits(
            wrong_score,
            wrong_quality.to(dtype=wrong_score.dtype).clamp(0.0, 1.0),
        )
    )
    total = float(weights.ranking) * ranking + float(weights.quality) * quality
    return total, {
        "loss_total": total.detach(),
        "loss_ranking": ranking.detach(),
        "loss_quality": quality.detach(),
        "mean_margin": (good_score - wrong_score).detach().mean(),
    }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
