from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backbone_dla import DLA34Backbone
from .fpn import SimpleFPN


class V22LaneFieldStageA(nn.Module):
    """Trainable global lane-evidence field used by the V22 Stage-A gate.

    The module deliberately has no proposal, slot, routing, or set decoder. It
    learns a new image representation whose only job is to expose lane centres
    in global image coordinates.  V7 weights are copied into ``backbone`` and
    ``fpn`` by the training tool, after which this branch is independent and
    fully trainable.
    """

    def __init__(
        self,
        *,
        input_h: int = 640,
        input_w: int = 1600,
        num_rows: int = 160,
        x_bins: int = 800,
        fpn_channels: int = 256,
        hidden_dim: int = 128,
        distance_limit_px: float = 96.0,
        freeze_batch_norm_stats: bool = True,
    ) -> None:
        super().__init__()
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.distance_limit_px = float(distance_limit_px)
        self.freeze_batch_norm_stats = bool(freeze_batch_norm_stats)
        if self.input_h <= 0 or self.input_w <= 0:
            raise ValueError("V22 field input dimensions must be positive")
        if self.num_rows <= 1 or self.x_bins <= 1:
            raise ValueError("V22 field grid must have at least two rows/bins")
        if self.distance_limit_px <= 0:
            raise ValueError("V22 field distance limit must be positive")

        self.backbone = DLA34Backbone(pretrained=False)
        self.fpn = SimpleFPN(
            in_channels=self.backbone.out_channels,
            out_channels=int(fpn_channels),
        )
        groups = _group_count(int(hidden_dim))
        self.field_trunk = nn.Sequential(
            # The student encoder remains full-resolution and trainable, but
            # the hidden field is decoded on native stride-4 P2. Only the
            # three scalar output maps are resized from 400 to 800 columns.
            # Keeping a 128-channel tensor at 800 columns doubled compute
            # without adding image evidence beyond P2's native sampling.
            nn.Conv2d(int(fpn_channels), int(hidden_dim), 1, bias=False),
            nn.GroupNorm(groups, int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(
                int(hidden_dim),
                int(hidden_dim),
                kernel_size=3,
                padding=1,
                groups=int(hidden_dim),
                bias=False,
            ),
            nn.GroupNorm(groups, int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), 1, bias=False),
            nn.GroupNorm(groups, int(hidden_dim)),
            nn.GELU(),
        )
        self.centerline_head = nn.Conv2d(int(hidden_dim), 1, 1)
        self.distance_head = nn.Conv2d(int(hidden_dim), 1, 1)
        self.support_head = nn.Conv2d(int(hidden_dim), 1, 1)
        nn.init.constant_(self.centerline_head.bias, -4.0)
        nn.init.zeros_(self.distance_head.weight)
        nn.init.zeros_(self.distance_head.bias)
        nn.init.constant_(self.support_head.bias, -1.0)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batch_norm_stats:
            for module in (*self.backbone.modules(), *self.fpn.modules()):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def copy_v7_encoder_(self, v7_model: nn.Module) -> dict[str, float | int]:
        encoder = getattr(v7_model, "encoder", None)
        if encoder is None or not hasattr(encoder, "backbone") or not hasattr(encoder, "fpn"):
            raise TypeError("V22 warm start requires a V7 model with encoder.backbone/fpn")
        self.backbone.load_state_dict(encoder.backbone.state_dict(), strict=True)
        self.fpn.load_state_dict(encoder.fpn.state_dict(), strict=True)
        source_backbone = dict(encoder.backbone.named_parameters())
        source_fpn = dict(encoder.fpn.named_parameters())
        max_difference = 0.0
        tensors = 0
        for name, value in self.backbone.named_parameters():
            max_difference = max(
                max_difference,
                float((value.detach() - source_backbone[name].detach()).abs().max()),
            )
            tensors += 1
        for name, value in self.fpn.named_parameters():
            max_difference = max(
                max_difference,
                float((value.detach() - source_fpn[name].detach()).abs().max()),
            )
            tensors += 1
        return {"parameter_tensors": tensors, "maximum_copy_difference": max_difference}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        p2 = self.fpn(features)
        field = self.field_trunk(p2)
        centerline_logits = self.centerline_head(field)
        distance_raw = self.distance_head(field)
        support_logits = self.support_head(field)
        if centerline_logits.shape[-2:] != (self.num_rows, self.x_bins):
            centerline_logits = F.interpolate(
                centerline_logits,
                size=(self.num_rows, self.x_bins),
                mode="bilinear",
                align_corners=False,
            )
            distance_raw = F.interpolate(
                distance_raw,
                size=(self.num_rows, self.x_bins),
                mode="bilinear",
                align_corners=False,
            )
            support_logits = F.interpolate(
                support_logits,
                size=(self.num_rows, self.x_bins),
                mode="bilinear",
                align_corners=False,
            )
        return {
            "centerline_logits": centerline_logits,
            "distance_raw": distance_raw,
            "support_logits": support_logits,
            "field_features": field,
        }


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def build_lane_field_targets(
    targets: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_rows: int,
    x_bins: int,
    input_w: int,
    centerline_sigma_px: float,
    distance_limit_px: float,
) -> dict[str, torch.Tensor]:
    """Rasterize row lanes into centre, nearest signed distance, and support."""

    if centerline_sigma_px <= 0 or distance_limit_px <= 0:
        raise ValueError("V22 target radii must be positive")
    bin_width = float(input_w) / float(x_bins)
    x_grid = (
        (torch.arange(x_bins, device=device, dtype=torch.float32) + 0.5)
        * bin_width
    ).view(1, x_bins)
    centerline_items: list[torch.Tensor] = []
    distance_items: list[torch.Tensor] = []
    support_items: list[torch.Tensor] = []
    for target in targets:
        x_rows = target["x_rows"].to(device=device, dtype=torch.float32)
        valid = target["valid_mask"].to(device=device).bool()
        row_count = min(int(x_rows.shape[-1]) if x_rows.ndim == 2 else 0, int(num_rows))
        centerline = torch.zeros((num_rows, x_bins), device=device, dtype=torch.float32)
        distance = torch.zeros_like(centerline)
        support = torch.zeros_like(centerline)
        if int(x_rows.shape[0]) > 0 and row_count > 0:
            x_rows = x_rows[:, :row_count]
            valid = valid[:, :row_count]
            valid = valid & torch.isfinite(x_rows) & (x_rows >= 0.0) & (x_rows < float(input_w))
            # [lanes, rows, bins], positive means the closest lane lies right
            # of the pixel. This exact sign is audited in the unit tests.
            signed = x_rows.unsqueeze(-1) - x_grid.view(1, 1, x_bins)
            absolute = signed.abs().masked_fill(~valid.unsqueeze(-1), float("inf"))
            nearest_absolute, nearest_lane = absolute.min(dim=0)
            has_lane = valid.any(dim=0).unsqueeze(-1).expand(row_count, x_bins)
            nearest_signed = signed.gather(
                0, nearest_lane.unsqueeze(0)
            ).squeeze(0)
            finite_absolute = torch.where(
                has_lane, nearest_absolute, torch.full_like(nearest_absolute, 1.0e6)
            )
            centerline[:row_count] = torch.exp(
                -0.5 * (finite_absolute / float(centerline_sigma_px)).pow(2)
            ) * has_lane.float()
            within = has_lane & (nearest_absolute <= float(distance_limit_px))
            support[:row_count] = within.float()
            distance[:row_count] = torch.where(
                within,
                nearest_signed.clamp(
                    -float(distance_limit_px), float(distance_limit_px)
                )
                / float(distance_limit_px),
                torch.zeros_like(nearest_signed),
            )
        centerline_items.append(centerline)
        distance_items.append(distance)
        support_items.append(support)
    return {
        "centerline": torch.stack(centerline_items).unsqueeze(1).to(dtype=dtype),
        "distance_norm": torch.stack(distance_items).unsqueeze(1).to(dtype=dtype),
        "support": torch.stack(support_items).unsqueeze(1).to(dtype=dtype),
    }


def lane_field_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    input_w: int,
    centerline_sigma_px: float = 3.0,
    distance_limit_px: float = 96.0,
    centerline_positive_weight: float = 16.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = outputs["centerline_logits"]
    distance_raw = outputs["distance_raw"]
    support_logits = outputs["support_logits"]
    if logits.shape != distance_raw.shape or logits.shape != support_logits.shape:
        raise ValueError("V22 field heads must share the same shape")
    batch, channels, rows, bins = logits.shape
    if channels != 1 or batch != len(targets):
        raise ValueError("V22 field output/target batch mismatch")
    built = build_lane_field_targets(
        targets,
        device=logits.device,
        dtype=torch.float32,
        num_rows=rows,
        x_bins=bins,
        input_w=int(input_w),
        centerline_sigma_px=float(centerline_sigma_px),
        distance_limit_px=float(distance_limit_px),
    )
    center_target = built["centerline"]
    support_target = built["support"]
    distance_target = built["distance_norm"]

    center_bce = F.binary_cross_entropy_with_logits(
        logits.float(), center_target, reduction="none"
    )
    center_weight = 1.0 + float(centerline_positive_weight) * center_target
    center_loss = (center_bce * center_weight).sum() / center_weight.sum().clamp_min(1.0)

    support_bce = F.binary_cross_entropy_with_logits(
        support_logits.float(), support_target, reduction="none"
    )
    # Balance the two broad support classes per batch without a tuned scalar.
    support_positive = support_target.sum().clamp_min(1.0)
    support_negative = (1.0 - support_target).sum().clamp_min(1.0)
    support_loss = 0.5 * (
        (support_bce * support_target).sum() / support_positive
        + (support_bce * (1.0 - support_target)).sum() / support_negative
    )

    predicted_distance = torch.tanh(distance_raw.float())
    distance_error = F.smooth_l1_loss(
        predicted_distance, distance_target, reduction="none", beta=0.05
    )
    # Rows near the lane centre matter most for candidate verification, while
    # the full 96-pixel corridor still teaches the correction direction.
    near_weight = 0.25 + 0.75 * torch.exp(
        -distance_target.abs() * float(distance_limit_px) / 32.0
    )
    distance_weight = support_target * near_weight
    distance_loss = (distance_error * distance_weight).sum() / distance_weight.sum().clamp_min(1.0)

    total = 2.0 * center_loss + distance_loss + 0.25 * support_loss
    with torch.no_grad():
        center_probability = torch.sigmoid(logits.float())
        diagnostics = {
            "loss_total": total.detach(),
            "loss_centerline": center_loss.detach(),
            "loss_distance": distance_loss.detach(),
            "loss_support": support_loss.detach(),
            "center_probability_on_target": (
                (center_probability * center_target).sum()
                / center_target.sum().clamp_min(1.0)
            ).detach(),
            "distance_mae_px": (
                (predicted_distance - distance_target).abs()
                * float(distance_limit_px)
                * support_target
            ).sum().div(support_target.sum().clamp_min(1.0)).detach(),
            "support_accuracy": (
                ((torch.sigmoid(support_logits.float()) >= 0.5) == support_target.bool())
                .float()
                .mean()
                .detach()
            ),
        }
    return total, diagnostics


def _pixel_x_to_grid(
    x_rows: torch.Tensor,
    *,
    input_w: int,
    x_bins: int,
) -> torch.Tensor:
    bin_width = float(input_w) / float(x_bins)
    bin_index = x_rows.float() / bin_width - 0.5
    return 2.0 * bin_index / float(max(x_bins - 1, 1)) - 1.0


def sample_lane_field_rows(
    field: torch.Tensor,
    x_rows: torch.Tensor,
    *,
    input_w: int,
) -> torch.Tensor:
    """Sample a dense [B,C,R,X] field at [B,N,R] curve coordinates."""

    if field.ndim != 4 or x_rows.ndim != 3:
        raise ValueError("V22 row sampler expects field [B,C,R,X] and x [B,N,R]")
    batch, channels, rows, bins = field.shape
    if x_rows.shape[0] != batch or x_rows.shape[2] != rows:
        raise ValueError("V22 row sampler shape mismatch")
    curves = int(x_rows.shape[1])
    x_grid = _pixel_x_to_grid(x_rows, input_w=input_w, x_bins=bins)
    y_grid = torch.linspace(
        -1.0, 1.0, rows, device=field.device, dtype=torch.float32
    ).view(1, 1, rows).expand(batch, curves, rows)
    grid = torch.stack((x_grid, y_grid), dim=-1)
    sampled = F.grid_sample(
        field.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.permute(0, 2, 3, 1).contiguous()


def soft_range_weights(
    range_norm: torch.Tensor,
    *,
    rows: int,
    temperature: float = 0.02,
) -> torch.Tensor:
    row = torch.linspace(
        0.0, 1.0, rows, device=range_norm.device, dtype=torch.float32
    )
    start = range_norm[..., 0].float().unsqueeze(-1)
    stop = range_norm[..., 1].float().unsqueeze(-1)
    return torch.sigmoid((row - start) / float(temperature)) * torch.sigmoid(
        (stop - row) / float(temperature)
    )


@torch.no_grad()
def score_candidates_from_lane_field(
    outputs: dict[str, torch.Tensor],
    *,
    source_x: torch.Tensor,
    source_range: torch.Tensor,
    candidate_x: torch.Tensor,
    candidate_range: torch.Tensor,
    candidate_valid: torch.Tensor,
    input_w: int,
    distance_limit_px: float,
) -> dict[str, torch.Tensor]:
    """Analytically score candidates against the field-corrected source lane.

    The source curve carries V7's semantic ownership.  The new field supplies
    only a signed image correction. No learned candidate reranker is involved.
    """

    if source_x.ndim != 3 or candidate_x.ndim != 4:
        raise ValueError("V22 candidate scorer expects source [B,S,R], candidates [B,S,K,R]")
    batch, slots, rows = source_x.shape
    if candidate_x.shape[:2] != (batch, slots) or candidate_x.shape[-1] != rows:
        raise ValueError("V22 candidate/source curve mismatch")
    source_sample_x = source_x.reshape(batch, slots, rows)
    distance_sample = sample_lane_field_rows(
        torch.tanh(outputs["distance_raw"].float()) * float(distance_limit_px),
        source_sample_x,
        input_w=input_w,
    )[..., 0]
    support_sample = sample_lane_field_rows(
        outputs["support_logits"].float(), source_sample_x, input_w=input_w
    )[..., 0].sigmoid()
    corrected_x = (
        source_x.float() + support_sample * distance_sample
    ).clamp(0.0, float(input_w - 1))

    source_weight = soft_range_weights(source_range, rows=rows)
    candidate_weight = soft_range_weights(candidate_range, rows=rows)
    valid_source_x = torch.isfinite(source_x) & (source_x >= 0.0) & (
        source_x < float(input_w)
    )
    valid_candidate_x = torch.isfinite(candidate_x) & (candidate_x >= 0.0) & (
        candidate_x < float(input_w)
    )
    weight = (
        source_weight.unsqueeze(2)
        * candidate_weight
        * valid_source_x.unsqueeze(2).float()
        * valid_candidate_x.float()
    )
    denominator = weight.sum(dim=-1).clamp_min(1.0)
    field_error = (
        (candidate_x.float() - corrected_x.unsqueeze(2)).abs() * weight
    ).sum(dim=-1) / denominator
    geometry_error = (
        (candidate_x.float() - source_x.float().unsqueeze(2)).abs() * weight
    ).sum(dim=-1) / denominator
    valid = candidate_valid.bool() & (weight.sum(dim=-1) >= 3.0)
    field_score = (-field_error).masked_fill(~valid, -1.0e4)
    geometry_score = (-geometry_error).masked_fill(~valid, -1.0e4)
    return {
        "field_score": field_score,
        "geometry_score": geometry_score,
        "field_error_px": field_error,
        "geometry_error_px": geometry_error,
        "corrected_source_x": corrected_x,
        "source_support": support_sample,
        "valid": valid,
    }


def model_contract(model: V22LaneFieldStageA) -> dict[str, Any]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "input_h": model.input_h,
        "input_w": model.input_w,
        "num_rows": model.num_rows,
        "x_bins": model.x_bins,
        "distance_limit_px": model.distance_limit_px,
        "trainable_parameters": int(trainable),
        "freeze_batch_norm_stats": model.freeze_batch_norm_stats,
    }
