from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DenoisingAnchorBatch:
    anchors_normalized: torch.Tensor
    target_x_rows: torch.Tensor
    valid_mask: torch.Tensor
    active: torch.Tensor
    mean_absolute_perturbation_px: torch.Tensor


def build_denoising_query_anchors(
    ordered: dict[str, torch.Tensor],
    *,
    input_w: int,
    seed: int,
    held_out: bool = False,
) -> DenoisingAnchorBatch:
    """Construct fixed-contract training-only lane-query perturbations.

    The function is stateless with respect to the process RNG.  This keeps the
    clean control/treatment augmentation stream paired while the treatment
    receives horizontal shifts, range truncation, row dropout and one
    contiguous occlusion.  Held-out magnitudes are disjoint from the training
    shifts and are used only by the G4 evaluator.
    """

    target_x = ordered["x_rows"].float()
    valid = ordered["valid_mask"].bool()
    active = ordered["active"].bool()
    if target_x.ndim != 3 or valid.shape != target_x.shape:
        raise ValueError("ordered denoising targets must be [B,S,R]")
    batch, slots, rows = target_x.shape
    device = target_x.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    canonical = torch.linspace(
        0.16,
        0.84,
        slots,
        device=device,
        dtype=torch.float32,
    ).view(1, slots, 1)
    anchor_px = canonical.expand(batch, slots, rows) * float(input_w)
    anchor_px = torch.where(valid, target_x, anchor_px)

    magnitudes = (
        torch.tensor((12.0, 24.0, 48.0), device=device)
        if held_out
        else torch.tensor((8.0, 16.0, 32.0, 64.0), device=device)
    )
    magnitude_index = torch.randint(
        0,
        int(magnitudes.numel()),
        (batch, slots),
        generator=generator,
        device=device,
    )
    sign = torch.where(
        torch.randint(
            0,
            2,
            (batch, slots),
            generator=generator,
            device=device,
        ).bool(),
        torch.ones((batch, slots), device=device),
        -torch.ones((batch, slots), device=device),
    )
    shift = magnitudes[magnitude_index] * sign
    shifted = anchor_px + shift.unsqueeze(-1)

    query_mask = valid.clone()
    maximum_trim = min(
        10 if held_out else 8,
        max(rows // 10, 1),
    )
    trim_top = torch.randint(
        0,
        maximum_trim + 1,
        (batch, slots),
        generator=generator,
        device=device,
    )
    trim_bottom = torch.randint(
        0,
        maximum_trim + 1,
        (batch, slots),
        generator=generator,
        device=device,
    )
    row_index = torch.arange(rows, device=device).view(1, 1, rows)
    first = torch.where(
        valid,
        row_index,
        torch.full_like(row_index, rows),
    ).amin(dim=-1)
    last = torch.where(
        valid,
        row_index,
        torch.full_like(row_index, -1),
    ).amax(dim=-1)
    query_mask &= row_index >= (first + trim_top).unsqueeze(-1)
    query_mask &= row_index <= (last - trim_bottom).unsqueeze(-1)

    dropout_probability = 0.20 if held_out else 0.15
    dropout = torch.rand(
        (batch, slots, rows),
        generator=generator,
        device=device,
    ) < dropout_probability
    query_mask &= ~dropout

    occlusion_length = min(
        20 if held_out else 14,
        max(rows // 8, 1),
    )
    if rows > occlusion_length:
        start = torch.randint(
            0,
            rows - occlusion_length + 1,
            (batch, slots),
            generator=generator,
            device=device,
        )
        occluded = (
            row_index >= start.unsqueeze(-1)
        ) & (
            row_index < (start + occlusion_length).unsqueeze(-1)
        )
        query_mask &= ~occluded

    fallback = canonical.expand(batch, slots, rows) * float(input_w)
    perturbed = torch.where(query_mask & active.unsqueeze(-1), shifted, fallback)
    perturbed = perturbed.clamp(0.0, float(input_w - 1))
    measured = valid & active.unsqueeze(-1)
    mean_perturbation = (
        (perturbed - target_x).abs() * measured.float()
    ).sum() / measured.sum().clamp_min(1).float()
    return DenoisingAnchorBatch(
        anchors_normalized=perturbed / float(input_w),
        target_x_rows=target_x,
        valid_mask=valid,
        active=active,
        mean_absolute_perturbation_px=mean_perturbation,
    )
