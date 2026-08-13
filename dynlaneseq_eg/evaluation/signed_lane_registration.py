from __future__ import annotations

from dataclasses import dataclass

import torch

from dynlaneseq_eg.modeling.common import fixed_row_fractions, sort_range_norm


@dataclass(frozen=True)
class SignedRegistrationResult:
    scores: torch.Tensor
    selected_indices: torch.Tensor
    sampled_relative_log_probability: torch.Tensor
    mean_signed_displacement_px: torch.Tensor
    p90_absolute_displacement_px: torch.Tensor
    valid_rows: torch.Tensor


def _sample_row_values(
    values: torch.Tensor,
    proposal_x_rows: torch.Tensor,
    *,
    input_w: int,
) -> torch.Tensor:
    """Bilinearly sample `[B,S,R,X]` values at `[B,N,R]` x coordinates."""

    if values.ndim != 4:
        raise ValueError("row values must have shape [B,S,R,X]")
    if proposal_x_rows.ndim != 3:
        raise ValueError("proposal x must have shape [B,N,R]")
    batch, slots, rows, x_bins = values.shape
    if tuple(proposal_x_rows.shape[:1] + proposal_x_rows.shape[2:]) != (
        batch,
        rows,
    ):
        raise ValueError("proposal x batch/row axes do not match row values")
    candidates = int(proposal_x_rows.shape[1])
    width = float(max(int(input_w) - 1, 1))
    feature_x = (
        proposal_x_rows.detach().float().clamp(0.0, width)
        * float(max(x_bins - 1, 0))
        / width
    )
    left = feature_x.floor().long()
    right = (left + 1).clamp(max=max(x_bins - 1, 0))
    alpha = feature_x - left.to(feature_x.dtype)

    # Gather over a compact [B*R,S,X] view instead of materializing B*S*N*R*X.
    by_row = values.permute(0, 2, 1, 3).reshape(
        batch * rows, slots, x_bins
    )
    left_by_row = left.permute(0, 2, 1).reshape(batch * rows, candidates)
    right_by_row = right.permute(0, 2, 1).reshape(batch * rows, candidates)
    left_value = by_row.gather(
        -1, left_by_row[:, None, :].expand(-1, slots, -1)
    )
    right_value = by_row.gather(
        -1, right_by_row[:, None, :].expand(-1, slots, -1)
    )
    sampled = torch.lerp(
        left_value,
        right_value,
        alpha.permute(0, 2, 1)
        .reshape(batch * rows, candidates)[:, None, :]
        .to(left_value.dtype),
    )
    return sampled.reshape(batch, rows, slots, candidates).permute(
        0, 2, 3, 1
    )


def _masked_weighted_quantile(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("quantile values and mask must have the same shape")
    expanded_weights = weights.expand_as(values)
    order = values.masked_fill(~mask, torch.inf).argsort(dim=-1)
    ordered_values = values.gather(-1, order)
    ordered_mask = mask.gather(-1, order)
    ordered_weights = expanded_weights.gather(-1, order)
    ordered_weights = ordered_weights * ordered_mask.to(values.dtype)
    total = ordered_weights.sum(dim=-1, keepdim=True)
    reached = ordered_weights.cumsum(dim=-1) >= total * float(quantile)
    index = reached.to(torch.int64).argmax(dim=-1)
    selected = ordered_values.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return torch.where(
        total.squeeze(-1) > 0.0,
        selected,
        torch.full_like(selected, -torch.inf),
    )


def signed_curve_registration(
    *,
    visual_logits: torch.Tensor,
    proposal_x_rows: torch.Tensor,
    proposal_range_norm: torch.Tensor,
    anchor_range_norm: torch.Tensor,
    proposal_visible: torch.Tensor,
    group_mask: torch.Tensor,
    writer_valid: torch.Tensor,
    input_w: int,
    min_valid_rows: int = 5,
) -> SignedRegistrationResult:
    """Rank intact proposals against a proposal-independent row posterior.

    The score contract is intentionally parameter-free beyond fixed 50/50
    mean/lower-tail aggregation and the perspective weights documented above.
    Signed displacement is returned only for causal diagnostics.
    """

    if visual_logits.ndim != 4:
        raise ValueError("visual logits must have shape [B,S,R,X]")
    batch, slots, rows, x_bins = visual_logits.shape
    if tuple(proposal_x_rows.shape[:1] + proposal_x_rows.shape[2:]) != (
        batch,
        rows,
    ):
        raise ValueError("proposal x shape does not match visual logits")
    candidates = int(proposal_x_rows.shape[1])
    if tuple(proposal_range_norm.shape) != (batch, candidates, 2):
        raise ValueError("proposal range must have shape [B,N,2]")
    if tuple(anchor_range_norm.shape) != (batch, slots, 2):
        raise ValueError("anchor range must have shape [B,S,2]")
    if tuple(proposal_visible.shape) != (batch, candidates, rows):
        raise ValueError("proposal visibility must have shape [B,N,R]")
    if tuple(group_mask.shape) != (batch, slots, candidates):
        raise ValueError("group mask must have shape [B,S,N]")
    if tuple(writer_valid.shape) != (batch, slots):
        raise ValueError("writer-valid mask must have shape [B,S]")
    if x_bins < 2:
        raise ValueError("signed registration requires at least two x bins")

    logits = visual_logits.detach().float()
    log_probability = torch.log_softmax(logits, dim=-1)
    relative_log_probability = log_probability - log_probability.amax(
        dim=-1, keepdim=True
    )
    sampled = _sample_row_values(
        relative_log_probability,
        proposal_x_rows,
        input_w=input_w,
    )

    row_y = fixed_row_fractions(
        rows, device=logits.device, dtype=torch.float32
    )
    anchor_range = sort_range_norm(anchor_range_norm.detach().float())
    anchor_visible = (
        (row_y.view(1, 1, rows) >= anchor_range[..., :1])
        & (row_y.view(1, 1, rows) <= anchor_range[..., 1:])
    )
    valid = (
        proposal_visible[:, None].detach().bool()
        & anchor_visible[:, :, None]
        & group_mask.unsqueeze(-1).detach().bool()
        & writer_valid[:, :, None, None].detach().bool()
        & torch.isfinite(proposal_x_rows[:, None])
    )
    row_weight = (0.10 + 0.90 * row_y.pow(3.0)).view(1, 1, 1, rows)
    weighted = row_weight * valid.to(row_weight.dtype)
    denominator = weighted.sum(dim=-1)
    mean_log_probability = (sampled * weighted).sum(dim=-1) / denominator.clamp_min(
        1.0e-12
    )
    lower_tail = _masked_weighted_quantile(
        sampled,
        valid,
        row_weight,
        0.10,
    )
    scores = 0.5 * mean_log_probability + 0.5 * lower_tail
    eligible = (
        group_mask.detach().bool()
        & writer_valid.unsqueeze(-1).detach().bool()
        & (valid.sum(dim=-1) >= int(min_valid_rows))
        & torch.isfinite(scores)
    )
    scores = scores.masked_fill(~eligible, -1.0e4)

    probability = torch.softmax(logits, dim=-1)
    x_grid = torch.linspace(
        0.0,
        float(max(int(input_w) - 1, 1)),
        x_bins,
        device=logits.device,
        dtype=torch.float32,
    )
    expected_x = torch.einsum("bsrx,x->bsr", probability, x_grid)
    signed = proposal_x_rows[:, None].detach().float() - expected_x[:, :, None]
    mean_signed = (signed * weighted).sum(dim=-1) / denominator.clamp_min(1.0e-12)
    p90_absolute = _masked_weighted_quantile(
        signed.abs(),
        valid,
        row_weight,
        0.90,
    )
    mean_signed = torch.where(eligible, mean_signed, torch.zeros_like(mean_signed))
    p90_absolute = torch.where(
        eligible, p90_absolute, torch.zeros_like(p90_absolute)
    )
    selected = scores.argmax(dim=-1)
    return SignedRegistrationResult(
        scores=scores,
        selected_indices=selected,
        sampled_relative_log_probability=sampled,
        mean_signed_displacement_px=mean_signed,
        p90_absolute_displacement_px=p90_absolute,
        valid_rows=valid,
    )


def gather_whole_proposals(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather one complete proposal tensor per slot without mixing members."""

    if values.ndim < 2 or indices.ndim != 2:
        raise ValueError("proposal gather expects values [B,N,...], ids [B,S]")
    if int(values.shape[0]) != int(indices.shape[0]):
        raise ValueError("proposal gather batch mismatch")
    suffix = tuple(int(value) for value in values.shape[2:])
    view = indices.long().clamp(0, max(int(values.shape[1]) - 1, 0))
    view = view.view(*indices.shape, *([1] * len(suffix)))
    return values.gather(1, view.expand(*indices.shape, *suffix))

