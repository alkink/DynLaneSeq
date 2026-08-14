from __future__ import annotations

import math
from itertools import combinations, permutations

import torch
from torch import nn
from torch.nn import functional as F

from .common import fixed_indices, fixed_row_fractions, sort_range_norm


_FOUR_POSITION_PERMUTATIONS_CPU = torch.tensor(
    tuple(permutations(range(4))), dtype=torch.long
)
_FOUR_POSITION_PERMUTATION_CACHE: dict[
    tuple[str, int | None], torch.Tensor
] = {}


def _four_position_permutations(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    cached = _FOUR_POSITION_PERMUTATION_CACHE.get(key)
    if cached is None:
        cached = _FOUR_POSITION_PERMUTATIONS_CPU.to(device=device)
        _FOUR_POSITION_PERMUTATION_CACHE[key] = cached
    return cached


def _position_basis(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            values,
            values.square(),
            torch.sin(math.pi * values),
            torch.cos(math.pi * values),
        ),
        dim=-1,
    )


def _row_slope(x_rows: torch.Tensor) -> torch.Tensor:
    if int(x_rows.shape[-1]) <= 1:
        return torch.zeros_like(x_rows)
    delta = x_rows[..., 1:] - x_rows[..., :-1]
    return torch.cat((delta[..., :1], delta), dim=-1)


def _symmetric_expectation(
    probability: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Return an exactly-zero displacement for a uniform distribution."""

    midpoint = int(probability.shape[-1]) // 2
    positive = probability[..., midpoint + 1 :]
    negative = probability[..., :midpoint].flip(-1)
    return torch.einsum(
        "...k,k->...",
        positive - negative,
        offsets[midpoint + 1 :].to(probability),
    )


def build_exact_four_set_tables(
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate unordered four-sets and their 24 ordered assignments."""

    if int(candidate_count) < 4:
        raise ValueError("V18 exact four-set search requires at least 4 candidates")
    combination_table = torch.tensor(
        tuple(combinations(range(int(candidate_count)), 4)),
        dtype=torch.long,
    )
    permutation_table = torch.tensor(
        tuple(permutations(range(4))),
        dtype=torch.long,
    )
    ordered = combination_table[:, permutation_table]
    return combination_table, permutation_table, ordered


def exact_ordered_set_energies(
    unary: torch.Tensor,
    pair: torch.Tensor,
    active_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    ordered_assignments: torch.Tensor,
    slot_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every ordered injective assignment without candidate pruning.

    Pair terms are multiplied by detached activity probabilities.  This makes
    semantic-lane compatibility meaningful for emitted lane pairs without
    pretending that unused slots in two- or three-lane images have a unique
    semantic proposal target.
    """

    if unary.ndim != 3:
        raise ValueError("V18 unary energy must have shape [B,4,N]")
    batch, slots, candidates = unary.shape
    if slots != 4:
        raise ValueError("V18 exact energy is defined for exactly four slots")
    if pair.shape != (batch, 6, candidates, candidates):
        raise ValueError("V18 pair energy must have shape [B,6,N,N]")
    if active_logits.shape != (batch, slots):
        raise ValueError("V18 active logits must have shape [B,4]")
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("V18 candidate validity must have shape [B,N]")

    assignments = ordered_assignments.to(device=unary.device)
    set_count, permutation_count, assignment_slots = assignments.shape
    if assignment_slots != slots:
        raise ValueError("V18 ordered assignment table has the wrong slot count")
    energy = unary.new_zeros((batch, set_count, permutation_count)).float()
    for slot in range(slots):
        index = assignments[..., slot].reshape(1, -1).expand(batch, -1)
        selected = unary[:, slot].float().gather(1, index)
        energy = energy + selected.view(batch, set_count, permutation_count)

    active_probability = torch.sigmoid(active_logits.detach().float())
    if tuple(slot_pairs.shape) != (6, 2):
        raise ValueError("V18 slot-pair table must have shape [6,2]")
    # Four slots always induce this fixed lexicographic pair order. Avoid
    # ``CUDA tensor.tolist()`` here: that tiny-looking conversion forced a
    # device synchronization on every V18 forward.
    fixed_slot_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for pair_index, (left, right) in enumerate(fixed_slot_pairs):
        left_index = assignments[..., left].reshape(1, -1).expand(batch, -1)
        right_index = assignments[..., right].reshape(1, -1).expand(batch, -1)
        flat_pair = pair[:, pair_index].float().reshape(batch, candidates * candidates)
        linear_index = left_index * candidates + right_index
        selected = flat_pair.gather(1, linear_index).view(
            batch, set_count, permutation_count
        )
        pair_weight = (
            active_probability[:, left] * active_probability[:, right]
        ).view(batch, 1, 1)
        energy = energy + pair_weight * selected

    combination_members = assignments[:, 0]
    valid_set = candidate_valid[:, combination_members].all(dim=-1)
    return energy, valid_set


def unordered_set_log_scores(
    ordered_energy: torch.Tensor,
    valid_set: torch.Tensor,
    *,
    permutation_temperature: float = 1.0,
) -> torch.Tensor:
    """Marginalize slot permutations into one score per physical four-set."""

    temperature = float(permutation_temperature)
    if temperature <= 0.0:
        raise ValueError("V18 permutation temperature must be positive")
    permutation_count = int(ordered_energy.shape[-1])
    score = temperature * (
        torch.logsumexp(ordered_energy.float() / temperature, dim=-1)
        - math.log(float(permutation_count))
    )
    # A finite sentinel is deliberate: downstream target multiplication must
    # never evaluate 0 * -inf.  The loss also indexes only valid sets.
    return score.masked_fill(~valid_set, -1.0e9)


def decode_exact_ordered_set(
    ordered_energy: torch.Tensor,
    valid_set: torch.Tensor,
    ordered_assignments: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Hard exact decode over all 32P4 ordered assignments."""

    batch, set_count, permutation_count = ordered_energy.shape
    masked = ordered_energy.float().masked_fill(
        ~valid_set.unsqueeze(-1), -1.0e9
    )
    flat_index = masked.reshape(batch, -1).argmax(dim=-1)
    set_index = torch.div(flat_index, permutation_count, rounding_mode="floor")
    permutation_index = flat_index.remainder(permutation_count)
    assignments = ordered_assignments.to(device=ordered_energy.device)
    indices = assignments[set_index, permutation_index]
    has_valid = valid_set.any(dim=-1)
    indices = torch.where(
        has_valid.unsqueeze(-1),
        indices,
        indices.new_full(indices.shape, -1),
    )
    return {
        "indices": indices,
        "set_index": set_index,
        "permutation_index": permutation_index,
        "energy": masked.reshape(batch, -1).gather(
            1, flat_index.unsqueeze(-1)
        ).squeeze(-1),
        "has_valid_set": has_valid,
    }


@torch.no_grad()
def exact_unordered_set_rewards(
    candidate_gt_quality: torch.Tensor,
    gt_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    combination_table: torch.Tensor,
    *,
    threshold_50_temperature: float = 0.03,
    threshold_75_temperature: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best injective GT matching reward for each unordered four-set.

    When an image has fewer than four GT lanes, unused members are
    intentionally marginalized. They are not mislabeled as semantic
    duplicates, and no arbitrary proposal ID target is invented for an
    inactive slot.
    """

    if candidate_gt_quality.ndim != 3:
        raise ValueError("V18 candidate quality must have shape [B,N,G]")
    batch, candidates, max_gt = candidate_gt_quality.shape
    if gt_valid.shape != (batch, max_gt):
        raise ValueError("V18 gt_valid must have shape [B,G]")
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("V18 candidate_valid must have shape [B,N]")
    combos = combination_table.to(device=candidate_gt_quality.device)
    set_count = int(combos.shape[0])
    rewards = candidate_gt_quality.new_zeros((batch, set_count)).float()
    valid_set = candidate_valid[:, combos].all(dim=-1)
    tau50 = float(threshold_50_temperature)
    tau75 = float(threshold_75_temperature)
    if tau50 <= 0.0 or tau75 <= 0.0:
        raise ValueError("V18 reward temperatures must be positive")

    if max_gt == 0:
        return rewards, valid_set

    # Select the first four valid GT lanes without a per-image CUDA
    # synchronization. Invalid/padded GT columns sort behind every real lane.
    gt_axis = torch.arange(max_gt, device=candidate_gt_quality.device)
    ordering_key = torch.where(
        gt_valid.bool(),
        gt_axis.view(1, -1),
        gt_axis.view(1, -1) + max_gt,
    )
    selected_gt = ordering_key.argsort(dim=-1)[:, : min(max_gt, 4)]
    selected_gt_valid = gt_valid.bool().gather(1, selected_gt)
    if int(selected_gt.shape[1]) < 4:
        padding = 4 - int(selected_gt.shape[1])
        selected_gt = torch.cat(
            (selected_gt, selected_gt.new_zeros((batch, padding))), dim=-1
        )
        selected_gt_valid = torch.cat(
            (
                selected_gt_valid,
                selected_gt_valid.new_zeros((batch, padding)),
            ),
            dim=-1,
        )

    quality = candidate_gt_quality[:, combos].float()
    gather_gt = selected_gt[:, None, None, :].expand(
        batch, set_count, 4, 4
    )
    quality = quality.gather(-1, gather_gt)
    lane_reward = (
        torch.sigmoid((quality - 0.50) / tau50)
        + 0.5 * torch.sigmoid((quality - 0.75) / tau75)
        + 0.1 * quality
    )
    position_assignments = _four_position_permutations(quality.device)
    permutation_count = int(position_assignments.shape[0])
    expanded_reward = lane_reward.unsqueeze(2).expand(
        batch, set_count, permutation_count, 4, 4
    )
    gather_position = position_assignments.view(1, 1, permutation_count, 1, 4)
    gather_position = gather_position.expand(
        batch, set_count, permutation_count, 1, 4
    )
    selected = expanded_reward.gather(3, gather_position).squeeze(3)
    selected = selected * selected_gt_valid[:, None, None, :].to(selected)
    rewards = selected.sum(dim=-1).amax(dim=-1)
    return rewards, valid_set


def masked_unordered_set_listwise_loss(
    set_scores: torch.Tensor,
    target_reward: torch.Tensor,
    valid_set: torch.Tensor,
    *,
    support_delta: float,
    target_temperature: float,
    model_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Finite listwise CE over physical sets, never ordered permutations."""

    if set_scores.shape != target_reward.shape or set_scores.shape != valid_set.shape:
        raise ValueError("V18 set score, reward and validity shapes must match")
    if float(support_delta) < 0.0:
        raise ValueError("V18 target support delta must be non-negative")
    if float(target_temperature) <= 0.0 or float(model_temperature) <= 0.0:
        raise ValueError("V18 listwise temperatures must be positive")
    valid = valid_set.bool()
    has_valid = valid.any(dim=-1)
    # Supply one private fallback element only for empty rows. Those rows are
    # multiplied out below; the fallback merely keeps both softmax operations
    # finite without a Python ``bool(CUDA_tensor)`` synchronization.
    fallback = torch.zeros_like(valid)
    fallback[:, 0] = ~has_valid
    safe_valid = valid | fallback

    score = set_scores.float()
    reward = target_reward.float()
    negative_infinity = torch.tensor(
        -torch.inf, device=score.device, dtype=score.dtype
    )
    masked_reward = reward.masked_fill(~safe_valid, negative_infinity)
    best_reward = masked_reward.amax(dim=-1, keepdim=True)
    support = safe_valid & (
        reward >= best_reward - float(support_delta)
    )
    target_logits = (reward / float(target_temperature)).masked_fill(
        ~support, negative_infinity
    )
    target_probability = torch.softmax(target_logits, dim=-1)
    model_logits = (score / float(model_temperature)).masked_fill(
        ~safe_valid, negative_infinity
    )
    model_log_probability = torch.log_softmax(model_logits, dim=-1)
    loss_per_batch = -(
        target_probability
        * torch.where(
            support,
            model_log_probability,
            torch.zeros_like(model_log_probability),
        )
    ).sum(dim=-1)
    target_entropy = -(
        target_probability
        * target_probability.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    support_size = support.float().sum(dim=-1)
    selected_index = model_logits.argmax(dim=-1, keepdim=True)
    selected_reward = reward.gather(1, selected_index).squeeze(-1)
    chosen_regret = best_reward.squeeze(-1) - selected_reward

    valid_weight = has_valid.to(score.dtype)
    denominator = valid_weight.sum().clamp_min(1.0)

    def valid_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * valid_weight).sum() / denominator

    # The zero anchor makes the all-invalid result retain an explicit zero
    # gradient to set_scores, matching the historical implementation.
    zero_anchor = set_scores.sum() * 0.0
    return zero_anchor + valid_mean(loss_per_batch), {
        "target_entropy": valid_mean(target_entropy).detach(),
        "support_size": valid_mean(support_size).detach(),
        "chosen_regret": valid_mean(chosen_regret).detach(),
    }


class _CandidateAssociationEncoder(nn.Module):
    """Live proposal/image representation used by the exact set objective."""

    def __init__(
        self,
        *,
        proposal_dim: int,
        feature_dim: int,
        hidden_dim: int,
        input_w: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        scale_names: tuple[str, ...],
        offsets_px: tuple[float, ...],
        sampling_backend: str,
    ) -> None:
        super().__init__()
        if not scale_names or "p2" not in scale_names:
            raise ValueError("V18 association scales must contain P2")
        if len(offsets_px) < 3 or len(offsets_px) % 2 != 1:
            raise ValueError("V18 association offsets must be odd and nontrivial")
        if float(offsets_px[len(offsets_px) // 2]) != 0.0:
            raise ValueError("V18 association offsets require zero center")
        self.hidden_dim = int(hidden_dim)
        self.input_w = int(input_w)
        self.scale_names = tuple(str(value) for value in scale_names)
        self.sampling_backend = str(sampling_backend).strip().lower()
        if self.sampling_backend not in {"grid_sample", "linear_gather"}:
            raise ValueError(
                "V18 sampling_backend must be grid_sample or linear_gather"
            )
        self.register_buffer(
            "offsets_px", torch.tensor(offsets_px, dtype=torch.float32)
        )
        self.proposal_norm = nn.LayerNorm(int(proposal_dim))
        self.proposal_projection = nn.Linear(
            int(proposal_dim), self.hidden_dim, bias=False
        )
        self.visual_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.scale_norms = nn.ModuleDict(
            {name: nn.LayerNorm(int(feature_dim)) for name in self.scale_names}
        )
        self.scale_keys = nn.ModuleDict(
            {
                name: nn.Linear(int(feature_dim), self.hidden_dim, bias=False)
                for name in self.scale_names
            }
        )
        self.scale_values = nn.ModuleDict(
            {
                name: nn.Linear(int(feature_dim), self.hidden_dim, bias=False)
                for name in self.scale_names
            }
        )
        self.offset_key = nn.Linear(4, self.hidden_dim, bias=False)
        self.offset_state = nn.Linear(4, self.hidden_dim, bias=False)
        self.visual_context = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.geometry_projection = nn.Linear(6, self.hidden_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )
        vertical_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical = nn.TransformerEncoder(
            vertical_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def _as_nchw(feature: torch.Tensor, rows: int) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("V18 image features must be rank four")
        if int(feature.shape[1]) == int(rows):
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature

    @classmethod
    def sample_feature_grid(
        cls,
        feature: torch.Tensor,
        x_rows: torch.Tensor,
        row_fraction: torch.Tensor,
        offsets_px: torch.Tensor,
        *,
        input_w: int,
    ) -> torch.Tensor:
        batch, candidates, rows = x_rows.shape
        feature_nchw = cls._as_nchw(feature, rows)
        sample_x = x_rows.unsqueeze(-1) + offsets_px.view(1, 1, 1, -1)
        sample_x = sample_x.clamp(0.0, float(max(input_w - 1, 1)))
        grid_x = sample_x / float(max(input_w - 1, 1)) * 2.0 - 1.0
        grid_y = row_fraction.view(1, 1, rows, 1).expand_as(grid_x)
        grid = torch.stack((grid_x, grid_y * 2.0 - 1.0), dim=-1).reshape(
            batch, candidates * rows, int(offsets_px.numel()), 2
        )
        sampled = F.grid_sample(
            feature_nchw.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        channels = int(sampled.shape[1])
        return sampled.view(
            batch, channels, candidates, rows, int(offsets_px.numel())
        ).permute(0, 2, 3, 4, 1).contiguous()

    @classmethod
    def sample_feature_linear(
        cls,
        feature: torch.Tensor,
        x_rows: torch.Tensor,
        row_fraction: torch.Tensor,
        offsets_px: torch.Tensor,
        *,
        input_w: int,
    ) -> torch.Tensor:
        """Exact row-aligned bilinear sampling without a 2-D FP32 grid.

        V18 queries the same fixed R rows for every candidate. P2 already has
        R rows; P3/P4 can therefore be vertically interpolated once and then
        sampled only along x. Bilinear interpolation is separable, so this is
        the specialized form of the legacy align-corners grid sample while
        avoiding three large FP32 NCHW/sample tensors per forward.
        """

        del row_fraction
        batch, candidates, rows = x_rows.shape
        if feature.ndim != 4:
            raise ValueError("V18 image features must be rank four")
        if int(feature.shape[1]) == rows:
            row_features = feature
        else:
            feature_nchw = cls._as_nchw(feature, rows)
            row_features = F.interpolate(
                feature_nchw,
                size=(rows, int(feature_nchw.shape[-1])),
                mode="bilinear",
                align_corners=True,
            ).permute(0, 2, 3, 1)
        if int(row_features.shape[0]) != batch or int(row_features.shape[1]) != rows:
            raise ValueError("V18 linear sampler feature/curve rows do not match")
        x_bins = int(row_features.shape[2])
        channels = int(row_features.shape[3])

        device_type = row_features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            sample_x = x_rows.detach().float().unsqueeze(-1)
            sample_x = sample_x + offsets_px.float().view(1, 1, 1, -1)
            sample_x = sample_x.clamp(
                0.0, float(max(int(input_w) - 1, 1))
            )
            feature_x = sample_x * float(max(x_bins - 1, 0)) / float(
                max(int(input_w) - 1, 1)
            )
            left = feature_x.floor().to(dtype=torch.long)
            right = (left + 1).clamp(max=max(x_bins - 1, 0))
            alpha = feature_x - left.to(feature_x)

        offsets_count = int(offsets_px.numel())
        flat_features = row_features.reshape(batch * rows, x_bins, channels)
        left = left.permute(0, 2, 1, 3).reshape(
            batch * rows, candidates * offsets_count
        )
        right = right.permute(0, 2, 1, 3).reshape(
            batch * rows, candidates * offsets_count
        )
        alpha = alpha.permute(0, 2, 1, 3).reshape(
            batch * rows, candidates * offsets_count, 1
        )
        row_index = fixed_indices(
            batch * rows,
            device=row_features.device,
            dtype=torch.long,
        ).view(-1, 1)
        paired_index = torch.stack((left, right), dim=-1).reshape(
            batch * rows, candidates * offsets_count * 2
        )
        paired_value = flat_features[row_index, paired_index].view(
            batch * rows,
            candidates * offsets_count,
            2,
            channels,
        )
        sampled = torch.lerp(
            paired_value[:, :, 0],
            paired_value[:, :, 1],
            alpha.to(row_features.dtype),
        )
        return (
            sampled.view(
                batch, rows, candidates, offsets_count, channels
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    @classmethod
    def sample_feature(
        cls,
        feature: torch.Tensor,
        x_rows: torch.Tensor,
        row_fraction: torch.Tensor,
        offsets_px: torch.Tensor,
        *,
        input_w: int,
        sampling_backend: str,
    ) -> torch.Tensor:
        if str(sampling_backend).strip().lower() == "linear_gather":
            return cls.sample_feature_linear(
                feature,
                x_rows,
                row_fraction,
                offsets_px,
                input_w=input_w,
            )
        return cls.sample_feature_grid(
            feature,
            x_rows,
            row_fraction,
            offsets_px,
            input_w=input_w,
        )

    def forward(
        self,
        *,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        image_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch, candidates, rows, _channels = proposal_rows.shape
        row_fraction = fixed_row_fractions(
            rows, device=proposal_rows.device, dtype=torch.float32
        )
        x = proposal_x.detach().float()
        ranges = sort_range_norm(proposal_range.detach().float())
        finite = torch.isfinite(x)
        safe_x = torch.where(finite, x, torch.zeros_like(x))
        visible = (
            (row_fraction.view(1, 1, rows) >= ranges[..., :1])
            & (row_fraction.view(1, 1, rows) <= ranges[..., 1:])
            & finite
            & candidate_valid.detach().bool().unsqueeze(-1)
        )

        proposal_state = self.proposal_projection(
            self.proposal_norm(proposal_rows.float())
        )
        offsets = self.offsets_px.to(safe_x)
        offset_basis = _position_basis(
            offsets / float(max(self.input_w - 1, 1))
        )
        visual_context: torch.Tensor | None = None
        attention_sum: torch.Tensor | None = None
        for name in self.scale_names:
            feature = image_features.get(name)
            if not isinstance(feature, torch.Tensor):
                raise ValueError(f"V18 association is missing image scale {name!r}")
            sampled = self.sample_feature(
                feature,
                safe_x,
                row_fraction,
                offsets,
                input_w=self.input_w,
                sampling_backend=self.sampling_backend,
            )
            normalized = self.scale_norms[name](sampled)
            keys = self.scale_keys[name](normalized)
            keys = keys + self.offset_key(offset_basis).view(
                1, 1, 1, -1, self.hidden_dim
            )
            values = self.scale_values[name](normalized)
            logits = torch.einsum(
                "bnrh,bnrkh->bnrk", self.visual_query(proposal_state), keys
            ) / math.sqrt(float(self.hidden_dim))
            probability = torch.softmax(logits.float(), dim=-1)
            context = torch.einsum("bnrk,bnrkh->bnrh", probability, values.float())
            expected_offset = torch.einsum(
                "bnrk,k->bnr", probability, offsets.float()
            )
            context = self.visual_context(context) + self.offset_state(
                _position_basis(
                    expected_offset / float(max(self.input_w - 1, 1))
                )
            )
            visual_context = context if visual_context is None else visual_context + context
            attention_sum = probability if attention_sum is None else attention_sum + probability
        if visual_context is None or attention_sum is None:
            raise RuntimeError("V18 association has no visual evidence")
        visual_context = visual_context / float(len(self.scale_names))
        attention = attention_sum / float(len(self.scale_names))

        width = float(max(self.input_w - 1, 1))
        slope = _row_slope(safe_x) / width
        geometry = torch.stack(
            (
                safe_x / width,
                slope,
                slope.abs(),
                ranges[..., 0].unsqueeze(-1).expand_as(safe_x),
                ranges[..., 1].unsqueeze(-1).expand_as(safe_x),
                visible.float(),
            ),
            dim=-1,
        )
        row_state = proposal_state + visual_context + self.geometry_projection(geometry)
        row_state = row_state + self.fusion(self.fusion_norm(row_state))
        row_state = self.vertical(
            row_state.reshape(batch * candidates, rows, self.hidden_dim)
        ).reshape(batch, candidates, rows, self.hidden_dim)
        row_state = self.output_norm(row_state)

        bottom_weight = 0.10 + 0.90 * row_fraction.pow(3)
        pooling_weight = visible.float() * bottom_weight.view(1, 1, rows)
        pooling_weight = pooling_weight / pooling_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        candidate_state = torch.einsum("bnr,bnrh->bnh", pooling_weight, row_state)
        candidate_state = candidate_state * candidate_valid.unsqueeze(-1).to(
            candidate_state.dtype
        )
        return {
            "candidate_state": candidate_state,
            "row_state": row_state,
            "visual_attention": attention,
            "visible": visible,
        }


class _OneShotKeepRefineGeometry(nn.Module):
    """One V7-anchored alternative plus an explicit KEEP/REFINE policy."""

    def __init__(
        self,
        *,
        proposal_dim: int,
        feature_dim: int,
        slot_dim: int,
        candidate_dim: int,
        hidden_dim: int,
        input_w: int,
        num_slots: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        scale_names: tuple[str, ...],
        visual_offsets_px: tuple[float, ...],
        delta_offsets_px: tuple[float, ...],
        range_offsets_norm: tuple[float, ...],
        keep_prior_probability: float,
        sampling_backend: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_w = int(input_w)
        self.num_slots = int(num_slots)
        self.scale_names = tuple(scale_names)
        self.sampling_backend = str(sampling_backend).strip().lower()
        if self.sampling_backend not in {"grid_sample", "linear_gather"}:
            raise ValueError(
                "V18 sampling_backend must be grid_sample or linear_gather"
            )
        for label, values in (
            ("visual", visual_offsets_px),
            ("delta", delta_offsets_px),
            ("range", range_offsets_norm),
        ):
            if len(values) < 3 or len(values) % 2 != 1:
                raise ValueError(f"V18 {label} offsets must be odd and nontrivial")
            if float(values[len(values) // 2]) != 0.0:
                raise ValueError(f"V18 {label} offsets require zero center")
        if not 0.5 < float(keep_prior_probability) < 1.0:
            raise ValueError("V18 KEEP prior must be in (0.5, 1)")
        self.register_buffer(
            "visual_offsets_px", torch.tensor(visual_offsets_px, dtype=torch.float32)
        )
        self.register_buffer(
            "delta_offsets_px", torch.tensor(delta_offsets_px, dtype=torch.float32)
        )
        self.register_buffer(
            "range_offsets_norm", torch.tensor(range_offsets_norm, dtype=torch.float32)
        )
        keep_logit = math.log(float(keep_prior_probability))
        refine_logit = math.log(1.0 - float(keep_prior_probability))
        self.register_buffer(
            "policy_prior", torch.tensor((keep_logit, refine_logit), dtype=torch.float32)
        )
        self.slot_norm = nn.LayerNorm(int(slot_dim))
        self.slot_projection = nn.Linear(int(slot_dim), self.hidden_dim)
        self.candidate_projection = nn.Linear(int(candidate_dim), self.hidden_dim)
        self.slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.row_position = nn.Linear(4, self.hidden_dim, bias=False)
        self.anchor_geometry = nn.Linear(4, self.hidden_dim, bias=False)
        self.initial_norm = nn.LayerNorm(self.hidden_dim)
        nn.init.normal_(self.slot_tokens.weight, std=0.02)

        self.scale_norms = nn.ModuleDict(
            {name: nn.LayerNorm(int(feature_dim)) for name in self.scale_names}
        )
        self.scale_keys = nn.ModuleDict(
            {
                name: nn.Linear(int(feature_dim), self.hidden_dim, bias=False)
                for name in self.scale_names
            }
        )
        self.scale_values = nn.ModuleDict(
            {
                name: nn.Linear(int(feature_dim), self.hidden_dim, bias=False)
                for name in self.scale_names
            }
        )
        self.visual_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.offset_key = nn.Linear(4, self.hidden_dim, bias=False)
        self.offset_state = nn.Linear(4, self.hidden_dim, bias=False)
        self.visual_context = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.proposal_norm = nn.LayerNorm(int(proposal_dim))
        self.proposal_key = nn.Linear(int(proposal_dim), self.hidden_dim, bias=False)
        self.proposal_value = nn.Linear(int(proposal_dim), self.hidden_dim, bias=False)
        self.proposal_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.relative_bias = nn.Sequential(
            nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 1, bias=False)
        )
        self.relative_value = nn.Sequential(
            nn.Linear(8, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.proposal_context = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )
        vertical_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical = nn.TransformerEncoder(
            vertical_layer, num_layers=1, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.delta_head = nn.Linear(
            self.hidden_dim, len(delta_offsets_px), bias=False
        )
        self.range_head = nn.Linear(
            self.hidden_dim, 2 * len(range_offsets_norm), bias=False
        )
        self.uncertainty_head = nn.Linear(self.hidden_dim, 1)
        self.policy_head = nn.Linear(self.hidden_dim + 4, 2)
        self.activity_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.range_head.weight)
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.activity_head.weight)
        nn.init.zeros_(self.activity_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    def _proposal_context(
        self,
        hidden: torch.Tensor,
        current_x: torch.Tensor,
        current_range: torch.Tensor,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, rows, _channels = hidden.shape
        finite = torch.isfinite(proposal_x)
        safe_x = torch.where(finite, proposal_x, torch.zeros_like(proposal_x))
        width = float(max(self.input_w - 1, 1))
        dx = (safe_x[:, None] - current_x.unsqueeze(2)) / width
        slope_delta = (
            _row_slope(safe_x)[:, None] - _row_slope(current_x).unsqueeze(2)
        ) / width
        row_y = row_fraction.view(1, 1, rows)
        proposal_visible = (
            (row_y >= proposal_range[..., :1])
            & (row_y <= proposal_range[..., 1:])
            & finite
        )
        range_start = proposal_range[:, None, :, None, 0] - current_range[
            :, :, None, None, 0
        ]
        range_end = proposal_range[:, None, :, None, 1] - current_range[
            :, :, None, None, 1
        ]
        relative = torch.stack(
            (
                dx,
                dx.abs(),
                dx.square(),
                slope_delta,
                slope_delta.abs(),
                proposal_visible[:, None].expand(-1, slots, -1, -1).float(),
                range_start.expand(-1, -1, -1, rows),
                range_end.expand(-1, -1, -1, rows),
            ),
            dim=-1,
        ).permute(0, 1, 3, 2, 4)
        keys = self.proposal_key(self.proposal_norm(proposal_rows.float()))
        values = self.proposal_value(self.proposal_norm(proposal_rows.float()))
        logits = torch.einsum(
            "bsrh,bnrh->bsrn", self.proposal_query(hidden), keys
        ) / math.sqrt(float(self.hidden_dim))
        logits = logits + self.relative_bias(relative).squeeze(-1)
        valid = candidate_valid[:, None, None, :].bool() & proposal_visible.permute(
            0, 2, 1
        )[:, None]
        probability = torch.softmax(logits.masked_fill(~valid, -1.0e4).float(), dim=-1)
        probability = probability * valid.float()
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        value = values.permute(0, 2, 1, 3)[:, None] + self.relative_value(relative)
        context = torch.einsum("bsrn,bsrnh->bsrh", probability, value)
        return hidden + self.proposal_context(context), probability

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        candidate_state: torch.Tensor,
        route_indices: torch.Tensor,
        anchor_x: torch.Tensor,
        anchor_range: torch.Tensor,
        geometry_valid: torch.Tensor,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        image_features: dict[str, torch.Tensor],
        legacy_active_logits: torch.Tensor,
        set_margin: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, slots, rows = anchor_x.shape
        row_fraction = fixed_row_fractions(
            rows, device=anchor_x.device, dtype=torch.float32
        )
        safe_route = route_indices.clamp(min=0)
        selected_candidate = candidate_state.gather(
            1,
            safe_route.unsqueeze(-1).expand(-1, -1, int(candidate_state.shape[-1])),
        )
        selected_candidate = torch.where(
            (route_indices >= 0).unsqueeze(-1),
            selected_candidate,
            torch.zeros_like(selected_candidate),
        )
        width = float(max(self.input_w - 1, 1))
        slot = self.slot_projection(self.slot_norm(slot_states.float())).unsqueeze(2)
        slot = slot + self.slot_tokens.weight.view(1, slots, 1, self.hidden_dim)
        slot = slot + self.candidate_projection(selected_candidate).unsqueeze(2)
        row = self.row_position(_position_basis(row_fraction)).view(
            1, 1, rows, self.hidden_dim
        )
        anchor_geometry = torch.stack(
            (
                anchor_x / width,
                _row_slope(anchor_x) / width,
                anchor_range[..., 0].unsqueeze(-1).expand_as(anchor_x),
                anchor_range[..., 1].unsqueeze(-1).expand_as(anchor_x),
            ),
            dim=-1,
        )
        hidden = self.initial_norm(slot + row + self.anchor_geometry(anchor_geometry))

        offsets = self.visual_offsets_px.to(anchor_x)
        key_sum: torch.Tensor | None = None
        value_sum: torch.Tensor | None = None
        for name in self.scale_names:
            feature = image_features.get(name)
            if not isinstance(feature, torch.Tensor):
                raise ValueError(f"V18 refiner is missing image scale {name!r}")
            sampled = _CandidateAssociationEncoder.sample_feature(
                feature,
                anchor_x,
                row_fraction,
                offsets,
                input_w=self.input_w,
                sampling_backend=self.sampling_backend,
            )
            normalized = self.scale_norms[name](sampled)
            key = self.scale_keys[name](normalized)
            value = self.scale_values[name](normalized)
            key_sum = key if key_sum is None else key_sum + key
            value_sum = value if value_sum is None else value_sum + value
        if key_sum is None or value_sum is None:
            raise RuntimeError("V18 refiner has no visual evidence")
        offset_basis = _position_basis(offsets / width)
        keys = key_sum / float(len(self.scale_names))
        keys = keys + self.offset_key(offset_basis).view(
            1, 1, 1, -1, self.hidden_dim
        )
        values = value_sum / float(len(self.scale_names))
        visual_logits = torch.einsum(
            "bsrh,bsrkh->bsrk", self.visual_query(hidden), keys
        ) / math.sqrt(float(self.hidden_dim))
        visual_probability = torch.softmax(visual_logits.float(), dim=-1)
        visual_context = torch.einsum(
            "bsrk,bsrkh->bsrh", visual_probability, values.float()
        )
        expected_offset = torch.einsum(
            "bsrk,k->bsr", visual_probability, offsets.float()
        )
        hidden = hidden + self.visual_context(visual_context)
        hidden = hidden + self.offset_state(_position_basis(expected_offset / width))

        hidden, proposal_probability = self._proposal_context(
            hidden,
            anchor_x,
            anchor_range,
            proposal_rows,
            proposal_x,
            proposal_range,
            candidate_valid,
            row_fraction,
        )
        hidden = hidden + self.fusion(self.fusion_norm(hidden))
        hidden = self.vertical(
            hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        normalized = self.output_norm(hidden)

        delta_logits = self.delta_head(normalized)
        delta_probability = torch.softmax(delta_logits.float(), dim=-1)
        delta = _symmetric_expectation(delta_probability, self.delta_offsets_px)
        refined_x = (anchor_x + delta).clamp(0.0, width)
        pooled = normalized.mean(dim=2)
        range_logits = self.range_head(pooled).view(
            batch, slots, 2, int(self.range_offsets_norm.numel())
        )
        range_probability = torch.softmax(range_logits.float(), dim=-1)
        range_delta = _symmetric_expectation(
            range_probability, self.range_offsets_norm
        )
        refined_range = sort_range_norm((anchor_range + range_delta).clamp(0.0, 1.0))
        log_sigma = self.uncertainty_head(normalized).squeeze(-1).clamp(-4.0, 4.0)

        uncertainty = log_sigma.exp().mean(dim=-1)
        mean_delta = delta.abs().mean(dim=-1) / width
        max_delta = delta.abs().amax(dim=-1) / width
        policy_features = torch.cat(
            (
                pooled,
                set_margin[:, None, None].expand(-1, slots, 1),
                uncertainty.unsqueeze(-1),
                mean_delta.unsqueeze(-1),
                max_delta.unsqueeze(-1),
            ),
            dim=-1,
        )
        policy_logits = self.policy_head(policy_features) + self.policy_prior
        policy = policy_logits.argmax(dim=-1)
        use_refine = policy == 1
        final_x = torch.where(use_refine.unsqueeze(-1), refined_x, anchor_x)
        final_range = torch.where(
            use_refine.unsqueeze(-1), refined_range, anchor_range
        )
        final_x = torch.where(geometry_valid.unsqueeze(-1), final_x, anchor_x)
        final_range = torch.where(
            geometry_valid.unsqueeze(-1), final_range, anchor_range
        )
        activity_residual = self.activity_head(pooled).squeeze(-1)
        final_active_logits = legacy_active_logits.float() + activity_residual
        return {
            "refined_x": refined_x,
            "refined_range": refined_range,
            "final_x": final_x,
            "final_range": final_range,
            "delta": delta,
            "delta_logits": delta_logits,
            "range_logits": range_logits,
            "log_sigma": log_sigma,
            "policy_logits": policy_logits,
            "policy": policy,
            "active_logits": final_active_logits,
            "visual_logits": visual_logits,
            "visual_attention": visual_probability,
            "proposal_attention": proposal_probability,
        }


class FourSlotJointExactSetEnergy(nn.Module):
    """V18 exact global four-set routing with one safe refinement option."""

    def __init__(
        self,
        proposal_dim: int,
        *,
        feature_dim: int,
        slot_dim: int,
        hidden_dim: int,
        input_w: int,
        candidate_count: int = 32,
        num_slots: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.0,
        scale_names: tuple[str, ...] = ("p2", "p3", "p4"),
        association_offsets_px: tuple[float, ...] = (
            -32.0,
            -16.0,
            -8.0,
            0.0,
            8.0,
            16.0,
            32.0,
        ),
        visual_offsets_px: tuple[float, ...] = (
            -64.0,
            -32.0,
            -16.0,
            0.0,
            16.0,
            32.0,
            64.0,
        ),
        delta_offsets_px: tuple[float, ...] = (
            -64.0,
            -32.0,
            -16.0,
            -8.0,
            0.0,
            8.0,
            16.0,
            32.0,
            64.0,
        ),
        range_offsets_norm: tuple[float, ...] = (
            -0.025,
            -0.0125,
            0.0,
            0.0125,
            0.025,
        ),
        permutation_temperature: float = 1.0,
        keep_prior_probability: float = 0.997,
        detach_association_for_set_loss: bool = False,
        sampling_backend: str = "grid_sample",
    ) -> None:
        super().__init__()
        if int(num_slots) != 4 or int(candidate_count) != 32:
            raise ValueError("V18 causal contract requires S=4 and N=32")
        self.candidate_count = int(candidate_count)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.permutation_temperature = float(permutation_temperature)
        self.detach_association_for_set_loss = bool(
            detach_association_for_set_loss
        )
        combination_table, permutation_table, ordered = build_exact_four_set_tables(
            self.candidate_count
        )
        self.register_buffer("combination_table", combination_table, persistent=False)
        self.register_buffer("permutation_table", permutation_table, persistent=False)
        self.register_buffer("ordered_assignments", ordered, persistent=False)
        slot_pairs = torch.tensor(tuple(combinations(range(4), 2)), dtype=torch.long)
        self.register_buffer("slot_pairs", slot_pairs, persistent=False)

        self.association = _CandidateAssociationEncoder(
            proposal_dim=int(proposal_dim),
            feature_dim=int(feature_dim),
            hidden_dim=self.hidden_dim,
            input_w=int(input_w),
            num_heads=int(num_heads),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            scale_names=tuple(scale_names),
            offsets_px=tuple(association_offsets_px),
            sampling_backend=str(sampling_backend),
        )
        self.slot_norm = nn.LayerNorm(int(slot_dim))
        self.slot_unary = nn.Linear(int(slot_dim), self.hidden_dim, bias=False)
        self.candidate_unary = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.unary_interaction = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        nn.init.zeros_(self.unary_interaction[-1].weight)

        self.pair_left = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.pair_right = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.pair_geometry = nn.Linear(10, self.hidden_dim, bias=False)
        self.pair_tokens = nn.Embedding(6, self.hidden_dim)
        self.pair_hidden = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.pair_output = nn.Linear(self.hidden_dim, 1, bias=False)
        nn.init.normal_(self.pair_tokens.weight, std=0.02)
        nn.init.zeros_(self.pair_output.weight)

        self.refiner = _OneShotKeepRefineGeometry(
            proposal_dim=int(proposal_dim),
            feature_dim=int(feature_dim),
            slot_dim=int(slot_dim),
            candidate_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            input_w=int(input_w),
            num_slots=self.num_slots,
            num_heads=int(num_heads),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            scale_names=tuple(scale_names),
            visual_offsets_px=tuple(visual_offsets_px),
            delta_offsets_px=tuple(delta_offsets_px),
            range_offsets_norm=tuple(range_offsets_norm),
            keep_prior_probability=float(keep_prior_probability),
            sampling_backend=str(sampling_backend),
        )

    @staticmethod
    def _pair_geometry(
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        *,
        input_w: int,
    ) -> torch.Tensor:
        x = proposal_x.detach().float()
        ranges = sort_range_norm(proposal_range.detach().float())
        finite = torch.isfinite(x)
        safe_x = torch.where(finite, x, torch.zeros_like(x))
        common = finite[:, :, None] & finite[:, None, :]
        denominator = common.sum(dim=-1).clamp_min(1)
        signed = (safe_x[:, :, None] - safe_x[:, None, :]) / float(
            max(input_w - 1, 1)
        )
        mean_signed = (signed * common.float()).sum(dim=-1) / denominator
        mean_abs = (signed.abs() * common.float()).sum(dim=-1) / denominator
        bottom_index = max(int(x.shape[-1]) - 1, 0)
        top_index = 0
        bottom_dx = signed[..., bottom_index]
        top_dx = signed[..., top_index]
        slope = _row_slope(safe_x) / float(max(input_w - 1, 1))
        slope_delta = (slope[:, :, None] - slope[:, None, :]).abs()
        mean_slope = (slope_delta * common.float()).sum(dim=-1) / denominator
        start_delta = ranges[:, :, None, 0] - ranges[:, None, :, 0]
        end_delta = ranges[:, :, None, 1] - ranges[:, None, :, 1]
        overlap = (
            torch.minimum(ranges[:, :, None, 1], ranges[:, None, :, 1])
            - torch.maximum(ranges[:, :, None, 0], ranges[:, None, :, 0])
        ).clamp_min(0.0)
        common_fraction = common.float().mean(dim=-1)
        crossing_fraction = (
            (signed[..., 1:] * signed[..., :-1] < 0.0)
            & common[..., 1:]
            & common[..., :-1]
        ).float().mean(dim=-1)
        valid_pair = (
            candidate_valid[:, :, None] & candidate_valid[:, None, :]
        ).float()
        return torch.stack(
            (
                mean_signed,
                mean_abs,
                bottom_dx,
                top_dx,
                mean_slope,
                start_delta,
                end_delta,
                overlap,
                common_fraction,
                crossing_fraction,
            ),
            dim=-1,
        ) * valid_pair.unsqueeze(-1)

    def route(
        self,
        *,
        slot_states: torch.Tensor,
        legacy_route_logits: torch.Tensor,
        legacy_active_logits: torch.Tensor,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        image_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        association = self.association(
            proposal_rows=proposal_rows,
            proposal_x=proposal_x,
            proposal_range=proposal_range,
            candidate_valid=candidate_valid,
            image_features=image_features,
        )
        live_candidate = association["candidate_state"]
        energy_candidate = (
            live_candidate.detach()
            if self.detach_association_for_set_loss
            else live_candidate
        )
        slot = self.slot_unary(self.slot_norm(slot_states.float()))
        candidate = self.candidate_unary(energy_candidate)
        slot_view = slot[:, :, None].expand(-1, -1, self.candidate_count, -1)
        candidate_view = candidate[:, None].expand(-1, self.num_slots, -1, -1)
        unary_feature = torch.cat(
            (slot_view, candidate_view, slot_view * candidate_view), dim=-1
        )
        unary_residual = self.unary_interaction(unary_feature).squeeze(-1)
        unary = legacy_route_logits.detach().float() + unary_residual
        unary = unary.masked_fill(~candidate_valid[:, None, :].bool(), -1.0e4)

        left = self.pair_left(energy_candidate)[:, :, None]
        right = self.pair_right(energy_candidate)[:, None, :]
        relation = self._pair_geometry(
            proposal_x,
            proposal_range,
            candidate_valid,
            input_w=self.association.input_w,
        )
        pair_base = left + right + self.pair_geometry(relation)
        pair_hidden = pair_base[:, None] + self.pair_tokens.weight.view(
            1, 6, 1, 1, self.hidden_dim
        )
        pair_energy = self.pair_output(self.pair_hidden(pair_hidden)).squeeze(-1)

        ordered_energy, valid_set = exact_ordered_set_energies(
            unary,
            pair_energy,
            legacy_active_logits,
            candidate_valid,
            self.ordered_assignments,
            self.slot_pairs,
        )
        set_scores = unordered_set_log_scores(
            ordered_energy,
            valid_set,
            permutation_temperature=self.permutation_temperature,
        )
        decoded = decode_exact_ordered_set(
            ordered_energy,
            valid_set,
            self.ordered_assignments,
        )
        top_values = ordered_energy.float().masked_fill(
            ~valid_set.unsqueeze(-1), -1.0e9
        ).reshape(int(unary.shape[0]), -1).topk(2, dim=-1).values
        margin = top_values[:, 0] - top_values[:, 1]
        return {
            "indices": decoded["indices"],
            "unary": unary,
            "unary_residual": unary_residual,
            "pair_energy": pair_energy,
            "ordered_energy": ordered_energy,
            "set_scores": set_scores,
            "valid_set": valid_set,
            "set_index": decoded["set_index"],
            "permutation_index": decoded["permutation_index"],
            "set_margin": margin,
            "candidate_state": live_candidate,
            "association_row_state": association["row_state"],
            "association_visual_attention": association["visual_attention"],
        }

    def refine(
        self,
        *,
        route_result: dict[str, torch.Tensor],
        slot_states: torch.Tensor,
        anchor_x: torch.Tensor,
        anchor_range: torch.Tensor,
        geometry_valid: torch.Tensor,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        image_features: dict[str, torch.Tensor],
        legacy_active_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.refiner(
            # V7 supplies an immutable deployment anchor/prior.  V18 may
            # reshape the live candidate/image representation, but geometry
            # and policy losses must not leak back through the mature V7
            # route, bounded refiner, or activity path.
            slot_states=slot_states.detach(),
            candidate_state=route_result["candidate_state"],
            route_indices=route_result["indices"],
            anchor_x=anchor_x.detach().float(),
            anchor_range=sort_range_norm(anchor_range.detach().float()),
            geometry_valid=geometry_valid.bool(),
            proposal_rows=proposal_rows,
            proposal_x=proposal_x.detach().float(),
            proposal_range=sort_range_norm(proposal_range.detach().float()),
            candidate_valid=candidate_valid.bool(),
            image_features=image_features,
            legacy_active_logits=legacy_active_logits.detach(),
            # Margin is a policy feature, not a second training signal for
            # the exact router.  Only the physical-set listwise objective is
            # allowed to shape unary/pair energies.
            set_margin=route_result["set_margin"].detach(),
        )
