from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backbone_dla import DLA34Backbone
from .common import fixed_indices, fixed_row_fractions, sort_range_norm
from .fpn import SimpleFPN
from .v23_ordered_slot_cost_volume import soft_viterbi_marginals


@dataclass(frozen=True)
class V25LossWeights:
    existence: float = 2.0
    row_distribution: float = 5.0
    point: float = 1.0
    strip_iou: float = 2.0
    range: float = 1.0
    quality50: float = 0.5
    quality75: float = 0.25
    smoothness: float = 0.25
    order: float = 0.25
    duplicate: float = 0.25
    line_width: float = 30.0
    minimum_valid_rows: int = 5
    minimum_spacing_px: float = 12.0


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _last_valid_x(x_rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    rows = int(x_rows.shape[-1])
    index = fixed_indices(rows, device=x_rows.device, dtype=torch.long).view(
        *([1] * (x_rows.ndim - 1)), rows
    )
    last = torch.where(valid, index, torch.full_like(index, -1)).amax(dim=-1)
    safe = last.clamp_min(0)
    value = x_rows.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return torch.where(last >= 0, value, torch.full_like(value, float("inf")))


def build_ordered_lane_targets(
    targets: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    slots: int,
    rows: int,
    input_w: int,
    minimum_valid_rows: int = 5,
) -> dict[str, torch.Tensor]:
    """Pad canonical left-to-right CULane targets to fixed lane objects."""

    batch = len(targets)
    x = torch.zeros((batch, slots, rows), dtype=torch.float32)
    valid = torch.zeros((batch, slots, rows), dtype=torch.bool)
    ranges = torch.zeros((batch, slots, 2), dtype=torch.float32)
    active = torch.zeros((batch, slots), dtype=torch.bool)
    counts: list[int] = []
    for batch_index, target in enumerate(targets):
        target_x = target["x_rows"].detach().cpu().float()
        target_valid = target["valid_mask"].detach().cpu().bool()
        if target_x.ndim != 2 or target_valid.shape != target_x.shape:
            raise ValueError("V25 targets require x_rows/valid_mask [G,R]")
        if int(target_x.shape[-1]) != rows:
            raise ValueError(
                f"V25 target rows={target_x.shape[-1]}, expected {rows}"
            )
        target_valid = (
            target_valid
            & torch.isfinite(target_x)
            & (target_x >= 0.0)
            & (target_x < float(input_w))
        )
        keep = target_valid.sum(dim=-1) >= int(minimum_valid_rows)
        target_x = target_x[keep]
        target_valid = target_valid[keep]
        count = int(target_x.shape[0])
        if count > slots:
            raise ValueError(
                f"V25 has {slots} final lane objects but target contains {count} lanes"
            )
        if count:
            order = _last_valid_x(target_x, target_valid).argsort(stable=True)
            target_x = target_x[order]
            target_valid = target_valid[order]
            row_index = fixed_indices(rows, device=torch.device("cpu"), dtype=torch.long)
            row_index = row_index.view(1, rows)
            first = torch.where(
                target_valid, row_index, torch.full_like(row_index, rows)
            ).amin(dim=-1)
            last = torch.where(
                target_valid, row_index, torch.full_like(row_index, -1)
            ).amax(dim=-1)
            target_range = torch.stack(
                (first.float() / float(rows), last.float() / float(rows)), dim=-1
            )
            x[batch_index, :count] = target_x
            valid[batch_index, :count] = target_valid
            ranges[batch_index, :count] = target_range
            active[batch_index, :count] = True
        counts.append(count)
    return {
        "x_rows": x.to(device=device, non_blocking=True),
        "valid_mask": valid.to(device=device, non_blocking=True),
        "range_norm": ranges.to(device=device, non_blocking=True),
        "active": active.to(device=device, non_blocking=True),
        "counts": torch.tensor(counts, device=device, dtype=torch.long),
    }


def hard_viterbi_paths(
    unary_logits: torch.Tensor,
    *,
    transition_radius_bins: int = 8,
    transition_penalty: float = 0.15,
) -> torch.Tensor:
    """Return the coherent maximum-score x-bin path for every lane object."""

    if unary_logits.ndim != 4:
        raise ValueError("hard Viterbi expects unary logits [B,S,R,X]")
    radius = int(transition_radius_bins)
    if radius < 0:
        raise ValueError("transition radius must be non-negative")
    batch, slots, rows, bins = unary_logits.shape
    flat = unary_logits.float().reshape(batch * slots, rows, bins)
    score = flat[:, 0]
    backpointers: list[torch.Tensor] = []
    if radius:
        offsets = fixed_indices(
            2 * radius + 1, device=flat.device, dtype=flat.dtype
        ) - radius
        penalty = offsets.abs() * float(transition_penalty)
        current_x = fixed_indices(
            bins, device=flat.device, dtype=torch.long
        ).view(1, bins)
    for row in range(1, rows):
        if radius:
            neighbours = F.pad(score, (radius, radius), value=-1.0e9).unfold(
                -1, 2 * radius + 1, 1
            )
            best_score, best_offset = (neighbours - penalty).max(dim=-1)
            predecessor = (current_x + best_offset - radius).clamp(0, bins - 1)
        else:
            best_score = score
            predecessor = fixed_indices(
                bins, device=flat.device, dtype=torch.long
            ).view(1, bins).expand(batch * slots, bins)
        score = flat[:, row] + best_score
        # A row-constant shift leaves the argmax path exact and prevents the
        # dynamic-programming score from growing unnecessarily under AMP.
        score = score - score.amax(dim=-1, keepdim=True)
        backpointers.append(predecessor.to(dtype=torch.int16))
    path = torch.empty(
        (batch * slots, rows), device=flat.device, dtype=torch.long
    )
    path[:, -1] = score.argmax(dim=-1)
    for row in range(rows - 1, 0, -1):
        pointer = backpointers[row - 1].long()
        path[:, row - 1] = pointer.gather(
            1, path[:, row].unsqueeze(-1)
        ).squeeze(-1)
    return path.view(batch, slots, rows)


def image_mediated_other_coverage(probability: torch.Tensor) -> torch.Tensor:
    """Probability that another final lane object owns the same image bin."""

    if probability.ndim != 4:
        raise ValueError("ownership probability must have shape [B,S,R,X]")
    slots = int(probability.shape[1])
    if slots < 2:
        return torch.zeros_like(probability)
    outputs = []
    bounded = probability.float().clamp(0.0, 1.0 - 1.0e-6)
    for slot in range(slots):
        not_owned = torch.ones_like(bounded[:, 0])
        for other in range(slots):
            if other != slot:
                not_owned = not_owned * (1.0 - bounded[:, other])
        outputs.append(1.0 - not_owned)
    return torch.stack(outputs, dim=1).to(dtype=probability.dtype)


class ImageMediatedLaneBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        enable_competition: bool,
        enable_slot_interaction: bool,
        competition_weight: float,
    ) -> None:
        super().__init__()
        self.enable_competition = bool(enable_competition)
        self.enable_slot_interaction = bool(enable_slot_interaction)
        self.competition_weight = float(competition_weight)
        self.context_projection = nn.Linear(dim, dim)
        self.context_norm = nn.LayerNorm(dim)
        vertical = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical = nn.TransformerEncoder(
            vertical, num_layers=1, norm=nn.LayerNorm(dim)
        )
        self.slot_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.slot_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        prior_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, slots, rows, dim = query.shape
        scale = 1.0 / math.sqrt(float(dim))
        logits = torch.einsum("bsrc,bcrx->bsrx", query, keys) * scale
        if prior_logits is not None:
            if prior_logits.shape != logits.shape:
                raise ValueError("proposal prior and image cost volume differ in shape")
            logits = logits + prior_logits
        probability = logits.float().softmax(dim=-1).to(dtype=query.dtype)
        other_coverage = image_mediated_other_coverage(probability)
        if self.enable_competition:
            logits = logits - self.competition_weight * other_coverage
            probability = logits.float().softmax(dim=-1).to(dtype=query.dtype)
            other_coverage = image_mediated_other_coverage(probability)
        context = torch.einsum("bsrx,bcrx->bsrc", probability, values)
        query = self.context_norm(query + self.context_projection(context))
        query = self.vertical(query.reshape(batch * slots, rows, dim)).reshape(
            batch, slots, rows, dim
        )
        if self.enable_slot_interaction:
            row_slots = query.permute(0, 2, 1, 3).reshape(
                batch * rows, slots, dim
            )
            delta, _ = self.slot_attention(
                row_slots, row_slots, row_slots, need_weights=False
            )
            row_slots = self.slot_norm(row_slots + delta)
            query = row_slots.reshape(batch, rows, slots, dim).permute(0, 2, 1, 3)
        query = self.ffn_norm(query + self.ffn(query))
        return query, logits, probability, other_coverage


class V25ImageMediatedLaneObjects(nn.Module):
    """Four image-owned lane objects with coherent continuous path output."""

    def __init__(
        self,
        *,
        input_h: int = 640,
        input_w: int = 1600,
        num_rows: int = 160,
        x_bins: int = 800,
        fpn_channels: int = 256,
        hidden_dim: int = 96,
        decoder_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        transition_radius_bins: int = 8,
        transition_penalty: float = 0.15,
        enable_competition: bool = False,
        enable_slot_interaction: bool = False,
        competition_weight: float = 2.0,
        anchor_prior_weight: float = 0.10,
        anchor_prior_sigma: float = 0.28,
        decode_mode: str = "hard_path",
        pretrained_backbone: bool = True,
        require_pretrained_backbone: bool = True,
        pretrained_weights_path: str = "",
        freeze_batch_norm_stats: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("V25 hidden_dim must be divisible by num_heads")
        if decoder_layers < 1:
            raise ValueError("V25 requires at least one decoder layer")
        if decode_mode not in {"hard_path", "row_argmax", "expectation"}:
            raise ValueError(f"unsupported V25 decode_mode={decode_mode!r}")
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.num_slots = 4
        self.transition_radius_bins = int(transition_radius_bins)
        self.transition_penalty = float(transition_penalty)
        self.anchor_prior_weight = float(anchor_prior_weight)
        self.anchor_prior_sigma = float(anchor_prior_sigma)
        self.decode_mode = str(decode_mode)
        self.freeze_batch_norm_stats = bool(freeze_batch_norm_stats)

        self.backbone = DLA34Backbone(
            pretrained=bool(pretrained_backbone),
            require_pretrained=bool(require_pretrained_backbone),
            weights_path=pretrained_weights_path or None,
        )
        self.fpn = SimpleFPN(
            in_channels=self.backbone.out_channels,
            out_channels=int(fpn_channels),
        )
        self.fine_stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.p2_projection = nn.Sequential(
            nn.Conv2d(int(fpn_channels), int(hidden_dim), 1, bias=False),
            nn.GroupNorm(_group_count(int(hidden_dim)), int(hidden_dim)),
            nn.GELU(),
        )
        self.fine_projection = nn.Sequential(
            nn.Conv2d(32, int(hidden_dim), 1, bias=False),
            nn.GroupNorm(_group_count(int(hidden_dim)), int(hidden_dim)),
            nn.GELU(),
        )
        self.image_fusion = nn.Sequential(
            nn.Conv2d(2 * int(hidden_dim), int(hidden_dim), 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(int(hidden_dim)), int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), 1, bias=False),
        )
        self.key_projection = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.value_projection = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.slot_embedding = nn.Parameter(torch.empty(self.num_slots, hidden_dim))
        self.row_embedding = nn.Parameter(torch.empty(self.num_rows, hidden_dim))
        self.anchor_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.ModuleList(
            [
                ImageMediatedLaneBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    enable_competition=enable_competition,
                    enable_slot_interaction=enable_slot_interaction,
                    competition_weight=competition_weight,
                )
                for _ in range(int(decoder_layers))
            ]
        )
        self.exist_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.range_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.quality_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.register_buffer(
            "canonical_slot_centres",
            torch.linspace(0.16, 0.84, self.num_slots),
            persistent=True,
        )
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        # Begin with a broad road-visible interval instead of two identical
        # sigmoid outputs (which would write a zero-height lane at step zero).
        nn.init.zeros_(self.range_head[-1].weight)
        self.range_head[-1].bias.data.copy_(
            torch.tensor((-1.3862944, 2.9444390))  # sigmoid -> 0.20 / 0.95
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batch_norm_stats:
            for module in (*self.backbone.modules(), *self.fpn.modules()):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def _image_features(self, images: torch.Tensor) -> torch.Tensor:
        coarse = self.p2_projection(self.fpn(self.backbone(images)))
        fine = self.fine_projection(self.fine_stem(images))
        size = (self.num_rows, self.x_bins)
        coarse = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        fine = F.interpolate(fine, size=size, mode="bilinear", align_corners=False)
        return self.image_fusion(torch.cat((coarse, fine), dim=1))

    def _initial_query(self, batch: int, *, device: torch.device) -> torch.Tensor:
        rows = fixed_row_fractions(
            self.num_rows, device=device, dtype=torch.float32
        ).view(1, 1, self.num_rows)
        centres = self.canonical_slot_centres.to(device=device).view(
            1, self.num_slots, 1
        )
        geometry = torch.stack(
            (
                centres.expand(batch, self.num_slots, self.num_rows),
                rows.expand(batch, self.num_slots, self.num_rows),
            ),
            dim=-1,
        )
        return (
            self.slot_embedding.view(1, self.num_slots, 1, -1)
            + self.row_embedding.view(1, 1, self.num_rows, -1)
            + self.anchor_projection(geometry)
        )

    def _anchor_prior(self, batch: int, *, device: torch.device) -> torch.Tensor:
        x = (
            fixed_indices(self.x_bins, device=device, dtype=torch.float32) + 0.5
        ) / float(self.x_bins)
        centres = self.canonical_slot_centres.to(device=device).view(
            1, self.num_slots, 1, 1
        )
        prior = -0.5 * ((x.view(1, 1, 1, -1) - centres) / self.anchor_prior_sigma).pow(2)
        return self.anchor_prior_weight * prior.expand(
            batch, self.num_slots, self.num_rows, self.x_bins
        )

    def _decode(
        self, unary_logits: torch.Tensor, path_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        bin_width = float(self.input_w) / float(self.x_bins)
        centres = (
            fixed_indices(
                self.x_bins, device=unary_logits.device, dtype=torch.float32
            )
            + 0.5
        ) * bin_width
        posterior = path_logits.float().softmax(dim=-1)
        expected = (posterior * centres.view(1, 1, 1, -1)).sum(dim=-1)
        row_index = unary_logits.float().argmax(dim=-1)
        row_argmax = (row_index.float() + 0.5) * bin_width
        # The discrete path is a writer/inference operation. Running another
        # 159-step recurrence during training adds no gradient and previously
        # consumed roughly one third of decoder time. Row argmax is retained
        # as a cheap training diagnostic; eval always computes the exact path.
        hard_index = (
            row_index
            if self.training
            else hard_viterbi_paths(
                unary_logits,
                transition_radius_bins=self.transition_radius_bins,
                transition_penalty=self.transition_penalty,
            )
        )
        hard = (hard_index.float() + 0.5) * bin_width
        selected = {
            "expectation": expected,
            "row_argmax": row_argmax,
            "hard_path": hard,
        }[self.decode_mode]
        return {
            "pred_x_rows": selected,
            "soft_x_rows": expected,
            "row_argmax_x_rows": row_argmax,
            "hard_path_x_rows": hard,
            "hard_path_indices": hard_index,
            "path_posterior": posterior,
        }

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(images.shape[0])
        features = self._image_features(images)
        keys = self.key_projection(features)
        values = self.value_projection(features)
        query = self._initial_query(batch, device=images.device).to(dtype=features.dtype)
        prior = self._anchor_prior(batch, device=images.device).to(dtype=features.dtype)
        final_layer_logits: torch.Tensor | None = None
        final_other_coverage: torch.Tensor | None = None
        cumulative = prior
        for layer in self.decoder:
            query, logits, _probability, other_coverage = layer(
                query, keys, values, prior_logits=cumulative
            )
            final_layer_logits = logits
            final_other_coverage = other_coverage
            # ``logits`` already contains the prior/cumulative evidence passed
            # into this block. Adding it again would double all earlier
            # evidence at every layer.
            cumulative = logits
        unary_logits = cumulative
        path_logits = soft_viterbi_marginals(
            unary_logits,
            transition_radius_bins=self.transition_radius_bins,
            transition_penalty=self.transition_penalty,
        )
        decoded = self._decode(unary_logits, path_logits)
        pooled = query.mean(dim=2)
        exist_logits = self.exist_head(pooled)
        range_norm = sort_range_norm(torch.sigmoid(self.range_head(pooled)))
        quality = self.quality_head(pooled)
        return {
            "exist_logits": exist_logits,
            "pred_x_rows": decoded["pred_x_rows"].clamp(
                0.0, float(self.input_w - 1)
            ),
            "range_norm": range_norm,
            "quality_logits": quality[..., 0],
            "quality50_logits": quality[..., 0],
            "quality75_logits": quality[..., 1],
            "unary_logits": unary_logits,
            "path_logits": path_logits,
            "soft_x_rows": decoded["soft_x_rows"],
            "row_argmax_x_rows": decoded["row_argmax_x_rows"],
            "hard_path_x_rows": decoded["hard_path_x_rows"],
            "hard_path_indices": decoded["hard_path_indices"],
            "path_posterior": decoded["path_posterior"],
            "lane_object_state": query,
            "image_features": features,
            "final_layer_image_logits": final_layer_logits,
            "final_other_coverage": final_other_coverage,
        }


def _aligned_strip_iou(
    pred_x: torch.Tensor,
    pred_range: torch.Tensor,
    target_x: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    line_width: float,
) -> torch.Tensor:
    rows = int(pred_x.shape[-1])
    row_fraction = fixed_row_fractions(
        rows, device=pred_x.device, dtype=torch.float32
    ).view(1, 1, rows)
    pred_range = sort_range_norm(pred_range.float())
    pred_valid = (
        torch.isfinite(pred_x)
        & (row_fraction >= pred_range[..., :1])
        & (row_fraction <= pred_range[..., 1:])
    )
    target_valid = target_valid.bool() & torch.isfinite(target_x)
    both = pred_valid & target_valid
    either = pred_valid | target_valid
    overlap = (float(line_width) - (pred_x.float() - target_x.float()).abs()).clamp_min(0.0)
    overlap = torch.where(both, overlap, torch.zeros_like(overlap))
    union = torch.where(
        both,
        2.0 * float(line_width) - overlap,
        torch.where(
            either,
            torch.full_like(overlap, float(line_width)),
            torch.zeros_like(overlap),
        ),
    )
    return overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1.0e-6)


def _row_distribution_loss(
    logits: torch.Tensor,
    target_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    input_w: int,
) -> torch.Tensor:
    bins = int(logits.shape[-1])
    position = target_x.float() / (float(input_w) / float(bins)) - 0.5
    lower = position.floor().long().clamp(0, bins - 1)
    upper = (lower + 1).clamp(max=bins - 1)
    upper_weight = (position - lower.float()).clamp(0.0, 1.0)
    log_probability = logits.float().log_softmax(dim=-1)
    lower_loss = -log_probability.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    upper_loss = -log_probability.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    loss = lower_loss * (1.0 - upper_weight) + upper_loss * upper_weight
    return (loss * valid.float()).sum() / valid.sum().clamp_min(1).float()


def v25_lane_object_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    input_w: int,
    weights: V25LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    soft_x = outputs["soft_x_rows"].float()
    batch, slots, rows = soft_x.shape
    ordered = build_ordered_lane_targets(
        targets,
        device=soft_x.device,
        slots=slots,
        rows=rows,
        input_w=input_w,
        minimum_valid_rows=weights.minimum_valid_rows,
    )
    active = ordered["active"]
    valid = ordered["valid_mask"] & active.unsqueeze(-1)
    existence_target = torch.where(
        active,
        torch.zeros_like(active, dtype=torch.long),
        torch.ones_like(active, dtype=torch.long),
    )
    existence = F.cross_entropy(
        outputs["exist_logits"].float().reshape(-1, 2),
        existence_target.reshape(-1),
    )
    row = _row_distribution_loss(
        # Direct x-distribution supervision belongs on the image-owned unary
        # cost volume. Path marginals provide a structured geometry posterior
        # for point/IoU losses but can be extremely sharp at random init.
        outputs["unary_logits"],
        ordered["x_rows"],
        valid,
        input_w=input_w,
    )
    point_error = F.smooth_l1_loss(
        soft_x / float(input_w),
        ordered["x_rows"] / float(input_w),
        reduction="none",
        beta=0.01,
    )
    point = (point_error * valid.float()).sum() / valid.sum().clamp_min(1).float()
    range_error = F.smooth_l1_loss(
        outputs["range_norm"].float(),
        ordered["range_norm"],
        reduction="none",
        beta=0.02,
    ).mean(dim=-1)
    range_loss = (range_error * active.float()).sum() / active.sum().clamp_min(1).float()
    iou = _aligned_strip_iou(
        soft_x,
        outputs["range_norm"],
        ordered["x_rows"],
        valid,
        line_width=weights.line_width,
    )
    strip_iou = ((1.0 - iou) * active.float()).sum() / active.sum().clamp_min(1).float()
    quality50 = F.binary_cross_entropy_with_logits(
        outputs["quality50_logits"].float(),
        ((iou.detach() >= 0.50) & active).float(),
        reduction="none",
    )
    quality75 = F.binary_cross_entropy_with_logits(
        outputs["quality75_logits"].float(),
        ((iou.detach() >= 0.75) & active).float(),
        reduction="none",
    )
    quality_mask = active.float()
    quality50 = (quality50 * quality_mask).sum() / quality_mask.sum().clamp_min(1.0)
    quality75 = (quality75 * quality_mask).sum() / quality_mask.sum().clamp_min(1.0)

    if rows >= 3:
        pred_curvature = soft_x[..., 2:] - 2.0 * soft_x[..., 1:-1] + soft_x[..., :-2]
        target_x = ordered["x_rows"]
        target_curvature = target_x[..., 2:] - 2.0 * target_x[..., 1:-1] + target_x[..., :-2]
        triple = valid[..., 2:] & valid[..., 1:-1] & valid[..., :-2]
        smooth_error = F.smooth_l1_loss(
            pred_curvature / float(input_w),
            target_curvature / float(input_w),
            reduction="none",
            beta=0.005,
        )
        smoothness = (smooth_error * triple.float()).sum() / triple.sum().clamp_min(1).float()
    else:
        smoothness = soft_x.new_zeros(())

    order_terms: list[torch.Tensor] = []
    duplicate_terms: list[torch.Tensor] = []
    posterior = outputs["path_posterior"].float()
    for left in range(slots - 1):
        right = left + 1
        pair_active = active[:, left] & active[:, right]
        common = valid[:, left] & valid[:, right] & pair_active.unsqueeze(-1)
        violation = (
            float(weights.minimum_spacing_px)
            - (soft_x[:, right] - soft_x[:, left])
        ).clamp_min(0.0) / float(input_w)
        order_terms.append(
            (violation * common.float()).sum() / common.sum().clamp_min(1).float()
        )
        overlap = (posterior[:, left] * posterior[:, right]).sum(dim=-1)
        duplicate_terms.append(
            (overlap * common.float()).sum() / common.sum().clamp_min(1).float()
        )
    order_loss = torch.stack(order_terms).mean() if order_terms else soft_x.new_zeros(())
    duplicate = (
        torch.stack(duplicate_terms).mean()
        if duplicate_terms
        else soft_x.new_zeros(())
    )
    total = (
        weights.existence * existence
        + weights.row_distribution * row
        + weights.point * point
        + weights.strip_iou * strip_iou
        + weights.range * range_loss
        + weights.quality50 * quality50
        + weights.quality75 * quality75
        + weights.smoothness * smoothness
        + weights.order * order_loss
        + weights.duplicate * duplicate
    )
    diagnostics = {
        "loss_total": total.detach(),
        "loss_existence": existence.detach(),
        "loss_row_distribution": row.detach(),
        "loss_point": point.detach(),
        "loss_strip_iou": strip_iou.detach(),
        "loss_range": range_loss.detach(),
        "loss_quality50": quality50.detach(),
        "loss_quality75": quality75.detach(),
        "loss_smoothness": smoothness.detach(),
        "loss_order": order_loss.detach(),
        "loss_duplicate": duplicate.detach(),
        "mean_soft_iou": (iou * active.float()).sum().detach()
        / active.sum().clamp_min(1).float(),
        "active_lane_fraction": active.float().mean().detach(),
        "predicted_lane_probability": torch.softmax(
            outputs["exist_logits"].float(), dim=-1
        )[..., 0].mean().detach(),
        "hard_soft_mean_abs_px": (
            outputs["hard_path_x_rows"].float() - soft_x
        ).abs().mean().detach(),
    }
    return total, diagnostics


def v25_model_contract(model: V25ImageMediatedLaneObjects) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "final_geometry_owner": "four_image_mediated_lane_objects",
        "writer_depends_on_v7": False,
        "global_geometry_gate_present": False,
        "source_raw_router_present": False,
        "proposal_id_owns_geometry": False,
        "num_final_lane_objects": model.num_slots,
        "num_rows": model.num_rows,
        "x_bins": model.x_bins,
        "decode_mode": model.decode_mode,
        "trainable_parameters": parameters,
        "competition_enabled": any(
            block.enable_competition for block in model.decoder
        ),
        "slot_interaction_enabled": any(
            block.enable_slot_interaction for block in model.decoder
        ),
    }
