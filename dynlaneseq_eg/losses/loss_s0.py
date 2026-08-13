from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import math

import torch
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.modeling.common import fixed_indices, sort_range_norm
from .matcher_s0 import HungarianMatcherS0
from .range_aware_iou import (
    batched_pairwise_range_aware_row_strip_iou,
    pairwise_range_aware_row_strip_iou,
)


_SLOT_ASSIGNMENT_PATH_CACHE: dict[
    tuple[int, int, str, int | None],
    tuple[torch.Tensor, torch.Tensor],
] = {}


@dataclass(frozen=True)
class _MatchedLaneBatch:
    """All Hungarian-matched lanes packed across the image batch.

    The normal and three intermediate decoder outputs reuse the same five
    geometry objectives.  Packing their tiny per-image selections once keeps
    the exact assignment while avoiding four independent advanced-indexing
    graphs in every objective.
    """

    pred_x: torch.Tensor
    gt_x: torch.Tensor
    valid: torch.Tensor
    pred_range: torch.Tensor | None
    gt_range: torch.Tensor | None
    row_logits: torch.Tensor | None
    input_reference: torch.Tensor | None


def _padded_lane_targets(
    targets: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-count lane targets for small batched IoU matrices."""

    batch = len(targets)
    max_gt = max(
        (int(target["x_rows"].shape[0]) for target in targets),
        default=0,
    )
    if max_gt == 0:
        return (
            torch.zeros((batch, 0, rows), device=device, dtype=dtype),
            torch.zeros((batch, 0, rows), device=device, dtype=torch.bool),
        )
    padded_x: list[torch.Tensor] = []
    padded_valid: list[torch.Tensor] = []
    for target in targets:
        x_rows = target["x_rows"].to(device=device, dtype=dtype)
        valid = target["valid_mask"].to(device=device).bool()
        if tuple(x_rows.shape) != tuple(valid.shape) or int(x_rows.shape[1]) != rows:
            raise ValueError("lane target rows/validity shape mismatch")
        padding = max_gt - int(x_rows.shape[0])
        if padding:
            x_rows = torch.cat(
                (x_rows, x_rows.new_zeros((padding, rows))),
                dim=0,
            )
            valid = torch.cat(
                (
                    valid,
                    torch.zeros(
                        (padding, rows),
                        dtype=torch.bool,
                        device=device,
                    ),
                ),
                dim=0,
            )
        padded_x.append(x_rows)
        padded_valid.append(valid)
    return torch.stack(padded_x), torch.stack(padded_valid)


def _slot_assignment_paths(
    slots: int,
    gt_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached ordered slot paths and their inactive-slot masks."""

    key = (int(slots), int(gt_count), device.type, device.index)
    cached = _SLOT_ASSIGNMENT_PATH_CACHE.get(key)
    if cached is not None:
        return cached
    path_values = tuple(permutations(range(int(slots)), int(gt_count)))
    paths = torch.tensor(path_values, dtype=torch.long, device=device).reshape(
        -1,
        int(gt_count),
    )
    assigned = torch.zeros(
        (int(paths.shape[0]), int(slots)),
        dtype=torch.bool,
        device=device,
    )
    if int(gt_count) > 0:
        assigned.scatter_(1, paths, True)
    result = (paths, ~assigned)
    _SLOT_ASSIGNMENT_PATH_CACHE[key] = result
    return result


def _vectorized_slot_path_costs(
    candidate_cost: torch.Tensor,
    inactive_cost: torch.Tensor,
    paths: torch.Tensor,
    inactive_mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate every GT-to-slot permutation without scalar tensor loops."""

    batch, slots, gt_count = candidate_cost.shape
    path_count = int(paths.shape[0])
    if int(gt_count) > 0:
        by_gt = candidate_cost.permute(0, 2, 1).unsqueeze(1).expand(
            -1,
            path_count,
            -1,
            -1,
        )
        gather_index = paths.view(1, path_count, int(gt_count), 1).expand(
            batch,
            -1,
            -1,
            -1,
        )
        selected = by_gt.gather(-1, gather_index).squeeze(-1).sum(dim=-1)
    else:
        selected = candidate_cost.new_zeros((batch, path_count))
    inactive = (
        inactive_cost.unsqueeze(1)
        * inactive_mask.to(dtype=inactive_cost.dtype).unsqueeze(0)
    ).sum(dim=-1)
    return selected + inactive


@torch.no_grad()
def build_pointer_sequence_targets(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    max_selections: int,
    input_h: int,
    line_width: float,
    min_valid_rows: int,
) -> torch.Tensor:
    """Build score-independent left-to-right lane targets followed by STOP.

    Candidate ownership is obtained from the complete detached range-aware IoU
    matrix, not from the deployment score.  Hungarian assignment makes the
    representatives unique; ordering the matched GT lanes left-to-right makes
    the autoregressive contract deterministic under candidate permutations.
    ``num_candidates`` is the STOP class and positions after STOP are ignored.
    """

    pred_x = outputs["pred_x_rows"].detach().float()
    ranges = outputs["range_norm"].detach().float()
    batch, candidates, _rows = pred_x.shape
    steps = int(max_selections)
    if steps < 1:
        raise ValueError("pointer max_selections must be positive")
    sequence = torch.full(
        (batch, steps),
        -100,
        dtype=torch.long,
        device=pred_x.device,
    )
    for batch_index, target in enumerate(targets):
        gt_x = target["x_rows"].to(
            device=pred_x.device,
            dtype=pred_x.dtype,
        )
        gt_valid = target["valid_mask"].to(pred_x.device).bool()
        quality, candidate_valid, valid_gt = pairwise_range_aware_row_strip_iou(
            pred_x[batch_index],
            ranges[batch_index],
            gt_x,
            gt_valid,
            input_h=int(input_h),
            line_width=float(line_width),
            min_valid_rows=int(min_valid_rows),
        )
        gt_ids = torch.nonzero(valid_gt, as_tuple=False).flatten()
        if gt_ids.numel() == 0:
            sequence[batch_index, 0] = candidates
            continue
        candidate_ids = torch.nonzero(
            candidate_valid,
            as_tuple=False,
        ).flatten()
        if candidate_ids.numel() == 0:
            # The frozen candidate pool has no deployable curve for this
            # image.  Teaching an invalid ID would contradict the pointer's
            # inference mask and used to crash the set-teacher reroll.
            sequence[batch_index, 0] = candidates
            continue
        quality = quality[candidate_ids][:, gt_ids]
        if int(gt_ids.numel()) > steps:
            # CULane normally has at most four lanes.  If an annotation exceeds
            # deployment cardinality, retain the lanes the frozen pool can
            # represent best instead of introducing an arbitrary file-order cut.
            keep = quality.amax(dim=0).topk(k=steps).indices
            quality = quality[:, keep]
            gt_ids = gt_ids[keep]
        pred_ids, local_gt_ids = HungarianMatcherS0._linear_sum_assignment(
            1.0 - quality.detach().cpu()
        )
        pairs: list[tuple[float, int]] = []
        for pred_value, local_gt_value in zip(
            pred_ids.tolist(),
            local_gt_ids.tolist(),
        ):
            gt_index = int(gt_ids[int(local_gt_value)])
            visible_ids = torch.nonzero(
                gt_valid[gt_index] & torch.isfinite(gt_x[gt_index]),
                as_tuple=False,
            ).flatten()
            if visible_ids.numel() == 0:
                bottom_x = float(gt_x[gt_index].nan_to_num().median())
            else:
                tail = visible_ids[-min(5, int(visible_ids.numel())) :]
                bottom_x = float(gt_x[gt_index, tail].median())
            pairs.append((bottom_x, int(candidate_ids[int(pred_value)])))
        pairs.sort(key=lambda row: row[0])
        count = min(len(pairs), steps)
        if count > 0:
            sequence[batch_index, :count] = torch.tensor(
                [candidate for _x, candidate in pairs[:count]],
                device=sequence.device,
                dtype=torch.long,
            )
        if count < steps:
            sequence[batch_index, count] = candidates
    return sequence


def _pointer_teacher_seed(
    base_seed: int,
    iteration: int,
    visit: int,
    batch_index: int,
) -> int:
    """Return a stable per-visit seed without Python's randomized ``hash``."""

    mask = (1 << 63) - 1
    value = int(base_seed) & mask
    for item in (iteration, visit, batch_index):
        value ^= (
            int(item)
            + 0x9E3779B97F4A7C15
            + ((value << 6) & mask)
            + (value >> 2)
        ) & mask
        value &= mask
    return value


@torch.no_grad()
def build_pointer_cluster_soft_targets(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    max_selections: int,
    input_h: int,
    line_width: float,
    min_valid_rows: int,
    representable_min: float,
    support_quality_delta: float,
    temperature: float,
    base_seed: int,
    iteration: int,
    visit: int,
    target_mode: str = "sampled_cluster",
) -> dict[str, torch.Tensor]:
    """Build GT-cluster soft targets and a discrete teacher rollout.

    The detached candidate/GT quality matrix is retained until each pointer
    step.  A cardinality-first one-to-one assignment decides only which GTs
    can be represented jointly; it does *not* choose the supervised candidate
    identity.  Each retained GT then supplies a near-best soft representative
    distribution.  GT order and the discrete candidate fed to the GRU are
    resampled reproducibly on every training visit.

    ``sampled_cluster`` is the original V4.5 contract: only the next randomly
    ordered GT cluster supplies the step loss.  ``remaining_cluster_mixture``
    keeps the same sampled teacher prefix but averages the normalized support
    distributions of *all* remaining jointly representable GTs.  It therefore
    removes false-negative gradients between still-valid lane clusters without
    introducing a model/free rollout.
    """

    pred_x = outputs["pred_x_rows"].detach().float()
    ranges = outputs["range_norm"].detach().float()
    batch, candidates, _rows = pred_x.shape
    steps = int(max_selections)
    if steps < 1:
        raise ValueError("pointer max_selections must be positive")
    if not 0.0 <= float(representable_min) < 1.0:
        raise ValueError("pointer representable_min must be in [0, 1)")
    if float(support_quality_delta) < 0.0:
        raise ValueError("pointer support_quality_delta must be non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("pointer cluster temperature must be positive")
    target_mode = str(target_mode).strip().lower()
    if target_mode not in {"sampled_cluster", "remaining_cluster_mixture"}:
        raise ValueError(
            "pointer cluster target_mode must be sampled_cluster or "
            "remaining_cluster_mixture"
        )

    device = pred_x.device
    stop_class = candidates
    teacher_indices = torch.full(
        (batch, steps),
        -100,
        dtype=torch.long,
        device=device,
    )
    probabilities = pred_x.new_zeros((batch, steps, candidates + 1))
    active = torch.zeros((batch, steps), dtype=torch.bool, device=device)
    support_sizes = pred_x.new_zeros((batch, steps))
    target_entropy = pred_x.new_zeros((batch, steps))
    target_quality = pred_x.new_zeros((batch, steps))
    remaining_cluster_count = pred_x.new_zeros((batch, steps))
    candidate_steps = torch.zeros((batch, steps), dtype=torch.bool, device=device)
    representable_count = pred_x.new_zeros((batch,))
    fallback_count = pred_x.new_zeros((batch,))
    reservation_exclusion_count = pred_x.new_zeros((batch,))

    for batch_index, target in enumerate(targets):
        gt_x = target["x_rows"].to(device=device, dtype=pred_x.dtype)
        gt_valid = target["valid_mask"].to(device).bool()
        quality, candidate_valid, valid_gt = pairwise_range_aware_row_strip_iou(
            pred_x[batch_index],
            ranges[batch_index],
            gt_x,
            gt_valid,
            input_h=int(input_h),
            line_width=float(line_width),
            min_valid_rows=int(min_valid_rows),
        )
        valid_candidate_ids = torch.nonzero(
            candidate_valid,
            as_tuple=False,
        ).flatten()
        valid_gt_ids = torch.nonzero(valid_gt, as_tuple=False).flatten()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _pointer_teacher_seed(
                base_seed,
                iteration,
                visit,
                batch_index,
            )
        )

        if valid_candidate_ids.numel() == 0 or valid_gt_ids.numel() == 0:
            teacher_indices[batch_index, 0] = stop_class
            probabilities[batch_index, 0, stop_class] = 1.0
            active[batch_index, 0] = True
            continue

        local_quality = quality[valid_candidate_ids][:, valid_gt_ids]
        qualified = local_quality > float(representable_min)
        assignment_size = min(
            int(local_quality.shape[0]),
            int(local_quality.shape[1]),
        )
        reward = (
            qualified.to(local_quality.dtype) * float(assignment_size + 1)
            + local_quality
        )
        local_candidate_assignment, local_gt_assignment = (
            HungarianMatcherS0._linear_sum_assignment(-reward.detach().cpu())
        )
        joint_pairs: list[tuple[int, int, float]] = []
        for local_candidate, local_gt in zip(
            local_candidate_assignment.tolist(),
            local_gt_assignment.tolist(),
        ):
            if not bool(qualified[local_candidate, local_gt]):
                continue
            joint_pairs.append(
                (
                    int(valid_gt_ids[local_gt]),
                    int(valid_candidate_ids[local_candidate]),
                    float(local_quality[local_candidate, local_gt]),
                )
            )
        if len(joint_pairs) > steps:
            joint_pairs.sort(key=lambda item: item[2], reverse=True)
            joint_pairs = joint_pairs[:steps]

        if not joint_pairs:
            teacher_indices[batch_index, 0] = stop_class
            probabilities[batch_index, 0, stop_class] = 1.0
            active[batch_index, 0] = True
            continue

        permutation = torch.randperm(len(joint_pairs), generator=generator).tolist()
        ordered_pairs = [joint_pairs[index] for index in permutation]
        representable_count[batch_index] = float(len(ordered_pairs))
        available = candidate_valid.detach().cpu().bool().clone()
        quality_cpu = quality.detach().cpu().float()

        for step, (gt_index, fallback_candidate, _assigned_quality) in enumerate(
            ordered_pairs
        ):
            remaining_pairs = ordered_pairs[step:]
            remaining_cluster_count[batch_index, step] = float(
                len(remaining_pairs)
            )

            if target_mode == "sampled_cluster":
                # Preserve the calibrated V4.5 target exactly.
                q = quality_cpu[:, gt_index]
                valid_q = q[available]
                if valid_q.numel() == 0:
                    raise RuntimeError(
                        "V4.5 teacher exhausted all valid candidates"
                    )
                q_best = float(valid_q.max())
                cutoff = max(
                    float(representable_min),
                    q_best - float(support_quality_delta),
                )
                support = (
                    available
                    & (q > float(representable_min))
                    & (q >= cutoff)
                )
                future_fallbacks = [
                    int(pair[1]) for pair in ordered_pairs[step + 1 :]
                ]
                if future_fallbacks:
                    future_ids = torch.tensor(
                        future_fallbacks,
                        dtype=torch.long,
                    )
                    reservation_exclusion_count[batch_index] += float(
                        support[future_ids].sum()
                    )
                    support[future_ids] = False
                if not bool(support.any()):
                    if not bool(available[fallback_candidate]):
                        raise RuntimeError(
                            "V4.5 collision guard lost a reserved representative"
                        )
                    support[fallback_candidate] = True
                    fallback_count[batch_index] += 1.0
                support_ids = torch.nonzero(
                    support,
                    as_tuple=False,
                ).flatten()
                distribution = torch.softmax(
                    q[support_ids] / float(temperature),
                    dim=0,
                )
                probabilities[
                    batch_index,
                    step,
                    support_ids.to(device=device),
                ] = distribution.to(
                    device=device,
                    dtype=probabilities.dtype,
                )
                target_quality_value = float(
                    (distribution * q[support_ids]).sum()
                )
            else:
                # V4.6: every remaining GT contributes equal probability mass.
                # Per-GT normalization prevents duplicate-rich clusters from
                # dominating the target.  Other GTs' unique Hungarian fallback
                # candidates are reserved so every target action retains a
                # feasible continuation path.
                cluster_targets: list[
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ] = []
                remaining_fallbacks = [int(pair[1]) for pair in remaining_pairs]
                for (
                    remaining_gt,
                    remaining_fallback,
                    _remaining_quality,
                ) in remaining_pairs:
                    q_remaining = quality_cpu[:, remaining_gt]
                    eligible = available.clone()
                    other_fallbacks = [
                        value
                        for value in remaining_fallbacks
                        if value != int(remaining_fallback)
                    ]
                    if other_fallbacks:
                        other_ids = torch.tensor(
                            other_fallbacks,
                            dtype=torch.long,
                        )
                        eligible[other_ids] = False
                    valid_q = q_remaining[eligible]
                    if valid_q.numel() == 0:
                        raise RuntimeError(
                            "V4.6 teacher exhausted all eligible candidates"
                        )
                    q_best = float(valid_q.max())
                    cutoff = max(
                        float(representable_min),
                        q_best - float(support_quality_delta),
                    )
                    raw_support = (
                        available
                        & (q_remaining > float(representable_min))
                        & (q_remaining >= cutoff)
                    )
                    reservation_exclusion_count[batch_index] += float(
                        (raw_support & ~eligible).sum()
                    )
                    support = raw_support & eligible
                    if not bool(support.any()):
                        if not bool(available[remaining_fallback]):
                            raise RuntimeError(
                                "V4.6 collision guard lost a reserved "
                                "representative"
                            )
                        support[remaining_fallback] = True
                        fallback_count[batch_index] += 1.0
                    remaining_support_ids = torch.nonzero(
                        support,
                        as_tuple=False,
                    ).flatten()
                    remaining_distribution = torch.softmax(
                        q_remaining[remaining_support_ids]
                        / float(temperature),
                        dim=0,
                    )
                    cluster_targets.append(
                        (
                            q_remaining,
                            remaining_support_ids,
                            remaining_distribution,
                        )
                    )

                q, support_ids, distribution = cluster_targets[0]
                target_row = probabilities[batch_index, step, :candidates]
                cluster_weight = 1.0 / float(len(cluster_targets))
                expected_quality = 0.0
                for (
                    cluster_quality,
                    cluster_support_ids,
                    cluster_distribution,
                ) in cluster_targets:
                    target_row[cluster_support_ids.to(device=device)] += (
                        cluster_distribution.to(
                            device=device,
                            dtype=target_row.dtype,
                        )
                        * cluster_weight
                    )
                    expected_quality += cluster_weight * float(
                        (
                            cluster_distribution
                            * cluster_quality[cluster_support_ids]
                        ).sum()
                    )
                target_row /= target_row.sum().clamp_min(1e-12)
                target_quality_value = expected_quality

            sampled_local = int(
                torch.multinomial(
                    distribution,
                    num_samples=1,
                    replacement=False,
                    generator=generator,
                )
            )
            sampled_candidate = int(support_ids[sampled_local])

            teacher_indices[batch_index, step] = sampled_candidate
            active[batch_index, step] = True
            candidate_steps[batch_index, step] = True
            target_row = probabilities[batch_index, step, :candidates]
            support_sizes[batch_index, step] = float((target_row > 0.0).sum())
            target_entropy[batch_index, step] = float(
                -(target_row * target_row.clamp_min(1e-12).log()).sum()
            )
            target_quality[batch_index, step] = target_quality_value
            available[sampled_candidate] = False

        count = len(ordered_pairs)
        if count < steps:
            teacher_indices[batch_index, count] = stop_class
            probabilities[batch_index, count, stop_class] = 1.0
            active[batch_index, count] = True

    return {
        "indices": teacher_indices,
        "probabilities": probabilities,
        "active": active,
        "candidate_steps": candidate_steps,
        "support_sizes": support_sizes,
        "target_entropy": target_entropy,
        "target_quality": target_quality,
        "remaining_cluster_count": remaining_cluster_count,
        "representable_count": representable_count,
        "fallback_count": fallback_count,
        "reservation_exclusion_count": reservation_exclusion_count,
    }


@torch.no_grad()
def build_four_slot_cluster_targets(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    num_slots: int,
    input_h: int,
    line_width: float,
    min_valid_rows: int,
    representable_min: float,
    cluster_min: float,
    cluster_delta: float,
    temperature: float,
    target_mode: str = "joint_threshold",
    padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, object]:
    """Keep the GT/object axis for four-slot proposal routing.

    Hungarian is used only to determine which GT lanes are jointly
    representable by the frozen proposal pool.  It does not choose a hard
    representative.  Every retained GT supplies a near-best soft row over
    all candidates, plus a zero-mass dustbin class.  Slot identity is left
    unspecified and is marginalized by :func:`four_slot_permutation_loss`.
    """

    slots = int(num_slots)
    if slots < 1:
        raise ValueError("four-slot num_slots must be positive")
    if not 0.0 <= float(representable_min) <= 1.0:
        raise ValueError("four-slot representable_min must be in [0, 1]")
    if not 0.0 <= float(cluster_min) <= 1.0:
        raise ValueError("four-slot cluster_min must be in [0, 1]")
    if float(cluster_delta) < 0.0:
        raise ValueError("four-slot cluster_delta must be non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("four-slot cluster temperature must be positive")
    mode = str(target_mode).strip().lower()
    if mode not in {"joint_threshold", "all_gt"}:
        raise ValueError(
            "four-slot target_mode must be joint_threshold or all_gt"
        )

    pred_x = outputs["pred_x_rows"].detach().float()
    pred_range = outputs["range_norm"].detach().float()
    batch, candidates, _rows = pred_x.shape
    rows_by_image: list[torch.Tensor] = []
    support_sizes: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    target_qualities: list[torch.Tensor] = []
    representable_counts = pred_x.new_zeros((batch,))

    if padded_targets is None:
        padded_gt_x, padded_gt_valid = _padded_lane_targets(
            targets,
            device=pred_x.device,
            dtype=pred_x.dtype,
            rows=int(pred_x.shape[-1]),
        )
    else:
        padded_gt_x, padded_gt_valid = padded_targets
    quality_batch, candidate_valid_batch, valid_gt_batch = (
        batched_pairwise_range_aware_row_strip_iou(
            pred_x,
            pred_range,
            padded_gt_x,
            padded_gt_valid,
            input_h=int(input_h),
            line_width=float(line_width),
            min_valid_rows=int(min_valid_rows),
        )
    )
    if mode == "all_gt" and max(
        (int(target["x_rows"].shape[0]) for target in targets),
        default=0,
    ) <= slots:
        # CULane's final object set has at most four lanes.  In all-GT mode
        # Hungarian and proposal-quality thresholding therefore cannot alter
        # which GT rows are supervised.  Form every near-best cluster in one
        # dense tensor instead of launching a softmax/indexing graph per GT.
        image_has_candidate = candidate_valid_batch.any(dim=-1)
        active_gt = valid_gt_batch & image_has_candidate.unsqueeze(-1)
        masked_quality = quality_batch.masked_fill(
            ~candidate_valid_batch.unsqueeze(-1),
            float("-inf"),
        )
        best = masked_quality.amax(dim=1)
        effective_floor = best.clamp(max=float(cluster_min))
        cutoff = torch.maximum(
            effective_floor,
            best - float(cluster_delta),
        )
        support = (
            candidate_valid_batch.unsqueeze(-1)
            & active_gt.unsqueeze(1)
            & (quality_batch >= cutoff.unsqueeze(1))
        )
        probability = torch.softmax(
            (quality_batch / float(temperature)).masked_fill(
                ~support,
                float("-inf"),
            ),
            dim=1,
        )
        probability = torch.where(
            support,
            probability,
            torch.zeros_like(probability),
        )
        dense_rows = torch.cat(
            (
                probability.permute(0, 2, 1),
                probability.new_zeros((batch, int(probability.shape[2]), 1)),
            ),
            dim=-1,
        )
        rows_by_image = [
            dense_rows[batch_index, active_gt[batch_index]]
            for batch_index in range(batch)
        ]
        active_float = active_gt.to(dtype=pred_x.dtype)
        normalizer = active_float.sum().clamp_min(1.0)
        support_count = support.sum(dim=1).to(dtype=pred_x.dtype)
        entropy = -(
            probability
            * probability.clamp_min(1.0e-12).log()
        ).sum(dim=1)
        expected_quality = (
            probability * quality_batch
        ).sum(dim=1)
        return {
            "rows": rows_by_image,
            "representable_count": active_float.sum(dim=-1),
            "mean_support_size": (
                support_count * active_float
            ).sum() / normalizer,
            "mean_entropy": (entropy * active_float).sum() / normalizer,
            "mean_target_quality": (
                expected_quality * active_float
            ).sum() / normalizer,
        }
    valid_metadata_cpu = (
        torch.cat((candidate_valid_batch, valid_gt_batch), dim=1).cpu()
        if mode == "all_gt"
        else None
    )

    for batch_index, _target in enumerate(targets):
        quality = quality_batch[batch_index]
        candidate_valid = candidate_valid_batch[batch_index]
        valid_gt = valid_gt_batch[batch_index]
        if valid_metadata_cpu is not None:
            metadata = valid_metadata_cpu[batch_index]
            candidate_id_values = torch.nonzero(
                metadata[:candidates],
                as_tuple=False,
            ).flatten().tolist()
            gt_id_values = torch.nonzero(
                metadata[candidates:],
                as_tuple=False,
            ).flatten().tolist()
            candidate_ids = None
            gt_ids = None
        else:
            candidate_ids = torch.nonzero(
                candidate_valid,
                as_tuple=False,
            ).flatten()
            gt_ids = torch.nonzero(valid_gt, as_tuple=False).flatten()
            candidate_id_values = candidate_ids.detach().cpu().tolist()
            gt_id_values = gt_ids.detach().cpu().tolist()
        empty = pred_x.new_zeros((0, candidates + 1))
        if not candidate_id_values or not gt_id_values:
            rows_by_image.append(empty)
            continue

        # In all-GT mode the assignment quality has no effect unless the
        # annotation contains more lanes than output slots.  CULane has at
        # most four, so skip an otherwise unconditional GPU->CPU Hungarian
        # round trip while producing exactly the same target rows and order.
        assignment_quality: dict[int, float] = {}
        needs_joint_assignment = mode != "all_gt" or len(gt_id_values) > slots
        if needs_joint_assignment:
            if candidate_ids is None or gt_ids is None:
                candidate_ids = torch.tensor(
                    candidate_id_values,
                    dtype=torch.long,
                    device=pred_x.device,
                )
                gt_ids = torch.tensor(
                    gt_id_values,
                    dtype=torch.long,
                    device=pred_x.device,
                )
            local_quality = quality[candidate_ids][:, gt_ids]
            cost_cpu = (1.0 - local_quality).detach().cpu()
            local_candidate_ids, local_gt_ids = (
                HungarianMatcherS0._linear_sum_assignment(cost_cpu)
            )
            for local_candidate, local_gt in zip(
                local_candidate_ids.tolist(),
                local_gt_ids.tolist(),
            ):
                assigned_quality = 1.0 - float(
                    cost_cpu[local_candidate, local_gt]
                )
                assignment_quality[int(gt_id_values[int(local_gt)])] = assigned_quality
        if mode == "all_gt":
            # Final slot cardinality belongs to the annotation, not to the
            # current proposal quality.  This avoids a cold-start feedback
            # loop where weak early proposals teach every slot to be dustbin.
            if needs_joint_assignment:
                jointly_representable = []
                for gt_index in gt_id_values:
                    gt_index = int(gt_index)
                    assigned_quality = assignment_quality.get(gt_index)
                    if assigned_quality is None:
                        assigned_quality = float(
                            quality[candidate_valid, gt_index].amax()
                        )
                    jointly_representable.append(
                        (gt_index, assigned_quality)
                    )
            else:
                jointly_representable = [
                    (int(gt_index), 0.0) for gt_index in gt_id_values
                ]
        else:
            jointly_representable = [
                (gt_index, assigned_quality)
                for gt_index, assigned_quality in assignment_quality.items()
                if assigned_quality >= float(representable_min)
            ]
        if len(jointly_representable) > slots:
            jointly_representable.sort(key=lambda item: item[1], reverse=True)
            jointly_representable = jointly_representable[:slots]

        image_rows: list[torch.Tensor] = []
        for gt_index, _assigned_quality in jointly_representable:
            gt_quality = quality[:, gt_index]
            best = gt_quality[candidate_valid].amax()
            if mode == "all_gt":
                # The floor may sharpen a mature proposal pool but may never
                # delete the best early-training proposal from the target.
                effective_floor = best.clamp(max=float(cluster_min))
                cutoff = torch.maximum(
                    effective_floor,
                    best - float(cluster_delta),
                )
            else:
                cutoff = torch.maximum(
                    best.new_tensor(float(cluster_min)),
                    best - float(cluster_delta),
                )
            support = candidate_valid & (gt_quality >= cutoff)
            if mode != "all_gt" and not bool(support.any()):
                continue
            probability = torch.softmax(
                gt_quality[support] / float(temperature),
                dim=0,
            )
            row = pred_x.new_zeros((candidates + 1,))
            row[:candidates][support] = probability
            image_rows.append(row)
            support_sizes.append(
                support.sum().to(dtype=pred_x.dtype)
            )
            entropies.append(
                -(probability * probability.clamp_min(1.0e-12).log()).sum()
            )
            target_qualities.append((probability * gt_quality[support]).sum())
        if image_rows:
            stacked = torch.stack(image_rows)
        else:
            stacked = empty
        rows_by_image.append(stacked)
        representable_counts[batch_index] = float(stacked.shape[0])

    zero = pred_x.sum() * 0.0

    def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    return {
        "rows": rows_by_image,
        "representable_count": representable_counts,
        "mean_support_size": mean_or_zero(support_sizes),
        "mean_entropy": mean_or_zero(entropies),
        "mean_target_quality": mean_or_zero(target_qualities),
    }


def four_slot_permutation_loss(
    route_logits: torch.Tensor,
    target_rows: list[torch.Tensor],
    *,
    permutation_temperature: float = 1.0,
    assignment_mode: str = "marginal",
) -> torch.Tensor:
    """Reduce every valid GT-to-slot permutation (at most ``4!``).

    ``marginal`` preserves the historical soft log-sum-exp objective.  The
    ``hard_min`` mode selects one detached minimum-cost permutation and only
    backpropagates through that path, matching DETR-style discrete ownership
    while retaining the soft candidate distribution inside each GT row.
    """

    if route_logits.ndim != 3:
        raise ValueError("four-slot logits must have shape [B,S,N+1]")
    batch, slots, classes = route_logits.shape
    if len(target_rows) != int(batch):
        raise ValueError("four-slot target batch size mismatch")
    temperature = float(permutation_temperature)
    if temperature <= 0.0:
        raise ValueError("four-slot permutation temperature must be positive")
    mode = str(assignment_mode).strip().lower()
    if mode not in {"marginal", "hard_min"}:
        raise ValueError(
            "four-slot assignment_mode must be marginal or hard_min"
        )
    dustbin_index = int(classes) - 1
    log_probability = F.log_softmax(route_logits.float(), dim=-1)
    losses: list[torch.Tensor] = []
    for batch_index, target_value in enumerate(target_rows):
        target = target_value.to(
            device=route_logits.device,
            dtype=log_probability.dtype,
        )
        if target.ndim != 2 or int(target.shape[1]) != int(classes):
            raise ValueError("four-slot target rows must have shape [G,N+1]")
        gt_count = int(target.shape[0])
        if gt_count > int(slots):
            raise ValueError("four-slot target contains more GTs than slots")
        dustbin_cost = -log_probability[batch_index, :, dustbin_index]
        if gt_count == 0:
            losses.append(dustbin_cost.sum())
            continue
        candidate_cost = -torch.einsum(
            "sc,gc->sg",
            log_probability[batch_index],
            target,
        )
        path_costs: list[torch.Tensor] = []
        for assigned_slots in permutations(range(int(slots)), gt_count):
            assigned = set(int(value) for value in assigned_slots)
            cost = candidate_cost.new_zeros(())
            for gt_index, slot_index in enumerate(assigned_slots):
                cost = cost + candidate_cost[int(slot_index), gt_index]
            for slot_index in range(int(slots)):
                if slot_index not in assigned:
                    cost = cost + dustbin_cost[slot_index]
            path_costs.append(cost)
        stacked = torch.stack(path_costs)
        if mode == "hard_min":
            best_path = stacked.detach().argmin()
            losses.append(stacked[best_path])
        else:
            losses.append(
                -temperature * torch.logsumexp(
                    -stacked / temperature,
                    dim=0,
                )
                + temperature * math.log(float(len(path_costs)))
            )
    return torch.stack(losses).mean() / float(max(int(slots), 1))


def four_slot_factorized_permutation_loss(
    active_logits: torch.Tensor,
    real_route_logits: torch.Tensor,
    target_rows: list[torch.Tensor],
    *,
    permutation_temperature: float = 1.0,
    assignment_mode: str = "marginal",
) -> torch.Tensor:
    """Permutation-marginal loss with separate cardinality and real route.

    An unmatched slot receives only a no-lane BCE target.  A matched slot
    receives an active BCE target plus cross entropy over real proposals.
    Geometry never consumes ``active_logits``; this factorization makes that
    separation explicit instead of hiding dustbin inside route normalization.
    """

    if active_logits.ndim != 2 or real_route_logits.ndim != 3:
        raise ValueError("factorized slot logits must be [B,S] and [B,S,N]")
    batch, slots, candidates = real_route_logits.shape
    if tuple(active_logits.shape) != (batch, slots):
        raise ValueError("factorized active logit shape mismatch")
    if len(target_rows) != int(batch):
        raise ValueError("factorized target batch size mismatch")
    temperature = float(permutation_temperature)
    if temperature <= 0.0:
        raise ValueError("four-slot permutation temperature must be positive")
    mode = str(assignment_mode).strip().lower()
    if mode not in {"marginal", "hard_min"}:
        raise ValueError(
            "four-slot assignment_mode must be marginal or hard_min"
        )

    real_log_probability = F.log_softmax(
        real_route_logits.float(),
        dim=-1,
    )
    active_cost = F.softplus(-active_logits.float())
    inactive_cost = F.softplus(active_logits.float())
    grouped: dict[int, list[int]] = {}
    for batch_index, target in enumerate(target_rows):
        if target.ndim != 2 or int(target.shape[1]) != int(candidates) + 1:
            raise ValueError("factorized target rows must have shape [G,N+1]")
        gt_count = int(target.shape[0])
        if gt_count > int(slots):
            raise ValueError("factorized target contains more GTs than slots")
        grouped.setdefault(gt_count, []).append(batch_index)

    losses: list[torch.Tensor | None] = [None] * int(batch)
    for gt_count, batch_indices in grouped.items():
        batch_ids = torch.tensor(
            batch_indices,
            dtype=torch.long,
            device=real_route_logits.device,
        )
        if gt_count == 0:
            group_losses = inactive_cost.index_select(0, batch_ids).sum(dim=-1)
        else:
            target = torch.stack(
                [target_rows[index] for index in batch_indices]
            ).to(
                device=real_route_logits.device,
                dtype=real_log_probability.dtype,
            )
            candidate_cost = -torch.einsum(
                "bsn,bgn->bsg",
                real_log_probability.index_select(0, batch_ids),
                target[..., :candidates],
            )
            group_active = active_cost.index_select(0, batch_ids)
            candidate_cost = candidate_cost + group_active.unsqueeze(-1)
            paths, inactive_mask = _slot_assignment_paths(
                int(slots),
                int(gt_count),
                real_route_logits.device,
            )
            stacked = _vectorized_slot_path_costs(
                candidate_cost,
                inactive_cost.index_select(0, batch_ids),
                paths,
                inactive_mask,
            )
            if mode == "hard_min":
                # The discrete path is deliberately outside autograd.
                best_path = stacked.detach().argmin(dim=-1, keepdim=True)
                group_losses = stacked.gather(-1, best_path).squeeze(-1)
            else:
                group_losses = (
                    -temperature
                    * torch.logsumexp(-stacked / temperature, dim=-1)
                    + temperature * math.log(float(int(paths.shape[0])))
                )
        for local_index, batch_index in enumerate(batch_indices):
            losses[batch_index] = group_losses[local_index]
    if any(value is None for value in losses):
        raise RuntimeError("missing factorized four-slot batch loss")
    return torch.stack([value for value in losses if value is not None]).mean() / float(
        max(int(slots), 1)
    )


def four_slot_collision_loss(
    route_logits: torch.Tensor,
    *,
    has_dustbin: bool = True,
) -> torch.Tensor:
    """Penalize different slots assigning probability to one proposal."""

    if route_logits.ndim != 3 or int(route_logits.shape[-1]) < 2:
        raise ValueError("four-slot logits must have shape [B,S,N+1]")
    proposal_logits = (
        route_logits[..., :-1] if bool(has_dustbin) else route_logits
    )
    proposal_probability = torch.softmax(proposal_logits.float(), dim=-1)
    gram = torch.einsum(
        "bsn,btn->bst",
        proposal_probability,
        proposal_probability,
    )
    slots = int(proposal_probability.shape[1])
    if slots <= 1:
        return gram.sum() * 0.0
    pair_count = int(slots * (slots - 1) // 2)
    return torch.triu(gram, diagonal=1).sum() / float(
        int(gram.shape[0]) * pair_count
    )


@dataclass
class LossConfig:
    w_exist: float = 2.0
    w_point: float = 5.0
    w_range: float = 1.0
    w_smooth: float = 0.0
    smooth_l1_beta: float = 0.01
    input_w: int = 800
    input_h: int = 288
    no_lane_weight: float = 1.0
    exist_loss_type: str = "ce"
    exist_target_mode: str = "binary"
    exist_quality_floor: float = 0.5
    exist_quality_beta: float = 2.0
    exist_quality_line_width: float = 30.0
    exist_quality_min_valid_rows: int = 5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    smoothness_contiguous: bool = True
    w_line_iou: float = 0.0
    line_iou_radius: float = 15.0
    w_seg: float = 0.0
    seg_pos_weight: float = 1.0
    seg_extra_weights: dict[str, float] = field(default_factory=dict)
    w_quality: float = 0.0
    w_cardinality: float = 0.0
    w_score_margin: float = 0.0
    score_margin: float = 0.5
    score_margin_topk_negatives: int = 8
    w_set_selection: float = 0.0
    set_selection_line_width: float = 30.0
    set_selection_focal_beta: float = 2.0
    set_selection_rank_weight: float = 0.25
    set_selection_target_margin: float = 0.10
    set_selection_min_valid_rows: int = 5
    set_selection_share_matcher_assignment: bool = False
    set_selection_negative_weight: float = 1.0
    set_selection_positive_floor: float = 0.0
    set_selection_coverage_weight: float = 0.0
    set_selection_duplicate_weight: float = 0.0
    set_selection_winner_weight: float = 0.0
    set_selection_count_weight: float = 0.0
    set_selection_duplicate_quality_min: float = 0.30
    set_selection_winner_quality_min: float = 0.30
    w_pointer_selection: float = 0.0
    pointer_quality_weight: float = 0.5
    pointer_cluster_listwise_weight: float = 0.0
    pointer_cluster_listwise_logit_temperature: float = 1.0
    pointer_stop_weight: float = 1.0
    pointer_unary_target_mode: str = "max_quality"
    w_four_slot_selection: float = 0.0
    four_slot_line_width: float = 30.0
    four_slot_min_valid_rows: int = 5
    four_slot_representable_min: float = 0.50
    four_slot_cluster_min: float = 0.30
    four_slot_cluster_delta: float = 0.05
    four_slot_cluster_temperature: float = 0.03
    four_slot_target_mode: str = "joint_threshold"
    four_slot_permutation_temperature: float = 1.0
    four_slot_assignment_mode: str = "marginal"
    four_slot_collision_weight: float = 0.10
    w_four_slot_geometry: float = 0.0
    four_slot_geometry_point_weight: float = 5.0
    four_slot_geometry_line_iou_weight: float = 2.0
    four_slot_geometry_dfl_weight: float = 1.0
    four_slot_geometry_range_weight: float = 0.0
    four_slot_geometry_match_min_quality: float = 0.20
    four_slot_geometry_match_all_slots: bool = False
    # V11 uses one final-slot Hungarian assignment for activity, global
    # proposal-memory attention and final slot-owned geometry.  This removes
    # the independent selection/geometry assignment contracts used by V7.
    w_four_slot_unified: float = 0.0
    four_slot_unified_active_weight: float = 1.0
    four_slot_unified_attention_weight: float = 1.0
    four_slot_unified_point_weight: float = 5.0
    four_slot_unified_range_weight: float = 1.0
    four_slot_unified_line_iou_weight: float = 2.0
    four_slot_unified_dfl_weight: float = 1.0
    four_slot_unified_aux_geometry_weight: float = 0.25
    # V12 Stage A trains image-first row localization and all-proposal
    # association under one source-stable V7 geometry assignment.  Deployment
    # geometry/activity remain exact V7 and receive no gradient.
    w_four_slot_visual_first: float = 0.0
    four_slot_visual_first_first_pass_weight: float = 0.5
    four_slot_visual_first_final_pass_weight: float = 1.0
    four_slot_visual_first_proposal_weight: float = 1.0
    # V13 bypasses proposal identity entirely.  A source-stable assignment
    # supervises full-width slot-owned x/range emitted from visual-first,
    # row-wise proposal context and precise local P2 evidence.
    w_four_slot_visual_precision: float = 0.0
    four_slot_visual_precision_point_weight: float = 5.0
    four_slot_visual_precision_range_weight: float = 1.0
    four_slot_visual_precision_line_iou_weight: float = 2.0
    four_slot_visual_precision_dfl_weight: float = 1.0
    # V14 Stage A repeats the image-first identifiability test with a fixed
    # source-active/writer-valid assignment and a jointly feasible
    # proposal/private-dustbin target transport.
    w_four_slot_v14_stage_a: float = 0.0
    four_slot_v14_visual_weight: float = 1.0
    four_slot_v14_association_weight: float = 1.0
    four_slot_v14_representable_min: float = 0.50
    four_slot_v14_cluster_delta: float = 0.10
    four_slot_v14_cluster_temperature: float = 0.03
    # V14 Stage B freezes Stage A and learns only a V7-parity-anchored
    # residual geometry consumer under the same fixed source assignment.
    w_four_slot_v14_stage_b: float = 0.0
    four_slot_v14_stage_b_point_weight: float = 5.0
    four_slot_v14_stage_b_range_weight: float = 1.0
    four_slot_v14_stage_b_line_iou_weight: float = 2.0
    four_slot_v14_stage_b_dfl_weight: float = 1.0
    # V15 removes proposal-ID supervision and hard cluster prototypes.  Direct
    # visual localization plus fixed-assignment final geometry train a
    # bottom-aware soft proposal graph used only as slot-row context.
    w_four_slot_v15: float = 0.0
    four_slot_v15_visual_weight: float = 1.0
    four_slot_v15_point_weight: float = 5.0
    four_slot_v15_range_weight: float = 1.0
    four_slot_v15_line_iou_weight: float = 2.0
    four_slot_v15_dfl_weight: float = 1.0
    # V16 directly ranks coherent proposal members inside GT-free,
    # anchor-owned variable groups.  No coordinate mixture is supervised.
    w_four_slot_v16: float = 0.0
    four_slot_v16_quality_weight: float = 1.0
    four_slot_v16_pairwise_weight: float = 1.0
    four_slot_v16_hard_weight: float = 1.0
    four_slot_v16_representable_min: float = 0.50
    four_slot_v16_pair_margin: float = 0.02
    four_slot_v16_hard_margin: float = 0.02
    # V17 directly supervises three bounded, re-centered continuous geometry
    # stages. Proposal IDs and coordinates never own the final curve.
    w_four_slot_v17: float = 0.0
    four_slot_v17_stage_weights: tuple[float, ...] = (0.25, 0.50, 1.0)
    four_slot_v17_visual_weight: float = 1.0
    four_slot_v17_point_weight: float = 5.0
    four_slot_v17_range_weight: float = 1.0
    four_slot_v17_line_iou_weight: float = 2.0
    four_slot_v17_dfl_weight: float = 1.0
    w_centerline: float = 0.0
    w_row_dfl: float = 0.0
    row_dfl_warmup_iters: int = 0
    centerline_sigma_bins: float = 1.5
    centerline_pos_weight: float = 1.0
    w_dynamic_proposal_heatmap: float = 0.0
    w_dynamic_proposal_x: float = 0.0
    w_dynamic_proposal_range: float = 0.0
    dynamic_proposal_sigma_bins: float = 1.5
    dynamic_proposal_seed_radius_bins: int = 2
    dynamic_proposal_heatmap_pos_weight: float = 1.0
    lambda_coarse: float = 0.0
    lambda_geometry_draft: float = 0.0
    lambda_intermediate: float = 0.0
    # ``None`` preserves the historical contract: intermediate decoder
    # layers use the same foreground weight as the final layer.  A separate
    # value lets a controlled experiment keep geometry deep supervision while
    # preventing independently matched auxiliary layers from teaching the
    # shared deployment score contradictory candidate identities.
    w_intermediate_exist: float | None = None
    intermediate_layer_weights: tuple[float, ...] = ()
    lambda_training_auxiliary: float = 0.0
    geometry_reduction: str = "global_rows"


class S0Criterion(nn.Module):
    def __init__(self, cfg: LossConfig | None = None, matcher: HungarianMatcherS0 | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.cfg.pointer_unary_target_mode = str(
            self.cfg.pointer_unary_target_mode
        ).strip().lower()
        self.matcher = matcher
        self._iteration = 0
        reduction = str(self.cfg.geometry_reduction).strip().lower()
        if reduction not in {
            "global",
            "global_rows",
            "row",
            "rows",
            "lane",
            "lane_mean",
            "per_lane",
        }:
            raise ValueError(
                f"Unsupported loss.geometry_reduction: {self.cfg.geometry_reduction!r}"
            )
        if not 0.0 <= float(self.cfg.set_selection_positive_floor) < 1.0:
            raise ValueError(
                "set_selection_positive_floor must be in [0, 1)"
            )
        for field_name in (
            "set_selection_coverage_weight",
            "set_selection_duplicate_weight",
            "set_selection_winner_weight",
            "set_selection_count_weight",
            "w_pointer_selection",
            "pointer_quality_weight",
            "pointer_cluster_listwise_weight",
            "w_four_slot_selection",
            "four_slot_collision_weight",
            "w_four_slot_geometry",
            "four_slot_geometry_point_weight",
            "four_slot_geometry_line_iou_weight",
            "four_slot_geometry_dfl_weight",
            "four_slot_geometry_range_weight",
        ):
            if float(getattr(self.cfg, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if float(self.cfg.pointer_stop_weight) <= 0.0:
            raise ValueError("pointer_stop_weight must be positive")
        if float(self.cfg.pointer_cluster_listwise_logit_temperature) <= 0.0:
            raise ValueError(
                "pointer_cluster_listwise_logit_temperature must be positive"
            )
        if float(self.cfg.four_slot_line_width) <= 0.0:
            raise ValueError("four_slot_line_width must be positive")
        if int(self.cfg.four_slot_min_valid_rows) < 1:
            raise ValueError("four_slot_min_valid_rows must be positive")
        for field_name in (
            "four_slot_representable_min",
            "four_slot_cluster_min",
        ):
            value = float(getattr(self.cfg, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if float(self.cfg.four_slot_cluster_delta) < 0.0:
            raise ValueError("four_slot_cluster_delta must be non-negative")
        if float(self.cfg.four_slot_cluster_temperature) <= 0.0:
            raise ValueError("four_slot_cluster_temperature must be positive")
        if float(self.cfg.four_slot_permutation_temperature) <= 0.0:
            raise ValueError(
                "four_slot_permutation_temperature must be positive"
            )
        self.cfg.four_slot_assignment_mode = str(
            self.cfg.four_slot_assignment_mode
        ).strip().lower()
        if self.cfg.four_slot_assignment_mode not in {
            "marginal",
            "hard_min",
        }:
            raise ValueError(
                "four_slot_assignment_mode must be marginal or hard_min"
            )
        self.cfg.four_slot_target_mode = str(
            self.cfg.four_slot_target_mode
        ).strip().lower()
        if self.cfg.four_slot_target_mode not in {
            "joint_threshold",
            "all_gt",
        }:
            raise ValueError(
                "four_slot_target_mode must be joint_threshold or all_gt"
            )
        if not 0.0 <= float(
            self.cfg.four_slot_geometry_match_min_quality
        ) <= 1.0:
            raise ValueError(
                "four_slot_geometry_match_min_quality must be in [0, 1]"
            )
        if self.cfg.pointer_unary_target_mode not in {
            "max_quality",
            "unique_representative",
        }:
            raise ValueError(
                "pointer_unary_target_mode must be max_quality or "
                "unique_representative"
            )
        for field_name in (
            "set_selection_duplicate_quality_min",
            "set_selection_winner_quality_min",
        ):
            value = float(getattr(self.cfg, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        exist_target_mode = str(self.cfg.exist_target_mode).strip().lower()
        if exist_target_mode not in {"binary", "iou_aware"}:
            raise ValueError("exist_target_mode must be binary or iou_aware")
        if not 0.0 <= float(self.cfg.exist_quality_floor) < 1.0:
            raise ValueError("exist_quality_floor must be in [0, 1)")
        if float(self.cfg.exist_quality_beta) < 0.0:
            raise ValueError("exist_quality_beta must be non-negative")
        if float(self.cfg.exist_quality_line_width) <= 0.0:
            raise ValueError("exist_quality_line_width must be positive")
        if int(self.cfg.exist_quality_min_valid_rows) < 1:
            raise ValueError("exist_quality_min_valid_rows must be positive")

    def lane_balanced_geometry(self) -> bool:
        return str(self.cfg.geometry_reduction).strip().lower() in {
            "lane",
            "lane_mean",
            "per_lane",
        }

    def set_iteration(self, iteration: int) -> None:
        self._iteration = int(iteration)
        if self.matcher is not None and hasattr(self.matcher, "set_iteration"):
            self.matcher.set_iteration(iteration)

    def row_dfl_weight(self) -> float:
        weight = float(self.cfg.w_row_dfl)
        warmup = int(self.cfg.row_dfl_warmup_iters)
        if weight == 0.0 or warmup <= 0:
            return weight
        return weight * min(1.0, float(self._iteration + 1) / float(warmup))

    def intermediate_exist_weight(self) -> float:
        value = self.cfg.w_intermediate_exist
        return float(self.cfg.w_exist if value is None else value)

    @staticmethod
    def _pack_matched_lanes(
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> _MatchedLaneBatch:
        """Gather one decoder output's matched geometry exactly once."""

        pred_x_all = outputs["pred_x_rows"]
        device = pred_x_all.device
        rows = int(pred_x_all.shape[-1])
        batch_parts: list[torch.Tensor] = []
        pred_parts: list[torch.Tensor] = []
        gt_x_parts: list[torch.Tensor] = []
        valid_parts: list[torch.Tensor] = []
        range_norm_all = outputs.get("range_norm")
        has_range = isinstance(range_norm_all, torch.Tensor) and all(
            "range_y" in target for target in targets
        )
        gt_range_parts: list[torch.Tensor] = []
        for batch_index, match in enumerate(matches):
            pred_indices = match["pred_indices"].to(device=device)
            if int(pred_indices.numel()) == 0:
                continue
            gt_indices = match["gt_indices"].to(device=device)
            batch_parts.append(
                torch.full_like(pred_indices, int(batch_index))
            )
            pred_parts.append(pred_indices)
            gt_x_parts.append(
                targets[batch_index]["x_rows"].to(device=device)[gt_indices]
            )
            valid_parts.append(
                targets[batch_index]["valid_mask"]
                .to(device=device)[gt_indices]
                .bool()
            )
            if has_range:
                gt_range_parts.append(
                    targets[batch_index]["range_y"].to(device=device)[gt_indices]
                )

        if batch_parts:
            batch_indices = torch.cat(batch_parts)
            pred_indices = torch.cat(pred_parts)
            pred_x = pred_x_all[batch_indices, pred_indices]
            pred_range = (
                range_norm_all[batch_indices, pred_indices]
                if isinstance(range_norm_all, torch.Tensor)
                else None
            )
            row_logits_all = outputs.get("row_x_logits")
            row_logits = (
                row_logits_all[batch_indices, pred_indices]
                if isinstance(row_logits_all, torch.Tensor)
                else None
            )
            input_reference_all = outputs.get("input_reference_x_rows")
            input_reference = (
                input_reference_all[batch_indices, pred_indices]
                if isinstance(input_reference_all, torch.Tensor)
                else None
            )
            gt_x = torch.cat(gt_x_parts)
            valid = torch.cat(valid_parts)
            gt_range = torch.cat(gt_range_parts) if has_range else None
        else:
            pred_x = pred_x_all.new_empty((0, rows))
            pred_range = (
                range_norm_all.new_empty((0, 2))
                if isinstance(range_norm_all, torch.Tensor)
                else None
            )
            row_logits_all = outputs.get("row_x_logits")
            row_logits = (
                row_logits_all.new_empty((0, rows, int(row_logits_all.shape[-1])))
                if isinstance(row_logits_all, torch.Tensor)
                else None
            )
            input_reference_all = outputs.get("input_reference_x_rows")
            input_reference = (
                input_reference_all.new_empty((0, rows))
                if isinstance(input_reference_all, torch.Tensor)
                else None
            )
            gt_x = pred_x_all.new_empty((0, rows))
            valid = torch.empty((0, rows), device=device, dtype=torch.bool)
            gt_range = (
                range_norm_all.new_empty((0, 2))
                if has_range and isinstance(range_norm_all, torch.Tensor)
                else None
            )
        return _MatchedLaneBatch(
            pred_x=pred_x,
            gt_x=gt_x,
            valid=valid,
            pred_range=pred_range,
            gt_range=gt_range,
            row_logits=row_logits,
            input_reference=input_reference,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        raw_outputs = outputs
        if "final" in outputs:
            outputs = outputs["final"]
        elif "stage2" in outputs:
            outputs = outputs["stage2"]
        zero = self._zero_anchor(raw_outputs).sum() * 0.0
        four_slot_targets: tuple[torch.Tensor, torch.Tensor] | None = None
        matched_lanes = (
            self._pack_matched_lanes(outputs, targets, matches)
            if any(
                float(weight) != 0.0
                for weight in (
                    self.cfg.w_point,
                    self.cfg.w_range,
                    self.cfg.w_line_iou,
                    self.cfg.w_row_dfl,
                )
            )
            else None
        )
        if (
            self.cfg.w_four_slot_selection != 0
            or self.cfg.w_four_slot_geometry != 0
            or self.cfg.w_four_slot_unified != 0
            or self.cfg.w_four_slot_visual_first != 0
            or self.cfg.w_four_slot_visual_precision != 0
            or self.cfg.w_four_slot_v14_stage_a != 0
            or self.cfg.w_four_slot_v14_stage_b != 0
            or self.cfg.w_four_slot_v15 != 0
            or self.cfg.w_four_slot_v16 != 0
            or self.cfg.w_four_slot_v17 != 0
        ):
            proposal_rows = outputs.get("pred_x_rows")
            if not isinstance(proposal_rows, torch.Tensor):
                raise ValueError("four-slot losses require proposal row geometry")
            # Selection targets, reference matching, and refined-quality
            # diagnostics consume the same GT row tensors.  Prepare their
            # padded batch once instead of repeating three device copies and
            # stacks per micro-batch.
            four_slot_targets = _padded_lane_targets(
                targets,
                device=proposal_rows.device,
                dtype=torch.float32,
                rows=int(proposal_rows.shape[-1]),
            )
        loss_exist = (
            self.compute_exist_loss(outputs, matches, targets)
            if self.cfg.w_exist != 0
            else zero
        )
        loss_point = self.compute_point_loss(outputs, targets, matches, matched_lanes) if self.cfg.w_point != 0 else zero
        loss_range = self.compute_range_loss(outputs, targets, matches, matched_lanes) if self.cfg.w_range != 0 else zero
        loss_smooth = self.compute_smoothness_loss(outputs, targets, matches) if self.cfg.w_smooth != 0 else zero
        loss_line_iou = self.compute_line_iou_loss(outputs, targets, matches, matched_lanes) if self.cfg.w_line_iou != 0 else zero
        loss_seg = self.compute_seg_loss(raw_outputs, targets) if self.cfg.w_seg != 0 else zero
        loss_quality = self.compute_quality_loss(outputs, targets, matches) if self.cfg.w_quality != 0 else zero
        loss_cardinality = (
            self.compute_cardinality_loss(outputs, targets)
            if self.cfg.w_cardinality != 0
            else zero
        )
        loss_score_margin = (
            self.compute_score_margin_loss(outputs, matches)
            if self.cfg.w_score_margin != 0
            else zero
        )
        if self.cfg.w_set_selection != 0:
            set_selection = self.compute_set_selection_loss(
                outputs,
                targets,
                matches,
            )
        else:
            set_selection = {
                "total": zero,
                "quality": zero,
                "ranking": zero,
                "coverage": zero,
                "duplicate": zero,
                "winner": zero,
                "count": zero,
                "target_mean": zero,
                "target_positive_fraction": zero,
                "delta_abs": zero,
            }
        if self.cfg.w_pointer_selection != 0:
            pointer_selection = self.compute_pointer_selection_loss(
                outputs,
                targets,
            )
        else:
            pointer_selection = {
                "total": zero,
                "sequence": zero,
                "quality": zero,
                "listwise": zero,
                "mean_emitted_target": zero,
                "mean_stop_probability": zero,
                "mean_cluster_support": zero,
                "mean_cluster_entropy": zero,
                "mean_cluster_quality": zero,
                "mean_remaining_cluster_count": zero,
                "mean_representable_count": zero,
                "teacher_fallback_count": zero,
                "teacher_reservation_exclusion_count": zero,
            }
        if self.cfg.w_four_slot_selection != 0:
            four_slot_selection = self.compute_four_slot_selection_loss(
                outputs,
                targets,
                four_slot_targets,
            )
        else:
            four_slot_selection = {
                "total": zero,
                "permutation": zero,
                "collision": zero,
                "mean_representable_count": zero,
                "mean_support_size": zero,
                "mean_target_entropy": zero,
                "mean_target_quality": zero,
                "mean_raw_collision_count": zero,
                "mean_route_entropy": zero,
            }
        if self.cfg.w_four_slot_geometry != 0:
            four_slot_geometry = self.compute_four_slot_geometry_loss(
                outputs,
                targets,
                four_slot_targets,
            )
        else:
            four_slot_geometry = {
                "total": zero,
                "point": zero,
                "range": zero,
                "line_iou": zero,
                "dfl": zero,
                "mean_matched": zero,
                "mean_reference_quality": zero,
                "mean_refined_quality": zero,
                "mean_quality_gain": zero,
            }
        if self.cfg.w_four_slot_unified != 0:
            four_slot_unified = self.compute_four_slot_unified_loss(
                outputs,
                targets,
                four_slot_targets,
            )
        else:
            four_slot_unified = {
                "total": zero,
                "active": zero,
                "attention": zero,
                "point": zero,
                "range": zero,
                "line_iou": zero,
                "dfl": zero,
                "aux_point": zero,
                "aux_range": zero,
                "aux_line_iou": zero,
                "mean_matched": zero,
                "mean_base_quality": zero,
                "mean_final_quality": zero,
                "mean_quality_gain": zero,
                "mean_target_support_mass": zero,
                "mean_active_probability": zero,
            }
        if self.cfg.w_four_slot_visual_first != 0:
            four_slot_visual_first = (
                self.compute_four_slot_visual_first_loss(
                    outputs,
                    targets,
                    four_slot_targets,
                )
            )
        else:
            four_slot_visual_first = {
                "total": zero,
                "first_visual": zero,
                "final_visual": zero,
                "proposal": zero,
                "mean_matched": zero,
                "mean_anchor_quality": zero,
                "mean_target_support_mass": zero,
                "mean_target_top1": zero,
                "mean_first_visual_mae_px": zero,
                "mean_final_visual_mae_px": zero,
            }
        if self.cfg.w_four_slot_visual_precision != 0:
            four_slot_visual_precision = (
                self.compute_four_slot_visual_precision_loss(
                    outputs,
                    targets,
                    four_slot_targets,
                )
            )
        else:
            four_slot_visual_precision = {
                "total": zero,
                "point": zero,
                "range": zero,
                "line_iou": zero,
                "dfl": zero,
                "mean_matched": zero,
                "mean_anchor_quality": zero,
                "mean_final_quality": zero,
                "mean_quality_gain": zero,
                "mean_abs_delta_px": zero,
                "mean_candidate_gate": zero,
            }
        if self.cfg.w_four_slot_v14_stage_a != 0:
            four_slot_v14_stage_a = self.compute_four_slot_v14_stage_a_loss(
                outputs,
                targets,
                four_slot_targets,
            )
        else:
            four_slot_v14_stage_a = {
                "total": zero,
                "visual": zero,
                "association": zero,
                "mean_matched": zero,
                "mean_representable": zero,
                "mean_anchor_quality": zero,
                "mean_target_support_mass": zero,
                "mean_v7_target_support_mass": zero,
                "mean_target_top1": zero,
                "mean_v7_target_top1": zero,
                "mean_visual_mae_px": zero,
                "target_attention_row_error": zero,
                "target_real_column_excess": zero,
            }
        if self.cfg.w_four_slot_v14_stage_b != 0:
            four_slot_v14_stage_b = self.compute_four_slot_v14_stage_b_loss(
                outputs,
                targets,
                four_slot_targets,
            )
        else:
            four_slot_v14_stage_b = {
                "total": zero,
                "point": zero,
                "range": zero,
                "line_iou": zero,
                "dfl": zero,
                "mean_matched": zero,
                "mean_anchor_quality": zero,
                "mean_final_quality": zero,
                "mean_quality_gain": zero,
                "mean_abs_delta_px": zero,
            }
        if self.cfg.w_four_slot_v15 != 0:
            four_slot_v15 = self.compute_four_slot_v15_loss(
                outputs,
                targets,
                four_slot_targets,
            )
        else:
            four_slot_v15 = {
                "total": zero,
                "visual": zero,
                "point": zero,
                "range": zero,
                "line_iou": zero,
                "dfl": zero,
                "mean_matched": zero,
                "mean_anchor_quality": zero,
                "mean_final_quality": zero,
                "mean_quality_gain": zero,
                "mean_abs_delta_px": zero,
            }
        if self.cfg.w_four_slot_v16 != 0:
            four_slot_v16 = self.compute_four_slot_v16_loss(
                outputs,
                targets,
                four_slot_targets,
            )
        else:
            four_slot_v16 = {
                "total": zero,
                "quality": zero,
                "pairwise": zero,
                "hard": zero,
                "mean_matched": zero,
                "mean_representable": zero,
                "mean_group_size": zero,
                "mean_anchor_quality": zero,
                "mean_selected_quality": zero,
                "mean_oracle_quality": zero,
                "mean_quality_gain": zero,
                "mean_target_top1": zero,
            }
        if self.cfg.w_four_slot_v17 != 0:
            four_slot_v17 = self.compute_four_slot_v17_loss(
                outputs,
                targets,
                four_slot_targets,
            )
        else:
            four_slot_v17 = {
                "total": zero,
                "visual": zero,
                "point": zero,
                "range": zero,
                "line_iou": zero,
                "dfl": zero,
                "mean_matched": zero,
                "mean_anchor_quality": zero,
                "mean_final_quality": zero,
                "mean_quality_gain": zero,
                "mean_abs_delta_px": zero,
            }
        loss_centerline = self.compute_centerline_loss(raw_outputs, targets) if self.cfg.w_centerline != 0 else zero
        row_dfl_weight = self.row_dfl_weight()
        loss_row_dfl = self.compute_row_dfl_loss(outputs, targets, matches, matched_lanes) if row_dfl_weight != 0 else zero
        if (
            self.cfg.w_dynamic_proposal_heatmap != 0
            or self.cfg.w_dynamic_proposal_x != 0
            or self.cfg.w_dynamic_proposal_range != 0
        ):
            dynamic_proposal_losses = self.compute_dynamic_proposal_losses(raw_outputs, targets)
        else:
            dynamic_proposal_losses = {"heatmap": zero, "x": zero, "range": zero}
        total = (
            self.cfg.w_exist * loss_exist
            + self.cfg.w_point * loss_point
            + self.cfg.w_range * loss_range
            + self.cfg.w_smooth * loss_smooth
            + self.cfg.w_line_iou * loss_line_iou
            + self.cfg.w_seg * loss_seg
            + self.cfg.w_quality * loss_quality
            + self.cfg.w_cardinality * loss_cardinality
            + self.cfg.w_score_margin * loss_score_margin
            + self.cfg.w_set_selection * set_selection["total"]
            + self.cfg.w_pointer_selection * pointer_selection["total"]
            + self.cfg.w_four_slot_selection * four_slot_selection["total"]
            + self.cfg.w_four_slot_geometry * four_slot_geometry["total"]
            + self.cfg.w_four_slot_unified * four_slot_unified["total"]
            + self.cfg.w_four_slot_visual_first
            * four_slot_visual_first["total"]
            + self.cfg.w_four_slot_visual_precision
            * four_slot_visual_precision["total"]
            + self.cfg.w_four_slot_v14_stage_a
            * four_slot_v14_stage_a["total"]
            + self.cfg.w_four_slot_v14_stage_b
            * four_slot_v14_stage_b["total"]
            + self.cfg.w_four_slot_v15 * four_slot_v15["total"]
            + self.cfg.w_four_slot_v16 * four_slot_v16["total"]
            + self.cfg.w_four_slot_v17 * four_slot_v17["total"]
            + self.cfg.w_centerline * loss_centerline
            + row_dfl_weight * loss_row_dfl
            + self.cfg.w_dynamic_proposal_heatmap * dynamic_proposal_losses["heatmap"]
            + self.cfg.w_dynamic_proposal_x * dynamic_proposal_losses["x"]
            + self.cfg.w_dynamic_proposal_range * dynamic_proposal_losses["range"]
        )
        out = {
            "loss_total": total,
            "loss_exist": loss_exist,
            "loss_point": loss_point,
            "loss_range": loss_range,
            "loss_smooth": loss_smooth,
            "loss_line_iou": loss_line_iou,
            "loss_seg": loss_seg,
            "loss_quality": loss_quality,
            "loss_cardinality": loss_cardinality,
            "loss_score_margin": loss_score_margin,
            "loss_set_selection": set_selection["total"],
            "loss_set_selection_quality": set_selection["quality"],
            "loss_set_selection_ranking": set_selection["ranking"],
            "loss_set_selection_coverage": set_selection["coverage"],
            "loss_set_selection_duplicate": set_selection["duplicate"],
            "loss_set_selection_winner": set_selection["winner"],
            "loss_set_selection_count": set_selection["count"],
            "set_selection_target_mean": set_selection["target_mean"],
            "set_selection_target_positive_fraction": set_selection[
                "target_positive_fraction"
            ],
            "set_selection_delta_abs": set_selection["delta_abs"],
            "loss_pointer_selection": pointer_selection["total"],
            "loss_pointer_sequence": pointer_selection["sequence"],
            "loss_pointer_quality": pointer_selection["quality"],
            "loss_pointer_listwise": pointer_selection["listwise"],
            "pointer_target_mean_emitted": pointer_selection[
                "mean_emitted_target"
            ],
            "pointer_target_stop_probability": pointer_selection[
                "mean_stop_probability"
            ],
            "pointer_target_mean_cluster_support": pointer_selection[
                "mean_cluster_support"
            ],
            "pointer_target_mean_cluster_entropy": pointer_selection[
                "mean_cluster_entropy"
            ],
            "pointer_target_mean_cluster_quality": pointer_selection[
                "mean_cluster_quality"
            ],
            "pointer_target_mean_remaining_cluster_count": pointer_selection[
                "mean_remaining_cluster_count"
            ],
            "pointer_target_mean_representable_count": pointer_selection[
                "mean_representable_count"
            ],
            "pointer_teacher_fallback_count": pointer_selection[
                "teacher_fallback_count"
            ],
            "pointer_teacher_reservation_exclusion_count": pointer_selection[
                "teacher_reservation_exclusion_count"
            ],
            "loss_four_slot_selection": four_slot_selection["total"],
            "loss_four_slot_permutation": four_slot_selection["permutation"],
            "loss_four_slot_collision": four_slot_selection["collision"],
            "four_slot_target_mean_representable_count": four_slot_selection[
                "mean_representable_count"
            ],
            "four_slot_target_mean_support_size": four_slot_selection[
                "mean_support_size"
            ],
            "four_slot_target_mean_entropy": four_slot_selection[
                "mean_target_entropy"
            ],
            "four_slot_target_mean_quality": four_slot_selection[
                "mean_target_quality"
            ],
            "four_slot_mean_raw_collision_count": four_slot_selection[
                "mean_raw_collision_count"
            ],
            "four_slot_mean_route_entropy": four_slot_selection[
                "mean_route_entropy"
            ],
            "loss_four_slot_geometry": four_slot_geometry["total"],
            "loss_four_slot_geometry_point": four_slot_geometry["point"],
            "loss_four_slot_geometry_range": four_slot_geometry["range"],
            "loss_four_slot_geometry_line_iou": four_slot_geometry[
                "line_iou"
            ],
            "loss_four_slot_geometry_dfl": four_slot_geometry["dfl"],
            "four_slot_geometry_mean_matched": four_slot_geometry[
                "mean_matched"
            ],
            "four_slot_geometry_mean_reference_quality": four_slot_geometry[
                "mean_reference_quality"
            ],
            "four_slot_geometry_mean_refined_quality": four_slot_geometry[
                "mean_refined_quality"
            ],
            "four_slot_geometry_mean_quality_gain": four_slot_geometry[
                "mean_quality_gain"
            ],
            "loss_four_slot_unified": four_slot_unified["total"],
            "loss_four_slot_unified_active": four_slot_unified["active"],
            "loss_four_slot_unified_attention": four_slot_unified[
                "attention"
            ],
            "loss_four_slot_unified_point": four_slot_unified["point"],
            "loss_four_slot_unified_range": four_slot_unified["range"],
            "loss_four_slot_unified_line_iou": four_slot_unified[
                "line_iou"
            ],
            "loss_four_slot_unified_dfl": four_slot_unified["dfl"],
            "loss_four_slot_unified_aux_point": four_slot_unified[
                "aux_point"
            ],
            "loss_four_slot_unified_aux_range": four_slot_unified[
                "aux_range"
            ],
            "loss_four_slot_unified_aux_line_iou": four_slot_unified[
                "aux_line_iou"
            ],
            "four_slot_unified_mean_matched": four_slot_unified[
                "mean_matched"
            ],
            "four_slot_unified_mean_base_quality": four_slot_unified[
                "mean_base_quality"
            ],
            "four_slot_unified_mean_final_quality": four_slot_unified[
                "mean_final_quality"
            ],
            "four_slot_unified_mean_quality_gain": four_slot_unified[
                "mean_quality_gain"
            ],
            "four_slot_unified_mean_target_support_mass": four_slot_unified[
                "mean_target_support_mass"
            ],
            "four_slot_unified_mean_active_probability": four_slot_unified[
                "mean_active_probability"
            ],
            "loss_four_slot_visual_first": four_slot_visual_first["total"],
            "loss_four_slot_visual_first_first": four_slot_visual_first[
                "first_visual"
            ],
            "loss_four_slot_visual_first_final": four_slot_visual_first[
                "final_visual"
            ],
            "loss_four_slot_visual_first_proposal": four_slot_visual_first[
                "proposal"
            ],
            "four_slot_visual_first_mean_matched": four_slot_visual_first[
                "mean_matched"
            ],
            "four_slot_visual_first_mean_anchor_quality": (
                four_slot_visual_first["mean_anchor_quality"]
            ),
            "four_slot_visual_first_mean_target_support_mass": (
                four_slot_visual_first["mean_target_support_mass"]
            ),
            "four_slot_visual_first_mean_target_top1": (
                four_slot_visual_first["mean_target_top1"]
            ),
            "four_slot_visual_first_mean_first_visual_mae_px": (
                four_slot_visual_first["mean_first_visual_mae_px"]
            ),
            "four_slot_visual_first_mean_final_visual_mae_px": (
                four_slot_visual_first["mean_final_visual_mae_px"]
            ),
            "loss_four_slot_visual_precision": (
                four_slot_visual_precision["total"]
            ),
            "loss_four_slot_visual_precision_point": (
                four_slot_visual_precision["point"]
            ),
            "loss_four_slot_visual_precision_range": (
                four_slot_visual_precision["range"]
            ),
            "loss_four_slot_visual_precision_line_iou": (
                four_slot_visual_precision["line_iou"]
            ),
            "loss_four_slot_visual_precision_dfl": (
                four_slot_visual_precision["dfl"]
            ),
            "four_slot_visual_precision_mean_matched": (
                four_slot_visual_precision["mean_matched"]
            ),
            "four_slot_visual_precision_mean_anchor_quality": (
                four_slot_visual_precision["mean_anchor_quality"]
            ),
            "four_slot_visual_precision_mean_final_quality": (
                four_slot_visual_precision["mean_final_quality"]
            ),
            "four_slot_visual_precision_mean_quality_gain": (
                four_slot_visual_precision["mean_quality_gain"]
            ),
            "four_slot_visual_precision_mean_abs_delta_px": (
                four_slot_visual_precision["mean_abs_delta_px"]
            ),
            "four_slot_visual_precision_mean_candidate_gate": (
                four_slot_visual_precision["mean_candidate_gate"]
            ),
            "loss_four_slot_v14_stage_a": four_slot_v14_stage_a["total"],
            "loss_four_slot_v14_visual": four_slot_v14_stage_a["visual"],
            "loss_four_slot_v14_association": four_slot_v14_stage_a[
                "association"
            ],
            "four_slot_v14_mean_matched": four_slot_v14_stage_a[
                "mean_matched"
            ],
            "four_slot_v14_mean_representable": four_slot_v14_stage_a[
                "mean_representable"
            ],
            "four_slot_v14_mean_anchor_quality": four_slot_v14_stage_a[
                "mean_anchor_quality"
            ],
            "four_slot_v14_mean_target_support_mass": four_slot_v14_stage_a[
                "mean_target_support_mass"
            ],
            "four_slot_v14_mean_v7_target_support_mass": (
                four_slot_v14_stage_a["mean_v7_target_support_mass"]
            ),
            "four_slot_v14_mean_target_top1": four_slot_v14_stage_a[
                "mean_target_top1"
            ],
            "four_slot_v14_mean_v7_target_top1": four_slot_v14_stage_a[
                "mean_v7_target_top1"
            ],
            "four_slot_v14_mean_visual_mae_px": four_slot_v14_stage_a[
                "mean_visual_mae_px"
            ],
            "four_slot_v14_target_attention_row_error": (
                four_slot_v14_stage_a["target_attention_row_error"]
            ),
            "four_slot_v14_target_real_column_excess": (
                four_slot_v14_stage_a["target_real_column_excess"]
            ),
            "loss_four_slot_v14_stage_b": four_slot_v14_stage_b["total"],
            "loss_four_slot_v14_stage_b_point": four_slot_v14_stage_b["point"],
            "loss_four_slot_v14_stage_b_range": four_slot_v14_stage_b["range"],
            "loss_four_slot_v14_stage_b_line_iou": four_slot_v14_stage_b[
                "line_iou"
            ],
            "loss_four_slot_v14_stage_b_dfl": four_slot_v14_stage_b["dfl"],
            "four_slot_v14_stage_b_mean_matched": four_slot_v14_stage_b[
                "mean_matched"
            ],
            "four_slot_v14_stage_b_mean_anchor_quality": four_slot_v14_stage_b[
                "mean_anchor_quality"
            ],
            "four_slot_v14_stage_b_mean_final_quality": four_slot_v14_stage_b[
                "mean_final_quality"
            ],
            "four_slot_v14_stage_b_mean_quality_gain": four_slot_v14_stage_b[
                "mean_quality_gain"
            ],
            "four_slot_v14_stage_b_mean_abs_delta_px": four_slot_v14_stage_b[
                "mean_abs_delta_px"
            ],
            "loss_four_slot_v15": four_slot_v15["total"],
            "loss_four_slot_v15_visual": four_slot_v15["visual"],
            "loss_four_slot_v15_point": four_slot_v15["point"],
            "loss_four_slot_v15_range": four_slot_v15["range"],
            "loss_four_slot_v15_line_iou": four_slot_v15["line_iou"],
            "loss_four_slot_v15_dfl": four_slot_v15["dfl"],
            "four_slot_v15_mean_matched": four_slot_v15["mean_matched"],
            "four_slot_v15_mean_anchor_quality": four_slot_v15[
                "mean_anchor_quality"
            ],
            "four_slot_v15_mean_final_quality": four_slot_v15[
                "mean_final_quality"
            ],
            "four_slot_v15_mean_quality_gain": four_slot_v15[
                "mean_quality_gain"
            ],
            "four_slot_v15_mean_abs_delta_px": four_slot_v15[
                "mean_abs_delta_px"
            ],
            "loss_four_slot_v16": four_slot_v16["total"],
            "loss_four_slot_v16_quality": four_slot_v16["quality"],
            "loss_four_slot_v16_pairwise": four_slot_v16["pairwise"],
            "loss_four_slot_v16_hard": four_slot_v16["hard"],
            "four_slot_v16_mean_matched": four_slot_v16["mean_matched"],
            "four_slot_v16_mean_representable": four_slot_v16[
                "mean_representable"
            ],
            "four_slot_v16_mean_group_size": four_slot_v16[
                "mean_group_size"
            ],
            "four_slot_v16_mean_anchor_quality": four_slot_v16[
                "mean_anchor_quality"
            ],
            "four_slot_v16_mean_selected_quality": four_slot_v16[
                "mean_selected_quality"
            ],
            "four_slot_v16_mean_oracle_quality": four_slot_v16[
                "mean_oracle_quality"
            ],
            "four_slot_v16_mean_quality_gain": four_slot_v16[
                "mean_quality_gain"
            ],
            "four_slot_v16_mean_target_top1": four_slot_v16[
                "mean_target_top1"
            ],
            "loss_four_slot_v17": four_slot_v17["total"],
            "loss_four_slot_v17_visual": four_slot_v17["visual"],
            "loss_four_slot_v17_point": four_slot_v17["point"],
            "loss_four_slot_v17_range": four_slot_v17["range"],
            "loss_four_slot_v17_line_iou": four_slot_v17["line_iou"],
            "loss_four_slot_v17_dfl": four_slot_v17["dfl"],
            "four_slot_v17_mean_matched": four_slot_v17["mean_matched"],
            "four_slot_v17_mean_anchor_quality": four_slot_v17[
                "mean_anchor_quality"
            ],
            "four_slot_v17_mean_final_quality": four_slot_v17[
                "mean_final_quality"
            ],
            "four_slot_v17_mean_quality_gain": four_slot_v17[
                "mean_quality_gain"
            ],
            "four_slot_v17_mean_abs_delta_px": four_slot_v17[
                "mean_abs_delta_px"
            ],
            "loss_centerline": loss_centerline,
            "loss_row_dfl": loss_row_dfl,
            "weight_row_dfl": zero.new_tensor(row_dfl_weight),
            "loss_dynamic_proposal_heatmap": dynamic_proposal_losses["heatmap"],
            "loss_dynamic_proposal_x": dynamic_proposal_losses["x"],
            "loss_dynamic_proposal_range": dynamic_proposal_losses["range"],
        }
        if self.cfg.lambda_coarse > 0 and isinstance(raw_outputs.get("coarse"), dict):
            coarse = raw_outputs["coarse"]
            coarse_exist = (
                self.compute_exist_loss(coarse, matches, targets)
                if self.cfg.w_exist != 0
                else zero
            )
            coarse_point = self.compute_point_loss(coarse, targets, matches) if self.cfg.w_point != 0 else zero
            coarse_range = self.compute_range_loss(coarse, targets, matches) if self.cfg.w_range != 0 else zero
            coarse_smooth = self.compute_smoothness_loss(coarse, targets, matches) if self.cfg.w_smooth != 0 else zero
            coarse_line_iou = self.compute_line_iou_loss(coarse, targets, matches) if self.cfg.w_line_iou != 0 else zero
            coarse_quality = self.compute_quality_loss(coarse, targets, matches) if self.cfg.w_quality != 0 else zero
            coarse_row_dfl = self.compute_row_dfl_loss(coarse, targets, matches) if row_dfl_weight != 0 else zero
            coarse_total = (
                self.cfg.w_exist * coarse_exist
                + self.cfg.w_point * coarse_point
                + self.cfg.w_range * coarse_range
                + self.cfg.w_smooth * coarse_smooth
                + self.cfg.w_line_iou * coarse_line_iou
                + self.cfg.w_quality * coarse_quality
                + row_dfl_weight * coarse_row_dfl
            )
            total = total + self.cfg.lambda_coarse * coarse_total
            out.update(
                {
                    "loss_total": total,
                    "loss_coarse_total": coarse_total,
                    "loss_exist_coarse": coarse_exist,
                    "loss_point_coarse": coarse_point,
                    "loss_range_coarse": coarse_range,
                    "loss_smooth_coarse": coarse_smooth,
                    "loss_line_iou_coarse": coarse_line_iou,
                    "loss_quality_coarse": coarse_quality,
                    "loss_row_dfl_coarse": coarse_row_dfl,
                }
            )
        out = self.add_geometry_draft_loss(out, raw_outputs, targets, matches)
        out = self.add_intermediate_losses(out, raw_outputs, targets)
        out = self.add_training_auxiliary_losses(out, raw_outputs, targets)
        return out

    def add_training_auxiliary_losses(
        self,
        losses: dict[str, torch.Tensor],
        outputs: dict[str, object],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Supervise train-only query groups without diluting the primary set.

        The normal criterion above sees only the 32 deployable candidates.
        Auxiliary groups receive their own grouped one-to-many assignments and
        contribute a separately weighted objective.  Encoder auxiliaries and
        set-selection are intentionally excluded because these query groups do
        not exist at inference time.
        """

        strength = float(self.cfg.lambda_training_auxiliary)
        auxiliary = outputs.get("_training_auxiliary_outputs")
        if strength <= 0.0:
            if isinstance(auxiliary, dict):
                raise ValueError(
                    "model emitted training auxiliary queries but "
                    "loss.lambda_training_auxiliary is not positive"
                )
            return losses
        if not isinstance(auxiliary, dict):
            raise ValueError(
                "lambda_training_auxiliary > 0 requires model training auxiliary outputs"
            )
        auxiliary_matches = outputs.get("_training_auxiliary_matches")
        if not isinstance(auxiliary_matches, (list, tuple)):
            raise ValueError("training auxiliary outputs require precomputed matches")

        zero = self._zero_anchor(auxiliary).sum() * 0.0
        row_dfl_weight = self.row_dfl_weight()
        aux_exist = (
            self.compute_exist_loss(auxiliary, auxiliary_matches, targets)
            if self.cfg.w_exist != 0
            else zero
        )
        aux_point = (
            self.compute_point_loss(auxiliary, targets, auxiliary_matches)
            if self.cfg.w_point != 0
            else zero
        )
        aux_range = (
            self.compute_range_loss(auxiliary, targets, auxiliary_matches)
            if self.cfg.w_range != 0
            else zero
        )
        aux_smooth = (
            self.compute_smoothness_loss(auxiliary, targets, auxiliary_matches)
            if self.cfg.w_smooth != 0
            else zero
        )
        aux_line_iou = (
            self.compute_line_iou_loss(auxiliary, targets, auxiliary_matches)
            if self.cfg.w_line_iou != 0
            else zero
        )
        aux_quality = (
            self.compute_quality_loss(auxiliary, targets, auxiliary_matches)
            if self.cfg.w_quality != 0
            else zero
        )
        aux_row_dfl = (
            self.compute_row_dfl_loss(auxiliary, targets, auxiliary_matches)
            if row_dfl_weight != 0
            else zero
        )
        auxiliary_total = (
            self.cfg.w_exist * aux_exist
            + self.cfg.w_point * aux_point
            + self.cfg.w_range * aux_range
            + self.cfg.w_smooth * aux_smooth
            + self.cfg.w_line_iou * aux_line_iou
            + self.cfg.w_quality * aux_quality
            + row_dfl_weight * aux_row_dfl
        )

        auxiliary_layers = outputs.get("_training_auxiliary_aux_outputs")
        auxiliary_layer_matches = outputs.get(
            "_training_auxiliary_aux_matches"
        )
        auxiliary_intermediate = zero
        if float(self.cfg.lambda_intermediate) > 0.0:
            if not isinstance(auxiliary_layers, (list, tuple)) or not auxiliary_layers:
                raise ValueError(
                    "deeply supervised training auxiliaries require intermediate outputs"
                )
            if not isinstance(auxiliary_layer_matches, (list, tuple)) or len(
                auxiliary_layer_matches
            ) != len(auxiliary_layers):
                raise ValueError(
                    "training auxiliary intermediate outputs require matching assignments"
                )
            configured_weights = tuple(
                float(value) for value in self.cfg.intermediate_layer_weights
            )
            layer_weights = configured_weights or tuple(
                1.0 for _ in auxiliary_layers
            )
            if len(layer_weights) != len(auxiliary_layers):
                raise ValueError(
                    "training auxiliary intermediate weights must match decoder layers"
                )
            normalizer = float(sum(layer_weights))
            if normalizer <= 0.0:
                raise ValueError("training auxiliary intermediate weights must sum positive")
            for layer, layer_matches, layer_weight in zip(
                auxiliary_layers,
                auxiliary_layer_matches,
                layer_weights,
            ):
                if not isinstance(layer, dict):
                    raise TypeError("training auxiliary decoder output must be a dictionary")
                layer_zero = self._zero_anchor(layer).sum() * 0.0
                intermediate_exist_weight = self.intermediate_exist_weight()
                layer_exist = (
                    self.compute_exist_loss(layer, layer_matches, targets)
                    if intermediate_exist_weight != 0.0
                    else layer_zero
                )
                layer_point = (
                    self.compute_point_loss(layer, targets, layer_matches)
                    if self.cfg.w_point != 0
                    else layer_zero
                )
                layer_range = (
                    self.compute_range_loss(layer, targets, layer_matches)
                    if self.cfg.w_range != 0
                    else layer_zero
                )
                layer_line_iou = (
                    self.compute_line_iou_loss(layer, targets, layer_matches)
                    if self.cfg.w_line_iou != 0
                    else layer_zero
                )
                layer_row_dfl = (
                    self.compute_row_dfl_loss(layer, targets, layer_matches)
                    if row_dfl_weight != 0
                    else layer_zero
                )
                layer_total = (
                    intermediate_exist_weight * layer_exist
                    + self.cfg.w_point * layer_point
                    + self.cfg.w_range * layer_range
                    + self.cfg.w_line_iou * layer_line_iou
                    + row_dfl_weight * layer_row_dfl
                )
                auxiliary_intermediate = auxiliary_intermediate + (
                    float(layer_weight) / normalizer
                ) * layer_total
            auxiliary_total = auxiliary_total + float(
                self.cfg.lambda_intermediate
            ) * auxiliary_intermediate

        out = dict(losses)
        out["loss_total"] = out["loss_total"] + strength * auxiliary_total
        out["loss_training_auxiliary_total"] = auxiliary_total
        out["loss_training_auxiliary_exist"] = aux_exist
        out["loss_training_auxiliary_point"] = aux_point
        out["loss_training_auxiliary_range"] = aux_range
        out["loss_training_auxiliary_smooth"] = aux_smooth
        out["loss_training_auxiliary_line_iou"] = aux_line_iou
        out["loss_training_auxiliary_quality"] = aux_quality
        out["loss_training_auxiliary_row_dfl"] = aux_row_dfl
        out["loss_training_auxiliary_intermediate"] = auxiliary_intermediate
        out["weight_training_auxiliary"] = zero.new_tensor(strength)
        return out

    def add_intermediate_losses(
        self,
        losses: dict[str, torch.Tensor],
        outputs: dict[str, object],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Deeply supervise intermediate structured decoder states.

        Each layer receives its own assignment, matching CondLSTR's training
        principle while retaining the final layer as the only quality-calibrated
        output.  Encoder auxiliary objectives, smoothness, and quality are not
        duplicated here.
        """
        strength = float(self.cfg.lambda_intermediate)
        if strength <= 0.0:
            return losses
        aux_outputs = outputs.get("aux_outputs")
        if not isinstance(aux_outputs, (list, tuple)) or not aux_outputs:
            raise ValueError("lambda_intermediate > 0 requires non-empty model aux_outputs")
        if self.matcher is None:
            raise ValueError("lambda_intermediate > 0 requires an auxiliary matcher")

        configured_weights = tuple(float(v) for v in self.cfg.intermediate_layer_weights)
        if configured_weights:
            if len(configured_weights) != len(aux_outputs):
                raise ValueError(
                    "intermediate_layer_weights must match the number of auxiliary decoder layers: "
                    f"got {len(configured_weights)} weights for {len(aux_outputs)} outputs"
                )
            layer_weights = configured_weights
        else:
            layer_weights = tuple(1.0 for _ in aux_outputs)
        if any(weight < 0.0 for weight in layer_weights) or sum(layer_weights) <= 0.0:
            raise ValueError("intermediate_layer_weights must be non-negative with a positive sum")

        normalizer = float(sum(layer_weights))
        row_dfl_weight = self.row_dfl_weight()
        intermediate_exist_weight = self.intermediate_exist_weight()
        aggregate = losses["loss_total"].new_zeros(())
        component_sums = {
            "exist": aggregate.clone(),
            "point": aggregate.clone(),
            "range": aggregate.clone(),
            "line_iou": aggregate.clone(),
            "row_dfl": aggregate.clone(),
        }
        out = dict(losses)
        precomputed_matches = outputs.get("_aux_matches")
        if isinstance(precomputed_matches, (list, tuple)):
            if len(precomputed_matches) != len(aux_outputs):
                raise ValueError(
                    "_aux_matches must match the number of auxiliary decoder layers: "
                    f"got {len(precomputed_matches)} matches for {len(aux_outputs)} outputs"
                )
            aux_matches_by_layer = precomputed_matches
        elif hasattr(self.matcher, "match_many"):
            aux_matches_by_layer = self.matcher.match_many(tuple(aux_outputs), targets)
        else:
            aux_matches_by_layer = [self.matcher(aux, targets) for aux in aux_outputs]

        for layer_index, (aux, layer_weight, aux_matches) in enumerate(
            zip(aux_outputs, layer_weights, aux_matches_by_layer),
            start=1,
        ):
            if not isinstance(aux, dict):
                raise TypeError("every auxiliary decoder output must be a dictionary")
            zero = self._zero_anchor(aux).sum() * 0.0
            packed = self._pack_matched_lanes(aux, targets, aux_matches)
            aux_exist = (
                self.compute_exist_loss(aux, aux_matches, targets)
                if intermediate_exist_weight != 0.0
                else zero
            )
            aux_point = self.compute_point_loss(aux, targets, aux_matches, packed) if self.cfg.w_point != 0 else zero
            aux_range = self.compute_range_loss(aux, targets, aux_matches, packed) if self.cfg.w_range != 0 else zero
            aux_line_iou = (
                self.compute_line_iou_loss(aux, targets, aux_matches, packed)
                if self.cfg.w_line_iou != 0
                else zero
            )
            aux_row_dfl = (
                self.compute_row_dfl_loss(aux, targets, aux_matches, packed)
                if row_dfl_weight != 0
                else zero
            )
            aux_total = (
                intermediate_exist_weight * aux_exist
                + self.cfg.w_point * aux_point
                + self.cfg.w_range * aux_range
                + self.cfg.w_line_iou * aux_line_iou
                + row_dfl_weight * aux_row_dfl
            )
            normalized_weight = float(layer_weight) / normalizer
            aggregate = aggregate + normalized_weight * aux_total
            component_sums["exist"] = component_sums["exist"] + normalized_weight * aux_exist
            component_sums["point"] = component_sums["point"] + normalized_weight * aux_point
            component_sums["range"] = component_sums["range"] + normalized_weight * aux_range
            component_sums["line_iou"] = component_sums["line_iou"] + normalized_weight * aux_line_iou
            component_sums["row_dfl"] = component_sums["row_dfl"] + normalized_weight * aux_row_dfl
            out[f"loss_intermediate_l{layer_index}_total"] = aux_total

        out["loss_total"] = out["loss_total"] + strength * aggregate
        out["loss_intermediate_total"] = aggregate
        out["loss_intermediate_exist"] = component_sums["exist"]
        out["loss_intermediate_point"] = component_sums["point"]
        out["loss_intermediate_range"] = component_sums["range"]
        out["loss_intermediate_line_iou"] = component_sums["line_iou"]
        out["loss_intermediate_row_dfl"] = component_sums["row_dfl"]
        out["weight_intermediate"] = aggregate.new_tensor(strength)
        out["weight_intermediate_exist"] = aggregate.new_tensor(
            intermediate_exist_weight
        )
        return out

    def add_geometry_draft_loss(
        self,
        losses: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if self.cfg.lambda_geometry_draft <= 0 or not isinstance(outputs.get("s0_geometry_draft"), dict):
            return losses
        draft = outputs["s0_geometry_draft"]
        zero = self._zero_anchor(draft).sum() * 0.0
        draft_exist = (
            self.compute_exist_loss(draft, matches, targets)
            if self.cfg.w_exist != 0
            else zero
        )
        draft_point = self.compute_point_loss(draft, targets, matches) if self.cfg.w_point != 0 else zero
        draft_range = self.compute_range_loss(draft, targets, matches) if self.cfg.w_range != 0 else zero
        draft_smooth = self.compute_smoothness_loss(draft, targets, matches) if self.cfg.w_smooth != 0 else zero
        draft_line_iou = self.compute_line_iou_loss(draft, targets, matches) if self.cfg.w_line_iou != 0 else zero
        draft_quality = self.compute_quality_loss(draft, targets, matches) if self.cfg.w_quality != 0 else zero
        row_dfl_weight = self.row_dfl_weight()
        draft_row_dfl = self.compute_row_dfl_loss(draft, targets, matches) if row_dfl_weight != 0 else zero
        draft_total = (
            self.cfg.w_exist * draft_exist
            + self.cfg.w_point * draft_point
            + self.cfg.w_range * draft_range
            + self.cfg.w_smooth * draft_smooth
            + self.cfg.w_line_iou * draft_line_iou
            + self.cfg.w_quality * draft_quality
            + row_dfl_weight * draft_row_dfl
        )
        losses = dict(losses)
        losses["loss_total"] = losses["loss_total"] + self.cfg.lambda_geometry_draft * draft_total
        losses.update(
            {
                "loss_geometry_draft_total": draft_total,
                "loss_exist_geometry_draft": draft_exist,
                "loss_point_geometry_draft": draft_point,
                "loss_range_geometry_draft": draft_range,
                "loss_smooth_geometry_draft": draft_smooth,
                "loss_line_iou_geometry_draft": draft_line_iou,
                "loss_quality_geometry_draft": draft_quality,
                "loss_row_dfl_geometry_draft": draft_row_dfl,
            }
        )
        return losses

    @torch.no_grad()
    def compute_exist_quality_targets(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Build detached localization-aware targets for the one deployable score."""

        pred_x = outputs["pred_x_rows"].detach().float()
        ranges = outputs["range_norm"].detach().float()
        batch, candidates, _rows = pred_x.shape
        if len(targets) != batch or len(matches) != batch:
            raise ValueError("exist quality targets require one target/match per image")
        score_target = pred_x.new_zeros((batch, candidates))
        floor = float(self.cfg.exist_quality_floor)
        for batch_index, (target, match) in enumerate(zip(targets, matches)):
            pred_indices = match["pred_indices"].to(pred_x.device)
            gt_indices = match["gt_indices"].to(pred_x.device)
            if pred_indices.numel() == 0:
                continue
            gt_x = target["x_rows"].to(
                device=pred_x.device,
                dtype=pred_x.dtype,
            )
            gt_valid = target["valid_mask"].to(pred_x.device).bool()
            quality, _candidate_valid, _valid_gt = (
                pairwise_range_aware_row_strip_iou(
                    pred_x[batch_index],
                    ranges[batch_index],
                    gt_x,
                    gt_valid,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.exist_quality_line_width),
                    min_valid_rows=int(self.cfg.exist_quality_min_valid_rows),
                )
            )
            matched_quality = quality[pred_indices, gt_indices].clamp(0.0, 1.0)
            score_target[batch_index, pred_indices] = floor + (
                1.0 - floor
            ) * matched_quality
        return score_target

    def compute_exist_loss(
        self,
        outputs: dict[str, torch.Tensor],
        matches: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        logits = outputs["exist_logits"]
        b, n, _ = logits.shape
        if str(self.cfg.exist_target_mode).strip().lower() == "iou_aware":
            if targets is None:
                raise ValueError("iou_aware existence supervision requires targets")
            lane_target = self.compute_exist_quality_targets(
                outputs,
                targets,
                matches,
            ).to(device=logits.device, dtype=torch.float32)
            lane_logit = (logits[..., 0] - logits[..., 1]).float()
            probability = torch.sigmoid(lane_logit)
            modulation = (lane_target - probability).abs().pow(
                float(self.cfg.exist_quality_beta)
            )
            return (
                modulation
                * F.binary_cross_entropy_with_logits(
                    lane_logit,
                    lane_target,
                    reduction="none",
                )
            ).mean()
        target = torch.ones((b, n), dtype=torch.long, device=logits.device)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(logits.device)
            if pred_idx.numel() > 0:
                target[bi, pred_idx] = 0
        if str(self.cfg.exist_loss_type).lower() == "focal":
            lane_target = (target == 0).to(dtype=logits.dtype)
            lane_logit = logits[..., 0] - logits[..., 1]
            ce = F.binary_cross_entropy_with_logits(lane_logit, lane_target, reduction="none")
            prob = torch.sigmoid(lane_logit)
            p_t = prob * lane_target + (1.0 - prob) * (1.0 - lane_target)
            alpha = float(self.cfg.focal_alpha)
            alpha_t = alpha * lane_target + (1.0 - alpha) * (1.0 - lane_target)
            loss = alpha_t * (1.0 - p_t).pow(float(self.cfg.focal_gamma)) * ce
            return loss.mean()
        weight = torch.tensor([1.0, self.cfg.no_lane_weight], device=logits.device, dtype=logits.dtype)
        # Candidate subsets are views in the hybrid primary/auxiliary decoder;
        # reshape handles their non-contiguous candidate dimension safely.
        return F.cross_entropy(logits.reshape(b * n, 2), target.reshape(b * n), weight=weight)

    def compute_cardinality_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Calibrate the NMS-free lane count without choosing a threshold.

        Per-query focal supervision says which slot is foreground, but it does
        not directly constrain the total probability mass emitted by a
        32-query set.  This differentiable count objective makes an image with
        four lanes carry approximately four foreground probabilities and a
        cross/no-lane image carry approximately zero.  Geometry is not used in
        the target, so the loss cannot improve its score by moving a curve.
        """

        logits = outputs["exist_logits"]
        probability = torch.softmax(logits.float(), dim=-1)[..., 0]
        predicted_count = probability.sum(dim=1)
        target_count = probability.new_tensor(
            [float(target["x_rows"].shape[0]) for target in targets]
        )
        return F.smooth_l1_loss(
            predicted_count,
            target_count,
            beta=1.0,
            reduction="mean",
        )

    def compute_score_margin_loss(
        self,
        outputs: dict[str, torch.Tensor],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Force every assigned lane above the hardest unmatched duplicates.

        This is a set-ranking objective, not a second score head.  It operates
        on the exact foreground logit used by the matcher and deployment and
        therefore cannot create the old existence/quality/selector mismatch.
        """

        logits = outputs["exist_logits"].float()
        foreground = logits[..., 0] - logits[..., 1]
        margin = float(self.cfg.score_margin)
        topk = int(self.cfg.score_margin_topk_negatives)
        if topk < 1:
            raise ValueError("score_margin_topk_negatives must be positive")
        total = foreground.sum() * 0.0
        count = foreground.new_zeros(())
        for batch_index, match in enumerate(matches):
            positive_indices = match["pred_indices"].to(foreground.device)
            if positive_indices.numel() == 0:
                continue
            negative_mask = torch.ones(
                foreground.shape[1],
                dtype=torch.bool,
                device=foreground.device,
            )
            negative_mask[positive_indices] = False
            negative = foreground[batch_index, negative_mask]
            if negative.numel() == 0:
                continue
            hardest = negative.topk(min(topk, int(negative.numel()))).values
            positive = foreground[batch_index, positive_indices]
            pair_loss = F.softplus(
                margin - positive.unsqueeze(-1) + hardest.unsqueeze(0)
            )
            total = total + pair_loss.sum()
            count = count + pair_loss.new_tensor(float(pair_loss.numel()))
        return total / count.clamp_min(1.0)

    @staticmethod
    def _zero_anchor(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, dict):
                try:
                    return S0Criterion._zero_anchor(value)
                except StopIteration:
                    continue
        raise StopIteration

    def compute_point_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
        packed: _MatchedLaneBatch | None = None,
    ) -> torch.Tensor:
        packed = packed or self._pack_matched_lanes(outputs, targets, matches)
        if int(packed.pred_x.shape[0]) == 0:
            return outputs["pred_x_rows"].sum() * 0.0
        pred = packed.pred_x / float(self.cfg.input_w)
        gt = packed.gt_x / float(self.cfg.input_w)
        valid = packed.valid.to(dtype=pred.dtype)
        loss = F.smooth_l1_loss(
            pred,
            gt,
            beta=self.cfg.smooth_l1_beta,
            reduction="none",
        )
        lane_balanced = self.lane_balanced_geometry()
        if lane_balanced:
            valid_count = valid.sum(dim=-1)
            lane_loss = (loss * valid).sum(dim=-1) / valid_count.clamp_min(1.0)
            valid_lane = valid_count > 0
            return (
                lane_loss * valid_lane.to(dtype=lane_loss.dtype)
            ).sum() / valid_lane.to(dtype=lane_loss.dtype).sum().clamp_min(1.0)
        return (loss * valid).sum() / valid.sum().clamp_min(1.0)

    def compute_row_dfl_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
        packed: _MatchedLaneBatch | None = None,
    ) -> torch.Tensor:
        logits = outputs.get("row_x_logits")
        if logits is None:
            return outputs["pred_x_rows"].sum() * 0.0
        packed = packed or self._pack_matched_lanes(outputs, targets, matches)
        pred_logits = packed.row_logits
        if pred_logits is None or int(pred_logits.shape[0]) == 0:
            return logits.sum(dtype=torch.float32) * 0.0
        num_rows, x_bins = int(pred_logits.shape[-2]), int(pred_logits.shape[-1])
        # Preserve a differentiable zero without materializing an FP32 copy of
        # every slot/row/bin.  Only matched lane logits contribute to DFL and
        # are converted below after indexing.
        lane_balanced = self.lane_balanced_geometry()
        bin_width = float(self.cfg.input_w) / float(x_bins)
        delta_offsets = outputs.get("row_x_offsets_px")
        input_reference = packed.input_reference
        local_delta_mode = isinstance(delta_offsets, torch.Tensor)
        if local_delta_mode:
            if not isinstance(input_reference, torch.Tensor):
                raise ValueError(
                    "local delta DFL requires input_reference_x_rows"
                )
            delta_offsets = delta_offsets.to(
                device=logits.device,
                dtype=torch.float32,
            ).reshape(-1)
            if int(delta_offsets.numel()) != int(x_bins):
                raise ValueError(
                    "row_x_offsets_px count must match local row logits"
                )
        row_count = min(int(packed.gt_x.shape[-1]), int(num_rows))
        if row_count <= 0:
            return logits.sum(dtype=torch.float32) * 0.0
        pred_logits = pred_logits[:, :row_count].float()
        gt_x = packed.gt_x[:, :row_count].float()
        mask = packed.valid[:, :row_count]
        valid = mask & torch.isfinite(gt_x) & (gt_x >= 0.0) & (gt_x <= float(self.cfg.input_w))

            # Avoid a Python boolean conversion of a CUDA tensor here.  Deep
            # supervision reaches this path once per image and decoder output;
            # the old ``if not valid.any()`` therefore serialized the stream
            # many times per optimizer step.  Invalid values are made safe
            # before indexing and remain exactly zero-weighted below.
        safe_gt_x = torch.where(valid, gt_x, torch.zeros_like(gt_x))
        if local_delta_mode:
            if input_reference is None:
                raise ValueError("local delta DFL requires matched input references")
            reference = input_reference[:, :row_count].detach().float()
            target_delta = torch.minimum(
                torch.maximum(safe_gt_x - reference, delta_offsets[0]),
                delta_offsets[-1],
            )
            right = torch.searchsorted(
                delta_offsets,
                target_delta.contiguous(),
            ).clamp(min=1, max=x_bins - 1)
            left = right - 1
            left_offset = delta_offsets[left]
            right_offset = delta_offsets[right]
            right_w = (target_delta - left_offset) / (
                right_offset - left_offset
            ).clamp_min(1e-6)
            right_w = right_w.clamp(0.0, 1.0)
            left_w = 1.0 - right_w
        else:
            target_bin = (safe_gt_x / bin_width).clamp(
                0.0,
                float(x_bins - 1),
            )
            left = target_bin.floor().long()
            right = (left + 1).clamp(max=x_bins - 1)
            right_w = target_bin - left.to(dtype=target_bin.dtype)
            left_w = 1.0 - right_w
            same = right == left
            left_w = torch.where(same, torch.ones_like(left_w), left_w)
            right_w = torch.where(same, torch.zeros_like(right_w), right_w)

        log_probs = F.log_softmax(pred_logits, dim=-1)
        left_lp = log_probs.gather(-1, left.unsqueeze(-1)).squeeze(-1)
        right_lp = log_probs.gather(-1, right.unsqueeze(-1)).squeeze(-1)
        loss = -(left_w * left_lp + right_w * right_lp)
        valid_f = valid.to(dtype=loss.dtype)
        if lane_balanced:
            valid_count = valid_f.sum(dim=-1)
            lane_loss = (loss * valid_f).sum(dim=-1) / valid_count.clamp_min(1.0)
            valid_lane = valid_count > 0
            return (
                lane_loss * valid_lane.to(dtype=lane_loss.dtype)
            ).sum() / valid_lane.to(dtype=lane_loss.dtype).sum().clamp_min(1.0)
        return (loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    def compute_line_iou_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
        packed: _MatchedLaneBatch | None = None,
    ) -> torch.Tensor:
        packed = packed or self._pack_matched_lanes(outputs, targets, matches)
        if int(packed.pred_x.shape[0]) == 0:
            return outputs["pred_x_rows"].sum() * 0.0
        radius = float(self.cfg.line_iou_radius)
        pred = packed.pred_x
        gt_x = packed.gt_x
        px1 = pred - radius
        px2 = pred + radius
        gx1 = gt_x - radius
        gx2 = gt_x + radius
        overlap = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
        union = (4.0 * radius - overlap).clamp(min=1e-6)
        iou = overlap / union
        enclosing = (torch.maximum(px2, gx2) - torch.minimum(px1, gx1)).clamp(min=1e-6)
        giou = iou - (enclosing - union) / enclosing
        valid = packed.valid.to(dtype=pred.dtype)
        valid_count = valid.sum(dim=-1)
        lane_loss = ((1.0 - giou) * valid).sum(dim=-1) / valid_count.clamp_min(1.0)
        valid_lane = valid_count > 0
        return (
            lane_loss * valid_lane.to(dtype=lane_loss.dtype)
        ).sum() / valid_lane.to(dtype=lane_loss.dtype).sum().clamp_min(1.0)

    def compute_quality_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        logits = outputs.get("quality_logits")
        if logits is None:
            return outputs["pred_x_rows"].sum() * 0.0
        quality_logits = logits.float()
        target_quality = torch.zeros_like(quality_logits)
        pred_x = outputs.get("quality_pred_x_rows", outputs["pred_x_rows"]).float()
        radius = float(self.cfg.line_iou_radius)
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            if pred_idx.numel() == 0:
                continue
            gt_x = targets[bi]["x_rows"].to(pred_x.device, dtype=pred_x.dtype)[gt_idx]
            mask = targets[bi]["valid_mask"].to(pred_x.device)[gt_idx].bool()
            pred = pred_x[bi, pred_idx]
            px1 = pred - radius
            px2 = pred + radius
            gx1 = gt_x - radius
            gx2 = gt_x + radius
            overlap = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
            union = (4.0 * radius - overlap).clamp(min=1e-6)
            valid = mask.to(dtype=pred_x.dtype)
            valid_count = valid.sum(dim=-1)
            qualities = ((overlap / union) * valid).sum(dim=-1) / valid_count.clamp_min(1.0)
            qualities = qualities * (valid_count > 0).to(dtype=qualities.dtype)
            target_quality[bi, pred_idx] = qualities.detach().to(dtype=target_quality.dtype)
        return F.binary_cross_entropy_with_logits(quality_logits, target_quality)

    @torch.no_grad()
    def compute_set_selection_pairwise_quality(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Keep the complete candidate-to-GT quality matrix for set losses."""
        pred_x = outputs["pred_x_rows"].detach().float()
        ranges = outputs["range_norm"].detach().float()
        _batch, candidates, _rows = pred_x.shape
        line_width = float(self.cfg.set_selection_line_width)
        min_valid_rows = int(self.cfg.set_selection_min_valid_rows)
        pairwise_rows: list[torch.Tensor] = []
        for batch_index, target in enumerate(targets):
            gt_x = target["x_rows"].to(
                device=pred_x.device,
                dtype=pred_x.dtype,
            )
            gt_valid = target["valid_mask"].to(pred_x.device).bool()
            if int(gt_x.shape[0]) == 0:
                pairwise_rows.append(pred_x.new_zeros((candidates, 0)))
                continue
            quality, _candidate_valid, _valid_gt = (
                pairwise_range_aware_row_strip_iou(
                    pred_x[batch_index],
                    ranges[batch_index],
                    gt_x,
                    gt_valid,
                    input_h=int(self.cfg.input_h),
                    line_width=line_width,
                    min_valid_rows=min_valid_rows,
                )
            )
            pairwise_rows.append(quality)
        return pairwise_rows

    @torch.no_grad()
    def compute_set_selection_targets(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]] | None = None,
        pairwise_quality: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Build one unique continuous-quality target per ground-truth lane.

        The target is a range-aware row-strip IoU surrogate aligned with the
        30-pixel CULane raster metric.  Assignment deliberately ignores the
        current proposal score: the selection objective must teach which
        geometry is useful instead of reproducing the existing ranking.
        """

        pred_x = outputs["pred_x_rows"].detach().float()
        batch, candidates, _rows = pred_x.shape
        pairwise_rows = (
            self.compute_set_selection_pairwise_quality(outputs, targets)
            if pairwise_quality is None
            else pairwise_quality
        )
        if len(pairwise_rows) != batch:
            raise ValueError("set-selection pairwise quality batch mismatch")

        if bool(self.cfg.set_selection_share_matcher_assignment):
            if matches is None or len(matches) != batch:
                raise ValueError(
                    "set_selection_share_matcher_assignment requires the final "
                    "matcher assignments"
                )
            target = pred_x.new_zeros((batch, candidates))
            for batch_index, (quality, match) in enumerate(
                zip(pairwise_rows, matches)
            ):
                pred_indices = match["pred_indices"].to(pred_x.device)
                gt_indices = match["gt_indices"].to(pred_x.device)
                if pred_indices.numel() > 0:
                    matched_quality = quality[
                        pred_indices,
                        gt_indices,
                    ]
                    positive_floor = float(self.cfg.set_selection_positive_floor)
                    # The assignment identity is supervised even while the
                    # from-scratch geometry is still poor.  As localization
                    # improves, the continuous IoU term raises the target
                    # toward one and supplies the desired quality ordering.
                    target[batch_index, pred_indices] = positive_floor + (
                        1.0 - positive_floor
                    ) * matched_quality
            return target

        # One D2H synchronization for the complete micro-batch, followed by
        # tiny per-image Hungarian solves on CPU and one H2D target transfer.
        nonempty = [row.reshape(-1) for row in pairwise_rows if row.numel() > 0]
        flat_cpu = (
            torch.cat(nonempty, dim=0).cpu()
            if nonempty
            else torch.empty(0, dtype=torch.float32)
        )
        target_cpu = torch.zeros(
            (batch, candidates),
            dtype=torch.float32,
        )
        offset = 0
        for batch_index, quality in enumerate(pairwise_rows):
            if quality.numel() == 0:
                continue
            elements = int(quality.numel())
            quality_cpu = flat_cpu[offset : offset + elements].view_as(quality)
            offset += elements
            pred_indices, gt_indices = HungarianMatcherS0._linear_sum_assignment(
                1.0 - quality_cpu
            )
            if pred_indices.numel() > 0:
                target_cpu[batch_index, pred_indices] = quality_cpu[
                    pred_indices,
                    gt_indices,
                ]
        return target_cpu.to(device=pred_x.device)

    def _compute_relation_set_losses(
        self,
        selection_logits: torch.Tensor,
        pairwise_quality: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Coverage, duplicate suppression, representative quality, and count.

        All quality matrices are detached geometry observations.  Gradients
        therefore update only the scalar set scorer while explicitly teaching
        it to cover distinct GT lanes and select the best localized member of
        each duplicate cluster.
        """

        probability = torch.sigmoid(selection_logits)
        zero = selection_logits.sum() * 0.0
        coverage_rows: list[torch.Tensor] = []
        duplicate_rows: list[torch.Tensor] = []
        winner_rows: list[torch.Tensor] = []
        gt_counts: list[float] = []
        duplicate_min = float(self.cfg.set_selection_duplicate_quality_min)
        winner_min = float(self.cfg.set_selection_winner_quality_min)

        for batch_index, raw_quality in enumerate(pairwise_quality):
            quality = raw_quality.detach().to(
                device=selection_logits.device,
                dtype=selection_logits.dtype,
            )
            candidate_probability = probability[batch_index]
            candidates = int(candidate_probability.numel())
            gt_count = int(quality.shape[1])
            gt_counts.append(float(gt_count))
            if gt_count == 0:
                duplicate_rows.append(zero)
                winner_rows.append(zero)
                continue

            # Probability that at least one selected candidate covers GT j.
            activation = (
                candidate_probability.unsqueeze(-1) * quality
            ).clamp(min=0.0, max=1.0 - 1e-6)
            log_missing = torch.log1p(-activation).sum(dim=0)
            coverage = (1.0 - torch.exp(log_missing)).clamp_min(1e-6)
            coverage_rows.append(-torch.log(coverage).mean())

            # Candidates have duplicate affinity when they both explain the
            # same GT.  Only the upper triangle is counted once.
            duplicate_affinity = torch.minimum(
                quality.unsqueeze(1),
                quality.unsqueeze(0),
            ).amax(dim=-1)
            duplicate_affinity = torch.where(
                duplicate_affinity >= duplicate_min,
                duplicate_affinity,
                torch.zeros_like(duplicate_affinity),
            )
            upper = torch.triu(
                torch.ones(
                    (candidates, candidates),
                    device=selection_logits.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            duplicate_weight = duplicate_affinity * upper.to(
                dtype=duplicate_affinity.dtype
            )
            coactivation = (
                candidate_probability.unsqueeze(1)
                * candidate_probability.unsqueeze(0)
            )
            duplicate_rows.append(
                (coactivation * duplicate_weight).sum()
                / duplicate_weight.sum().clamp_min(1e-6)
            )

            # Within every GT cluster, rank the strict-IoU winner above worse
            # but still plausible duplicate representatives.
            best_quality, best_index = quality.max(dim=0)
            best_logit = selection_logits[batch_index, best_index]
            quality_gap = (best_quality.unsqueeze(0) - quality).clamp_min(0.0)
            eligible = (quality >= winner_min) & (
                best_quality.unsqueeze(0) >= winner_min
            )
            winner_weight = quality_gap * eligible.to(dtype=quality.dtype)
            winner_penalty = F.softplus(
                selection_logits[batch_index].unsqueeze(-1)
                - best_logit.unsqueeze(0)
            )
            winner_rows.append(
                (winner_penalty * winner_weight).sum()
                / winner_weight.sum().clamp_min(1e-6)
            )

        coverage_loss = (
            torch.stack(coverage_rows).mean() if coverage_rows else zero
        )
        duplicate_loss = (
            torch.stack(duplicate_rows).mean() if duplicate_rows else zero
        )
        winner_loss = torch.stack(winner_rows).mean() if winner_rows else zero
        count_target = selection_logits.new_tensor(gt_counts)
        count_loss = F.smooth_l1_loss(
            probability.sum(dim=-1),
            count_target,
        )
        return {
            "coverage": coverage_loss,
            "duplicate": duplicate_loss,
            "winner": winner_loss,
            "count": count_loss,
        }

    def compute_set_selection_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = outputs.get("selection_logits")
        if logits is None:
            raise ValueError(
                "w_set_selection > 0 requires model.structured_query."
                "set_selection.enabled=true"
            )
        selection_logits = logits.float()
        pairwise_quality = self.compute_set_selection_pairwise_quality(
            outputs,
            targets,
        )
        selection_targets = self.compute_set_selection_targets(
            outputs,
            targets,
            matches,
            pairwise_quality=pairwise_quality,
        ).to(dtype=selection_logits.dtype)
        probability = torch.sigmoid(selection_logits)
        modulation = (
            selection_targets - probability
        ).abs().pow(float(self.cfg.set_selection_focal_beta))
        per_candidate_quality = (
            modulation
            * F.binary_cross_entropy_with_logits(
                selection_logits,
                selection_targets,
                reduction="none",
            )
        )
        negative_weight = float(self.cfg.set_selection_negative_weight)
        if negative_weight <= 0.0:
            raise ValueError("set_selection_negative_weight must be positive")
        candidate_weight = torch.where(
            selection_targets > 0.0,
            torch.ones_like(selection_targets),
            torch.full_like(selection_targets, negative_weight),
        )
        quality_loss = (
            per_candidate_quality * candidate_weight
        ).sum() / candidate_weight.sum().clamp_min(1.0)

        target_delta = (
            selection_targets.unsqueeze(-1)
            - selection_targets.unsqueeze(-2)
        )
        pair_weight = (
            target_delta - float(self.cfg.set_selection_target_margin)
        ).clamp_min(0.0)
        logit_delta = (
            selection_logits.unsqueeze(-1)
            - selection_logits.unsqueeze(-2)
        )
        ranking_loss = (
            F.softplus(-logit_delta) * pair_weight.detach()
        ).sum() / pair_weight.sum().clamp_min(1e-6)
        relation_losses = self._compute_relation_set_losses(
            selection_logits,
            pairwise_quality,
        )
        total = quality_loss + float(
            self.cfg.set_selection_rank_weight
        ) * ranking_loss
        total = (
            total
            + float(self.cfg.set_selection_coverage_weight)
            * relation_losses["coverage"]
            + float(self.cfg.set_selection_duplicate_weight)
            * relation_losses["duplicate"]
            + float(self.cfg.set_selection_winner_weight)
            * relation_losses["winner"]
            + float(self.cfg.set_selection_count_weight)
            * relation_losses["count"]
        )
        delta = outputs.get("selection_delta_logits")
        delta_abs = (
            delta.detach().float().abs().mean()
            if isinstance(delta, torch.Tensor)
            else selection_logits.detach().sum() * 0.0
        )
        return {
            "total": total,
            "quality": quality_loss,
            "ranking": ranking_loss,
            "coverage": relation_losses["coverage"],
            "duplicate": relation_losses["duplicate"],
            "winner": relation_losses["winner"],
            "count": relation_losses["count"],
            "target_mean": selection_targets.detach().mean(),
            "target_positive_fraction": (
                selection_targets.detach() > 0.0
            ).float().mean(),
            "delta_abs": delta_abs,
        }

    def compute_four_slot_selection_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Train four object slots while the 32-proposal detector is frozen."""

        logits_value = outputs.get("selection_slot_logits")
        if not isinstance(logits_value, torch.Tensor):
            raise ValueError(
                "w_four_slot_selection > 0 requires four-slot routing logits"
            )
        route_logits = logits_value.float()
        if route_logits.ndim != 3:
            raise ValueError("selection_slot_logits must have shape [B,S,N+1]")
        target_data = build_four_slot_cluster_targets(
            outputs,
            targets,
            num_slots=int(route_logits.shape[1]),
            input_h=int(self.cfg.input_h),
            line_width=float(self.cfg.four_slot_line_width),
            min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
            representable_min=float(self.cfg.four_slot_representable_min),
            cluster_min=float(self.cfg.four_slot_cluster_min),
            cluster_delta=float(self.cfg.four_slot_cluster_delta),
            temperature=float(self.cfg.four_slot_cluster_temperature),
            target_mode=str(self.cfg.four_slot_target_mode),
            padded_targets=padded_targets,
        )
        target_rows = target_data["rows"]
        if not isinstance(target_rows, list):
            raise TypeError("four-slot target builder returned invalid rows")
        active_logits = outputs.get("selection_slot_active_logits")
        real_route_logits = outputs.get("selection_slot_real_route_logits")
        if isinstance(active_logits, torch.Tensor) and isinstance(
            real_route_logits,
            torch.Tensor,
        ):
            permutation_loss = four_slot_factorized_permutation_loss(
                active_logits,
                real_route_logits,
                target_rows,
                permutation_temperature=float(
                    self.cfg.four_slot_permutation_temperature
                ),
                assignment_mode=str(self.cfg.four_slot_assignment_mode),
            )
            # Collision is a conditional real-route regularizer.  Active/no-
            # lane probability is excluded so it cannot become an escape path.
            collision_loss = four_slot_collision_loss(
                real_route_logits,
                has_dustbin=False,
            )
        else:
            permutation_loss = four_slot_permutation_loss(
                route_logits,
                target_rows,
                permutation_temperature=float(
                    self.cfg.four_slot_permutation_temperature
                ),
                assignment_mode=str(self.cfg.four_slot_assignment_mode),
            )
            collision_loss = four_slot_collision_loss(route_logits)
        total = permutation_loss + float(
            self.cfg.four_slot_collision_weight
        ) * collision_loss
        representable = target_data["representable_count"]
        if not isinstance(representable, torch.Tensor):
            raise TypeError("four-slot target builder returned invalid counts")
        raw_collision = outputs.get("selection_slot_raw_collision_count")
        route_entropy = outputs.get("selection_slot_route_entropy")
        zero = total.detach() * 0.0
        return {
            "total": total,
            "permutation": permutation_loss,
            "collision": collision_loss,
            "mean_representable_count": representable.float().mean().detach(),
            "mean_support_size": target_data["mean_support_size"],
            "mean_target_entropy": target_data["mean_entropy"],
            "mean_target_quality": target_data["mean_target_quality"],
            "mean_raw_collision_count": (
                raw_collision.float().mean().detach()
                if isinstance(raw_collision, torch.Tensor)
                else zero
            ),
            "mean_route_entropy": (
                route_entropy.float().mean().detach()
                if isinstance(route_entropy, torch.Tensor)
                else zero
            ),
        }

    @torch.no_grad()
    def _match_four_slot_unified_final(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        list[dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build the sole V11 slot-to-GT assignment from final geometry.

        Activity confidence is intentionally absent from this detached cost.
        It therefore cannot select its own positive target.  The returned
        assignment is reused by activity, proposal-memory attention and every
        final/coarse geometry term.
        """

        refined = outputs.get("selection_slot_pred_x_rows")
        ranges = outputs.get("selection_slot_range_norm")
        geometry_valid = outputs.get("selection_slot_geometry_valid")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (refined, ranges, geometry_valid)
        ):
            raise ValueError(
                "unified four-slot loss requires final geometry and validity"
            )
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=refined.device,
                dtype=torch.float32,
                rows=int(refined.shape[-1]),
            )
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        quality, slot_valid, gt_valid = (
            batched_pairwise_range_aware_row_strip_iou(
                refined.detach().float(),
                ranges.detach().float(),
                padded_gt_x,
                padded_gt_valid,
                input_h=int(self.cfg.input_h),
                line_width=float(self.cfg.four_slot_line_width),
                min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
            )
        )
        pair_valid = (
            geometry_valid.bool() & slot_valid
        ).unsqueeze(-1) & gt_valid.unsqueeze(1)
        cost_batch = (1.0 - quality).masked_fill(
            ~pair_valid,
            torch.inf,
        ).detach().cpu()
        cpu_pairs: list[torch.Tensor] = []
        counts: list[int] = []
        for cost in cost_batch:
            finite = torch.isfinite(cost)
            slot_ids = torch.nonzero(
                finite.any(dim=1),
                as_tuple=False,
            ).flatten()
            gt_ids = torch.nonzero(
                finite.any(dim=0),
                as_tuple=False,
            ).flatten()
            if slot_ids.numel() == 0 or gt_ids.numel() == 0:
                pairs = torch.empty((0, 2), dtype=torch.long)
            else:
                local_cost = cost.index_select(0, slot_ids).index_select(
                    1,
                    gt_ids,
                )
                local_slot, local_gt = (
                    HungarianMatcherS0._linear_sum_assignment(local_cost)
                )
                pairs = torch.stack(
                    (
                        slot_ids.index_select(0, local_slot),
                        gt_ids.index_select(0, local_gt),
                    ),
                    dim=-1,
                )
            cpu_pairs.append(pairs)
            counts.append(int(pairs.shape[0]))
        packed = (
            torch.cat(cpu_pairs, dim=0).to(refined.device)
            if any(counts)
            else torch.empty(
                (0, 2),
                dtype=torch.long,
                device=refined.device,
            )
        )
        matches: list[dict[str, torch.Tensor]] = []
        offset = 0
        for count in counts:
            pairs = packed[offset : offset + count]
            offset += count
            matches.append(
                {
                    "pred_indices": pairs[:, 0],
                    "gt_indices": pairs[:, 1],
                }
            )
        return (
            matches,
            quality,
            refined.new_tensor(counts, dtype=torch.float32),
        )

    @torch.no_grad()
    def _four_slot_unified_proposal_targets(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the V7 near-best proposal distribution for every GT.

        Unlike the old independent selection loss, V11 indexes these rows
        with the *same* final slot-to-GT assignment used by geometry.  The
        soft distribution is appropriate here because proposal attention is
        memory retrieval, not a deployment hard-ID score.
        """

        proposal_x = outputs["pred_x_rows"].detach().float()
        proposal_range = outputs["range_norm"].detach().float()
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=proposal_x.device,
                dtype=torch.float32,
                rows=int(proposal_x.shape[-1]),
            )
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        quality, candidate_valid, gt_valid = (
            batched_pairwise_range_aware_row_strip_iou(
                proposal_x,
                proposal_range,
                padded_gt_x,
                padded_gt_valid,
                input_h=int(self.cfg.input_h),
                line_width=float(self.cfg.four_slot_line_width),
                min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
            )
        )
        if int(quality.shape[-1]) == 0:
            return (
                quality.permute(0, 2, 1),
                torch.zeros_like(quality, dtype=torch.bool).permute(0, 2, 1),
            )
        masked_quality = quality.masked_fill(
            ~candidate_valid.unsqueeze(-1),
            float("-inf"),
        )
        best = masked_quality.amax(dim=1)
        effective_floor = best.clamp(max=float(self.cfg.four_slot_cluster_min))
        cutoff = torch.maximum(
            effective_floor,
            best - float(self.cfg.four_slot_cluster_delta),
        )
        support = (
            candidate_valid.unsqueeze(-1)
            & gt_valid.unsqueeze(1)
            & (quality >= cutoff.unsqueeze(1))
        )
        probability = torch.softmax(
            (quality / float(self.cfg.four_slot_cluster_temperature)).masked_fill(
                ~support,
                float("-inf"),
            ),
            dim=1,
        )
        probability = torch.where(
            support,
            probability,
            torch.zeros_like(probability),
        )
        return probability.permute(0, 2, 1), support.permute(0, 2, 1)

    @torch.no_grad()
    def _match_four_slot_visual_first_anchor(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """Match GTs once against frozen V7 final geometry.

        V12 Stage A must not let a fresh visual prediction choose the target
        that supervises it.  The detached V7 anchor is stable across every
        new association/visual loss and is independent of V12 parameters.
        Activity is intentionally absent so an inactive fourth geometry can
        still represent an annotated lane.
        """

        anchor_x = outputs.get("selection_slot_v12_anchor_x_rows")
        anchor_range = outputs.get("selection_slot_v12_anchor_range_norm")
        geometry_valid = outputs.get("selection_slot_v12_geometry_valid")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (anchor_x, anchor_range, geometry_valid)
        ):
            raise ValueError(
                "visual-first loss requires frozen V7 anchor geometry"
            )
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=anchor_x.device,
                dtype=torch.float32,
                rows=int(anchor_x.shape[-1]),
            )
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        quality, slot_valid, gt_valid = (
            batched_pairwise_range_aware_row_strip_iou(
                anchor_x.detach().float(),
                anchor_range.detach().float(),
                padded_gt_x,
                padded_gt_valid,
                input_h=int(self.cfg.input_h),
                line_width=float(self.cfg.four_slot_line_width),
                min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
            )
        )
        pair_valid = (
            geometry_valid.bool() & slot_valid
        ).unsqueeze(-1) & gt_valid.unsqueeze(1)
        cost_batch = (1.0 - quality).masked_fill(
            ~pair_valid,
            torch.inf,
        ).detach().cpu()
        cpu_pairs: list[torch.Tensor] = []
        counts: list[int] = []
        for cost in cost_batch:
            finite = torch.isfinite(cost)
            slot_ids = torch.nonzero(
                finite.any(dim=1), as_tuple=False
            ).flatten()
            gt_ids = torch.nonzero(
                finite.any(dim=0), as_tuple=False
            ).flatten()
            if slot_ids.numel() == 0 or gt_ids.numel() == 0:
                pairs = torch.empty((0, 2), dtype=torch.long)
            else:
                local_cost = cost.index_select(0, slot_ids).index_select(
                    1, gt_ids
                )
                local_slot, local_gt = (
                    HungarianMatcherS0._linear_sum_assignment(local_cost)
                )
                pairs = torch.stack(
                    (
                        slot_ids.index_select(0, local_slot),
                        gt_ids.index_select(0, local_gt),
                    ),
                    dim=-1,
                )
            cpu_pairs.append(pairs)
            counts.append(int(pairs.shape[0]))
        packed = (
            torch.cat(cpu_pairs, dim=0).to(anchor_x.device)
            if any(counts)
            else torch.empty(
                (0, 2), dtype=torch.long, device=anchor_x.device
            )
        )
        matches: list[dict[str, torch.Tensor]] = []
        offset = 0
        for count in counts:
            pairs = packed[offset : offset + count]
            offset += count
            matches.append(
                {
                    "pred_indices": pairs[:, 0],
                    "gt_indices": pairs[:, 1],
                }
            )
        return matches, quality, anchor_x.new_tensor(counts, dtype=torch.float32)

    def _four_slot_visual_distribution_loss(
        self,
        logits: torch.Tensor,
        matches: list[dict[str, torch.Tensor]],
        padded_gt_x: torch.Tensor,
        padded_gt_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Linearly interpolated x-bin CE and expected-x MAE."""

        x_bins = int(logits.shape[-1])
        if x_bins < 2:
            raise ValueError("visual-first attention needs at least two x bins")
        losses: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []
        x_scale = float(x_bins - 1) / float(max(self.cfg.input_w - 1, 1))
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            gt_ids = match["gt_indices"]
            if pred_ids.numel() == 0:
                continue
            selected_logits = logits[batch_index, pred_ids].float()
            target_x = padded_gt_x[batch_index, gt_ids].float()
            valid = padded_gt_valid[batch_index, gt_ids].bool()
            valid = valid & torch.isfinite(target_x)
            valid = valid & (target_x >= 0.0)
            valid = valid & (
                target_x <= float(max(self.cfg.input_w - 1, 1))
            )
            if not bool(valid.any()):
                continue
            target_bin = (target_x * x_scale).clamp(0.0, float(x_bins - 1))
            lower = target_bin.floor().long()
            upper = (lower + 1).clamp(max=x_bins - 1)
            upper_weight = target_bin - lower.to(target_bin.dtype)
            lower_weight = 1.0 - upper_weight
            log_probability = F.log_softmax(selected_logits, dim=-1)
            lower_log = log_probability.gather(
                -1, lower.unsqueeze(-1)
            ).squeeze(-1)
            upper_log = log_probability.gather(
                -1, upper.unsqueeze(-1)
            ).squeeze(-1)
            row_loss = -(
                lower_weight * lower_log + upper_weight * upper_log
            )
            losses.append(row_loss[valid])
            probability = torch.softmax(selected_logits, dim=-1)
            bin_position = torch.linspace(
                0.0,
                float(max(self.cfg.input_w - 1, 1)),
                x_bins,
                device=logits.device,
                dtype=probability.dtype,
            )
            expected_x = torch.einsum("mrx,x->mr", probability, bin_position)
            errors.append((expected_x - target_x).abs()[valid])
        if not losses:
            zero = logits.sum() * 0.0
            return zero, zero.detach()
        return torch.cat(losses).mean(), torch.cat(errors).mean().detach()

    def compute_four_slot_visual_first_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """V12 Stage-A image-first association objective."""

        first_logits = outputs.get("selection_slot_v12_first_visual_logits")
        visual_logits = outputs.get("selection_slot_v12_visual_logits")
        proposal_attention = outputs.get(
            "selection_slot_v12_proposal_attention"
        )
        if not all(
            isinstance(value, torch.Tensor)
            for value in (first_logits, visual_logits, proposal_attention)
        ):
            raise ValueError(
                "w_four_slot_visual_first > 0 requires V12 outputs"
            )
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=visual_logits.device,
                dtype=torch.float32,
                rows=int(visual_logits.shape[-2]),
            )
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        matches, anchor_quality, match_count = (
            self._match_four_slot_visual_first_anchor(
                outputs, targets, (padded_gt_x, padded_gt_valid)
            )
        )
        first_visual, first_mae = self._four_slot_visual_distribution_loss(
            first_logits,
            matches,
            padded_gt_x,
            padded_gt_valid,
        )
        final_visual, final_mae = self._four_slot_visual_distribution_loss(
            visual_logits,
            matches,
            padded_gt_x,
            padded_gt_valid,
        )

        proposal_target, proposal_support = (
            self._four_slot_unified_proposal_targets(
                outputs,
                targets,
                (padded_gt_x, padded_gt_valid),
            )
        )
        proposal_losses: list[torch.Tensor] = []
        support_masses: list[torch.Tensor] = []
        top1_rows: list[torch.Tensor] = []
        matched_anchor: list[torch.Tensor] = []
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            gt_ids = match["gt_indices"]
            if pred_ids.numel() == 0:
                continue
            predicted = proposal_attention[batch_index, pred_ids].float()
            target_probability = proposal_target[
                batch_index, gt_ids
            ].to(predicted.dtype)
            support = proposal_support[batch_index, gt_ids]
            proposal_losses.append(
                -(
                    target_probability
                    * predicted.clamp_min(1.0e-12).log()
                ).sum(dim=-1)
            )
            support_masses.append(
                (predicted * support.to(predicted.dtype)).sum(dim=-1)
            )
            top1_rows.append(
                (
                    predicted.argmax(dim=-1)
                    == target_probability.argmax(dim=-1)
                ).float()
            )
            matched_anchor.append(
                anchor_quality[batch_index, pred_ids, gt_ids]
            )
        if proposal_losses:
            proposal_loss = torch.cat(proposal_losses).mean()
            mean_support_mass = torch.cat(support_masses).mean().detach()
            mean_target_top1 = torch.cat(top1_rows).mean().detach()
            mean_anchor_quality = torch.cat(matched_anchor).mean().detach()
        else:
            proposal_loss = proposal_attention.sum() * 0.0
            mean_support_mass = proposal_loss.detach()
            mean_target_top1 = proposal_loss.detach()
            mean_anchor_quality = proposal_loss.detach()

        total = (
            float(self.cfg.four_slot_visual_first_first_pass_weight)
            * first_visual
            + float(self.cfg.four_slot_visual_first_final_pass_weight)
            * final_visual
            + float(self.cfg.four_slot_visual_first_proposal_weight)
            * proposal_loss
        )
        return {
            "total": total,
            "first_visual": first_visual,
            "final_visual": final_visual,
            "proposal": proposal_loss,
            "mean_matched": match_count.mean().detach(),
            "mean_anchor_quality": mean_anchor_quality,
            "mean_target_support_mass": mean_support_mass,
            "mean_target_top1": mean_target_top1,
            "mean_first_visual_mae_px": first_mae,
            "mean_final_visual_mae_px": final_mae,
        }

    @torch.no_grad()
    def _match_four_slot_v14_anchor(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """One score-independent assignment from deployable V7 slots to GT."""

        anchor_x = outputs.get("selection_slot_v14_anchor_x_rows")
        anchor_range = outputs.get("selection_slot_v14_anchor_range_norm")
        writer_valid = outputs.get("selection_slot_v14_writer_valid")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (anchor_x, anchor_range, writer_valid)
        ):
            raise ValueError("V14 loss requires frozen deployable V7 anchors")
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=anchor_x.device,
                dtype=torch.float32,
                rows=int(anchor_x.shape[-1]),
            )
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        quality, slot_valid, gt_valid = (
            batched_pairwise_range_aware_row_strip_iou(
                anchor_x.detach().float(),
                anchor_range.detach().float(),
                padded_gt_x,
                padded_gt_valid,
                input_h=int(self.cfg.input_h),
                line_width=float(self.cfg.four_slot_line_width),
                min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
            )
        )
        pair_valid = (
            writer_valid.detach().bool() & slot_valid
        ).unsqueeze(-1) & gt_valid.unsqueeze(1)
        cost_batch = (1.0 - quality).masked_fill(~pair_valid, torch.inf)
        matches: list[dict[str, torch.Tensor]] = []
        counts: list[int] = []
        for cost in cost_batch.detach().cpu():
            finite = torch.isfinite(cost)
            slot_ids = torch.nonzero(
                finite.any(dim=1), as_tuple=False
            ).flatten()
            gt_ids = torch.nonzero(
                finite.any(dim=0), as_tuple=False
            ).flatten()
            if slot_ids.numel() == 0 or gt_ids.numel() == 0:
                pairs = torch.empty((0, 2), dtype=torch.long)
            else:
                local = cost.index_select(0, slot_ids).index_select(1, gt_ids)
                local_slot, local_gt = (
                    HungarianMatcherS0._linear_sum_assignment(local)
                )
                pairs = torch.stack(
                    (
                        slot_ids.index_select(0, local_slot),
                        gt_ids.index_select(0, local_gt),
                    ),
                    dim=-1,
                )
            matches.append(
                {
                    "pred_indices": pairs[:, 0].to(anchor_x.device),
                    "gt_indices": pairs[:, 1].to(anchor_x.device),
                }
            )
            counts.append(int(pairs.shape[0]))
        return (
            matches,
            quality,
            anchor_x.new_tensor(counts, dtype=torch.float32),
        )

    def _four_slot_v14_joint_targets(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build capacity-compatible proposal/private-dustbin targets."""

        from dynlaneseq_eg.modeling.four_slot_selection import (
            structured_unique_route_marginals_with_private_dustbins,
        )

        proposal_x = outputs["pred_x_rows"].detach().float()
        proposal_range = outputs["range_norm"].detach().float()
        candidate_valid = outputs["selection_slot_candidate_valid"].detach().bool()
        predicted = outputs["selection_slot_v14_proposal_attention"]
        batch, slots, classes = predicted.shape
        candidates = int(proposal_x.shape[1])
        if int(classes) != candidates + int(slots):
            raise ValueError("V14 proposal attention must include private dustbins")
        padded_gt_x, padded_gt_valid = padded_targets
        quality, quality_candidate_valid, gt_valid = (
            batched_pairwise_range_aware_row_strip_iou(
                proposal_x,
                proposal_range,
                padded_gt_x,
                padded_gt_valid,
                input_h=int(self.cfg.input_h),
                line_width=float(self.cfg.four_slot_line_width),
                min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
            )
        )
        candidate_valid = candidate_valid & quality_candidate_valid
        real_target_logits = proposal_x.new_full(
            (batch, slots, candidates),
            -30.0,
        )
        dustbin_target_logits = proposal_x.new_zeros((batch, slots))
        association_loss_valid = torch.ones(
            (batch, slots),
            device=proposal_x.device,
            dtype=torch.bool,
        )
        representable_match = torch.zeros_like(association_loss_valid)
        target_support = torch.zeros(
            (batch, slots, candidates),
            device=proposal_x.device,
            dtype=torch.bool,
        )
        target_id = torch.full(
            (batch, slots),
            -1,
            device=proposal_x.device,
            dtype=torch.long,
        )
        matched_slot = torch.zeros_like(association_loss_valid)

        delta = float(self.cfg.four_slot_v14_cluster_delta)
        temperature = float(self.cfg.four_slot_v14_cluster_temperature)
        representable_min = float(self.cfg.four_slot_v14_representable_min)
        if temperature <= 0.0:
            raise ValueError("V14 target temperature must be positive")
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            gt_ids = match["gt_indices"]
            for slot_tensor, gt_tensor in zip(pred_ids, gt_ids):
                slot_id = int(slot_tensor)
                gt_id = int(gt_tensor)
                matched_slot[batch_index, slot_id] = True
                valid = candidate_valid[batch_index]
                gt_quality = quality[batch_index, :, gt_id]
                if not bool(gt_valid[batch_index, gt_id]) or not bool(valid.any()):
                    association_loss_valid[batch_index, slot_id] = False
                    continue
                masked = gt_quality.masked_fill(~valid, float("-inf"))
                best_quality, best_id = masked.max(dim=0)
                if float(best_quality) < representable_min:
                    # The slot remains a real deployed lane and receives its
                    # visual row target, but the frozen proposal pool cannot
                    # supply a defensible association target.
                    association_loss_valid[batch_index, slot_id] = False
                    continue
                support = valid & (gt_quality >= best_quality - delta)
                real_target_logits[batch_index, slot_id] = (
                    gt_quality / temperature
                ).masked_fill(~support, -30.0)
                dustbin_target_logits[batch_index, slot_id] = -30.0
                representable_match[batch_index, slot_id] = True
                target_support[batch_index, slot_id] = support
                target_id[batch_index, slot_id] = best_id

        # Unmatched slots target their own dustbin.  Matched but
        # unrepresentable slots use a dustbin row only to complete the target
        # transport and are excluded from the association loss/metrics.
        target_attention = (
            structured_unique_route_marginals_with_private_dustbins(
                real_target_logits.detach(),
                candidate_valid,
                dustbin_target_logits.detach(),
                temperature=1.0,
                # Near-hard target supports converge more slowly than the
                # learned soft transport.  This is target construction only
                # (no backward graph), so use a strict fixed iteration count.
                iterations=512,
            ).detach()
        )
        return {
            "attention": target_attention,
            "association_loss_valid": association_loss_valid,
            "representable_match": representable_match,
            "target_support": target_support,
            "target_id": target_id,
            "matched_slot": matched_slot,
            "quality": quality,
        }

    def compute_four_slot_v14_stage_a_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Corrected visual-first identifiability objective."""

        visual_logits = outputs.get("selection_slot_v14_visual_logits")
        predicted_attention = outputs.get(
            "selection_slot_v14_proposal_attention"
        )
        if not all(
            isinstance(value, torch.Tensor)
            for value in (visual_logits, predicted_attention)
        ):
            raise ValueError("w_four_slot_v14_stage_a requires V14 outputs")
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=visual_logits.device,
                dtype=torch.float32,
                rows=int(visual_logits.shape[-2]),
            )
            padded_targets = (padded_gt_x, padded_gt_valid)
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        matches, anchor_quality, match_count = self._match_four_slot_v14_anchor(
            outputs,
            targets,
            padded_targets,
        )
        visual_loss, visual_mae = self._four_slot_visual_distribution_loss(
            visual_logits,
            matches,
            padded_gt_x,
            padded_gt_valid,
        )
        target = self._four_slot_v14_joint_targets(
            outputs,
            targets,
            matches,
            padded_targets,
        )
        target_attention = target["attention"].to(predicted_attention.dtype)
        log_predicted = predicted_attention.float().clamp_min(1.0e-12).log()
        log_target = target_attention.float().clamp_min(1.0e-12).log()
        per_slot_kl = (
            target_attention.float() * (log_target - log_predicted)
        ).sum(dim=-1)
        association_valid = target["association_loss_valid"]
        association_loss = (
            per_slot_kl[association_valid].mean()
            if bool(association_valid.any())
            else predicted_attention.sum() * 0.0
        )

        representable = target["representable_match"]
        real_attention = predicted_attention[..., : target["target_support"].shape[-1]]
        support_mass = (
            real_attention
            * target["target_support"].to(real_attention.dtype)
        ).sum(dim=-1)
        predicted_id = real_attention.argmax(dim=-1)
        v7_logits = outputs["selection_slot_real_route_logits"].detach().float()
        v7_probability = torch.softmax(v7_logits, dim=-1)
        v7_support_mass = (
            v7_probability
            * target["target_support"].to(v7_probability.dtype)
        ).sum(dim=-1)
        v7_id = v7_logits.argmax(dim=-1)
        if bool(representable.any()):
            mean_support_mass = support_mass[representable].mean().detach()
            mean_v7_support_mass = v7_support_mass[representable].mean().detach()
            mean_top1 = (
                predicted_id[representable] == target["target_id"][representable]
            ).float().mean().detach()
            mean_v7_top1 = (
                v7_id[representable] == target["target_id"][representable]
            ).float().mean().detach()
            mean_anchor_quality = torch.cat(
                [
                    anchor_quality[batch_index, match["pred_indices"], match["gt_indices"]]
                    for batch_index, match in enumerate(matches)
                    if match["pred_indices"].numel()
                ]
            ).mean().detach()
        else:
            mean_support_mass = association_loss.detach() * 0.0
            mean_v7_support_mass = association_loss.detach() * 0.0
            mean_top1 = association_loss.detach() * 0.0
            mean_v7_top1 = association_loss.detach() * 0.0
            mean_anchor_quality = association_loss.detach() * 0.0
        total = (
            float(self.cfg.four_slot_v14_visual_weight) * visual_loss
            + float(self.cfg.four_slot_v14_association_weight)
            * association_loss
        )
        return {
            "total": total,
            "visual": visual_loss,
            "association": association_loss,
            "mean_matched": match_count.mean().detach(),
            "mean_representable": representable.float().sum(dim=-1).mean().detach(),
            "mean_anchor_quality": mean_anchor_quality,
            "mean_target_support_mass": mean_support_mass,
            "mean_v7_target_support_mass": mean_v7_support_mass,
            "mean_target_top1": mean_top1,
            "mean_v7_target_top1": mean_v7_top1,
            "mean_visual_mae_px": visual_mae,
            "target_attention_row_error": (
                target_attention.sum(dim=-1) - 1.0
            ).abs().max().detach(),
            "target_real_column_excess": (
                target_attention[..., : real_attention.shape[-1]].sum(dim=1)
                - 1.0
            ).clamp_min(0.0).max().detach(),
        }

    def compute_four_slot_visual_precision_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Train V13 slot-owned geometry under the stable V7 assignment."""

        refined = outputs.get("selection_slot_pred_x_rows")
        ranges = outputs.get("selection_slot_range_norm")
        delta_logits = outputs.get("selection_slot_row_delta_logits")
        delta_offsets = outputs.get("selection_slot_row_delta_offsets_px")
        reference = outputs.get("selection_slot_v13_anchor_x_rows")
        candidate_gate = outputs.get("selection_slot_v13_candidate_gate")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                refined,
                ranges,
                delta_logits,
                delta_offsets,
                reference,
                candidate_gate,
            )
        ):
            raise ValueError(
                "w_four_slot_visual_precision > 0 requires V13 outputs"
            )
        matches, anchor_quality_batch, match_count = (
            self._match_four_slot_visual_first_anchor(
                outputs,
                targets,
                padded_targets,
            )
        )
        slot_outputs = {
            "pred_x_rows": refined,
            "range_norm": ranges,
            "row_x_logits": delta_logits,
            "row_x_offsets_px": delta_offsets,
            "input_reference_x_rows": reference,
        }
        point = self.compute_point_loss(slot_outputs, targets, matches)
        range_loss = self.compute_range_loss(slot_outputs, targets, matches)
        line_iou = self.compute_line_iou_loss(slot_outputs, targets, matches)
        dfl = self.compute_row_dfl_loss(slot_outputs, targets, matches)
        total = (
            float(self.cfg.four_slot_visual_precision_point_weight) * point
            + float(self.cfg.four_slot_visual_precision_range_weight)
            * range_loss
            + float(self.cfg.four_slot_visual_precision_line_iou_weight)
            * line_iou
            + float(self.cfg.four_slot_visual_precision_dfl_weight) * dfl
        )

        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=refined.device,
                dtype=torch.float32,
                rows=int(refined.shape[-1]),
            )
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        with torch.no_grad():
            final_quality_batch, _slot_valid, _gt_valid = (
                batched_pairwise_range_aware_row_strip_iou(
                    refined.detach().float(),
                    ranges.detach().float(),
                    padded_gt_x,
                    padded_gt_valid,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.four_slot_line_width),
                    min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
                )
            )
        anchor_rows: list[torch.Tensor] = []
        final_rows: list[torch.Tensor] = []
        delta_rows: list[torch.Tensor] = []
        gate_rows: list[torch.Tensor] = []
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            gt_ids = match["gt_indices"]
            if pred_ids.numel() == 0:
                continue
            anchor_rows.append(
                anchor_quality_batch[batch_index, pred_ids, gt_ids]
            )
            final_rows.append(
                final_quality_batch[batch_index, pred_ids, gt_ids]
            )
            delta_rows.append(
                (
                    refined[batch_index, pred_ids].float()
                    - reference[batch_index, pred_ids].float()
                ).abs().mean(dim=-1)
            )
            gate_rows.append(
                candidate_gate[batch_index, pred_ids].float().abs().mean(
                    dim=-1
                )
            )
        zero = total.detach() * 0.0
        mean_anchor = (
            torch.cat(anchor_rows).mean().detach() if anchor_rows else zero
        )
        mean_final = (
            torch.cat(final_rows).mean().detach() if final_rows else zero
        )
        mean_delta = (
            torch.cat(delta_rows).mean().detach() if delta_rows else zero
        )
        mean_gate = (
            torch.cat(gate_rows).mean().detach() if gate_rows else zero
        )
        return {
            "total": total,
            "point": point,
            "range": range_loss,
            "line_iou": line_iou,
            "dfl": dfl,
            "mean_matched": match_count.mean().detach(),
            "mean_anchor_quality": mean_anchor,
            "mean_final_quality": mean_final,
            "mean_quality_gain": (mean_final - mean_anchor).detach(),
            "mean_abs_delta_px": mean_delta,
            "mean_candidate_gate": mean_gate,
        }

    def compute_four_slot_v14_stage_b_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Parity-anchored geometry under the immutable V7 assignment."""

        refined = outputs.get("selection_slot_pred_x_rows")
        ranges = outputs.get("selection_slot_range_norm")
        delta_logits = outputs.get("selection_slot_row_delta_logits")
        delta_offsets = outputs.get("selection_slot_row_delta_offsets_px")
        anchor_x = outputs.get("selection_slot_v14_stage_b_anchor_x_rows")
        anchor_range = outputs.get(
            "selection_slot_v14_stage_b_anchor_range_norm"
        )
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                refined,
                ranges,
                delta_logits,
                delta_offsets,
                anchor_x,
                anchor_range,
            )
        ):
            raise ValueError("w_four_slot_v14_stage_b requires Stage-B outputs")
        matches, anchor_quality_batch, match_count = (
            self._match_four_slot_v14_anchor(
                outputs,
                targets,
                padded_targets,
            )
        )
        slot_outputs = {
            "pred_x_rows": refined,
            "range_norm": ranges,
            "row_x_logits": delta_logits,
            "row_x_offsets_px": delta_offsets,
            "input_reference_x_rows": anchor_x,
        }
        point = self.compute_point_loss(slot_outputs, targets, matches)
        range_loss = self.compute_range_loss(slot_outputs, targets, matches)
        line_iou = self.compute_line_iou_loss(slot_outputs, targets, matches)
        dfl = self.compute_row_dfl_loss(slot_outputs, targets, matches)
        total = (
            float(self.cfg.four_slot_v14_stage_b_point_weight) * point
            + float(self.cfg.four_slot_v14_stage_b_range_weight) * range_loss
            + float(self.cfg.four_slot_v14_stage_b_line_iou_weight) * line_iou
            + float(self.cfg.four_slot_v14_stage_b_dfl_weight) * dfl
        )
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=refined.device,
                dtype=torch.float32,
                rows=int(refined.shape[-1]),
            )
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        with torch.no_grad():
            final_quality_batch, _slot_valid, _gt_valid = (
                batched_pairwise_range_aware_row_strip_iou(
                    refined.detach().float(),
                    ranges.detach().float(),
                    padded_gt_x,
                    padded_gt_valid,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.four_slot_line_width),
                    min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
                )
            )
        anchor_rows: list[torch.Tensor] = []
        final_rows: list[torch.Tensor] = []
        delta_rows: list[torch.Tensor] = []
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            gt_ids = match["gt_indices"]
            if pred_ids.numel() == 0:
                continue
            anchor_rows.append(
                anchor_quality_batch[batch_index, pred_ids, gt_ids]
            )
            final_rows.append(
                final_quality_batch[batch_index, pred_ids, gt_ids]
            )
            delta_rows.append(
                (
                    refined[batch_index, pred_ids].float()
                    - anchor_x[batch_index, pred_ids].float()
                )
                .abs()
                .mean(dim=-1)
            )
        zero = total.detach() * 0.0
        mean_anchor = torch.cat(anchor_rows).mean().detach() if anchor_rows else zero
        mean_final = torch.cat(final_rows).mean().detach() if final_rows else zero
        mean_delta = torch.cat(delta_rows).mean().detach() if delta_rows else zero
        return {
            "total": total,
            "point": point,
            "range": range_loss,
            "line_iou": line_iou,
            "dfl": dfl,
            "mean_matched": match_count.mean().detach(),
            "mean_anchor_quality": mean_anchor,
            "mean_final_quality": mean_final,
            "mean_quality_gain": (mean_final - mean_anchor).detach(),
            "mean_abs_delta_px": mean_delta,
        }

    def compute_four_slot_v15_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Visual localization plus parity-anchored slot-owned geometry.

        V15 intentionally has no proposal-ID or cluster target.  The proposal
        graph is optimized only insofar as its contextual messages improve the
        final geometry under the immutable source assignment.
        """

        visual_logits = outputs.get("selection_slot_v15_visual_logits")
        refined = outputs.get("selection_slot_pred_x_rows")
        ranges = outputs.get("selection_slot_range_norm")
        delta_logits = outputs.get("selection_slot_row_delta_logits")
        delta_offsets = outputs.get("selection_slot_row_delta_offsets_px")
        anchor_x = outputs.get("selection_slot_v15_anchor_x_rows")
        anchor_range = outputs.get("selection_slot_v15_anchor_range_norm")
        writer_valid = outputs.get("selection_slot_v15_writer_valid")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                visual_logits,
                refined,
                ranges,
                delta_logits,
                delta_offsets,
                anchor_x,
                anchor_range,
                writer_valid,
            )
        ):
            raise ValueError("w_four_slot_v15 requires complete V15 outputs")
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=refined.device,
                dtype=torch.float32,
                rows=int(refined.shape[-1]),
            )
            padded_targets = (padded_gt_x, padded_gt_valid)
        else:
            padded_gt_x, padded_gt_valid = padded_targets

        # Reuse the audited V14 source-active/writer-valid matcher contract by
        # supplying only immutable V15 anchor aliases.  No learned V15 tensor
        # or score participates in assignment.
        anchor_view = dict(outputs)
        anchor_view["selection_slot_v14_anchor_x_rows"] = anchor_x
        anchor_view["selection_slot_v14_anchor_range_norm"] = anchor_range
        anchor_view["selection_slot_v14_writer_valid"] = writer_valid
        matches, anchor_quality_batch, match_count = (
            self._match_four_slot_v14_anchor(
                anchor_view,
                targets,
                padded_targets,
            )
        )
        visual, _visual_mae = self._four_slot_visual_distribution_loss(
            visual_logits,
            matches,
            padded_gt_x,
            padded_gt_valid,
        )
        slot_outputs = {
            "pred_x_rows": refined,
            "range_norm": ranges,
            "row_x_logits": delta_logits,
            "row_x_offsets_px": delta_offsets,
            "input_reference_x_rows": anchor_x,
        }
        point = self.compute_point_loss(slot_outputs, targets, matches)
        range_loss = self.compute_range_loss(slot_outputs, targets, matches)
        line_iou = self.compute_line_iou_loss(slot_outputs, targets, matches)
        dfl = self.compute_row_dfl_loss(slot_outputs, targets, matches)
        total = (
            float(self.cfg.four_slot_v15_visual_weight) * visual
            + float(self.cfg.four_slot_v15_point_weight) * point
            + float(self.cfg.four_slot_v15_range_weight) * range_loss
            + float(self.cfg.four_slot_v15_line_iou_weight) * line_iou
            + float(self.cfg.four_slot_v15_dfl_weight) * dfl
        )

        with torch.no_grad():
            final_quality_batch, _slot_valid, _gt_valid = (
                batched_pairwise_range_aware_row_strip_iou(
                    refined.detach().float(),
                    ranges.detach().float(),
                    padded_gt_x,
                    padded_gt_valid,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.four_slot_line_width),
                    min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
                )
            )
        anchor_rows: list[torch.Tensor] = []
        final_rows: list[torch.Tensor] = []
        delta_rows: list[torch.Tensor] = []
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            gt_ids = match["gt_indices"]
            if pred_ids.numel() == 0:
                continue
            anchor_rows.append(
                anchor_quality_batch[batch_index, pred_ids, gt_ids]
            )
            final_rows.append(
                final_quality_batch[batch_index, pred_ids, gt_ids]
            )
            delta_rows.append(
                (
                    refined[batch_index, pred_ids].float()
                    - anchor_x[batch_index, pred_ids].float()
                )
                .abs()
                .mean(dim=-1)
            )
        zero = total.detach() * 0.0
        mean_anchor = torch.cat(anchor_rows).mean().detach() if anchor_rows else zero
        mean_final = torch.cat(final_rows).mean().detach() if final_rows else zero
        mean_delta = torch.cat(delta_rows).mean().detach() if delta_rows else zero
        return {
            "total": total,
            "visual": visual,
            "point": point,
            "range": range_loss,
            "line_iou": line_iou,
            "dfl": dfl,
            "mean_matched": match_count.mean().detach(),
            "mean_anchor_quality": mean_anchor,
            "mean_final_quality": mean_final,
            "mean_quality_gain": (mean_final - mean_anchor).detach(),
            "mean_abs_delta_px": mean_delta,
        }

    def compute_four_slot_v17_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Three re-centered slot-owned stages under immutable V7 identity."""

        stage_x = outputs.get("selection_slot_v17_stage_x_rows")
        stage_range = outputs.get("selection_slot_v17_stage_range_norm")
        stage_input_x = outputs.get("selection_slot_v17_stage_input_x_rows")
        stage_delta_logits = outputs.get(
            "selection_slot_v17_stage_delta_logits"
        )
        stage_visual_logits = outputs.get(
            "selection_slot_v17_stage_visual_logits"
        )
        visual_offsets = outputs.get("selection_slot_v17_visual_offsets_px")
        delta_offsets = outputs.get("selection_slot_v17_delta_offsets_px")
        anchor_x = outputs.get("selection_slot_v17_anchor_x_rows")
        anchor_range = outputs.get("selection_slot_v17_anchor_range_norm")
        writer_valid = outputs.get("selection_slot_v17_writer_valid")
        required = (
            stage_x,
            stage_range,
            stage_input_x,
            stage_delta_logits,
            stage_visual_logits,
            visual_offsets,
            delta_offsets,
            anchor_x,
            anchor_range,
            writer_valid,
        )
        if not all(isinstance(value, torch.Tensor) for value in required):
            raise ValueError("w_four_slot_v17 requires complete V17 outputs")
        stages = int(stage_x.shape[1])
        stage_weights = tuple(float(value) for value in self.cfg.four_slot_v17_stage_weights)
        if len(stage_weights) != stages or any(value <= 0.0 for value in stage_weights):
            raise ValueError("V17 stage weights must be positive and match stages")
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=stage_x.device,
                dtype=torch.float32,
                rows=int(stage_x.shape[-1]),
            )
            padded_targets = (padded_gt_x, padded_gt_valid)
        else:
            padded_gt_x, padded_gt_valid = padded_targets

        anchor_view = dict(outputs)
        anchor_view["selection_slot_v14_anchor_x_rows"] = anchor_x
        anchor_view["selection_slot_v14_anchor_range_norm"] = anchor_range
        anchor_view["selection_slot_v14_writer_valid"] = writer_valid
        matches, anchor_quality_batch, match_count = (
            self._match_four_slot_v14_anchor(
                anchor_view,
                targets,
                padded_targets,
            )
        )

        weighted_visual = anchor_x.sum() * 0.0
        weighted_point = weighted_visual
        weighted_range = weighted_visual
        weighted_line_iou = weighted_visual
        weighted_dfl = weighted_visual
        for stage_index, stage_weight in enumerate(stage_weights):
            common = {
                "pred_x_rows": stage_x[:, stage_index],
                "range_norm": stage_range[:, stage_index],
                "input_reference_x_rows": stage_input_x[:, stage_index],
            }
            visual_outputs = dict(common)
            visual_outputs["row_x_logits"] = stage_visual_logits[:, stage_index]
            visual_outputs["row_x_offsets_px"] = visual_offsets
            geometry_outputs = dict(common)
            geometry_outputs["row_x_logits"] = stage_delta_logits[:, stage_index]
            geometry_outputs["row_x_offsets_px"] = delta_offsets

            visual = self.compute_row_dfl_loss(
                visual_outputs, targets, matches
            )
            point = self.compute_point_loss(
                geometry_outputs, targets, matches
            )
            range_loss = self.compute_range_loss(
                geometry_outputs, targets, matches
            )
            line_iou = self.compute_line_iou_loss(
                geometry_outputs, targets, matches
            )
            dfl = self.compute_row_dfl_loss(
                geometry_outputs, targets, matches
            )
            weighted_visual = weighted_visual + stage_weight * visual
            weighted_point = weighted_point + stage_weight * point
            weighted_range = weighted_range + stage_weight * range_loss
            weighted_line_iou = weighted_line_iou + stage_weight * line_iou
            weighted_dfl = weighted_dfl + stage_weight * dfl

        total = (
            float(self.cfg.four_slot_v17_visual_weight) * weighted_visual
            + float(self.cfg.four_slot_v17_point_weight) * weighted_point
            + float(self.cfg.four_slot_v17_range_weight) * weighted_range
            + float(self.cfg.four_slot_v17_line_iou_weight)
            * weighted_line_iou
            + float(self.cfg.four_slot_v17_dfl_weight) * weighted_dfl
        )
        with torch.no_grad():
            final_quality_batch, _slot_valid, _gt_valid = (
                batched_pairwise_range_aware_row_strip_iou(
                    stage_x[:, -1].detach().float(),
                    stage_range[:, -1].detach().float(),
                    padded_gt_x,
                    padded_gt_valid,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.four_slot_line_width),
                    min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
                )
            )
        anchor_rows: list[torch.Tensor] = []
        final_rows: list[torch.Tensor] = []
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            gt_ids = match["gt_indices"]
            if pred_ids.numel() == 0:
                continue
            anchor_rows.append(
                anchor_quality_batch[batch_index, pred_ids, gt_ids]
            )
            final_rows.append(
                final_quality_batch[batch_index, pred_ids, gt_ids]
            )
        zero = total.detach() * 0.0
        mean_anchor = torch.cat(anchor_rows).mean().detach() if anchor_rows else zero
        mean_final = torch.cat(final_rows).mean().detach() if final_rows else zero
        mean_delta = outputs[
            "selection_slot_v17_stage_delta_x_rows"
        ].detach().abs().mean()
        return {
            "total": total,
            "visual": weighted_visual,
            "point": weighted_point,
            "range": weighted_range,
            "line_iou": weighted_line_iou,
            "dfl": weighted_dfl,
            "mean_matched": match_count.mean().detach(),
            "mean_anchor_quality": mean_anchor,
            "mean_final_quality": mean_final,
            "mean_quality_gain": (mean_final - mean_anchor).detach(),
            "mean_abs_delta_px": mean_delta,
        }

    def compute_four_slot_v16_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Directly learn one coherent proposal quality order per V7 slot.

        V16 geometry is target-free and only defines the variable candidate
        mask.  A frozen V7 writer-valid assignment supplies the GT identity.
        Regression calibrates every local member, pairwise logistic loss
        orders meaningfully different members, and hard CE is used only when
        the best member has a defensible quality margin.  Near ties are never
        forced into an arbitrary exact-ID classification.
        """

        scores = outputs.get("selection_slot_v16_candidate_scores")
        group_mask = outputs.get("selection_slot_v16_group_mask")
        selected_ids = outputs.get("selection_slot_v16_selected_indices")
        anchor_ids = outputs.get("selection_slot_v16_anchor_indices")
        anchor_x = outputs.get("selection_slot_v16_anchor_x_rows")
        anchor_range = outputs.get("selection_slot_v16_anchor_range_norm")
        writer_valid = outputs.get("selection_slot_v16_writer_valid")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                scores,
                group_mask,
                selected_ids,
                anchor_ids,
                anchor_x,
                anchor_range,
                writer_valid,
            )
        ):
            raise ValueError("w_four_slot_v16 requires complete V16 outputs")
        proposal_x = outputs["pred_x_rows"].detach().float()
        proposal_range = outputs["range_norm"].detach().float()
        candidate_valid = outputs["selection_slot_candidate_valid"].detach().bool()
        batch, slots, candidates = scores.shape
        if tuple(group_mask.shape) != (batch, slots, candidates):
            raise ValueError("V16 group mask shape mismatch")

        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=scores.device,
                dtype=torch.float32,
                rows=int(proposal_x.shape[-1]),
            )
            padded_targets = (padded_gt_x, padded_gt_valid)
        else:
            padded_gt_x, padded_gt_valid = padded_targets

        anchor_view = dict(outputs)
        anchor_view["selection_slot_v14_anchor_x_rows"] = anchor_x
        anchor_view["selection_slot_v14_anchor_range_norm"] = anchor_range
        anchor_view["selection_slot_v14_writer_valid"] = writer_valid
        matches, _anchor_slot_quality, match_count = (
            self._match_four_slot_v14_anchor(
                anchor_view,
                targets,
                padded_targets,
            )
        )
        with torch.no_grad():
            quality, quality_candidate_valid, gt_valid = (
                batched_pairwise_range_aware_row_strip_iou(
                    proposal_x,
                    proposal_range,
                    padded_gt_x,
                    padded_gt_valid,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.four_slot_line_width),
                    min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
                )
            )
        candidate_valid = candidate_valid & quality_candidate_valid

        quality_losses: list[torch.Tensor] = []
        pairwise_losses: list[torch.Tensor] = []
        hard_losses: list[torch.Tensor] = []
        anchor_rows: list[torch.Tensor] = []
        selected_rows: list[torch.Tensor] = []
        oracle_rows: list[torch.Tensor] = []
        group_sizes: list[torch.Tensor] = []
        top1_rows: list[torch.Tensor] = []
        representable_counts = scores.new_zeros((batch,), dtype=torch.float32)
        representable_min = float(self.cfg.four_slot_v16_representable_min)
        pair_margin = float(self.cfg.four_slot_v16_pair_margin)
        hard_margin = float(self.cfg.four_slot_v16_hard_margin)
        for batch_index, match in enumerate(matches):
            for slot_tensor, gt_tensor in zip(
                match["pred_indices"], match["gt_indices"]
            ):
                slot = int(slot_tensor)
                gt = int(gt_tensor)
                if not bool(gt_valid[batch_index, gt]):
                    continue
                local_mask = (
                    group_mask[batch_index, slot].detach().bool()
                    & candidate_valid[batch_index]
                )
                local_ids = torch.nonzero(
                    local_mask, as_tuple=False
                ).flatten()
                if int(local_ids.numel()) == 0:
                    continue
                local_quality = quality[batch_index, local_ids, gt].detach()
                best_quality, best_offset = local_quality.max(dim=0)
                if float(best_quality) < representable_min:
                    continue
                representable_counts[batch_index] += 1.0
                local_scores = scores[batch_index, slot, local_ids].float()
                quality_losses.append(
                    F.smooth_l1_loss(
                        torch.sigmoid(local_scores),
                        local_quality.to(local_scores.dtype),
                        reduction="mean",
                    )
                )

                quality_gap = local_quality[:, None] - local_quality[None, :]
                ordered_pair = quality_gap >= pair_margin
                if bool(ordered_pair.any()):
                    score_gap = local_scores[:, None] - local_scores[None, :]
                    pairwise_losses.append(
                        F.softplus(-score_gap[ordered_pair]).mean()
                    )

                if int(local_ids.numel()) == 1:
                    second_quality = best_quality.new_tensor(float("-inf"))
                else:
                    second_quality = local_quality.topk(2).values[1]
                if float(best_quality - second_quality) >= hard_margin:
                    hard_losses.append(
                        -F.log_softmax(local_scores, dim=-1)[best_offset]
                    )

                best_id = local_ids[best_offset]
                selected_id = selected_ids[batch_index, slot]
                anchor_id = anchor_ids[batch_index, slot]
                anchor_rows.append(quality[batch_index, anchor_id, gt])
                selected_rows.append(quality[batch_index, selected_id, gt])
                oracle_rows.append(best_quality)
                group_sizes.append(scores.new_tensor(float(local_ids.numel())))
                top1_rows.append((selected_id == best_id).to(torch.float32))

        zero = scores.sum() * 0.0
        quality_loss = (
            torch.stack(quality_losses).mean() if quality_losses else zero
        )
        pairwise_loss = (
            torch.stack(pairwise_losses).mean() if pairwise_losses else zero
        )
        hard_loss = torch.stack(hard_losses).mean() if hard_losses else zero
        total = (
            float(self.cfg.four_slot_v16_quality_weight) * quality_loss
            + float(self.cfg.four_slot_v16_pairwise_weight) * pairwise_loss
            + float(self.cfg.four_slot_v16_hard_weight) * hard_loss
        )
        detached_zero = total.detach() * 0.0

        def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
            return (
                torch.stack(values).float().mean().detach()
                if values
                else detached_zero
            )

        mean_anchor = mean_or_zero(anchor_rows)
        mean_selected = mean_or_zero(selected_rows)
        return {
            "total": total,
            "quality": quality_loss,
            "pairwise": pairwise_loss,
            "hard": hard_loss,
            "mean_matched": match_count.mean().detach(),
            "mean_representable": representable_counts.mean().detach(),
            "mean_group_size": mean_or_zero(group_sizes),
            "mean_anchor_quality": mean_anchor,
            "mean_selected_quality": mean_selected,
            "mean_oracle_quality": mean_or_zero(oracle_rows),
            "mean_quality_gain": (mean_selected - mean_anchor).detach(),
            "mean_target_top1": mean_or_zero(top1_rows),
        }

    def compute_four_slot_unified_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Joint V11 activity, global-memory and slot-geometry objective."""

        refined = outputs.get("selection_slot_pred_x_rows")
        ranges = outputs.get("selection_slot_range_norm")
        active_logits = outputs.get("selection_slot_active_logits")
        delta_logits = outputs.get("selection_slot_row_delta_logits")
        delta_offsets = outputs.get("selection_slot_row_delta_offsets_px")
        reference = outputs.get("selection_slot_input_reference_x_rows")
        attention = outputs.get("selection_slot_unified_proposal_attention")
        aux_x = outputs.get("selection_slot_unified_aux_x_rows")
        aux_range = outputs.get("selection_slot_unified_aux_range_norm")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                refined,
                ranges,
                active_logits,
                delta_logits,
                delta_offsets,
                reference,
                attention,
                aux_x,
                aux_range,
            )
        ):
            raise ValueError(
                "w_four_slot_unified > 0 requires V11 unified outputs"
            )
        matches, final_quality_batch, match_count = (
            self._match_four_slot_unified_final(
                outputs,
                targets,
                padded_targets,
            )
        )

        slot_outputs = {
            "pred_x_rows": refined,
            "range_norm": ranges,
            "row_x_logits": delta_logits,
            "row_x_offsets_px": delta_offsets,
            "input_reference_x_rows": reference,
        }
        point = self.compute_point_loss(slot_outputs, targets, matches)
        range_loss = self.compute_range_loss(slot_outputs, targets, matches)
        line_iou = self.compute_line_iou_loss(slot_outputs, targets, matches)
        dfl = self.compute_row_dfl_loss(slot_outputs, targets, matches)

        auxiliary_outputs = {
            "pred_x_rows": aux_x,
            "range_norm": aux_range,
        }
        aux_point = self.compute_point_loss(
            auxiliary_outputs,
            targets,
            matches,
        )
        aux_range_loss = self.compute_range_loss(
            auxiliary_outputs,
            targets,
            matches,
        )
        aux_line_iou = self.compute_line_iou_loss(
            auxiliary_outputs,
            targets,
            matches,
        )

        active_target = torch.zeros_like(active_logits, dtype=torch.float32)
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            if pred_ids.numel():
                active_target[batch_index, pred_ids] = 1.0
        active_loss = F.binary_cross_entropy_with_logits(
            active_logits.float(),
            active_target,
        )

        proposal_target, proposal_support = (
            self._four_slot_unified_proposal_targets(
                outputs,
                targets,
                padded_targets,
            )
        )
        attention_rows: list[torch.Tensor] = []
        support_mass_rows: list[torch.Tensor] = []
        for batch_index, match in enumerate(matches):
            pred_ids = match["pred_indices"]
            gt_ids = match["gt_indices"]
            if pred_ids.numel() == 0:
                continue
            predicted = attention[batch_index, pred_ids].float()
            target_probability = proposal_target[
                batch_index,
                gt_ids,
            ].to(predicted.dtype)
            support = proposal_support[batch_index, gt_ids]
            attention_rows.append(
                -(
                    target_probability
                    * predicted.clamp_min(1.0e-12).log()
                ).sum(dim=-1)
            )
            support_mass_rows.append(
                (predicted * support.to(predicted.dtype)).sum(dim=-1)
            )
        if attention_rows:
            attention_loss = torch.cat(attention_rows).mean()
            mean_support_mass = torch.cat(support_mass_rows).mean().detach()
        else:
            attention_loss = attention.sum() * 0.0
            mean_support_mass = attention_loss.detach()

        aux_total = (
            float(self.cfg.four_slot_unified_point_weight) * aux_point
            + float(self.cfg.four_slot_unified_range_weight) * aux_range_loss
            + float(self.cfg.four_slot_unified_line_iou_weight)
            * aux_line_iou
        )
        total = (
            float(self.cfg.four_slot_unified_active_weight) * active_loss
            + float(self.cfg.four_slot_unified_attention_weight)
            * attention_loss
            + float(self.cfg.four_slot_unified_point_weight) * point
            + float(self.cfg.four_slot_unified_range_weight) * range_loss
            + float(self.cfg.four_slot_unified_line_iou_weight) * line_iou
            + float(self.cfg.four_slot_unified_dfl_weight) * dfl
            + float(self.cfg.four_slot_unified_aux_geometry_weight) * aux_total
        )

        base_x = outputs.get("selection_slot_unified_base_x_rows")
        base_range = outputs.get("selection_slot_unified_base_range_norm")
        matched_final: list[torch.Tensor] = []
        matched_base: list[torch.Tensor] = []
        if isinstance(base_x, torch.Tensor) and isinstance(
            base_range,
            torch.Tensor,
        ):
            if padded_targets is None:
                padded_gt_x, padded_gt_valid = _padded_lane_targets(
                    targets,
                    device=base_x.device,
                    dtype=torch.float32,
                    rows=int(base_x.shape[-1]),
                )
            else:
                padded_gt_x, padded_gt_valid = padded_targets
            with torch.no_grad():
                base_quality_batch, _base_valid, _gt_valid = (
                    batched_pairwise_range_aware_row_strip_iou(
                        base_x.detach().float(),
                        base_range.detach().float(),
                        padded_gt_x,
                        padded_gt_valid,
                        input_h=int(self.cfg.input_h),
                        line_width=float(self.cfg.four_slot_line_width),
                        min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
                    )
                )
            for batch_index, match in enumerate(matches):
                pred_ids = match["pred_indices"]
                gt_ids = match["gt_indices"]
                if pred_ids.numel() == 0:
                    continue
                matched_final.append(
                    final_quality_batch[batch_index, pred_ids, gt_ids]
                )
                matched_base.append(
                    base_quality_batch[batch_index, pred_ids, gt_ids]
                )
        zero = total.detach() * 0.0
        mean_final = (
            torch.cat(matched_final).mean().detach()
            if matched_final
            else zero
        )
        mean_base = (
            torch.cat(matched_base).mean().detach()
            if matched_base
            else zero
        )
        return {
            "total": total,
            "active": active_loss,
            "attention": attention_loss,
            "point": point,
            "range": range_loss,
            "line_iou": line_iou,
            "dfl": dfl,
            "aux_point": aux_point,
            "aux_range": aux_range_loss,
            "aux_line_iou": aux_line_iou,
            "mean_matched": match_count.mean().detach(),
            "mean_base_quality": mean_base,
            "mean_final_quality": mean_final,
            "mean_quality_gain": (mean_final - mean_base).detach(),
            "mean_target_support_mass": mean_support_mass,
            "mean_active_probability": torch.sigmoid(
                active_logits.float()
            ).mean().detach(),
        }

    @torch.no_grad()
    def _match_four_slot_reference_geometry(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """Match routed *reference* curves once, before learned refinement.

        Keeping assignment on the immutable routed proposal prevents the
        refiner from changing its own target identity while it learns a local
        bounded correction.  Costs for the whole batch cross to CPU once.
        """

        reference = outputs.get("selection_slot_input_reference_x_rows")
        ranges = outputs.get("selection_slot_input_range_norm")
        if not isinstance(ranges, torch.Tensor):
            ranges = outputs.get("selection_slot_range_norm")
        active = (
            outputs.get("selection_slot_geometry_valid")
            if bool(self.cfg.four_slot_geometry_match_all_slots)
            else outputs.get("selection_slot_active")
        )
        if not all(
            isinstance(value, torch.Tensor)
            for value in (reference, ranges, active)
        ):
            raise ValueError(
                "four-slot geometry loss requires routed reference geometry"
            )
        if padded_targets is None:
            padded_gt_x, padded_gt_valid = _padded_lane_targets(
                targets,
                device=reference.device,
                dtype=torch.float32,
                rows=int(reference.shape[-1]),
            )
        else:
            padded_gt_x, padded_gt_valid = padded_targets
        quality_batch, slot_valid_batch, gt_lane_valid_batch = (
            batched_pairwise_range_aware_row_strip_iou(
                reference.detach().float(),
                ranges.detach().float(),
                padded_gt_x,
                padded_gt_valid,
                input_h=int(self.cfg.input_h),
                line_width=float(self.cfg.four_slot_line_width),
                min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
            )
        )
        pair_valid = (
            active.bool() & slot_valid_batch
        ).unsqueeze(-1) & gt_lane_valid_batch.unsqueeze(1)
        # Transfer one fixed, tiny [B,S,G] matrix.  Encoding invalid rows and
        # columns as infinity lets CPU recover the exact same ascending slot
        # and GT index sets without CUDA ``nonzero`` or per-pair GPU scalar
        # extraction.  The sliced SciPy cost matrix is bit-identical to the
        # historical matcher input.
        cost_cpu_batch = (1.0 - quality_batch).masked_fill(
            ~pair_valid,
            torch.inf,
        ).detach().cpu()
        if int(cost_cpu_batch.shape[-1]) == 0:
            cost_cpu_batch = torch.empty(
                (len(targets), int(reference.shape[1]), 0),
                dtype=torch.float32,
            )
        cpu_pairs: list[torch.Tensor] = []
        pair_counts: list[int] = []
        reference_qualities: list[float] = []
        threshold = (
            0.0
            if bool(self.cfg.four_slot_geometry_match_all_slots)
            else float(self.cfg.four_slot_geometry_match_min_quality)
        )
        for cost_cpu in cost_cpu_batch:
            finite = torch.isfinite(cost_cpu)
            slot_ids = torch.nonzero(finite.any(dim=1), as_tuple=False).flatten()
            gt_ids = torch.nonzero(finite.any(dim=0), as_tuple=False).flatten()
            if slot_ids.numel() == 0 or gt_ids.numel() == 0:
                cpu_pairs.append(torch.empty((0, 2), dtype=torch.long))
                pair_counts.append(0)
                continue
            local_cost = cost_cpu.index_select(0, slot_ids).index_select(1, gt_ids)
            local_slot, local_gt = HungarianMatcherS0._linear_sum_assignment(
                local_cost
            )
            kept: list[tuple[int, int]] = []
            slot_id_values = slot_ids.tolist()
            gt_id_values = gt_ids.tolist()
            for slot_value, gt_value in zip(
                local_slot.tolist(),
                local_gt.tolist(),
            ):
                quality_value = 1.0 - float(
                    local_cost[int(slot_value), int(gt_value)]
                )
                if quality_value < threshold:
                    continue
                kept.append(
                    (
                        int(slot_id_values[int(slot_value)]),
                        int(gt_id_values[int(gt_value)]),
                    )
                )
                reference_qualities.append(quality_value)
            pair_tensor = (
                torch.tensor(kept, dtype=torch.long)
                if kept
                else torch.empty((0, 2), dtype=torch.long)
            )
            cpu_pairs.append(pair_tensor)
            pair_counts.append(int(pair_tensor.shape[0]))
        packed = (
            torch.cat(cpu_pairs, dim=0).to(reference.device)
            if any(pair_counts)
            else torch.empty((0, 2), dtype=torch.long, device=reference.device)
        )
        matches: list[dict[str, torch.Tensor]] = []
        pair_offset = 0
        for count in pair_counts:
            pairs = packed[pair_offset : pair_offset + count]
            pair_offset += count
            matches.append(
                {
                    "pred_indices": pairs[:, 0],
                    "gt_indices": pairs[:, 1],
                }
            )
        reference_quality = reference.new_tensor(
            reference_qualities,
            dtype=torch.float32,
        )
        match_count = reference.new_tensor(pair_counts, dtype=torch.float32)
        return matches, reference_quality, match_count

    def compute_four_slot_geometry_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padded_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Train bounded slot geometry on top of frozen routed proposals."""

        refined = outputs.get("selection_slot_pred_x_rows")
        ranges = outputs.get("selection_slot_range_norm")
        delta_logits = outputs.get("selection_slot_row_delta_logits")
        delta_offsets = outputs.get("selection_slot_row_delta_offsets_px")
        reference = outputs.get("selection_slot_input_reference_x_rows")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                refined,
                ranges,
                delta_logits,
                delta_offsets,
                reference,
            )
        ):
            raise ValueError(
                "w_four_slot_geometry > 0 requires bounded slot refinement outputs"
            )
        matches, reference_quality, match_count = (
            self._match_four_slot_reference_geometry(
                outputs,
                targets,
                padded_targets,
            )
        )
        slot_outputs = {
            "pred_x_rows": refined,
            "range_norm": ranges,
            "row_x_logits": delta_logits,
            "row_x_offsets_px": delta_offsets,
            "input_reference_x_rows": reference,
        }
        point = self.compute_point_loss(slot_outputs, targets, matches)
        range_loss = self.compute_range_loss(slot_outputs, targets, matches)
        line_iou = self.compute_line_iou_loss(slot_outputs, targets, matches)
        dfl = self.compute_row_dfl_loss(slot_outputs, targets, matches)
        total = (
            float(self.cfg.four_slot_geometry_point_weight) * point
            + float(self.cfg.four_slot_geometry_range_weight) * range_loss
            + float(self.cfg.four_slot_geometry_line_iou_weight) * line_iou
            + float(self.cfg.four_slot_geometry_dfl_weight) * dfl
        )

        refined_quality_values: list[torch.Tensor] = []
        with torch.no_grad():
            if padded_targets is None:
                padded_gt_x, padded_gt_valid = _padded_lane_targets(
                    targets,
                    device=refined.device,
                    dtype=torch.float32,
                    rows=int(refined.shape[-1]),
                )
            else:
                padded_gt_x, padded_gt_valid = padded_targets
            refined_quality_batch, _slot_valid, _gt_valid = (
                batched_pairwise_range_aware_row_strip_iou(
                    refined.detach().float(),
                    ranges.detach().float(),
                    padded_gt_x,
                    padded_gt_valid,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.four_slot_line_width),
                    min_valid_rows=int(self.cfg.four_slot_min_valid_rows),
                )
            )
            for batch_index, match in enumerate(matches):
                pred_ids = match["pred_indices"]
                gt_ids = match["gt_indices"]
                if pred_ids.numel() == 0:
                    continue
                refined_quality_values.append(
                    refined_quality_batch[batch_index, pred_ids, gt_ids]
                )
        refined_quality = (
            torch.cat(refined_quality_values)
            if refined_quality_values
            else total.detach().new_zeros((0,))
        )
        zero = total.detach() * 0.0
        mean_reference = (
            reference_quality.mean() if reference_quality.numel() else zero
        )
        mean_refined = (
            refined_quality.mean() if refined_quality.numel() else zero
        )
        return {
            "total": total,
            "point": point,
            "range": range_loss,
            "line_iou": line_iou,
            "dfl": dfl,
            "mean_matched": match_count.mean().detach(),
            "mean_reference_quality": mean_reference.detach(),
            "mean_refined_quality": mean_refined.detach(),
            "mean_quality_gain": (mean_refined - mean_reference).detach(),
        }

    def compute_pointer_selection_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Train the sequential candidate/STOP policy and its unary quality.

        The sequence logits come from a teacher-forced state rollout created
        after the score-independent geometry assignment is known.  A separate
        continuous unary target supplies candidate quality in the historical
        mode.  The V4.4 set-oracle mode instead retains quality only on the
        unique Hungarian representatives and supervises probability mass over
        every still-valid target ordering.  Both contracts keep geometry
        detached from selection gradients.
        """

        logits_value = outputs.get("selection_pointer_logits")
        teacher_value = outputs.get("selection_pointer_teacher_indices")
        unary_value = outputs.get("selection_logits")
        if not isinstance(logits_value, torch.Tensor):
            raise ValueError(
                "w_pointer_selection > 0 requires sequential pointer logits"
            )
        if not isinstance(teacher_value, torch.Tensor):
            raise ValueError(
                "pointer training requires teacher targets from forward_with_matches"
            )
        if not isinstance(unary_value, torch.Tensor):
            raise ValueError("pointer training requires unary selection_logits")
        pointer_logits = logits_value.float()
        teacher = teacher_value.to(
            device=pointer_logits.device,
            dtype=torch.long,
        )
        batch, steps, classes = pointer_logits.shape
        candidates = classes - 1
        if teacher.shape != (batch, steps):
            raise ValueError("pointer teacher/logit shape mismatch")
        teacher_probabilities = outputs.get(
            "selection_pointer_teacher_probabilities"
        )
        teacher_class_mask = outputs.get("selection_pointer_teacher_class_mask")
        teacher_active = outputs.get("selection_pointer_teacher_active")
        if isinstance(teacher_probabilities, torch.Tensor):
            if teacher_probabilities.shape != pointer_logits.shape:
                raise ValueError("pointer soft-target/logit shape mismatch")
            if not isinstance(teacher_active, torch.Tensor) or teacher_active.shape != (
                batch,
                steps,
            ):
                raise ValueError("pointer soft-target active-mask shape mismatch")
            target_probability = teacher_probabilities.to(
                device=pointer_logits.device,
                dtype=pointer_logits.dtype,
            )
            active = teacher_active.to(
                device=pointer_logits.device,
                dtype=torch.bool,
            )
            if bool((target_probability < 0.0).any()):
                raise ValueError("pointer soft targets must be non-negative")
            target_sum = target_probability.sum(dim=-1)
            if not torch.allclose(
                target_sum[active],
                torch.ones_like(target_sum[active]),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError("active pointer soft targets must sum to one")
            if bool((target_sum[~active].abs() > 1e-6).any()):
                raise ValueError("inactive pointer soft targets must be zero")
            log_probability = F.log_softmax(pointer_logits, dim=-1)
            per_step = -(target_probability * log_probability).sum(dim=-1)
            per_step = torch.where(active, per_step, torch.zeros_like(per_step))
            stop_target = (target_probability[..., candidates] > 0.5) & active
        elif isinstance(teacher_class_mask, torch.Tensor):
            if teacher_class_mask.shape != pointer_logits.shape:
                raise ValueError("pointer set-target mask/logit shape mismatch")
            if not isinstance(teacher_active, torch.Tensor) or teacher_active.shape != (
                batch,
                steps,
            ):
                raise ValueError("pointer set-target active-mask shape mismatch")
            class_mask = teacher_class_mask.to(
                device=pointer_logits.device,
                dtype=torch.bool,
            )
            active = teacher_active.to(
                device=pointer_logits.device,
                dtype=torch.bool,
            )
            if bool((active & ~class_mask.any(dim=-1)).any()):
                raise ValueError("active pointer set target has no valid class")
            # Inactive positions occur after the supervised STOP.  Give those
            # rows a harmless finite STOP target before masking their loss so
            # logsumexp never produces an unused infinity.
            safe_class_mask = class_mask.clone()
            safe_class_mask[..., candidates] |= ~active
            log_probability = F.log_softmax(pointer_logits, dim=-1)
            target_log_mass = torch.logsumexp(
                log_probability.masked_fill(~safe_class_mask, -torch.inf),
                dim=-1,
            )
            per_step = torch.where(
                active,
                -target_log_mass,
                torch.zeros_like(target_log_mass),
            )
            stop_target = class_mask[..., candidates] & active
        else:
            per_step = F.cross_entropy(
                pointer_logits.reshape(batch * steps, classes),
                teacher.reshape(batch * steps),
                ignore_index=-100,
                reduction="none",
            ).view(batch, steps)
            active = teacher >= 0
            stop_target = teacher == candidates
        step_weight = torch.where(
            stop_target,
            torch.full_like(per_step, float(self.cfg.pointer_stop_weight)),
            torch.ones_like(per_step),
        )
        sequence_loss = (
            per_step * step_weight * active.to(per_step.dtype)
        ).sum() / (
            step_weight * active.to(step_weight.dtype)
        ).sum().clamp_min(1.0)

        pairwise_quality = self.compute_set_selection_pairwise_quality(
            outputs,
            targets,
        )
        quality_target = unary_value.new_zeros((batch, candidates)).float()
        unique_target_value = outputs.get("selection_pointer_unique_target_mask")
        unique_target_mask = (
            unique_target_value.to(
                device=quality_target.device,
                dtype=torch.bool,
            )
            if isinstance(unique_target_value, torch.Tensor)
            else None
        )
        if unique_target_mask is not None and unique_target_mask.shape != (
            batch,
            candidates,
        ):
            raise ValueError("pointer unique-target mask shape mismatch")
        for batch_index, quality in enumerate(pairwise_quality):
            if quality.numel() > 0:
                candidate_quality = quality.amax(dim=-1).to(
                    device=quality_target.device,
                    dtype=quality_target.dtype,
                )
                if self.cfg.pointer_unary_target_mode == "unique_representative":
                    if unique_target_mask is None:
                        raise ValueError(
                            "unique representative unary supervision requires "
                            "selection_pointer_unique_target_mask"
                        )
                    candidate_quality = candidate_quality * unique_target_mask[
                        batch_index
                    ].to(candidate_quality.dtype)
                quality_target[batch_index] = candidate_quality
        unary_logits = unary_value.float()
        probability = torch.sigmoid(unary_logits)
        modulation = (quality_target - probability).abs().pow(
            float(self.cfg.set_selection_focal_beta)
        )
        per_candidate_quality = (
            modulation
            * F.binary_cross_entropy_with_logits(
                unary_logits,
                quality_target,
                reduction="none",
            )
        )
        if self.cfg.pointer_unary_target_mode == "unique_representative":
            if unique_target_mask is None:
                raise ValueError("pointer unique unary target mask is missing")
            image_losses: list[torch.Tensor] = []
            for batch_index in range(batch):
                positive = unique_target_mask[batch_index]
                negative = ~positive
                if bool(positive.any()):
                    positive_loss = per_candidate_quality[batch_index][positive].mean()
                    negative_loss = per_candidate_quality[batch_index][negative].mean()
                    image_losses.append(0.5 * (positive_loss + negative_loss))
                else:
                    image_losses.append(per_candidate_quality[batch_index].mean())
            quality_loss = torch.stack(image_losses).mean()
        else:
            quality_loss = per_candidate_quality.mean()
        listwise_loss = unary_logits.sum() * 0.0
        listwise_weight = float(self.cfg.pointer_cluster_listwise_weight)
        if listwise_weight > 0.0:
            if not isinstance(teacher_probabilities, torch.Tensor):
                raise ValueError(
                    "pointer cluster-listwise supervision requires soft "
                    "cluster teacher probabilities"
                )
            candidate_target = target_probability[..., :candidates]
            candidate_step = active & (candidate_target.sum(dim=-1) > 0.5)
            support = candidate_target > 0.0
            temperature = float(
                self.cfg.pointer_cluster_listwise_logit_temperature
            )
            listwise_logits = (
                unary_logits.unsqueeze(1).expand(-1, steps, -1) / temperature
            )
            listwise_log_probability = F.log_softmax(
                listwise_logits.masked_fill(~support, -1e4),
                dim=-1,
            )
            per_cluster = -(
                candidate_target * listwise_log_probability
            ).sum(dim=-1)
            listwise_loss = (
                per_cluster * candidate_step.to(per_cluster.dtype)
            ).sum() / candidate_step.sum().clamp_min(1).to(per_cluster.dtype)
        total = (
            sequence_loss
            + float(self.cfg.pointer_quality_weight) * quality_loss
            + listwise_weight * listwise_loss
        )

        stop_rows = torch.nonzero(stop_target, as_tuple=False)
        if stop_rows.numel() > 0:
            stop_probability = torch.softmax(pointer_logits, dim=-1)[
                stop_rows[:, 0],
                stop_rows[:, 1],
                candidates,
            ].mean()
        else:
            stop_probability = total.detach() * 0.0
        emitted = ((teacher >= 0) & (teacher < candidates)).sum(dim=-1)

        candidate_steps_value = outputs.get(
            "selection_pointer_teacher_candidate_steps"
        )
        candidate_steps = (
            candidate_steps_value.to(device=pointer_logits.device, dtype=torch.bool)
            if isinstance(candidate_steps_value, torch.Tensor)
            else ((teacher >= 0) & (teacher < candidates))
        )

        def candidate_step_mean(name: str) -> torch.Tensor:
            value = outputs.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (
                batch,
                steps,
            ):
                return total.detach() * 0.0
            selected = value.to(
                device=pointer_logits.device,
                dtype=torch.float32,
            )[candidate_steps]
            return (
                selected.mean().detach()
                if selected.numel() > 0
                else total.detach() * 0.0
            )

        representable_value = outputs.get(
            "selection_pointer_teacher_representable_count"
        )
        fallback_value = outputs.get("selection_pointer_teacher_fallback_count")
        reservation_value = outputs.get(
            "selection_pointer_teacher_reservation_exclusion_count"
        )
        return {
            "total": total,
            "sequence": sequence_loss,
            "quality": quality_loss,
            "listwise": listwise_loss,
            "mean_emitted_target": emitted.float().mean().detach(),
            "mean_stop_probability": stop_probability.detach(),
            "mean_cluster_support": candidate_step_mean(
                "selection_pointer_teacher_support_sizes"
            ),
            "mean_cluster_entropy": candidate_step_mean(
                "selection_pointer_teacher_target_entropy"
            ),
            "mean_cluster_quality": candidate_step_mean(
                "selection_pointer_teacher_target_quality"
            ),
            "mean_remaining_cluster_count": candidate_step_mean(
                "selection_pointer_teacher_remaining_cluster_count"
            ),
            "mean_representable_count": (
                representable_value.float().mean().detach()
                if isinstance(representable_value, torch.Tensor)
                else emitted.float().mean().detach()
            ),
            "teacher_fallback_count": (
                fallback_value.float().sum().detach()
                if isinstance(fallback_value, torch.Tensor)
                else total.detach() * 0.0
            ),
            "teacher_reservation_exclusion_count": (
                reservation_value.float().sum().detach()
                if isinstance(reservation_value, torch.Tensor)
                else total.detach() * 0.0
            ),
        }

    def compute_seg_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        seg_logits = outputs.get("seg_logits")
        seg_items: list[tuple[torch.Tensor, float]] = []
        if seg_logits is not None:
            seg_items.append((seg_logits, 1.0))
        for scale_name, weight in self.cfg.seg_extra_weights.items():
            extra_logits = outputs.get(f"seg_logits_{scale_name}")
            if extra_logits is not None and float(weight) != 0.0:
                seg_items.append((extra_logits, float(weight)))
        if not seg_items:
            flat = outputs.get("final") or outputs.get("stage2") or {}
            seg_logits = flat.get("seg_logits") if isinstance(flat, dict) else None
            if seg_logits is not None:
                seg_items.append((seg_logits, 1.0))
        if not seg_items:
            anchor = outputs.get("exist_logits")
            if anchor is None:
                flat = outputs.get("final") or outputs.get("stage2") or outputs.get("coarse") or {}
                anchor = flat.get("exist_logits") if isinstance(flat, dict) else None
            if anchor is None:
                anchor = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
            return anchor.sum() * 0.0
        seg_logits = seg_items[0][0]
        seg_targets = []
        valid_weights = []
        for target in targets:
            if "seg_mask" not in target:
                return seg_logits.sum() * 0.0
            seg_mask = target["seg_mask"].to(seg_logits.device, dtype=seg_logits.dtype)
            seg_valid = target.get("seg_valid", True)
            if isinstance(seg_valid, torch.Tensor):
                valid = seg_valid.to(seg_logits.device, dtype=seg_logits.dtype).reshape(-1)[0]
            else:
                valid = torch.tensor(float(bool(seg_valid)), device=seg_logits.device, dtype=seg_logits.dtype)
            has_lane = int(target["x_rows"].shape[0]) > 0
            if has_lane:
                valid = valid * (seg_mask.detach().amax() > 0).to(dtype=seg_logits.dtype)
            seg_targets.append(seg_mask)
            valid_weights.append(valid)
        seg_target = torch.stack(seg_targets, dim=0)
        sample_weights = torch.stack(valid_weights, dim=0).to(device=seg_logits.device, dtype=seg_logits.dtype)
        sample_denom = sample_weights.sum().clamp_min(1.0)
        pos_weight = None
        if self.cfg.seg_pos_weight != 1.0:
            pos_weight = torch.tensor([self.cfg.seg_pos_weight], device=seg_logits.device, dtype=seg_logits.dtype)
        total = seg_logits.sum() * 0.0
        for logits, weight in seg_items:
            target = seg_target
            if target.shape[-2:] != logits.shape[-2:]:
                target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
            pw = pos_weight
            if pw is not None and pw.dtype != logits.dtype:
                pw = pw.to(dtype=logits.dtype)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
            loss = loss.flatten(1).mean(dim=1)
            total = total + float(weight) * (loss * sample_weights).sum() / sample_denom
        return total

    def compute_centerline_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        logits = outputs.get("centerline_logits")
        if logits is None:
            anchor = outputs.get("exist_logits")
            if anchor is None:
                flat = outputs.get("final") or outputs.get("stage2") or outputs.get("coarse") or {}
                anchor = flat.get("exist_logits") if isinstance(flat, dict) else None
            if anchor is None:
                anchor = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
            return anchor.sum() * 0.0

        b, _, num_rows, x_bins = logits.shape
        device = logits.device
        dtype = logits.dtype
        target_map = torch.zeros((b, 1, num_rows, x_bins), device=device, dtype=dtype)
        grid = fixed_indices(
            x_bins,
            device=device,
            dtype=dtype,
        ).view(1, 1, x_bins)
        sigma = max(float(self.cfg.centerline_sigma_bins), 1e-3)
        bin_width = float(self.cfg.input_w) / float(x_bins)
        for bi, target in enumerate(targets):
            x_rows = target["x_rows"].to(device=device, dtype=dtype)
            valid_mask = target["valid_mask"].to(device=device).bool()
            if x_rows.numel() == 0:
                continue
            row_count = min(int(x_rows.shape[1]), int(num_rows))
            x_rows = x_rows[:, :row_count]
            valid_mask = valid_mask[:, :row_count]
            centers = (x_rows / bin_width).clamp(min=0.0, max=float(x_bins - 1))
            valid = valid_mask & torch.isfinite(centers) & (x_rows >= 0.0) & (x_rows <= float(self.cfg.input_w))
            # Keep empty/invalid images on the tensor path instead of forcing
            # a device synchronization through ``Tensor.__bool__``.
            safe_centers = torch.where(valid, centers, torch.zeros_like(centers))
            diff = grid - safe_centers.unsqueeze(-1)
            gauss = torch.exp(-0.5 * (diff / sigma).pow(2))
            gauss = gauss * valid.unsqueeze(-1).to(dtype=dtype)
            target_map[bi, 0, :row_count] = gauss.amax(dim=0)
        pos_weight = None
        if self.cfg.centerline_pos_weight != 1.0:
            pos_weight = torch.tensor([self.cfg.centerline_pos_weight], device=device, dtype=dtype)
        return F.binary_cross_entropy_with_logits(logits, target_map, pos_weight=pos_weight)

    def compute_dynamic_proposal_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        dynamic = outputs.get("dynamic_proposals")
        if not isinstance(dynamic, dict) or not isinstance(dynamic.get("dense"), dict):
            anchor = self._zero_anchor(outputs)
            zero = anchor.sum() * 0.0
            return {"heatmap": zero, "x": zero, "range": zero}

        dense = dynamic["dense"]
        heatmap_logits = dense["heatmap_logits"]
        dense_x = dense["x_rows"]
        dense_range = sort_range_norm(dense["range_norm"].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        b, _, feat_h, feat_w = heatmap_logits.shape
        _, num_rows_pred, _, _ = dense_x.shape
        device = heatmap_logits.device
        dtype = heatmap_logits.dtype
        heatmap_target = torch.zeros((b, 1, feat_h, feat_w), device=device, dtype=dtype)
        grid_x = fixed_indices(feat_w, device=device, dtype=dtype)
        sigma = max(float(self.cfg.dynamic_proposal_sigma_bins), 1e-3)
        radius = max(int(self.cfg.dynamic_proposal_seed_radius_bins), 0)
        x_loss = dense_x.sum() * 0.0
        range_loss = dense_range.sum() * 0.0
        x_denom = dense_x.new_tensor(0.0)
        range_denom = dense_range.new_tensor(0.0)

        for bi, target in enumerate(targets):
            x_rows = target["x_rows"].to(device=device, dtype=dense_x.dtype)
            valid_mask = target["valid_mask"].to(device=device).bool()
            if x_rows.numel() == 0:
                continue
            lane_count = int(x_rows.shape[0])
            row_count = min(int(x_rows.shape[1]), int(num_rows_pred))
            if row_count <= 0:
                continue
            for lane_idx in range(lane_count):
                lane_x = x_rows[lane_idx, :row_count]
                lane_valid = valid_mask[lane_idx, :row_count]
                finite_valid = lane_valid & torch.isfinite(lane_x) & (lane_x >= 0.0) & (lane_x <= float(self.cfg.input_w))
                valid_rows = finite_valid.nonzero(as_tuple=False).flatten()
                if valid_rows.numel() == 0:
                    continue

                seed_row = valid_rows[-1]
                if row_count == 1:
                    feat_y = torch.zeros((), device=device, dtype=torch.long)
                else:
                    feat_y = torch.round(seed_row.to(dtype=dense_x.dtype) * float(feat_h - 1) / float(row_count - 1)).long()
                seed_x = lane_x[seed_row]
                seed_x_bin = (seed_x / float(self.cfg.input_w) * float(feat_w)).clamp(0.0, float(feat_w - 1))
                heat = torch.exp(-0.5 * ((grid_x - seed_x_bin.to(dtype=dtype)) / sigma).pow(2))
                heatmap_target[bi, 0, feat_y] = torch.maximum(heatmap_target[bi, 0, feat_y], heat)

                center_bin = int(torch.round(seed_x_bin).clamp(0, feat_w - 1).item())
                for offset in range(-radius, radius + 1):
                    feat_x = center_bin + offset
                    if feat_x < 0 or feat_x >= feat_w:
                        continue
                    weight = dense_x.new_tensor(float(torch.exp(torch.tensor(-0.5 * (float(offset) / sigma) ** 2))))
                    pred_lane = dense_x[bi, :row_count, feat_y, feat_x]
                    gt_lane = lane_x[:row_count]
                    mask = finite_valid[:row_count]
                    if mask.any():
                        x_loss = x_loss + weight * F.smooth_l1_loss(
                            pred_lane[mask] / float(self.cfg.input_w),
                            gt_lane[mask] / float(self.cfg.input_w),
                            beta=self.cfg.smooth_l1_beta,
                            reduction="sum",
                        )
                        x_denom = x_denom + weight * mask.to(dtype=dense_x.dtype).sum()
                    if "range_y" in target:
                        gt_range = target["range_y"].to(device=device, dtype=dense_range.dtype)[lane_idx] / float(self.cfg.input_h)
                        pred_range = dense_range[bi, :, feat_y, feat_x]
                        range_loss = range_loss + weight * F.smooth_l1_loss(
                            pred_range,
                            sort_range_norm(gt_range.view(1, 1, 2)).view(2),
                            beta=self.cfg.smooth_l1_beta,
                            reduction="sum",
                        )
                        range_denom = range_denom + weight * 2.0

        pos_weight = None
        if self.cfg.dynamic_proposal_heatmap_pos_weight != 1.0:
            pos_weight = torch.tensor([self.cfg.dynamic_proposal_heatmap_pos_weight], device=device, dtype=dtype)
        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_logits, heatmap_target, pos_weight=pos_weight)
        x_loss = x_loss / x_denom.clamp_min(1.0)
        range_loss = range_loss / range_denom.clamp_min(1.0)
        return {"heatmap": heatmap_loss, "x": x_loss, "range": range_loss}

    def _zero_anchor(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, dict):
                try:
                    return self._zero_anchor(value)
                except StopIteration:
                    pass
        raise StopIteration("No tensor found in outputs")

    def compute_range_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
        packed: _MatchedLaneBatch | None = None,
    ) -> torch.Tensor:
        packed = packed or self._pack_matched_lanes(outputs, targets, matches)
        if packed.pred_range is None:
            raise KeyError("range_norm")
        if packed.gt_range is None:
            raise KeyError("range_y")
        if int(packed.pred_range.shape[0]) == 0:
            return outputs["range_norm"].sum() * 0.0
        pred_range = sort_range_norm(packed.pred_range)
        gt_range = packed.gt_range / float(self.cfg.input_h)
        return F.smooth_l1_loss(
            pred_range,
            gt_range,
            beta=self.cfg.smooth_l1_beta,
            reduction="sum",
        ) / float(max(int(pred_range.numel()), 1))

    def compute_smoothness_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        matches: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"]
        total = pred_x.sum() * 0.0
        count = 0
        for bi, match in enumerate(matches):
            pred_idx = match["pred_indices"].to(pred_x.device)
            gt_idx = match["gt_indices"].to(pred_x.device)
            for pidx, gidx in zip(pred_idx.tolist(), gt_idx.tolist()):
                mask = targets[bi]["valid_mask"].to(pred_x.device)[gidx].bool()
                if not self.cfg.smoothness_contiguous:
                    if int(mask.sum().item()) >= 3:
                        lane_x = pred_x[bi, pidx][mask]
                        d2 = lane_x[2:] - 2.0 * lane_x[1:-1] + lane_x[:-2]
                        total = total + (d2 / float(self.cfg.input_w)).abs().mean()
                        count += 1
                    continue
                triplet_mask = mask[2:] & mask[1:-1] & mask[:-2]
                if triplet_mask.any():
                    lane_x = pred_x[bi, pidx]
                    d2 = lane_x[2:] - 2.0 * lane_x[1:-1] + lane_x[:-2]
                    total = total + (d2[triplet_mask] / float(self.cfg.input_w)).abs().sum()
                    count += int(triplet_mask.sum().item())
        return total / max(count, 1)
