from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import math

import torch

from dynlaneseq_eg.modeling.common import sort_range_norm
from .range_aware_iou import (
    batched_pairwise_range_aware_row_strip_iou,
    pairwise_range_aware_row_strip_iou,
)


@dataclass
class MatcherConfig:
    lambda_obj: float = 2.0
    lambda_obj_start: float | None = None
    lambda_obj_end: float | None = None
    lambda_obj_ramp_start_iter: int = 0
    lambda_obj_ramp_end_iter: int = 0
    lambda_point: float = 5.0
    lambda_range: float = 1.0
    lambda_line_iou: float = 0.0
    line_iou_radius: float = 7.5
    input_w: int = 800
    input_h: int = 288
    eps: float = 1e-6
    assignment: str = "hungarian"
    num_groups: int = 1
    object_cost_type: str = "neg_log_probability"
    cost_type: str = "composite"
    range_aware_line_width: float = 30.0
    range_aware_min_valid_rows: int = 5


class HungarianMatcherS0:
    def __init__(self, cfg: MatcherConfig | None = None):
        self.cfg = cfg or MatcherConfig()
        self._iteration = 0
        start = self._lambda_obj_endpoint(self.cfg.lambda_obj_start)
        end = self._lambda_obj_endpoint(self.cfg.lambda_obj_end)
        if not math.isfinite(start) or start < 0.0:
            raise ValueError("matcher lambda_obj_start must be finite and non-negative")
        if not math.isfinite(end) or end < 0.0:
            raise ValueError("matcher lambda_obj_end must be finite and non-negative")
        if int(self.cfg.lambda_obj_ramp_start_iter) < 0:
            raise ValueError("matcher lambda_obj_ramp_start_iter must be non-negative")
        if int(self.cfg.lambda_obj_ramp_end_iter) < int(
            self.cfg.lambda_obj_ramp_start_iter
        ):
            raise ValueError(
                "matcher lambda_obj_ramp_end_iter must be >= ramp_start_iter"
            )

    def _lambda_obj_endpoint(self, value: float | None) -> float:
        return float(self.cfg.lambda_obj if value is None else value)

    def set_iteration(self, iteration: int) -> None:
        self._iteration = max(int(iteration), 0)

    def effective_lambda_obj(self, iteration: int | None = None) -> float:
        """Return the bounded ownership cost used by the current assignment.

        With no explicit endpoints this is exactly the historical static
        ``lambda_obj`` behavior.  V5-B sets both endpoints and uses a warm-up
        followed by a linear ramp; the matcher remains non-differentiable.
        """

        start = self._lambda_obj_endpoint(self.cfg.lambda_obj_start)
        end = self._lambda_obj_endpoint(self.cfg.lambda_obj_end)
        ramp_start = int(self.cfg.lambda_obj_ramp_start_iter)
        ramp_end = int(self.cfg.lambda_obj_ramp_end_iter)
        active_iteration = self._iteration if iteration is None else max(
            int(iteration),
            0,
        )
        if active_iteration <= ramp_start:
            return start
        if ramp_end <= ramp_start or active_iteration >= ramp_end:
            return end
        fraction = float(active_iteration - ramp_start) / float(
            ramp_end - ramp_start
        )
        return (1.0 - fraction) * start + fraction * end

    @torch.no_grad()
    def __call__(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        return self.match_many((outputs,), targets)[0]

    @torch.no_grad()
    def match_many(
        self,
        output_sequence: list[dict[str, torch.Tensor]] | tuple[dict[str, torch.Tensor], ...],
        targets: list[dict[str, torch.Tensor]],
        *,
        assignment: str | None = None,
        group_sizes: tuple[int, ...] | list[int] | None = None,
        assignment_specs: list[
            tuple[str, tuple[int, ...] | list[int] | None]
        ]
        | tuple[tuple[str, tuple[int, ...] | list[int] | None], ...]
        | None = None,
    ) -> list[list[dict[str, torch.Tensor]]]:
        """Match several decoder outputs with one device round trip.

        Deep supervision needs an independent assignment for every decoder
        layer.  Computing those assignments one matcher call at a time forced
        one D2H synchronization and one H2D index transfer per layer.  Cost
        construction and SciPy's per-image Hungarian solve remain identical;
        only the transport of the already-computed cost matrices and integer
        pairs is batched across layers.
        """

        if assignment_specs is not None and (assignment is not None or group_sizes is not None):
            raise ValueError(
                "assignment_specs cannot be combined with assignment/group_sizes"
            )

        normalized_outputs = []
        for outputs in output_sequence:
            if "coarse" in outputs:
                outputs = outputs["coarse"]
            normalized_outputs.append(outputs)

        if not normalized_outputs:
            return []

        # Outputs from ordinary deep supervision have the same candidate
        # shape and use one batched graph.  Hybrid train-only auxiliary groups
        # may contain fewer candidates; group only those incompatible shapes
        # instead of falling back to a per-layer/per-image matcher.
        shape_groups: dict[tuple[tuple[int, ...], ...], list[int]] = {}
        for output_index, output in enumerate(normalized_outputs):
            key = (
                tuple(output["pred_x_rows"].shape),
                tuple(output["exist_logits"].shape),
                tuple(output["range_norm"].shape),
            )
            shape_groups.setdefault(key, []).append(output_index)
        pending_by_output: list[
            list[tuple[torch.Tensor, dict[str, torch.Tensor], int]] | None
        ] = [None for _ in normalized_outputs]
        gt_counts: tuple[int, ...] | None = None
        for output_indices in shape_groups.values():
            group_outputs = [normalized_outputs[index] for index in output_indices]
            batched_cost, batched_stats, group_gt_counts = self.compute_cost_many(
                group_outputs,
                targets,
            )
            if gt_counts is None:
                gt_counts = group_gt_counts
            elif gt_counts != group_gt_counts:
                raise RuntimeError("matcher target counts changed between shape groups")
            for group_index, output_index in enumerate(output_indices):
                pending = []
                for batch_index, num_gt in enumerate(group_gt_counts):
                    cost = batched_cost[
                        group_index,
                        batch_index,
                        :,
                        :num_gt,
                    ]
                    stats = {
                        name: value[group_index, batch_index]
                        for name, value in batched_stats.items()
                    }
                    pending.append((cost, stats, num_gt))
                pending_by_output[output_index] = pending
        if gt_counts is None or any(pending is None for pending in pending_by_output):
            raise RuntimeError("matcher failed to construct every output group")
        concrete_pending = [pending for pending in pending_by_output if pending is not None]

        if assignment_specs is None:
            active_assignment = self.cfg.assignment if assignment is None else str(assignment)
            active_group_sizes = (
                None if group_sizes is None else tuple(int(size) for size in group_sizes)
            )
            normalized_specs = [
                (active_assignment, active_group_sizes)
                for _ in normalized_outputs
            ]
        else:
            if len(assignment_specs) != len(normalized_outputs):
                raise ValueError(
                    "assignment_specs must match output_sequence length: "
                    f"got {len(assignment_specs)} specs for {len(normalized_outputs)} outputs"
                )
            normalized_specs = [
                (
                    str(spec_assignment),
                    None
                    if spec_sizes is None
                    else tuple(int(size) for size in spec_sizes),
                )
                for spec_assignment, spec_sizes in assignment_specs
            ]
        for spec_assignment, spec_sizes in normalized_specs:
            if spec_sizes is not None and (
                not spec_sizes or any(size < 1 for size in spec_sizes)
            ):
                raise ValueError("group_sizes must contain positive integers")
            if spec_sizes is not None and spec_assignment != "grouped_one_to_many":
                raise ValueError(
                    "explicit group_sizes require grouped_one_to_many assignment"
                )

        # Deep supervision calls the matcher once for every decoder output.
        # The matrices are tiny, so concatenate every layer and image and pay
        # for exactly one device synchronization while preserving the
        # identical per-image SciPy assignment below.
        nonempty_costs = [
            cost.reshape(-1)
            for pending in concrete_pending
            for cost, _, num_gt in pending
            if num_gt > 0
        ]
        flat_cost_cpu = (
            torch.cat(nonempty_costs, dim=0).detach().cpu()
            if nonempty_costs
            else torch.empty(0)
        )

        solved_by_output = []
        flat_offset = 0
        for pending, (active_assignment, active_group_sizes) in zip(
            concrete_pending,
            normalized_specs,
        ):
            solved = []
            for cost, stats, num_gt in pending:
                if num_gt == 0:
                    pred_idx = torch.empty(0, dtype=torch.long)
                    gt_idx = torch.empty(0, dtype=torch.long)
                else:
                    numel = int(cost.numel())
                    cost_cpu = flat_cost_cpu[flat_offset : flat_offset + numel].view(
                        int(cost.shape[0]),
                        int(cost.shape[1]),
                    )
                    flat_offset += numel
                    if active_assignment == "grouped_one_to_many":
                        if active_group_sizes is not None:
                            pred_idx, gt_idx = self._grouped_assignment_with_sizes(
                                cost_cpu,
                                active_group_sizes,
                            )
                        else:
                            pred_idx, gt_idx = self._grouped_assignment(
                                cost_cpu,
                                num_groups=max(1, int(self.cfg.num_groups)),
                            )
                    else:
                        pred_idx, gt_idx = self._linear_sum_assignment(cost_cpu)
                solved.append((pred_idx, gt_idx, stats, num_gt))
            solved_by_output.append(solved)

        # Losses consume the same assignment repeatedly (existence, point,
        # range, LineIoU, DFL, and quality).  Returning CPU indices makes every
        # one of those losses launch its own tiny H2D copy.  Pack every layer's
        # pairs into one transfer and keep them on the output device.  SciPy
        # still decides the exact same integer assignment.
        output_device = normalized_outputs[0]["exist_logits"].device
        pair_parts = [
            torch.stack((pred_idx, gt_idx), dim=-1)
            for solved in solved_by_output
            for pred_idx, gt_idx, _, _ in solved
            if pred_idx.numel() > 0
        ]
        if pair_parts:
            flat_pairs = torch.cat(pair_parts, dim=0).to(output_device)
        else:
            flat_pairs = torch.empty((0, 2), dtype=torch.long, device=output_device)

        matches_by_output = []
        pair_offset = 0
        for solved in solved_by_output:
            matches = []
            for pred_idx_cpu, _, stats, num_gt in solved:
                num_matched = int(pred_idx_cpu.numel())
                pairs = flat_pairs[pair_offset : pair_offset + num_matched]
                pair_offset += num_matched
                pred_idx = pairs[:, 0]
                gt_idx = pairs[:, 1]
                matches.append(
                    {
                        "pred_indices": pred_idx,
                        "gt_indices": gt_idx,
                        "num_gt": torch.tensor(num_gt, dtype=torch.long),
                        "num_matched": torch.tensor(num_matched, dtype=torch.long),
                        **stats,
                    }
                )
            matches_by_output.append(matches)
        return matches_by_output

    def compute_cost_many(
        self,
        output_sequence: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        tuple[int, ...],
    ]:
        """Construct all layer/image cost matrices in one GPU graph.

        V7 has four decoder outputs and a physical batch of four.  The old
        path launched the same small point/range/LineIoU kernels sixteen
        times before its single CPU transfer.  Padding the at-most-four GT
        lanes keeps every real cost entry mathematically identical while
        issuing each tensor operation once.
        """

        first = output_sequence[0]
        pred_x = torch.stack(
            [output["pred_x_rows"] for output in output_sequence],
            dim=0,
        )
        exist_logits = torch.stack(
            [output["exist_logits"] for output in output_sequence],
            dim=0,
        )
        range_norm = torch.stack(
            [output["range_norm"] for output in output_sequence],
            dim=0,
        )
        layers, batch, candidates, rows = pred_x.shape
        if int(first["exist_logits"].shape[0]) != batch or len(targets) != batch:
            raise ValueError("matcher outputs and targets must share the batch axis")
        gt_counts = tuple(int(target["x_rows"].shape[0]) for target in targets)
        max_gt = max(gt_counts, default=0)
        device = pred_x.device
        if max_gt == 0:
            empty = pred_x.new_empty((layers, batch, candidates, 0))
            zero = pred_x.new_zeros((layers, batch), dtype=torch.float32)
            return empty, {
                "mean_cost_obj": zero,
                "mean_cost_point": zero.clone(),
                "mean_cost_range": zero.clone(),
                "mean_cost_line_iou": zero.clone(),
                "matcher_lambda_obj": zero.new_full(
                    zero.shape,
                    self.effective_lambda_obj(),
                ),
            }, gt_counts

        padded_x: list[torch.Tensor] = []
        padded_mask: list[torch.Tensor] = []
        padded_range: list[torch.Tensor] = []
        for target, count in zip(targets, gt_counts):
            gt_x = target["x_rows"].to(device=device)
            gt_mask = target["valid_mask"].to(device=device).bool()
            gt_range = target["range_y"].to(device=device)
            if int(gt_x.shape[-1]) != rows or gt_x.shape != gt_mask.shape:
                raise ValueError("matcher target row shapes are incompatible")
            padding = max_gt - count
            if padding:
                gt_x = torch.cat((gt_x, gt_x.new_zeros((padding, rows))), dim=0)
                gt_mask = torch.cat(
                    (
                        gt_mask,
                        torch.zeros(
                            (padding, rows),
                            device=device,
                            dtype=torch.bool,
                        ),
                    ),
                    dim=0,
                )
                gt_range = torch.cat(
                    (gt_range, gt_range.new_zeros((padding, 2))),
                    dim=0,
                )
            padded_x.append(gt_x)
            padded_mask.append(gt_mask)
            padded_range.append(gt_range)
        gt_x = torch.stack(padded_x)
        gt_mask = torch.stack(padded_mask)
        gt_range = torch.stack(padded_range)
        column_exists = (
            torch.arange(max_gt, device=device).view(1, max_gt)
            < torch.tensor(gt_counts, device=device).view(batch, 1)
        )

        def component_mean(value: torch.Tensor) -> torch.Tensor:
            mask = column_exists.view(1, batch, 1, max_gt).to(value.dtype)
            numerator = (value.detach() * mask).sum(dim=(-2, -1))
            denominator = (
                column_exists.sum(dim=-1).view(1, batch).to(value.dtype)
                * float(candidates)
            )
            return torch.where(
                denominator > 0,
                numerator / denominator.clamp_min(1.0),
                torch.zeros_like(numerator),
            )

        cost_type = str(self.cfg.cost_type).strip().lower()
        if cost_type in {
            "range_aware_iou",
            "range_aware_raster_iou",
            "official_iou_surrogate",
        }:
            flat_pred = pred_x.reshape(layers * batch, candidates, rows)
            flat_range = range_norm.reshape(layers * batch, candidates, 2)
            repeated_gt_x = gt_x.unsqueeze(0).expand(layers, -1, -1, -1).reshape(
                layers * batch,
                max_gt,
                rows,
            )
            repeated_gt_mask = gt_mask.unsqueeze(0).expand(
                layers,
                -1,
                -1,
                -1,
            ).reshape(layers * batch, max_gt, rows)
            pairwise_iou, _candidate_valid, gt_lane_valid = (
                batched_pairwise_range_aware_row_strip_iou(
                    flat_pred,
                    flat_range,
                    repeated_gt_x,
                    repeated_gt_mask,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.range_aware_line_width),
                    min_valid_rows=int(self.cfg.range_aware_min_valid_rows),
                )
            )
            cost = (1.0 - pairwise_iou).reshape(
                layers,
                batch,
                candidates,
                max_gt,
            )
            valid_lane = gt_lane_valid.reshape(layers, batch, max_gt)
            cost = torch.where(
                valid_lane.unsqueeze(2),
                cost,
                torch.full_like(cost, 1e6),
            )
            zero = cost.new_zeros((layers, batch))
            return cost, {
                "mean_cost_obj": zero,
                "mean_cost_point": zero.clone(),
                "mean_cost_range": zero.clone(),
                "mean_cost_line_iou": component_mean(cost),
                "matcher_lambda_obj": zero.new_full(
                    zero.shape,
                    self.effective_lambda_obj(),
                ),
            }, gt_counts
        if cost_type not in {"composite", "legacy"}:
            raise ValueError(f"Unsupported matcher.cost_type: {self.cfg.cost_type!r}")

        p_lane = torch.softmax(exist_logits, dim=-1)[..., 0]
        object_cost_type = str(self.cfg.object_cost_type).strip().lower()
        if object_cost_type in {
            "neg_probability",
            "negative_probability",
            "minus_p",
        }:
            cost_obj = -p_lane.unsqueeze(-1).expand(
                layers,
                batch,
                candidates,
                max_gt,
            )
        elif object_cost_type in {
            "neg_log_probability",
            "negative_log_probability",
            "nll",
        }:
            cost_obj = -torch.log(p_lane.clamp_min(self.cfg.eps)).unsqueeze(
                -1
            ).expand(layers, batch, candidates, max_gt)
        else:
            raise ValueError(
                f"Unsupported matcher.object_cost_type: {self.cfg.object_cost_type!r}"
            )

        diff = (
            pred_x.unsqueeze(3) - gt_x.view(1, batch, 1, max_gt, rows)
        ).abs() / float(self.cfg.input_w)
        row_mask = gt_mask.view(1, batch, 1, max_gt, rows)
        valid_count = row_mask.sum(dim=-1).clamp_min(1)
        cost_point = (diff * row_mask.float()).sum(dim=-1) / valid_count
        valid_lane = gt_mask.sum(dim=-1) > 0
        cost_point = torch.where(
            valid_lane.view(1, batch, 1, max_gt),
            cost_point,
            torch.full_like(cost_point, 1e6),
        )

        pred_range = sort_range_norm(range_norm)
        gt_range_norm = gt_range / float(self.cfg.input_h)
        cost_range = (
            pred_range[..., 0].unsqueeze(-1)
            - gt_range_norm[:, :, 0].view(1, batch, 1, max_gt)
        ).abs() + (
            pred_range[..., 1].unsqueeze(-1)
            - gt_range_norm[:, :, 1].view(1, batch, 1, max_gt)
        ).abs()

        radius = float(self.cfg.line_iou_radius)
        pred = pred_x.unsqueeze(3)
        gt = gt_x.view(1, batch, 1, max_gt, rows)
        px1, px2 = pred - radius, pred + radius
        gx1, gx2 = gt - radius, gt + radius
        overlap = (
            torch.minimum(px2, gx2) - torch.maximum(px1, gx1)
        ).clamp(min=0.0)
        union = (4.0 * radius - overlap).clamp(min=self.cfg.eps)
        iou = overlap / union
        enclosing = (
            torch.maximum(px2, gx2) - torch.minimum(px1, gx1)
        ).clamp(min=self.cfg.eps)
        giou = iou - (enclosing - union) / enclosing
        line_cost_rows = 1.0 - giou
        cost_line_iou = (
            line_cost_rows * row_mask.float()
        ).sum(dim=-1) / valid_count
        cost_line_iou = torch.where(
            valid_lane.view(1, batch, 1, max_gt),
            cost_line_iou,
            torch.full_like(cost_line_iou, 1e6),
        )

        lambda_obj = self.effective_lambda_obj()
        cost = (
            lambda_obj * cost_obj
            + self.cfg.lambda_point * cost_point
            + self.cfg.lambda_range * cost_range
            + self.cfg.lambda_line_iou * cost_line_iou
        )
        return cost, {
            "mean_cost_obj": component_mean(cost_obj),
            "mean_cost_point": component_mean(cost_point),
            "mean_cost_range": component_mean(cost_range),
            "mean_cost_line_iou": component_mean(cost_line_iou),
            "matcher_lambda_obj": cost_obj.new_full(
                (layers, batch),
                lambda_obj,
            ),
        }, gt_counts

    def compute_cost_for_image(
        self,
        exist_logits: torch.Tensor,
        pred_x_rows: torch.Tensor,
        range_norm: torch.Tensor,
        target: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = pred_x_rows.device
        gt_x = target["x_rows"].to(device)
        gt_mask = target["valid_mask"].to(device).bool()
        gt_range = target["range_y"].to(device)
        n = int(pred_x_rows.shape[0])
        m = int(gt_x.shape[0])
        if m == 0:
            empty = torch.zeros((n, 0), device=device)
            return empty, {
                "mean_cost_obj": torch.tensor(0.0, device=device),
                "mean_cost_point": torch.tensor(0.0, device=device),
                "mean_cost_range": torch.tensor(0.0, device=device),
                "mean_cost_line_iou": torch.tensor(0.0, device=device),
                "matcher_lambda_obj": torch.tensor(
                    self.effective_lambda_obj(),
                    device=device,
                ),
            }

        cost_type = str(self.cfg.cost_type).strip().lower()
        if cost_type in {
            "range_aware_iou",
            "range_aware_raster_iou",
            "official_iou_surrogate",
        }:
            pairwise_iou, _candidate_valid, gt_lane_valid = (
                pairwise_range_aware_row_strip_iou(
                    pred_x_rows,
                    range_norm,
                    gt_x,
                    gt_mask,
                    input_h=int(self.cfg.input_h),
                    line_width=float(self.cfg.range_aware_line_width),
                    min_valid_rows=int(self.cfg.range_aware_min_valid_rows),
                )
            )
            cost = 1.0 - pairwise_iou
            cost = torch.where(
                gt_lane_valid.view(1, m),
                cost,
                torch.full_like(cost, 1e6),
            )
            zero = cost.detach().new_zeros(())
            return cost, {
                "mean_cost_obj": zero,
                "mean_cost_point": zero,
                "mean_cost_range": zero,
                "mean_cost_line_iou": cost.detach().mean(),
                "matcher_lambda_obj": zero.new_tensor(
                    self.effective_lambda_obj()
                ),
            }
        if cost_type not in {"composite", "legacy"}:
            raise ValueError(f"Unsupported matcher.cost_type: {self.cfg.cost_type!r}")

        p_lane = torch.softmax(exist_logits, dim=-1)[:, 0]
        object_cost_type = str(self.cfg.object_cost_type).strip().lower()
        if object_cost_type in {"neg_probability", "negative_probability", "minus_p"}:
            # Bounded classification cost, matching DETR-style assignment.
            # Unlike -log(p), this cannot overwhelm geometry merely because a
            # still-learning query has low confidence.
            cost_obj = -p_lane.view(n, 1).expand(n, m)
        elif object_cost_type in {"neg_log_probability", "negative_log_probability", "nll"}:
            cost_obj = -torch.log(p_lane.clamp_min(self.cfg.eps)).view(n, 1).expand(n, m)
        else:
            raise ValueError(f"Unsupported matcher.object_cost_type: {self.cfg.object_cost_type!r}")

        diff = (pred_x_rows[:, None, :] - gt_x[None, :, :]).abs() / float(self.cfg.input_w)
        mask = gt_mask[None, :, :].expand(n, m, -1)
        valid_count = mask.sum(dim=-1).clamp_min(1)
        cost_point = (diff * mask.float()).sum(dim=-1) / valid_count
        cost_point = torch.where(gt_mask.sum(dim=-1).view(1, m) > 0, cost_point, torch.full_like(cost_point, 1e6))

        pred_range = sort_range_norm(range_norm)
        gt_range_norm = gt_range / float(self.cfg.input_h)
        cost_range = (
            pred_range[:, None, 0].sub(gt_range_norm[None, :, 0]).abs()
            + pred_range[:, None, 1].sub(gt_range_norm[None, :, 1]).abs()
        )
        cost_line_iou = self.compute_line_iou_cost(pred_x_rows, gt_x, gt_mask)
        lambda_obj = self.effective_lambda_obj()
        cost = (
            lambda_obj * cost_obj
            + self.cfg.lambda_point * cost_point
            + self.cfg.lambda_range * cost_range
            + self.cfg.lambda_line_iou * cost_line_iou
        )
        return cost, {
            "mean_cost_obj": cost_obj.mean().detach(),
            "mean_cost_point": cost_point.mean().detach(),
            "mean_cost_range": cost_range.mean().detach(),
            "mean_cost_line_iou": cost_line_iou.mean().detach(),
            "matcher_lambda_obj": cost_obj.detach().new_tensor(lambda_obj),
        }

    def compute_line_iou_cost(
        self,
        pred_x_rows: torch.Tensor,
        gt_x: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        n = int(pred_x_rows.shape[0])
        m = int(gt_x.shape[0])
        radius = float(self.cfg.line_iou_radius)
        pred = pred_x_rows[:, None, :]
        gt = gt_x[None, :, :]
        px1 = pred - radius
        px2 = pred + radius
        gx1 = gt - radius
        gx2 = gt + radius
        overlap = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
        union = (4.0 * radius - overlap).clamp(min=self.cfg.eps)
        iou = overlap / union
        enclosing = (torch.maximum(px2, gx2) - torch.minimum(px1, gx1)).clamp(min=self.cfg.eps)
        giou = iou - (enclosing - union) / enclosing
        cost = 1.0 - giou
        mask = gt_mask[None, :, :].expand(n, m, -1)
        valid_count = mask.sum(dim=-1).clamp_min(1)
        cost = (cost * mask.float()).sum(dim=-1) / valid_count
        valid_lane = gt_mask.sum(dim=-1).view(1, m) > 0
        return torch.where(valid_lane, cost, torch.full_like(cost, 1e6))

    @staticmethod
    def _linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cost_cpu = cost.detach().cpu()
        try:
            from scipy.optimize import linear_sum_assignment

            row, col = linear_sum_assignment(cost_cpu.numpy())
            return torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)
        except Exception:
            n, m = cost_cpu.shape
            if m > 6:
                used: set[int] = set()
                rows = []
                cols = []
                for j in range(m):
                    values = cost_cpu[:, j].clone()
                    for r in used:
                        values[r] = float("inf")
                    r = int(values.argmin().item())
                    used.add(r)
                    rows.append(r)
                    cols.append(j)
                return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)
            best = None
            best_rows: tuple[int, ...] | None = None
            for rows in combinations(range(n), m):
                for row_perm in permutations(rows):
                    val = sum(float(cost_cpu[row_perm[j], j]) for j in range(m))
                    if best is None or val < best:
                        best = val
                        best_rows = row_perm
            assert best_rows is not None
            return torch.tensor(best_rows, dtype=torch.long), torch.arange(m, dtype=torch.long)

    def _grouped_assignment(self, cost: torch.Tensor, num_groups: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        n, _ = cost.shape
        num_groups = max(1, min(int(num_groups), n))
        edges = torch.linspace(0, n, num_groups + 1, dtype=torch.long, device=cost.device)
        pred_parts = []
        gt_parts = []
        for group_idx in range(num_groups):
            start = int(edges[group_idx].item())
            end = int(edges[group_idx + 1].item())
            if end <= start:
                continue
            row, col = self._linear_sum_assignment(cost[start:end])
            if row.numel() == 0:
                continue
            pred_parts.append(row + start)
            gt_parts.append(col)
        if not pred_parts:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        return torch.cat(pred_parts, dim=0), torch.cat(gt_parts, dim=0)

    def _grouped_assignment_with_sizes(
        self,
        cost: torch.Tensor,
        group_sizes: tuple[int, ...] | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = tuple(int(size) for size in group_sizes)
        if not sizes or any(size < 1 for size in sizes):
            raise ValueError("group_sizes must contain positive integers")
        if sum(sizes) != int(cost.shape[0]):
            raise ValueError(
                f"group_sizes sum to {sum(sizes)}, expected {int(cost.shape[0])} predictions"
            )
        pred_parts = []
        gt_parts = []
        start = 0
        for size in sizes:
            end = start + size
            row, col = self._linear_sum_assignment(cost[start:end])
            if row.numel() > 0:
                pred_parts.append(row + start)
                gt_parts.append(col)
            start = end
        if not pred_parts:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        return torch.cat(pred_parts, dim=0), torch.cat(gt_parts, dim=0)
