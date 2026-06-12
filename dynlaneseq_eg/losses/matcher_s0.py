from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import torch

from dynlaneseq_eg.modeling.common import sort_range_norm


@dataclass
class MatcherConfig:
    lambda_obj: float = 2.0
    lambda_point: float = 5.0
    lambda_range: float = 1.0
    lambda_line_iou: float = 0.0
    line_iou_radius: float = 7.5
    input_w: int = 800
    input_h: int = 288
    eps: float = 1e-6
    assignment: str = "hungarian"
    num_groups: int = 1


class HungarianMatcherS0:
    def __init__(self, cfg: MatcherConfig | None = None):
        self.cfg = cfg or MatcherConfig()

    @torch.no_grad()
    def __call__(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        if "coarse" in outputs:
            outputs = outputs["coarse"]
        matches = []
        for b, target in enumerate(targets):
            cost, stats = self.compute_cost_for_image(
                outputs["exist_logits"][b],
                outputs["pred_x_rows"][b],
                outputs["range_norm"][b],
                target,
            )
            num_gt = int(target["x_rows"].shape[0])
            if num_gt == 0:
                pred_idx = torch.empty(0, dtype=torch.long)
                gt_idx = torch.empty(0, dtype=torch.long)
            elif self.cfg.assignment == "grouped_one_to_many":
                pred_idx, gt_idx = self._grouped_assignment(cost, num_groups=max(1, int(self.cfg.num_groups)))
            else:
                pred_idx, gt_idx = self._linear_sum_assignment(cost)
            matches.append(
                {
                    "pred_indices": pred_idx,
                    "gt_indices": gt_idx,
                    "num_gt": torch.tensor(num_gt, dtype=torch.long),
                    "num_matched": torch.tensor(int(pred_idx.numel()), dtype=torch.long),
                    **stats,
                }
            )
        return matches

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
            }

        p_lane = torch.softmax(exist_logits, dim=-1)[:, 0]
        cost_obj = -torch.log(p_lane.clamp_min(self.cfg.eps)).view(n, 1).expand(n, m)

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
        cost = (
            self.cfg.lambda_obj * cost_obj
            + self.cfg.lambda_point * cost_point
            + self.cfg.lambda_range * cost_range
            + self.cfg.lambda_line_iou * cost_line_iou
        )
        return cost, {
            "mean_cost_obj": cost_obj.mean().detach(),
            "mean_cost_point": cost_point.mean().detach(),
            "mean_cost_range": cost_range.mean().detach(),
            "mean_cost_line_iou": cost_line_iou.mean().detach(),
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
