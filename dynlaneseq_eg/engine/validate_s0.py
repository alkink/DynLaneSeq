from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.common import nested_to_device


@torch.no_grad()
def validate_simple(
    model,
    dataloader,
    matcher,
    device: torch.device,
    max_batches: int | None = None,
    score_thresh: float = 0.5,
) -> dict[str, float]:
    model.eval()
    total_abs = 0.0
    total_points = 0
    total_gt = 0
    total_pred_lanes = 0
    matched_scores = []
    unmatched_scores = []
    num_images = 0
    for step, (images, targets, metas) in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device)
        targets = nested_to_device(targets, device)
        outputs = model(images)
        flat = outputs.get("stage2") or outputs.get("final") or outputs
        matches = matcher(outputs, targets)
        for b, match in enumerate(matches):
            num_images += 1
            pred_idx = match["pred_indices"].to(device)
            gt_idx = match["gt_indices"].to(device)
            p_lane = torch.softmax(flat["exist_logits"][b], dim=-1)[:, 0]
            slot_mask = torch.zeros_like(p_lane, dtype=torch.bool)
            if pred_idx.numel() > 0:
                slot_mask[pred_idx] = True
                matched_scores.append(p_lane[pred_idx].detach())
            unmatched_scores.append(p_lane[~slot_mask].detach())
            total_gt += int(targets[b]["x_rows"].shape[0])
            total_pred_lanes += int((p_lane > score_thresh).sum().item())
            if pred_idx.numel() == 0:
                continue
            pred = flat["pred_x_rows"][b, pred_idx]
            gt = targets[b]["x_rows"][gt_idx]
            mask = targets[b]["valid_mask"][gt_idx].bool()
            total_abs += float((pred[mask] - gt[mask]).abs().sum().item())
            total_points += int(mask.sum().item())
    matched_cat = torch.cat(matched_scores) if matched_scores else torch.tensor([0.0], device=device)
    unmatched_cat = torch.cat(unmatched_scores) if unmatched_scores else torch.tensor([0.0], device=device)
    return {
        "mean_point_error": total_abs / max(total_points, 1),
        "total_points": float(total_points),
        "total_gt_lanes": float(total_gt),
        "avg_gt_lanes_per_image": total_gt / max(num_images, 1),
        "avg_pred_lanes_per_image": total_pred_lanes / max(num_images, 1),
        "score_thresh": float(score_thresh),
        "mean_p_lane_matched": float(matched_cat.mean().detach().cpu()),
        "mean_p_lane_unmatched": float(unmatched_cat.mean().detach().cpu()),
        "max_p_lane_matched": float(matched_cat.max().detach().cpu()),
        "max_p_lane_unmatched": float(unmatched_cat.max().detach().cpu()),
    }
