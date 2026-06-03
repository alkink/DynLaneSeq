from __future__ import annotations

from collections import defaultdict

import torch


class SmoothedLogger:
    def __init__(self):
        self.values = defaultdict(list)

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = float(v.detach().cpu())
            self.values[k].append(float(v))

    def format_and_reset(self, prefix: str = "") -> str:
        parts = []
        for key, vals in sorted(self.values.items()):
            if vals:
                value = sum(vals) / len(vals)
                parts.append(f"{key} {_format_value(value)}")
        self.values.clear()
        return prefix + " | ".join(parts)


def _format_value(value: float) -> str:
    if value == 0.0:
        return "0.0000"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def match_stats(outputs, matches):
    flat = outputs["coarse"] if "coarse" in outputs else outputs
    p_lane = torch.softmax(flat["exist_logits"], dim=-1)[..., 0]
    matched = []
    unmatched = []
    for b, match in enumerate(matches):
        mask = torch.zeros(p_lane.shape[1], dtype=torch.bool, device=p_lane.device)
        pred_idx = match["pred_indices"].to(p_lane.device)
        if pred_idx.numel() > 0:
            mask[pred_idx] = True
            matched.append(p_lane[b, mask])
        unmatched.append(p_lane[b, ~mask])
    mean_matched = torch.cat(matched).mean() if matched else p_lane.sum() * 0.0
    mean_unmatched = torch.cat(unmatched).mean() if unmatched else p_lane.sum() * 0.0
    return {"mean_p_lane_matched": mean_matched, "mean_p_lane_unmatched": mean_unmatched}
