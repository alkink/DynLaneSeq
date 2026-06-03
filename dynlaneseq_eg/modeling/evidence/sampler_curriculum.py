from __future__ import annotations

import torch


class SamplerCurriculum:
    def __init__(
        self,
        noise_std: float = 3.0,
        detach_sample_coords: bool = True,
        input_w: int = 800,
    ):
        self.noise_std = noise_std
        self.detach_sample_coords = detach_sample_coords
        self.input_w = input_w

    @staticmethod
    def alpha_from_iter(iteration: int, warmup_iters: int = 1000, decay_iters: int = 1000) -> float:
        if iteration < warmup_iters:
            return 1.0
        if iteration >= warmup_iters + decay_iters:
            return 0.0
        return 1.0 - float(iteration - warmup_iters) / float(decay_iters)

    def build_sample_x(
        self,
        coarse_x: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None,
        matches: list[dict[str, torch.Tensor]] | None,
        alpha: float = 0.0,
        add_noise: bool = True,
    ) -> torch.Tensor:
        sample_x = coarse_x.clone()
        if targets is not None and matches is not None and alpha > 0:
            for b, match in enumerate(matches):
                pred_idx = match["pred_indices"].to(coarse_x.device)
                gt_idx = match["gt_indices"].to(coarse_x.device)
                if pred_idx.numel() == 0:
                    continue
                gt_x = targets[b]["x_rows"].to(coarse_x.device)[gt_idx]
                valid = targets[b]["valid_mask"].to(coarse_x.device)[gt_idx].bool()
                pred = coarse_x[b, pred_idx]
                mixed = alpha * gt_x + (1.0 - alpha) * pred
                if add_noise and alpha >= 1.0 and self.noise_std > 0:
                    mixed = mixed + torch.randn_like(mixed) * self.noise_std
                sample_x[b, pred_idx] = torch.where(valid, mixed, pred)
        sample_x = sample_x.clamp(0, self.input_w - 1)
        return sample_x.detach() if self.detach_sample_coords else sample_x

