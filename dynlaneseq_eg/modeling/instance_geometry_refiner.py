from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class RowStateRefinerBlock(nn.Module):
    """Gated 1D row-state block for per-lane continuous geometry tokens."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("RowStateRefinerBlock expects an odd kernel_size")
        padding = (kernel_size // 2) * int(dilation)
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=int(dilation), groups=dim)
        self.in_proj = nn.Conv1d(dim, int(hidden_dim) * 2, kernel_size=1)
        self.out_proj = nn.Conv1d(int(hidden_dim), dim, kernel_size=1)
        self.drop = nn.Dropout(float(dropout))
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        x = self.norm(tokens).transpose(1, 2).contiguous()
        x = self.depthwise(x)
        value, gate = self.in_proj(x).chunk(2, dim=1)
        x = F.gelu(value) * torch.sigmoid(gate)
        x = self.out_proj(self.drop(x)).transpose(1, 2).contiguous()
        return residual + self.drop(x)


class GatedInstanceMixer(nn.Module):
    """Non-softmax lane-instance mixer using group and scene context."""

    def __init__(
        self,
        dim: int,
        num_groups: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_groups = max(1, int(num_groups))
        hidden = int(hidden_dim or dim * 2)
        self.norm = nn.LayerNorm(self.dim * 5)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 5, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.dim * 2),
        )
        self.out_norm = nn.LayerNorm(self.dim)
        self.drop = nn.Dropout(float(dropout))
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def _group_context(self, lane_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, c = lane_tokens.shape
        if self.num_groups <= 1 or n % self.num_groups != 0:
            mean = lane_tokens.mean(dim=1, keepdim=True).expand(b, n, c)
            maxv = lane_tokens.amax(dim=1, keepdim=True).expand(b, n, c)
            return mean, maxv
        per_group = n // self.num_groups
        grouped = lane_tokens.view(b, self.num_groups, per_group, c)
        mean = grouped.mean(dim=2, keepdim=True).expand(b, self.num_groups, per_group, c).reshape(b, n, c)
        maxv = grouped.amax(dim=2, keepdim=True).expand(b, self.num_groups, per_group, c).reshape(b, n, c)
        return mean, maxv

    def forward(self, lane_tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = lane_tokens.shape
        group_mean, group_max = self._group_context(lane_tokens)
        scene_mean = lane_tokens.mean(dim=1, keepdim=True).expand(b, n, c)
        scene_max = lane_tokens.amax(dim=1, keepdim=True).expand(b, n, c)
        context = torch.cat([lane_tokens, group_mean, group_max, scene_mean, scene_max], dim=-1)
        delta, gate = self.mlp(self.norm(context)).chunk(2, dim=-1)
        mixed = lane_tokens + self.drop(torch.sigmoid(gate) * delta)
        return self.out_norm(mixed)


class InstanceGeometryTopologyRefiner(nn.Module):
    """S1 refiner that preserves q_ins/q_geo structure instead of flattening row tokens."""

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        x_bins: int = 200,
        num_groups: int = 1,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        kernel_size: int = 9,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
        zero_init: bool = True,
        visibility_head: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.visibility_head_enabled = bool(visibility_head)
        self.coarse_x_embed = nn.Linear(1, self.dim)
        self.aux_proj = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self.dim),
        )
        dilation_values = list(dilations or [])
        if not dilation_values:
            dilation_values = [1] * int(num_blocks)
        if len(dilation_values) < int(num_blocks):
            dilation_values = dilation_values + [dilation_values[-1]] * (int(num_blocks) - len(dilation_values))
        self.row_blocks = nn.ModuleList(
            [
                RowStateRefinerBlock(
                    dim=self.dim,
                    hidden_dim=int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation_values[idx]),
                    dropout=float(dropout),
                    zero_init=bool(zero_init),
                )
                for idx in range(int(num_blocks))
            ]
        )
        self.geometry_to_instance = nn.Sequential(
            nn.LayerNorm(self.dim * 3),
            nn.Linear(self.dim * 3, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.dim),
        )
        self.instance_mixer = GatedInstanceMixer(
            dim=self.dim,
            num_groups=int(num_groups),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            zero_init=bool(zero_init),
        )
        self.instance_to_geometry = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.dim * 2),
        )
        self.out_norm = nn.LayerNorm(self.dim)
        self.delta_head = nn.Linear(self.dim, self.x_bins)
        self.quality_head = nn.Sequential(
            nn.LayerNorm(self.dim * 3),
            nn.Linear(self.dim * 3, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.visibility_head = nn.Linear(self.dim, 1) if self.visibility_head_enabled else None
        self.drop = nn.Dropout(float(dropout))
        if zero_init:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
            nn.init.zeros_(self.quality_head[-1].weight)
            nn.init.zeros_(self.quality_head[-1].bias)
            if self.visibility_head is not None:
                nn.init.zeros_(self.visibility_head.weight)
                nn.init.zeros_(self.visibility_head.bias)

    def forward(
        self,
        q_ins: torch.Tensor,
        q_geo: torch.Tensor,
        coarse_x_norm: torch.Tensor,
        coarse_quality_logits: torch.Tensor | None = None,
        range_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        b, n, p, c = q_geo.shape
        if p != self.num_rows:
            raise ValueError(f"Expected {self.num_rows} rows, got {p}")
        if c != self.dim or q_ins.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got q_geo={c}, q_ins={q_ins.shape[-1]}")
        if coarse_x_norm.shape[:3] != (b, n, p):
            raise ValueError("coarse_x_norm must have shape [B, N, num_rows, 1]")

        row_tokens = q_geo + self.coarse_x_embed(coarse_x_norm)
        flat = row_tokens.reshape(b * n, p, c)
        for block in self.row_blocks:
            flat = block(flat)
        row_state = flat.reshape(b, n, p, c)

        geo_mean = row_state.mean(dim=2)
        geo_max = row_state.amax(dim=2)
        instance_update = self.geometry_to_instance(torch.cat([q_ins, geo_mean, geo_max], dim=-1))
        aux = row_state.new_zeros((b, n, 3))
        if coarse_quality_logits is not None:
            aux[..., 0] = coarse_quality_logits.to(dtype=row_state.dtype)
        if range_norm is not None:
            aux[..., 1:] = range_norm.to(dtype=row_state.dtype)
        lane_tokens = q_ins + instance_update + self.aux_proj(aux)
        lane_tokens = self.instance_mixer(lane_tokens)

        film = self.instance_to_geometry(lane_tokens)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(2)
        beta = beta.unsqueeze(2)
        hidden = self.out_norm(row_state + self.drop(gamma * row_state + beta))

        delta_logits = self.delta_head(hidden)
        hidden_mean = hidden.mean(dim=2)
        hidden_max = hidden.amax(dim=2)
        quality_delta = self.quality_head(torch.cat([lane_tokens, hidden_mean, hidden_max], dim=-1)).squeeze(-1)
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "row_x_logits": delta_logits,
            "row_hidden": hidden,
            "quality_delta_logits": quality_delta,
            "igt_debug": {
                "igt_row_hidden_abs": hidden.detach().abs().mean(),
                "igt_instance_update_abs": instance_update.detach().abs().mean(),
                "igt_delta_logits_abs": delta_logits.detach().abs().mean(),
                "igt_quality_delta_abs": quality_delta.detach().abs().mean(),
            },
        }
        if self.visibility_head is not None:
            out["row_visibility_logits"] = self.visibility_head(hidden).squeeze(-1)
        return out


def build_instance_geometry_refiner(model_cfg: dict[str, Any]) -> InstanceGeometryTopologyRefiner:
    cfg = model_cfg.get("igt_refiner", {})
    structured_cfg = model_cfg.get("structured_query", {})
    return InstanceGeometryTopologyRefiner(
        dim=int(model_cfg.get("dim", 256)),
        num_rows=int(model_cfg.get("num_rows", 72)),
        x_bins=int(model_cfg.get("x_bins", 200)),
        num_groups=int(cfg.get("num_groups", structured_cfg.get("num_groups", 1))),
        hidden_dim=int(cfg.get("hidden_dim", model_cfg.get("row_decoder_ff_dim", 512))),
        num_blocks=int(cfg.get("num_blocks", 3)),
        kernel_size=int(cfg.get("kernel_size", 9)),
        dilations=list(cfg.get("dilations", [1, 2, 4])),
        dropout=float(cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        zero_init=bool(cfg.get("zero_init", True)),
        visibility_head=bool(cfg.get("visibility_head", model_cfg.get("row_visibility", {}).get("enabled", False))),
    )
