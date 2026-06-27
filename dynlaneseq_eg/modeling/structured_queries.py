from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import soft_expected_x, sort_range_norm
from .evidence.orthogonal_sampler import OrthogonalEvidenceSampler


class RowAwareCrossAttentionLayer(nn.Module):
    """Let lane-row tokens read row evidence and exchange structured context."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        num_groups: int = 1,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        if self.num_groups < 1:
            raise ValueError("structured_query.num_groups must be >= 1")
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.inter_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.intra_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_inter = nn.LayerNorm(dim)
        self.norm_intra = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def _grouped_inter_attention(self, q: torch.Tensor, batch_rows: int, num_instances: int) -> torch.Tensor:
        if self.num_groups == 1:
            return self.inter_attn(q, q, q, need_weights=False)[0]
        if num_instances % self.num_groups != 0:
            raise ValueError(
                f"num_instances={num_instances} must be divisible by structured_query.num_groups={self.num_groups}"
            )
        group_size = num_instances // self.num_groups
        grouped = q.view(batch_rows, self.num_groups, group_size, q.shape[-1])
        grouped = grouped.reshape(batch_rows * self.num_groups, group_size, q.shape[-1])
        delta = self.inter_attn(grouped, grouped, grouped, need_weights=False)[0]
        return delta.view(batch_rows, self.num_groups, group_size, q.shape[-1]).reshape(
            batch_rows, num_instances, q.shape[-1]
        )

    def forward(self, row_tokens: torch.Tensor, row_value_features: torch.Tensor, row_key_features: torch.Tensor) -> torch.Tensor:
        b, n, r, c = row_tokens.shape
        _, rv, x_bins, _ = row_value_features.shape
        _, rk, key_x_bins, _ = row_key_features.shape
        if rv != r or rk != r:
            raise ValueError(f"row_features has value/key rows {rv}/{rk}, expected {r}")
        if key_x_bins != x_bins:
            raise ValueError(f"row key/value x bins differ: {key_x_bins} vs {x_bins}")

        # Row-local cross-attention: each row sees only horizontal evidence from the same row.
        q = row_tokens.permute(0, 2, 1, 3).reshape(b * r, n, c)
        q_norm = self.norm_cross(q)
        key = row_key_features.reshape(b * r, x_bins, c)
        value = row_value_features.reshape(b * r, x_bins, c)
        q = q + self.drop(self.cross_attn(q_norm, key, value, need_weights=False)[0])

        # Group-isolated interaction avoids letting one-to-many training groups suppress each other.
        q_norm = self.norm_inter(q)
        q = q + self.drop(self._grouped_inter_attention(q_norm, batch_rows=b * r, num_instances=n))
        q = q.view(b, r, n, c).permute(0, 2, 1, 3).contiguous()

        # Vertical interaction lets rows of the same lane share continuity and curvature context.
        lane_rows = q.reshape(b * n, r, c)
        lane_rows_norm = self.norm_intra(lane_rows)
        lane_rows = lane_rows + self.drop(
            self.intra_attn(lane_rows_norm, lane_rows_norm, lane_rows_norm, need_weights=False)[0]
        )
        lane_rows_norm = self.norm_ffn(lane_rows)
        lane_rows = lane_rows + self.drop(self.ffn(lane_rows_norm))
        return lane_rows.view(b, n, r, c).contiguous()


class QueryConditionedRowEvidenceHead(nn.Module):
    """Dynamic, query-conditioned row evidence logits for structured S0.

    This head lets each instance-row token produce a small channel kernel and
    scores every x-bin by a dot product against image features from the same
    row. In residual mode it starts as an exact no-op over the legacy MLP row
    logits, which makes checkpoint continuation controlled.
    """

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        x_bins: int = 200,
        evidence_dim: int = 64,
        mode: str = "residual",
        normalize: bool = True,
        dropout: float = 0.0,
        gamma_init: float = 0.0,
        logit_scale: float | None = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.evidence_dim = int(evidence_dim)
        self.mode = str(mode).lower()
        self.normalize = bool(normalize)
        self.logit_scale = float(1.0 if logit_scale is None else logit_scale)
        if self.evidence_dim < 1:
            raise ValueError("dynamic_row_evidence.evidence_dim must be >= 1")
        if self.mode not in {"residual", "replace"}:
            raise ValueError("dynamic_row_evidence.mode must be 'residual' or 'replace'")

        norm_groups = min(8, self.evidence_dim)
        while self.evidence_dim % norm_groups != 0:
            norm_groups -= 1
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.dim, self.evidence_dim, kernel_size=1),
            nn.GroupNorm(norm_groups, self.evidence_dim),
            nn.GELU(),
        )
        self.kernel_proj = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.evidence_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.evidence_dim, self.evidence_dim),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _evidence_features(self, features: torch.Tensor) -> torch.Tensor:
        evidence = self.feature_proj(features)
        if evidence.shape[-2:] != (self.num_rows, self.x_bins):
            evidence = F.interpolate(evidence, size=(self.num_rows, self.x_bins), mode="bilinear", align_corners=False)
        if self.normalize:
            evidence = F.normalize(evidence, dim=1)
        return evidence

    def forward(
        self,
        features: torch.Tensor,
        row_tokens: torch.Tensor,
        base_row_x_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        evidence = self._evidence_features(features)
        kernels = self.kernel_proj(row_tokens)
        if self.normalize:
            kernels = F.normalize(kernels, dim=-1)
        # evidence: [B, E, R, X], kernels: [B, N, R, E] -> [B, N, R, X]
        evidence_logits = torch.einsum("bnre,berx->bnrx", kernels, evidence)
        if not self.normalize:
            evidence_logits = evidence_logits * (1.0 / math.sqrt(float(self.evidence_dim)))
        evidence_logits = evidence_logits * self.logit_scale
        if self.mode == "replace":
            row_x_logits = self.gamma * evidence_logits
        else:
            row_x_logits = base_row_x_logits + self.gamma * evidence_logits
        debug = {
            "dynamic_row_evidence_gamma": self.gamma.detach(),
            "dynamic_row_evidence_logits_abs": evidence_logits.detach().abs().mean(),
            "dynamic_row_evidence_feature_abs": evidence.detach().abs().mean(),
            "dynamic_row_evidence_kernel_abs": kernels.detach().abs().mean(),
        }
        return row_x_logits, debug


class OrthogonalQualityVerifier(nn.Module):
    """Quality-only verifier from normal-direction visual evidence profiles."""

    def __init__(
        self,
        dim: int = 256,
        input_w: int = 800,
        input_h: int = 288,
        num_rows: int = 72,
        evidence_dim: int = 64,
        hidden_dim: int | None = None,
        offsets_px: list[float] | None = None,
        dropout: float = 0.0,
        detach_sample_x: bool = True,
        range_pad: float = 0.05,
        zero_init: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.evidence_dim = int(evidence_dim)
        hidden_dim = int(hidden_dim or dim)
        if self.evidence_dim < 1:
            raise ValueError("orthogonal_verifier.evidence_dim must be >= 1")

        norm_groups = min(8, self.evidence_dim)
        while self.evidence_dim % norm_groups != 0:
            norm_groups -= 1
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.dim, self.evidence_dim, kernel_size=1),
            nn.GroupNorm(norm_groups, self.evidence_dim),
            nn.GELU(),
        )
        self.sampler = OrthogonalEvidenceSampler(
            input_w=input_w,
            input_h=input_h,
            num_rows=num_rows,
            offsets_px=offsets_px,
            detach_sample_x=detach_sample_x,
            range_pad=range_pad,
        )
        offsets = self.sampler.offsets_px.detach().clone()
        self.register_buffer("offset_signs", offsets.sign().view(1, 1, 1, -1, 1))
        self.center_index = int(torch.argmin(offsets.abs()).item())
        profile_dim = self.evidence_dim * 4
        self.profile_mlp = nn.Sequential(
            nn.LayerNorm(profile_dim),
            nn.Linear(profile_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.dim),
            nn.GELU(),
        )
        self.row_adapter = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.dim),
        )
        self.quality_delta = nn.Sequential(
            nn.LayerNorm(self.dim * 3),
            nn.Linear(self.dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        if bool(zero_init):
            nn.init.zeros_(self.row_adapter[-1].weight)
            nn.init.zeros_(self.row_adapter[-1].bias)
            nn.init.zeros_(self.quality_delta[-1].weight)
            nn.init.zeros_(self.quality_delta[-1].bias)

    @staticmethod
    def _masked_mean(samples: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask_f = mask.to(dtype=samples.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=dim).clamp_min(1.0)
        return (samples * mask_f).sum(dim=dim) / denom

    def _offset_group_mean(self, samples: torch.Tensor, valid: torch.Tensor, sign: int) -> torch.Tensor:
        group = (self.offset_signs == float(sign)).to(device=samples.device)
        group_mask = valid.unsqueeze(-1) & group.bool()
        return self._masked_mean(samples, group_mask.squeeze(-1), dim=3)

    def forward(
        self,
        features: torch.Tensor,
        row_tokens: torch.Tensor,
        lane_query: torch.Tensor,
        pred_x_rows: torch.Tensor,
        range_norm: torch.Tensor,
        base_quality_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        evidence_features = self.feature_proj(features)
        samples, valid, sampler_debug = self.sampler(evidence_features, pred_x_rows, range_norm=range_norm)
        center = samples[:, :, :, self.center_index, :]
        center_valid = valid[:, :, :, self.center_index].unsqueeze(-1).to(dtype=samples.dtype)
        center = center * center_valid
        mean_all = self._masked_mean(samples, valid, dim=3)
        mean_left = self._offset_group_mean(samples, valid, sign=-1)
        mean_right = self._offset_group_mean(samples, valid, sign=1)
        profile = torch.cat([center, mean_all, mean_left, mean_right], dim=-1)
        row_evidence = self.profile_mlp(profile)

        row_valid = valid.any(dim=3)
        lane_evidence = self._masked_mean(row_evidence, row_valid, dim=2)
        adapted_rows = row_tokens + self.row_adapter(row_evidence)
        lane_row_context = self._masked_mean(adapted_rows, row_valid, dim=2)
        quality_input = torch.cat([lane_query, lane_evidence, lane_row_context], dim=-1)
        delta = self.quality_delta(quality_input).squeeze(-1)
        quality_logits = base_quality_logits + delta
        debug = {
            **sampler_debug,
            "orthogonal_evidence_abs": samples.detach().abs().mean(),
            "orthogonal_row_evidence_abs": row_evidence.detach().abs().mean(),
            "orthogonal_quality_delta_abs": delta.detach().abs().mean(),
        }
        return quality_logits, debug


class OrthogonalRowTokenGrounder(nn.Module):
    """Inject hypothesis-conditioned visual evidence into structured row tokens.

    This is intentionally earlier than the quality-only verifier: it samples
    normal-direction evidence around a draft lane, converts the profile into
    row evidence, and updates row tokens before the final row_x/range/quality
    heads run.  With zero initialization it is an exact no-op at iteration 0,
    which keeps checkpoint initialization controlled.
    """

    def __init__(
        self,
        dim: int = 256,
        input_w: int = 800,
        input_h: int = 288,
        num_rows: int = 72,
        evidence_dim: int = 64,
        hidden_dim: int | None = None,
        offsets_px: list[float] | None = None,
        dropout: float = 0.0,
        detach_sample_x: bool = True,
        range_pad: float = 0.05,
        zero_init: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.evidence_dim = int(evidence_dim)
        hidden_dim = int(hidden_dim or dim)
        if self.evidence_dim < 1:
            raise ValueError("orthogonal_grounder.evidence_dim must be >= 1")

        norm_groups = min(8, self.evidence_dim)
        while self.evidence_dim % norm_groups != 0:
            norm_groups -= 1
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.dim, self.evidence_dim, kernel_size=1),
            nn.GroupNorm(norm_groups, self.evidence_dim),
            nn.GELU(),
        )
        self.sampler = OrthogonalEvidenceSampler(
            input_w=input_w,
            input_h=input_h,
            num_rows=num_rows,
            offsets_px=offsets_px,
            detach_sample_x=detach_sample_x,
            range_pad=range_pad,
        )
        offsets = self.sampler.offsets_px.detach().clone()
        self.register_buffer("offset_signs", offsets.sign().view(1, 1, 1, -1, 1))
        self.center_index = int(torch.argmin(offsets.abs()).item())
        profile_dim = self.evidence_dim * 4
        self.profile_mlp = nn.Sequential(
            nn.LayerNorm(profile_dim),
            nn.Linear(profile_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.dim),
            nn.GELU(),
        )
        self.row_update = nn.Sequential(
            nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.dim),
        )
        if bool(zero_init):
            nn.init.zeros_(self.row_update[-1].weight)
            nn.init.zeros_(self.row_update[-1].bias)

    @staticmethod
    def _masked_mean(samples: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask_f = mask.to(dtype=samples.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=dim).clamp_min(1.0)
        return (samples * mask_f).sum(dim=dim) / denom

    def _offset_group_mean(self, samples: torch.Tensor, valid: torch.Tensor, sign: int) -> torch.Tensor:
        group = (self.offset_signs == float(sign)).to(device=samples.device)
        group_mask = valid.unsqueeze(-1) & group.bool()
        return self._masked_mean(samples, group_mask.squeeze(-1), dim=3)

    def forward(
        self,
        features: torch.Tensor,
        row_tokens: torch.Tensor,
        draft_x_rows: torch.Tensor,
        draft_range_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        evidence_features = self.feature_proj(features)
        samples, valid, sampler_debug = self.sampler(evidence_features, draft_x_rows, range_norm=draft_range_norm)
        center = samples[:, :, :, self.center_index, :]
        center_valid = valid[:, :, :, self.center_index].unsqueeze(-1).to(dtype=samples.dtype)
        center = center * center_valid
        mean_all = self._masked_mean(samples, valid, dim=3)
        mean_left = self._offset_group_mean(samples, valid, sign=-1)
        mean_right = self._offset_group_mean(samples, valid, sign=1)
        profile = torch.cat([center, mean_all, mean_left, mean_right], dim=-1)
        row_evidence = self.profile_mlp(profile)

        row_valid = valid.any(dim=3).unsqueeze(-1).to(dtype=row_tokens.dtype)
        delta = self.row_update(torch.cat([row_tokens, row_evidence.to(dtype=row_tokens.dtype)], dim=-1))
        delta = delta * row_valid
        grounded_rows = row_tokens + delta
        debug = {
            **sampler_debug,
            "orthogonal_grounder_evidence_abs": samples.detach().abs().mean(),
            "orthogonal_grounder_row_evidence_abs": row_evidence.detach().abs().mean(),
            "orthogonal_grounder_delta_abs": delta.detach().abs().mean(),
        }
        return grounded_rows, debug


class StructuredLaneQueryHead(nn.Module):
    """Instance-geometry S0 head with row-wise image evidence.

    The head keeps the DynLaneSeq output contract but replaces a single slot
    vector with a lane instance token plus per-row geometry tokens.
    """

    def __init__(
        self,
        dim: int = 256,
        num_instances: int = 64,
        num_rows: int = 72,
        x_bins: int = 200,
        input_w: int = 800,
        input_h: int = 288,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        use_x_pos: bool = True,
        num_groups: int = 1,
        exist_prior_prob: float | None = None,
        dynamic_row_evidence: dict[str, Any] | None = None,
        orthogonal_grounder: dict[str, Any] | None = None,
        orthogonal_verifier: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_instances = int(num_instances)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.use_x_pos = bool(use_x_pos)
        self.num_groups = int(num_groups)
        self.exist_prior_prob = None if exist_prior_prob is None else float(exist_prior_prob)
        dynamic_row_evidence = dynamic_row_evidence or {}
        orthogonal_grounder = orthogonal_grounder or {}
        orthogonal_verifier = orthogonal_verifier or {}
        self.dynamic_row_evidence_enabled = bool(dynamic_row_evidence.get("enabled", False))
        self.orthogonal_grounder_enabled = bool(orthogonal_grounder.get("enabled", False))
        self.orthogonal_verifier_enabled = bool(orthogonal_verifier.get("enabled", False))
        if self.num_groups < 1:
            raise ValueError("structured_query.num_groups must be >= 1")
        if self.num_instances % self.num_groups != 0:
            raise ValueError(
                f"structured_query.num_instances={self.num_instances} must be divisible by num_groups={self.num_groups}"
            )
        if self.exist_prior_prob is not None and not 0.0 < self.exist_prior_prob < 1.0:
            raise ValueError("structured_query.exist_prior_prob must be between 0 and 1")

        self.instance_tokens = nn.Embedding(self.num_instances, self.dim)
        self.row_tokens = nn.Embedding(self.num_rows, self.dim)
        self.x_tokens = nn.Embedding(self.x_bins, self.dim) if self.use_x_pos else None
        nn.init.normal_(self.instance_tokens.weight, std=0.02)
        nn.init.normal_(self.row_tokens.weight, std=0.02)
        if self.x_tokens is not None:
            nn.init.normal_(self.x_tokens.weight, std=0.02)

        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1),
            nn.GroupNorm(8, self.dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                RowAwareCrossAttentionLayer(
                    dim=self.dim,
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                    num_groups=self.num_groups,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.row_norm = nn.LayerNorm(self.dim)
        self.lane_norm = nn.LayerNorm(self.dim)
        self.row_x = nn.Linear(self.dim, self.x_bins)
        self.dynamic_row_evidence_head = (
            QueryConditionedRowEvidenceHead(
                dim=self.dim,
                num_rows=self.num_rows,
                x_bins=self.x_bins,
                evidence_dim=int(dynamic_row_evidence.get("evidence_dim", 64)),
                mode=str(dynamic_row_evidence.get("mode", "residual")),
                normalize=bool(dynamic_row_evidence.get("normalize", True)),
                dropout=float(dynamic_row_evidence.get("dropout", 0.0)),
                gamma_init=float(dynamic_row_evidence.get("gamma_init", 0.0)),
                logit_scale=dynamic_row_evidence.get("logit_scale", 1.0),
            )
            if self.dynamic_row_evidence_enabled
            else None
        )
        self.orthogonal_grounder = (
            OrthogonalRowTokenGrounder(
                dim=self.dim,
                input_w=self.input_w,
                input_h=self.input_h,
                num_rows=self.num_rows,
                evidence_dim=int(orthogonal_grounder.get("evidence_dim", 64)),
                hidden_dim=int(orthogonal_grounder.get("hidden_dim", self.dim)),
                offsets_px=orthogonal_grounder.get("offsets_px"),
                dropout=float(orthogonal_grounder.get("dropout", 0.0)),
                detach_sample_x=bool(orthogonal_grounder.get("detach_sample_x", True)),
                range_pad=float(orthogonal_grounder.get("range_pad", 0.05)),
                zero_init=bool(orthogonal_grounder.get("zero_init", True)),
            )
            if self.orthogonal_grounder_enabled
            else None
        )
        self.orthogonal_verifier = (
            OrthogonalQualityVerifier(
                dim=self.dim,
                input_w=self.input_w,
                input_h=self.input_h,
                num_rows=self.num_rows,
                evidence_dim=int(orthogonal_verifier.get("evidence_dim", 64)),
                hidden_dim=int(orthogonal_verifier.get("hidden_dim", self.dim)),
                offsets_px=orthogonal_verifier.get("offsets_px"),
                dropout=float(orthogonal_verifier.get("dropout", 0.0)),
                detach_sample_x=bool(orthogonal_verifier.get("detach_sample_x", True)),
                range_pad=float(orthogonal_verifier.get("range_pad", 0.05)),
                zero_init=bool(orthogonal_verifier.get("zero_init", True)),
            )
            if self.orthogonal_verifier_enabled
            else None
        )
        self.exist = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 2))
        self.range = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 2))
        self.quality = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 1))
        nn.init.constant_(self.range[-1].weight, 0.0)
        with torch.no_grad():
            self.range[-1].bias.copy_(torch.tensor([-2.0, 2.0]))
            if self.exist_prior_prob is not None:
                lane_logit = 0.5 * math.log(self.exist_prior_prob / (1.0 - self.exist_prior_prob))
                self.exist[-1].bias.copy_(self.exist[-1].bias.new_tensor([lane_logit, -lane_logit]))

    def _row_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature_proj(features)
        if feat.shape[-2:] != (self.num_rows, self.x_bins):
            feat = F.interpolate(feat, size=(self.num_rows, self.x_bins), mode="bilinear", align_corners=False)
        b, c, r, x = feat.shape
        feat_value = feat.permute(0, 2, 3, 1).contiguous()
        feat_key = feat_value
        if self.x_tokens is not None:
            x_pos = self.x_tokens.weight.to(device=features.device, dtype=features.dtype).view(1, 1, x, c)
            feat_key = feat_key + x_pos
        return feat_value, feat_key

    def _lane_query_from_rows(self, row_tokens: torch.Tensor, instance: torch.Tensor) -> torch.Tensor:
        b = int(row_tokens.shape[0])
        instance_residual = instance.unsqueeze(0).expand(b, -1, -1)
        return self.lane_norm(row_tokens.mean(dim=2) + row_tokens.amax(dim=2) + instance_residual)

    def _row_logits_from_rows(
        self,
        features: torch.Tensor,
        row_tokens: torch.Tensor,
        structured_debug: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        base_row_x_logits = self.row_x(row_tokens)
        if self.dynamic_row_evidence_head is not None:
            row_x_logits, dynamic_debug = self.dynamic_row_evidence_head(features, row_tokens, base_row_x_logits)
            structured_debug.update(dynamic_debug)
            return row_x_logits
        return base_row_x_logits

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        b = int(features.shape[0])
        dtype = features.dtype
        device = features.device
        instance = self.instance_tokens.weight.to(device=device, dtype=dtype)
        row = self.row_tokens.weight.to(device=device, dtype=dtype)
        row_tokens = instance[:, None, :] + row[None, :, :]
        row_tokens = row_tokens.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        row_value_features, row_key_features = self._row_features(features)

        for layer in self.layers:
            row_tokens = layer(row_tokens, row_value_features, row_key_features)

        row_tokens = self.row_norm(row_tokens)
        structured_debug = {
            "structured_row_abs": row_tokens.detach()[..., :16].abs().mean(),
            "structured_feature_abs": row_value_features.detach()[..., :16].abs().mean(),
        }
        lane_query = self._lane_query_from_rows(row_tokens, instance)
        row_x_logits = self._row_logits_from_rows(features, row_tokens, structured_debug)
        pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
        range_raw = self.range(lane_query)
        range_norm = sort_range_norm(torch.sigmoid(range_raw))

        if self.orthogonal_grounder is not None:
            draft_x_rows = pred_x_rows
            draft_range_norm = range_norm
            row_tokens, grounder_debug = self.orthogonal_grounder(
                features,
                row_tokens,
                draft_x_rows,
                draft_range_norm,
            )
            row_tokens = self.row_norm(row_tokens)
            structured_debug.update(grounder_debug)
            lane_query = self._lane_query_from_rows(row_tokens, instance)
            row_x_logits = self._row_logits_from_rows(features, row_tokens, structured_debug)
            pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
            range_raw = self.range(lane_query)
            range_norm = sort_range_norm(torch.sigmoid(range_raw))

        quality_logits = self.quality(lane_query).squeeze(-1)
        if self.orthogonal_verifier is not None:
            quality_logits, verifier_debug = self.orthogonal_verifier(
                features,
                row_tokens,
                lane_query,
                pred_x_rows,
                range_norm,
                quality_logits,
            )
            structured_debug.update(verifier_debug)
        return {
            "exist_logits": self.exist(lane_query),
            "row_x_logits": row_x_logits,
            "pred_x_rows": pred_x_rows,
            "range_raw": range_raw,
            "range_norm": range_norm,
            "quality_logits": quality_logits,
            "queries": lane_query,
            "structured_row_tokens": row_tokens,
            "structured_debug": structured_debug,
        }


def build_structured_query_head(model_cfg: dict[str, Any]) -> StructuredLaneQueryHead | None:
    structured_cfg = model_cfg.get("structured_query", {})
    if not bool(structured_cfg.get("enabled", False)):
        return None
    return StructuredLaneQueryHead(
        dim=int(model_cfg.get("dim", 256)),
        num_instances=int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 20))),
        num_rows=int(model_cfg.get("num_rows", 72)),
        x_bins=int(model_cfg.get("x_bins", 200)),
        input_w=int(model_cfg.get("input_w", 800)),
        input_h=int(model_cfg.get("input_h", 288)),
        num_heads=int(structured_cfg.get("num_heads", model_cfg.get("num_heads", 8))),
        num_layers=int(structured_cfg.get("num_layers", 2)),
        ff_dim=int(structured_cfg.get("ff_dim", model_cfg.get("decoder_ff_dim", 1024))),
        dropout=float(structured_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        use_x_pos=bool(structured_cfg.get("use_x_pos", True)),
        num_groups=int(structured_cfg.get("num_groups", 1)),
        exist_prior_prob=structured_cfg.get("exist_prior_prob"),
        dynamic_row_evidence=structured_cfg.get("dynamic_row_evidence"),
        orthogonal_grounder=structured_cfg.get("orthogonal_grounder"),
        orthogonal_verifier=structured_cfg.get("orthogonal_verifier"),
    )
