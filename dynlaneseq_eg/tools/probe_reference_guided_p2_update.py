from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, input_to_grid, nested_to_device
from dynlaneseq_eg.tools.analyze_attention_acquisition import (
    _group_zero_assignments,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_attention_coordinate_adapter import (
    GeometryStats,
    _best_iou,
    seed_everything,
)


PROBE_MODES: dict[str, tuple[bool, bool]] = {
    "anchor_only": (False, False),
    "state_only": (True, False),
    "local_p2": (False, True),
    "state_local_p2": (True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a trained LaneRowNet detector at an intermediate decoder "
            "layer and fit equal-capacity reference-update probes. Each probe "
            "sees the same explicit row reference; controlled variants add the "
            "intermediate row state, local P2 samples, or both. The held-out "
            "comparison tests the core precondition of a dynamic row-reference "
            "decoder without committing to a full detector training run."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--anchor-layer", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--offsets-px",
        type=float,
        nargs="+",
        default=[-64, -32, -16, -8, 0, 8, 16, 32, 64],
    )
    parser.add_argument("--smooth-l1-beta-px", type=float, default=3.0)
    parser.add_argument("--point-loss-weight", type=float, default=2.0)
    parser.add_argument("--line-iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument(
        "--train-group-mode",
        choices=("all", "group0"),
        default="all",
        help="Use all train-time group assignments or only deployment group zero.",
    )
    parser.add_argument("--eval-max-batches", type=int, default=32)
    parser.add_argument(
        "--eval-sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--save-probes", default="")
    parser.add_argument("--load-probes", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


class ReferenceGuidedP2Probe(nn.Module):
    """A tiny row-reference update used only as a falsification probe.

    All controlled variants have identical parameters. ``use_state`` and
    ``use_visual`` only replace the corresponding inputs with zeros. Absolute
    sample coordinates and relative offsets are available to every variant, so
    the visual comparison asks whether P2 appearance adds information beyond a
    coordinate/shape prior.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        feature_dim: int,
        hidden_dim: int,
        num_rows: int,
        offsets_px: list[float],
        input_w: int,
        use_state: bool,
        use_visual: bool,
    ) -> None:
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four")
        offsets = torch.tensor(offsets_px, dtype=torch.float32)
        if offsets.ndim != 1 or offsets.numel() < 3:
            raise ValueError("At least three offsets are required")
        if not bool((offsets[1:] > offsets[:-1]).all()):
            raise ValueError("offsets_px must be strictly increasing")
        if not bool(torch.isclose(offsets, -offsets.flip(0)).all()):
            raise ValueError("offsets_px must be symmetric around zero")

        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_rows = int(num_rows)
        self.input_w = int(input_w)
        self.use_state = bool(use_state)
        self.use_visual = bool(use_visual)
        self.register_buffer("offsets_px", offsets, persistent=True)

        self.state_norm = nn.LayerNorm(self.state_dim)
        self.state_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.profile_norm = nn.LayerNorm(2 * self.feature_dim)
        self.profile_proj = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.coordinate_proj = nn.Sequential(
            nn.Linear(5, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.row_embedding = nn.Embedding(self.num_rows, self.hidden_dim)
        self.offset_embedding = nn.Embedding(offsets.numel(), self.hidden_dim)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=4 * self.hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.local_score = nn.Linear(self.hidden_dim, 1)
        self.context_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.offset_embedding.weight, std=0.02)
        # Symmetric offsets plus zero logits make every probe an exact no-op at
        # initialization, so hit retention is not confounded by random motion.
        nn.init.zeros_(self.local_score.weight)
        nn.init.zeros_(self.local_score.bias)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(
        self,
        row_state: torch.Tensor,
        profiles: torch.Tensor,
        anchor_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if row_state.ndim != 3:
            raise ValueError("row_state must be [lanes,rows,state_dim]")
        lanes, rows, state_dim = row_state.shape
        offsets = int(self.offsets_px.numel())
        if rows != self.num_rows or state_dim != self.state_dim:
            raise ValueError(
                f"row_state shape {tuple(row_state.shape)} is incompatible with "
                f"[lanes,{self.num_rows},{self.state_dim}]"
            )
        if profiles.shape != (lanes, rows, offsets, self.feature_dim):
            raise ValueError(
                f"profiles must be {(lanes, rows, offsets, self.feature_dim)}, "
                f"got {tuple(profiles.shape)}"
            )
        if anchor_x.shape != (lanes, rows):
            raise ValueError(f"anchor_x must be {(lanes, rows)}")

        state = row_state.float()
        if not self.use_state:
            state = torch.zeros_like(state)
        visual = F.normalize(profiles.float(), p=2.0, dim=-1, eps=1e-6)
        center_index = int(self.offsets_px.abs().argmin())
        visual = torch.cat(
            (visual, visual - visual[:, :, center_index : center_index + 1]),
            dim=-1,
        )
        if not self.use_visual:
            visual = torch.zeros_like(visual)

        offset_values = self.offsets_px.to(device=anchor_x.device, dtype=torch.float32)
        sample_x = anchor_x.float().unsqueeze(-1) + offset_values.view(1, 1, offsets)
        width_scale = max(float(self.input_w - 1), 1.0)
        sample_norm = 2.0 * sample_x / width_scale - 1.0
        offset_norm = offset_values / max(float(offset_values.abs().max()), 1.0)
        coordinate = torch.stack(
            (
                sample_norm,
                torch.sin(math.pi * sample_norm),
                torch.cos(math.pi * sample_norm),
                torch.sin(2.0 * math.pi * sample_norm),
                offset_norm.view(1, 1, offsets).expand(lanes, rows, offsets),
            ),
            dim=-1,
        )

        hidden = (
            self.profile_proj(self.profile_norm(visual))
            + self.state_proj(self.state_norm(state)).unsqueeze(2)
            + self.coordinate_proj(coordinate)
            + self.row_embedding.weight.view(1, rows, 1, self.hidden_dim)
            + self.offset_embedding.weight.view(1, 1, offsets, self.hidden_dim)
        )
        mixed = self.local_mixer(
            hidden.reshape(lanes * rows, offsets, self.hidden_dim).transpose(1, 2)
        ).transpose(1, 2)
        hidden = hidden + mixed.reshape(lanes, rows, offsets, self.hidden_dim)

        local_logits = self.local_score(hidden).squeeze(-1)
        local_weights = torch.softmax(local_logits, dim=-1)
        row_summary = (local_weights.unsqueeze(-1) * hidden).sum(dim=2)
        row_context = self.row_encoder(row_summary)
        context = self.context_proj(row_context)
        logits = local_logits + (
            hidden * context.unsqueeze(2)
        ).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        probabilities = torch.softmax(logits, dim=-1)
        residual = (
            probabilities
            * offset_values.view(1, 1, offsets)
        ).sum(dim=-1)
        return {"logits": logits, "residual": residual}


@dataclass
class MatchedExamples:
    row_state: torch.Tensor
    profiles: torch.Tensor
    anchor_x: torch.Tensor
    gt_x: torch.Tensor
    valid: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.anchor_x.shape[0])


class AssignedComparisonStats:
    def __init__(self) -> None:
        self.lanes = 0
        self.rows = 0
        self.anchor_error = 0.0
        self.final_error = 0.0
        self.corrected_error = 0.0
        self.update_abs = 0.0

    def update(
        self,
        *,
        anchor: torch.Tensor,
        final: torch.Tensor,
        corrected: torch.Tensor,
        gt_x: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        valid = valid.bool()
        if int(valid.sum()) < 5:
            return
        self.lanes += 1
        self.rows += int(valid.sum())
        self.anchor_error += float((anchor[valid] - gt_x[valid]).abs().sum())
        self.final_error += float((final[valid] - gt_x[valid]).abs().sum())
        self.corrected_error += float((corrected[valid] - gt_x[valid]).abs().sum())
        self.update_abs += float((corrected[valid] - anchor[valid]).abs().sum())

    def summary(self) -> dict[str, float | int]:
        rows = max(self.rows, 1)
        return {
            "assigned_lanes": self.lanes,
            "valid_rows": self.rows,
            "anchor_row_mae_px": self.anchor_error / rows,
            "production_final_row_mae_px": self.final_error / rows,
            "corrected_row_mae_px": self.corrected_error / rows,
            "corrected_vs_anchor_mae_gain_px": (
                self.anchor_error - self.corrected_error
            )
            / rows,
            "corrected_vs_final_mae_gain_px": (
                self.final_error - self.corrected_error
            )
            / rows,
            "mean_abs_update_px": self.update_abs / rows,
        }


def _amp_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def _extract_stages(
    model: nn.Module,
    images: torch.Tensor,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    head = model.structured_query_head
    if head is None:
        raise ValueError("reference probe requires structured_query")
    with _amp_context(images.device, amp_dtype):
        p2 = model.encoder.forward_features(
            images,
            inference_only=True,
            structured_only=True,
        )["features"]
        previous = bool(head.intermediate_supervision)
        head.intermediate_supervision = True
        try:
            outputs = head(p2, inference_only=False)
        finally:
            head.intermediate_supervision = previous
    auxiliary = outputs.get("aux_outputs")
    if not isinstance(auxiliary, (list, tuple)):
        raise RuntimeError("Intermediate decoder outputs were not produced")
    stages = [*auxiliary, outputs]
    return p2.detach(), stages


def _sample_p2_profiles(
    p2: torch.Tensor,
    anchor_x: torch.Tensor,
    offsets_px: torch.Tensor,
    *,
    input_w: int,
    input_h: int,
) -> torch.Tensor:
    """Sample P2 around references.

    ``p2`` is [B,C,H,W], ``anchor_x`` is [B,N,R], and the returned tensor is
    [B,N,R,K,C]. Sampling is FP32 to avoid BF16 grid_sample limitations.
    """

    if p2.ndim != 4 or anchor_x.ndim != 3:
        raise ValueError("p2 must be [B,C,H,W] and anchor_x [B,N,R]")
    batch, instances, rows = anchor_x.shape
    offsets = int(offsets_px.numel())
    sample_x = anchor_x.float().unsqueeze(-1) + offsets_px.view(1, 1, 1, offsets)
    y = fixed_y_rows(rows, input_h, device=anchor_x.device, dtype=torch.float32)
    y = y.view(1, 1, rows, 1).expand(batch, instances, rows, offsets)
    grid = input_to_grid(
        sample_x.clamp(0.0, float(input_w - 1)),
        y,
        input_w=input_w,
        input_h=input_h,
    )
    sampled = F.grid_sample(
        p2.float(),
        grid.reshape(batch, instances * rows * offsets, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    channels = int(p2.shape[1])
    return sampled.squeeze(-1).transpose(1, 2).reshape(
        batch,
        instances,
        rows,
        offsets,
        channels,
    )


def _matched_examples(
    *,
    stage: dict[str, torch.Tensor],
    profiles: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    group_size: int,
    group_mode: str,
) -> MatchedExamples | None:
    state_parts: list[torch.Tensor] = []
    profile_parts: list[torch.Tensor] = []
    anchor_parts: list[torch.Tensor] = []
    gt_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    row_state = stage["structured_row_tokens"].float()
    anchor_x = stage["pred_x_rows"].float()
    for batch_index, (target, match) in enumerate(zip(targets, matches)):
        gt_x = target["x_rows"].to(device=anchor_x.device, dtype=torch.float32)
        valid = target["valid_mask"].to(device=anchor_x.device).bool()
        for pred_index, gt_index in zip(
            match["pred_indices"].tolist(),
            match["gt_indices"].tolist(),
        ):
            if group_mode == "group0" and int(pred_index) >= int(group_size):
                continue
            if int(valid[int(gt_index)].sum()) < 5:
                continue
            state_parts.append(row_state[batch_index, int(pred_index)])
            profile_parts.append(profiles[batch_index, int(pred_index)])
            anchor_parts.append(anchor_x[batch_index, int(pred_index)])
            gt_parts.append(gt_x[int(gt_index)])
            valid_parts.append(valid[int(gt_index)])
    if not state_parts:
        return None
    return MatchedExamples(
        row_state=torch.stack(state_parts),
        profiles=torch.stack(profile_parts),
        anchor_x=torch.stack(anchor_parts),
        gt_x=torch.stack(gt_parts),
        valid=torch.stack(valid_parts),
    )


def _paired_line_iou_loss(
    pred_x: torch.Tensor,
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> torch.Tensor:
    radius = 0.5 * float(line_width)
    overlap = (
        torch.minimum(pred_x + radius, gt_x + radius)
        - torch.maximum(pred_x - radius, gt_x - radius)
    ).clamp_min(0.0)
    union = (2.0 * float(line_width) - overlap).clamp_min(1e-6)
    mask = valid.float()
    iou = (overlap * mask).sum(dim=1) / (union * mask).sum(dim=1).clamp_min(1e-6)
    return (1.0 - iou).mean()


def _probe_loss(
    outputs: dict[str, torch.Tensor],
    examples: MatchedExamples,
    offsets_px: torch.Tensor,
    *,
    smooth_l1_beta_px: float,
    point_loss_weight: float,
    line_iou_loss_weight: float,
    line_width: float,
    input_w: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = examples.valid
    target_residual = examples.gt_x - examples.anchor_x
    target_index = (
        target_residual.unsqueeze(-1)
        - offsets_px.view(1, 1, -1)
    ).abs().argmin(dim=-1)
    classification = F.cross_entropy(
        outputs["logits"][valid].float(),
        target_index[valid],
    )
    corrected = (
        examples.anchor_x + outputs["residual"]
    ).clamp(0.0, float(input_w - 1))
    bounded_target = target_residual.clamp(
        min=float(offsets_px.min()),
        max=float(offsets_px.max()),
    )
    point = F.smooth_l1_loss(
        outputs["residual"][valid].float(),
        bounded_target[valid].float(),
        beta=float(smooth_l1_beta_px),
    ) / max(float(offsets_px.abs().max()), 1.0)
    line = _paired_line_iou_loss(
        corrected,
        examples.gt_x,
        valid,
        line_width=float(line_width),
    )
    total = (
        classification
        + float(point_loss_weight) * point
        + float(line_iou_loss_weight) * line
    )
    return total, {
        "total": float(total.detach()),
        "classification": float(classification.detach()),
        "point": float(point.detach()),
        "line_iou": float(line.detach()),
        "row_mae_px": float((corrected[valid] - examples.gt_x[valid]).abs().mean()),
    }


@torch.inference_mode()
def _evaluate(
    *,
    model: nn.Module,
    probes: dict[str, ReferenceGuidedP2Probe],
    matcher,
    loader,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    anchor_layer: int,
    offsets_px: torch.Tensor,
    input_w: int,
    input_h: int,
    group_size: int,
    max_batches: int,
    line_width: float,
) -> dict[str, Any]:
    for probe in probes.values():
        probe.eval()
    bucket_names = ("all", "production_hit_050", "production_miss_050")
    geometry = {
        name: {bucket: GeometryStats() for bucket in bucket_names}
        for name in PROBE_MODES
    }
    assigned = {name: AssignedComparisonStats() for name in PROBE_MODES}
    images_seen = 0
    gt_lanes = 0
    anchor_hits_050 = 0
    production_hits_050 = 0
    displayed_batches = len(loader)
    if int(max_batches) > 0:
        displayed_batches = min(displayed_batches, int(max_batches))

    progress = tqdm(
        enumerate(loader),
        total=displayed_batches,
        desc="reference-P2 probe eval",
        ncols=100,
    )
    for batch_index, (images, targets, _metas) in progress:
        if int(max_batches) > 0 and batch_index >= int(max_batches):
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last
                if channels_last
                else torch.contiguous_format
            ),
        )
        targets = nested_to_device(targets, device)
        p2, stages = _extract_stages(model, images, amp_dtype)
        anchor_stage = stages[int(anchor_layer) - 1]
        final_stage = stages[-1]
        matches = matcher(final_stage, targets)
        anchor_x = anchor_stage["pred_x_rows"].float()
        final_x = final_stage["pred_x_rows"].float()
        row_state = anchor_stage["structured_row_tokens"].float()
        profiles = _sample_p2_profiles(
            p2,
            anchor_x,
            offsets_px,
            input_w=input_w,
            input_h=input_h,
        )

        group_state = row_state[:, :group_size].reshape(
            -1, row_state.shape[2], row_state.shape[3]
        )
        group_profiles = profiles[:, :group_size].reshape(
            -1,
            profiles.shape[2],
            profiles.shape[3],
            profiles.shape[4],
        )
        group_anchor = anchor_x[:, :group_size].reshape(-1, anchor_x.shape[2])
        corrected: dict[str, torch.Tensor] = {}
        for name, probe in probes.items():
            delta = probe(
                group_state,
                group_profiles,
                group_anchor,
            )["residual"].reshape(
                anchor_x.shape[0],
                group_size,
                anchor_x.shape[2],
            )
            corrected[name] = (
                anchor_x[:, :group_size] + delta
            ).clamp(0.0, float(input_w - 1))

        for image_index, (target, match) in enumerate(zip(targets, matches)):
            gt_x = target["x_rows"].float()
            valid = target["valid_mask"].bool()
            group_assignment = _group_zero_assignments(
                match,
                group_size=group_size,
            )
            anchor_candidates = anchor_x[image_index, :group_size]
            final_candidates = final_x[image_index, :group_size]
            for gt_index in range(int(gt_x.shape[0])):
                if int(valid[gt_index].sum()) < 5:
                    continue
                gt_lanes += 1
                anchor_iou = _best_iou(
                    anchor_candidates,
                    gt_x[gt_index],
                    valid[gt_index],
                    line_width=line_width,
                )
                production_iou = _best_iou(
                    final_candidates,
                    gt_x[gt_index],
                    valid[gt_index],
                    line_width=line_width,
                )
                anchor_hits_050 += int(anchor_iou >= 0.5)
                production_hits_050 += int(production_iou >= 0.5)
                buckets = ["all"]
                buckets.append(
                    "production_hit_050"
                    if production_iou >= 0.5
                    else "production_miss_050"
                )
                for name in PROBE_MODES:
                    after = _best_iou(
                        corrected[name][image_index],
                        gt_x[gt_index],
                        valid[gt_index],
                        line_width=line_width,
                    )
                    for bucket in buckets:
                        geometry[name][bucket].update(production_iou, after)

                    pred_index = group_assignment.get(gt_index)
                    if pred_index is not None:
                        assigned[name].update(
                            anchor=anchor_x[image_index, pred_index],
                            final=final_x[image_index, pred_index],
                            corrected=corrected[name][image_index, pred_index],
                            gt_x=gt_x[gt_index],
                            valid=valid[gt_index],
                        )
        images_seen += int(images.shape[0])

    summaries = {
        name: {
            "raw_group0_vs_production": {
                bucket: stats.summary()
                for bucket, stats in geometry[name].items()
            },
            "assigned_query_rows": assigned[name].summary(),
        }
        for name in PROBE_MODES
    }
    state_miss = summaries["state_only"]["raw_group0_vs_production"][
        "production_miss_050"
    ]
    combined_miss = summaries["state_local_p2"]["raw_group0_vs_production"][
        "production_miss_050"
    ]
    combined_all = summaries["state_local_p2"]["raw_group0_vs_production"]["all"]
    visual_increment = (
        float(combined_miss["after_recall_050"])
        - float(state_miss["after_recall_050"])
    )
    production_recall = production_hits_050 / max(gt_lanes, 1)
    positive_gate = (
        visual_increment >= 0.10
        and float(combined_all["after_recall_050"]) >= production_recall - 0.01
        and int(combined_all["lost_050"])
        <= max(1, round(0.02 * int(combined_all["before_hits_050"])))
    )
    return {
        "images": images_seen,
        "gt_lanes": gt_lanes,
        "anchor_layer_group0_recall_050": anchor_hits_050 / max(gt_lanes, 1),
        "production_final_group0_recall_050": production_recall,
        "probes": summaries,
        "controlled_local_p2_signal": {
            "state_local_p2_minus_state_miss_recall_050_points": (
                100.0 * visual_increment
            ),
            "positive_gate": bool(positive_gate),
            "gate_definition": (
                "state+local-P2 must improve production-miss recall@0.50 by at "
                "least 10 points over the equal-capacity state-only control, "
                "retain overall production recall within one point, and lose "
                "no more than 2% of production hits."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    offsets_px = torch.tensor(args.offsets_px, dtype=torch.float32)
    if not bool(torch.isclose(offsets_px, -offsets_px.flip(0)).all()):
        raise ValueError("--offsets-px must be symmetric around zero")
    device = torch.device(args.device)
    offsets_device = offsets_px.to(device)
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.structured_query_head
    if head is None:
        raise ValueError("reference probe requires structured_query")
    layer_count = len(head.layers)
    if not 1 <= int(args.anchor_layer) < layer_count:
        raise ValueError(
            f"--anchor-layer must be before final layer in [1,{layer_count - 1}]"
        )
    channels_last = (
        bool(cfg.get("training", {}).get("channels_last", False))
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    num_instances = int(head.num_instances)
    num_groups = int(head.num_groups)
    group_size = num_instances // max(num_groups, 1)
    input_w = int(model_cfg.get("input_w", head.input_w))
    input_h = int(model_cfg.get("input_h", 288))
    feature_dim = int(model_cfg.get("dim", head.dim))
    probes = {
        name: ReferenceGuidedP2Probe(
            state_dim=int(head.dim),
            feature_dim=feature_dim,
            hidden_dim=int(args.hidden_dim),
            num_rows=int(head.num_rows),
            offsets_px=[float(value) for value in args.offsets_px],
            input_w=input_w,
            use_state=mode[0],
            use_visual=mode[1],
        ).to(device)
        for name, mode in PROBE_MODES.items()
    }
    if not args.load_probes:
        reference_state = probes["anchor_only"].state_dict()
        for name in tuple(PROBE_MODES)[1:]:
            probes[name].load_state_dict(reference_state, strict=True)
    parameter_counts = {
        name: sum(parameter.numel() for parameter in probe.parameters())
        for name, probe in probes.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Probe parameter counts differ: {parameter_counts}")

    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    matcher = build_matcher(cfg)
    history: list[dict[str, Any]] = []
    trained_steps = int(args.train_steps)
    if args.load_probes:
        saved = torch.load(args.load_probes, map_location="cpu")
        for name, probe in probes.items():
            probe.load_state_dict(saved["probes"][name], strict=True)
        trained_steps = int(saved.get("args", {}).get("train_steps", trained_steps))
    elif int(args.train_steps) > 0:
        train_loader = build_dataloader(cfg, split="train", training=True)
        optimizer = torch.optim.AdamW(
            [
                parameter
                for probe in probes.values()
                for parameter in probe.parameters()
            ],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        train_iterator = iter(train_loader)
        for probe in probes.values():
            probe.train()
        progress = tqdm(
            range(1, int(args.train_steps) + 1),
            desc="reference-P2 probe train",
            ncols=100,
        )
        for step in progress:
            try:
                images, targets, _metas = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                images, targets, _metas = next(train_iterator)
            images = images.to(
                device,
                non_blocking=True,
                memory_format=(
                    torch.channels_last
                    if channels_last
                    else torch.contiguous_format
                ),
            )
            targets = nested_to_device(targets, device)
            p2, stages = _extract_stages(model, images, amp_dtype)
            anchor_stage = stages[int(args.anchor_layer) - 1]
            final_stage = stages[-1]
            matches = matcher(final_stage, targets)
            anchor_x = anchor_stage["pred_x_rows"].float()
            profiles = _sample_p2_profiles(
                p2,
                anchor_x,
                offsets_device,
                input_w=input_w,
                input_h=input_h,
            )
            examples = _matched_examples(
                stage=anchor_stage,
                profiles=profiles,
                targets=targets,
                matches=matches,
                group_size=group_size,
                group_mode=str(args.train_group_mode),
            )
            if examples is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            losses: dict[str, torch.Tensor] = {}
            metrics: dict[str, dict[str, float]] = {}
            for name, probe in probes.items():
                outputs = probe(
                    examples.row_state,
                    examples.profiles,
                    examples.anchor_x,
                )
                loss, probe_metrics = _probe_loss(
                    outputs,
                    examples,
                    offsets_device,
                    smooth_l1_beta_px=float(args.smooth_l1_beta_px),
                    point_loss_weight=float(args.point_loss_weight),
                    line_iou_loss_weight=float(args.line_iou_loss_weight),
                    line_width=float(args.line_width),
                    input_w=input_w,
                )
                losses[name] = loss
                metrics[name] = probe_metrics
            total_loss = torch.stack(list(losses.values())).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for probe in probes.values()
                    for parameter in probe.parameters()
                ],
                max_norm=5.0,
            )
            optimizer.step()
            record = {
                "step": step,
                "matched_lane_examples": examples.count,
                "total_loss": float(total_loss.detach()),
                **{
                    f"{name}/{metric}": value
                    for name, values in metrics.items()
                    for metric, value in values.items()
                },
            }
            if (
                step == 1
                or step % int(args.log_interval) == 0
                or step == int(args.train_steps)
            ):
                history.append(record)
                progress.set_postfix(
                    {
                        name: f"{metrics[name]['row_mae_px']:.1f}px"
                        for name in PROBE_MODES
                    }
                )

    if args.save_probes:
        save_path = Path(args.save_probes)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "probes": {
                    name: probe.state_dict()
                    for name, probe in probes.items()
                },
                "args": vars(args),
                "parameter_counts": parameter_counts,
            },
            save_path,
        )

    eval_loader = build_dataloader(cfg, split="val", training=False)
    eval_loader, sampled_indices = select_diagnostic_loader(
        eval_loader,
        strategy=str(args.eval_sample_strategy),
        max_batches=int(args.eval_max_batches),
        num_workers=int(args.num_workers),
    )
    evaluation = _evaluate(
        model=model,
        probes=probes,
        matcher=matcher,
        loader=eval_loader,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        anchor_layer=int(args.anchor_layer),
        offsets_px=offsets_device,
        input_w=input_w,
        input_h=input_h,
        group_size=group_size,
        max_batches=int(args.eval_max_batches),
        line_width=float(args.line_width),
    )
    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "warning": (
            "The detector is frozen and tiny probes are fitted on CULane train. "
            "Held-out raw proposal recall is mechanistic evidence, not an "
            "official benchmark result."
        ),
        "hypothesis": (
            "If state+local-P2 beats the equal-capacity state-only control on "
            "production-final misses while retaining final hits, a dynamic "
            "row-reference decoder has a supported visual correction pathway."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "seed": int(args.seed),
        "anchor_layer": int(args.anchor_layer),
        "train_steps": trained_steps,
        "train_group_mode": str(args.train_group_mode),
        "eval_sample_strategy": str(args.eval_sample_strategy),
        "sampled_dataset_indices": sampled_indices,
        "offsets_px": [float(value) for value in args.offsets_px],
        "probe_parameter_counts": parameter_counts,
        "history": history,
        "evaluation": evaluation,
    }
    print(json.dumps(payload, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
