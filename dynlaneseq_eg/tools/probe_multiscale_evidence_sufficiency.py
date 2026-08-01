from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.diagnostic_sampling import uniformly_spaced_indices
from dynlaneseq_eg.tools.probe_curve_aligned_visual_verification import (
    sample_curve_aligned_profiles,
    sampled_range_weights,
)
from dynlaneseq_eg.tools.probe_four_slot_coverage_selector import (
    _load_cache,
    _validate_cache,
    source_scores,
)
from dynlaneseq_eg.tools.probe_hard_decision_selector import (
    ClusterCoveragePointerProbe,
    build_hard_decision_targets,
    cluster_iou_matrix,
    coverage_pointer_loss,
    representative_listwise_loss,
)
from dynlaneseq_eg.tools.probe_hierarchical_cluster_selector import (
    RepresentativeQualityProbe,
    _finish_counts,
    _new_counts,
    _representative_ids,
    _update_counts,
    build_hierarchical_supervision,
)


ARM_NAMES = ("state", "p2_center", "p2_strip", "p34_context", "joint")
SCALE_NAMES = ("p2", "p3", "p4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen module-compatibility probe for LaneRowNet. It tests whether "
            "row-preserving decoder state, local P2 strips, or broad P3/P4 "
            "context can resolve the hard NMS-cluster and representative "
            "decisions left unresolved by the deployed selector."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--amp-dtype", choices=("none", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--sequence-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--curve-samples", type=int, default=20)
    parser.add_argument(
        "--p2-offsets-px",
        type=float,
        nargs="+",
        default=[-16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument(
        "--context-offsets-px",
        type=float,
        nargs="+",
        default=[-96.0, -48.0, -24.0, 0.0, 24.0, 48.0, 96.0],
    )
    parser.add_argument("--range-temperature", type=float, default=0.02)
    parser.add_argument("--coverage-temperature", type=float, default=0.10)
    parser.add_argument("--coverage-weight-050", type=float, default=1.0)
    parser.add_argument("--coverage-weight-070", type=float, default=0.5)
    parser.add_argument("--coverage-weight-iou", type=float, default=0.1)
    parser.add_argument("--coverage-min-gain", type=float, default=1e-5)
    parser.add_argument("--representative-temperature", type=float, default=0.05)
    parser.add_argument("--representative-positive-iou", type=float, default=0.30)
    parser.add_argument("--representative-min-spread", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--train-eval-images", type=int, default=256)
    parser.add_argument("--val-eval-images", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probe", required=True)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _validate_offsets(values: Iterable[float], name: str) -> list[float]:
    offsets = [float(value) for value in values]
    if len(offsets) < 3 or len(offsets) % 2 == 0:
        raise ValueError(f"{name} must contain an odd number of at least three offsets")
    if offsets != sorted(offsets) or len(offsets) != len(set(offsets)):
        raise ValueError(f"{name} must be unique and strictly increasing")
    tensor = torch.tensor(offsets)
    if not bool(torch.isclose(tensor, -tensor.flip(0)).all()):
        raise ValueError(f"{name} must be symmetric around zero")
    if 0.0 not in offsets:
        raise ValueError(f"{name} must contain zero")
    return offsets


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "eval_batch_size",
        "train_steps",
        "hidden_dim",
        "sequence_dim",
        "num_heads",
        "ff_dim",
        "curve_samples",
        "top_k",
        "min_valid_rows",
        "train_eval_images",
        "val_eval_images",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(args.hidden_dim) % int(args.num_heads):
        raise ValueError("hidden_dim must be divisible by num_heads")
    _validate_offsets(args.p2_offsets_px, "p2_offsets_px")
    context = _validate_offsets(args.context_offsets_px, "context_offsets_px")
    if len(context) != len(args.p2_offsets_px):
        raise ValueError("P2 and context offset lists must have equal length")
    if int(args.eval_batch_size) < 2:
        raise ValueError("eval_batch_size must be >=2 for the wrong-image control")


def evidence_masks(offset_count: int) -> dict[str, torch.Tensor]:
    """Return equal-shape [P2,P3,P4]xK modality masks for every arm."""

    if int(offset_count) < 3 or int(offset_count) % 2 == 0:
        raise ValueError("offset_count must be odd and >=3")
    center = int(offset_count) // 2
    masks = {
        "state": torch.zeros((3, int(offset_count)), dtype=torch.bool),
        "p2_center": torch.zeros((3, int(offset_count)), dtype=torch.bool),
        "p2_strip": torch.zeros((3, int(offset_count)), dtype=torch.bool),
        "p34_context": torch.zeros((3, int(offset_count)), dtype=torch.bool),
        "joint": torch.ones((3, int(offset_count)), dtype=torch.bool),
    }
    masks["p2_center"][0, center] = True
    masks["p2_strip"][0] = True
    masks["p34_context"][1:] = True
    return masks


class CandidateSequenceDescriptor(nn.Module):
    """Equal-capacity candidate descriptor that keeps the row axis explicit.

    Every diagnostic arm owns the exact same module. The registered evidence
    mask is the only difference between state, centerline, strip, context, and
    joint arms. This prevents a larger visual head from winning merely through
    parameter count.
    """

    def __init__(
        self,
        *,
        base_dim: int,
        row_state_dim: int,
        feature_dim: int,
        rows: int,
        offsets_px: torch.Tensor,
        evidence_mask: torch.Tensor,
        sequence_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if tuple(evidence_mask.shape) != tuple(offsets_px.shape):
            raise ValueError("evidence_mask and offsets_px must share [S,K]")
        scales, offsets = evidence_mask.shape
        if int(scales) != len(SCALE_NAMES):
            raise ValueError("the probe expects exactly P2/P3/P4")
        self.rows = int(rows)
        self.scales = int(scales)
        self.offsets = int(offsets)
        self.sequence_dim = int(sequence_dim)
        self.register_buffer("evidence_mask", evidence_mask.bool(), persistent=True)
        normalized_offsets = offsets_px.float() / offsets_px.abs().amax(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        self.register_buffer("normalized_offsets", normalized_offsets, persistent=True)

        self.base = nn.Sequential(
            nn.LayerNorm(int(base_dim)),
            nn.Linear(int(base_dim), int(output_dim)),
            nn.GELU(),
        )
        self.state = nn.Sequential(
            nn.LayerNorm(int(row_state_dim)),
            nn.Linear(int(row_state_dim), int(sequence_dim)),
        )
        self.profile_norms = nn.ModuleList(
            nn.LayerNorm(int(feature_dim)) for _ in range(self.scales)
        )
        self.profile_projections = nn.ModuleList(
            nn.Linear(int(feature_dim), int(sequence_dim)) for _ in range(self.scales)
        )
        self.scale_embedding = nn.Parameter(
            torch.zeros(self.scales, int(sequence_dim))
        )
        self.offset_coordinate = nn.Sequential(
            nn.Linear(1, int(sequence_dim)),
            nn.GELU(),
            nn.Linear(int(sequence_dim), int(sequence_dim)),
        )
        self.row_embedding = nn.Parameter(torch.zeros(self.rows, int(sequence_dim)))
        self.local_mixer = nn.Sequential(
            nn.LayerNorm(int(sequence_dim)),
            nn.Linear(int(sequence_dim), int(sequence_dim)),
            nn.GELU(),
        )
        self.offset_score = nn.Linear(int(sequence_dim), 1)
        self.scale_score = nn.Linear(int(sequence_dim), 1)
        self.row_fusion = nn.Sequential(
            nn.LayerNorm(2 * int(sequence_dim)),
            nn.Linear(2 * int(sequence_dim), int(sequence_dim)),
            nn.GELU(),
        )
        self.row_conv = nn.Sequential(
            nn.Conv1d(int(sequence_dim), int(sequence_dim), 3, padding=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(
                int(sequence_dim),
                int(sequence_dim),
                3,
                padding=2,
                dilation=2,
            ),
            nn.GELU(),
        )
        self.row_attention = nn.Linear(int(sequence_dim), 1)
        self.sequence_summary = nn.Sequential(
            nn.LayerNorm(3 * int(sequence_dim)),
            nn.Linear(3 * int(sequence_dim), int(output_dim)),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(2 * int(output_dim)),
            nn.Linear(2 * int(output_dim), int(output_dim)),
            nn.GELU(),
        )
        nn.init.normal_(self.scale_embedding, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)

    def forward(
        self,
        base_features: torch.Tensor,
        row_states: torch.Tensor,
        profiles: torch.Tensor,
        row_weights: torch.Tensor,
    ) -> torch.Tensor:
        if profiles.ndim != 6:
            raise ValueError("profiles must have shape [B,N,R,S,K,C]")
        batch, candidates, rows, scales, offsets, _channels = profiles.shape
        if (rows, scales, offsets) != (self.rows, self.scales, self.offsets):
            raise ValueError(
                f"expected profile contract {self.rows}x{self.scales}x{self.offsets}, "
                f"got {rows}x{scales}x{offsets}"
            )
        if row_states.shape[:3] != (batch, candidates, rows):
            raise ValueError("row_states do not align with profiles")
        if row_weights.shape != (batch, candidates, rows):
            raise ValueError("row_weights must have shape [B,N,R]")

        state = self.state(row_states.float())
        offset_coord = self.offset_coordinate(
            self.normalized_offsets.view(scales, offsets, 1)
        )
        scale_rows: list[torch.Tensor] = []
        active_scales: list[bool] = []
        for scale_index in range(scales):
            active = self.evidence_mask[scale_index]
            active_scales.append(bool(active.any()))
            if not bool(active.any()):
                scale_rows.append(torch.zeros_like(state))
                continue
            visual = self.profile_projections[scale_index](
                self.profile_norms[scale_index](profiles[:, :, :, scale_index].float())
            )
            visual = visual + state.unsqueeze(3)
            visual = visual + self.scale_embedding[scale_index].view(1, 1, 1, 1, -1)
            visual = visual + offset_coord[scale_index].view(1, 1, 1, offsets, -1)
            visual = visual + self.row_embedding.view(1, 1, rows, 1, -1)
            visual = visual + self.local_mixer(visual)
            logits = self.offset_score(visual).squeeze(-1)
            logits = logits.masked_fill(
                ~active.view(1, 1, 1, offsets),
                -1e4,
            )
            probability = torch.softmax(logits, dim=-1)
            scale_rows.append((probability.unsqueeze(-1) * visual).sum(dim=3))

        stacked = torch.stack(scale_rows, dim=3)
        scale_mask = torch.tensor(
            active_scales,
            device=stacked.device,
            dtype=torch.bool,
        )
        if bool(scale_mask.any()):
            scale_logits = self.scale_score(stacked).squeeze(-1)
            scale_logits = scale_logits.masked_fill(
                ~scale_mask.view(1, 1, 1, scales),
                -1e4,
            )
            scale_probability = torch.softmax(scale_logits, dim=-1)
            evidence = (scale_probability.unsqueeze(-1) * stacked).sum(dim=3)
        else:
            evidence = torch.zeros_like(state)

        row = self.row_fusion(torch.cat((state, evidence), dim=-1))
        flat = row.reshape(batch * candidates, rows, self.sequence_dim)
        mixed = self.row_conv(flat.transpose(1, 2)).transpose(1, 2)
        encoded = (flat + mixed).reshape(batch, candidates, rows, self.sequence_dim)

        weights = row_weights.float().clamp_min(1e-6)
        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mean = (encoded * normalized.unsqueeze(-1)).sum(dim=2)
        valid = weights > 0.05
        maximum = encoded.masked_fill(~valid.unsqueeze(-1), -1e4).amax(dim=2)
        maximum = torch.where(
            valid.any(dim=-1, keepdim=True),
            maximum,
            torch.zeros_like(maximum),
        )
        attention_logits = self.row_attention(encoded).squeeze(-1) + weights.log()
        attention = torch.softmax(attention_logits, dim=-1)
        attended = (encoded * attention.unsqueeze(-1)).sum(dim=2)
        sequence = self.sequence_summary(torch.cat((mean, maximum, attended), dim=-1))
        base = self.base(base_features.float())
        return self.output(torch.cat((base, sequence), dim=-1))


class EvidenceSelectionArm(nn.Module):
    def __init__(
        self,
        *,
        base_dim: int,
        row_state_dim: int,
        feature_dim: int,
        rows: int,
        offsets_px: torch.Tensor,
        evidence_mask: torch.Tensor,
        sequence_dim: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        top_k: int,
    ) -> None:
        super().__init__()
        self.descriptor = CandidateSequenceDescriptor(
            base_dim=base_dim,
            row_state_dim=row_state_dim,
            feature_dim=feature_dim,
            rows=rows,
            offsets_px=offsets_px,
            evidence_mask=evidence_mask,
            sequence_dim=sequence_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.cluster = ClusterCoveragePointerProbe(
            hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            top_k=top_k,
        )
        self.representative = RepresentativeQualityProbe(
            hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )


def _channels_last_images(
    images: torch.Tensor,
    *,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    if channels_last:
        return images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return images.to(device, non_blocking=True)


@torch.no_grad()
def extract_frozen_sources(
    model: nn.Module,
    images: torch.Tensor,
    *,
    cached_curves: torch.Tensor,
    row_indices: torch.Tensor,
    p2_offsets: torch.Tensor,
    context_offsets: torch.Tensor,
    input_h: int,
    input_w: int,
    range_temperature: float,
    amp_dtype: torch.dtype | None,
    include_wrong_image: bool,
) -> dict[str, torch.Tensor]:
    """Run the deployable 32-query decoder and expose trained FPN states.

    P3/P4 are the trained top-down tensors that feed P2. The optional
    ``pyramid_outputs`` convolutions are deliberately not used because they
    never participate when the source config consumes only P2 and therefore
    are not trained by the source checkpoint.
    """

    encoder = model.encoder
    head = model.structured_query_head
    if head is None:
        raise ValueError("the diagnostic requires a structured query head")
    captured: dict[str, torch.Tensor] = {}

    def capture_final_rows(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("final row-reference layer must return a tensor")
        captured["rows"] = output.detach()

    handle = head.layers[-1].register_forward_hook(capture_final_rows)
    try:
        with _amp_context(images.device, amp_dtype):
            backbone = encoder.backbone(images)
            fpn = encoder.fpn
            p5 = fpn.lateral["c5"](backbone["c5"])
            p4 = fpn.lateral["c4"](backbone["c4"]) + F.interpolate(
                p5,
                size=backbone["c4"].shape[-2:],
                mode="nearest",
            )
            p3 = fpn.lateral["c3"](backbone["c3"]) + F.interpolate(
                p4,
                size=backbone["c3"].shape[-2:],
                mode="nearest",
            )
            p2_topdown = fpn.lateral["c2"](backbone["c2"]) + F.interpolate(
                p3,
                size=backbone["c2"].shape[-2:],
                mode="nearest",
            )
            p2 = encoder.proj(fpn.output(p2_topdown))
            # The frozen selector cache was produced by ``_frozen_outputs``,
            # which deliberately calls the structured head with
            # ``inference_only=False``.  In the unified-selector experiment
            # that path evaluates the 32 primary queries together with the
            # train-only auxiliary groups and slices the public predictions
            # back to the primary 32 afterwards.  Although those groups are
            # attention-isolated, changing the packed GEMM/attention shapes
            # under BF16 can move soft-expected-x predictions measurably.  Run
            # the exact cache-producing path here; otherwise the parity guard
            # compares two numerically different execution contracts.
            outputs = head(p2, inference_only=False)
    finally:
        handle.remove()
    row_states = captured.get("rows")
    if row_states is None:
        raise RuntimeError("failed to capture final deployable row states")

    cached_curves = cached_curves.to(device=images.device, dtype=torch.float32)
    primary_candidates = int(outputs["pred_x_rows"].shape[1])
    if int(cached_curves.shape[1]) != primary_candidates:
        raise ValueError(
            "cache/model primary-candidate mismatch: "
            f"cache={int(cached_curves.shape[1])}, model={primary_candidates}"
        )
    # The layer hook observes the packed primary+auxiliary tensor before the
    # head slices its public outputs.  Only the leading primary candidates
    # correspond to the cached/deployable candidate set.
    if int(row_states.shape[1]) < primary_candidates:
        raise ValueError("captured fewer row states than deployable candidates")
    row_states = row_states[:, :primary_candidates]
    parity = (outputs["pred_x_rows"].float() - cached_curves).abs().amax()
    profiles = []
    feature_maps = (p2, p3, p4)
    offset_rows = (p2_offsets, context_offsets, context_offsets)
    for feature, offsets in zip(feature_maps, offset_rows):
        profiles.append(
            sample_curve_aligned_profiles(
                feature,
                cached_curves,
                row_indices=row_indices,
                offsets_px=offsets,
                input_h=input_h,
                input_w=input_w,
            )
        )
    packed = torch.stack(profiles, dim=3)
    result = {
        "profiles": packed.detach(),
        "row_states": row_states.index_select(
            2,
            row_indices.to(device=row_states.device, dtype=torch.long),
        ).detach(),
        "row_weights": sampled_range_weights(
            outputs["range_norm"],
            row_indices=row_indices,
            total_rows=int(outputs["pred_x_rows"].shape[-1]),
            temperature=range_temperature,
        ).detach(),
        "prediction_parity_max_abs_px": parity.detach(),
    }
    if include_wrong_image:
        if int(images.shape[0]) < 2:
            raise ValueError("wrong-image control requires a batch of at least two")
        wrong_profiles = []
        for feature, offsets in zip(feature_maps, offset_rows):
            wrong_profiles.append(
                sample_curve_aligned_profiles(
                    torch.roll(feature, shifts=1, dims=0),
                    cached_curves,
                    row_indices=row_indices,
                    offsets_px=offsets,
                    input_h=input_h,
                    input_w=input_w,
                )
            )
        result["wrong_profiles"] = torch.stack(wrong_profiles, dim=3).detach()
    return result


def _loader_for_cache_positions(
    cfg: dict[str, Any],
    cache: dict[str, Any],
    *,
    split: str,
    positions: list[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    local = deepcopy(cfg)
    local.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    local["dataloader"]["num_workers"] = int(num_workers)
    if int(num_workers) == 0:
        local["dataloader"]["persistent_workers"] = False
    base = build_dataloader(local, split=split, training=False)
    dataset_indices = cache["metadata"].get("dataset_indices")
    if not isinstance(dataset_indices, list):
        raise ValueError(f"{split} cache has no dataset_indices")
    selected = [int(dataset_indices[position]) for position in positions]
    subset = Subset(base.dataset, selected)
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(getattr(base, "pin_memory", True)),
        persistent_workers=bool(int(num_workers) > 0),
        collate_fn=base.collate_fn,
        worker_init_fn=getattr(base, "worker_init_fn", None),
        drop_last=False,
    )


def _paths_for_positions(cache: dict[str, Any], positions: list[int]) -> list[str]:
    paths = cache["metadata"].get("image_paths")
    if not isinstance(paths, list):
        raise ValueError("cache has no image_paths")
    return [str(paths[position]) for position in positions]


def _assert_batch_paths(
    metas: list[dict[str, Any]],
    expected_paths: list[str],
) -> None:
    actual = [str(meta.get("image_path", "")) for meta in metas]
    if actual != expected_paths:
        raise ValueError(
            "diagnostic loader/cache order mismatch; refusing cross-image supervision"
        )


def _descriptor_inputs(
    evidence: dict[str, torch.Tensor],
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    profiles = evidence["profiles"]
    states = evidence["row_states"]
    if condition == "correct":
        return profiles, states
    if condition == "wrong_image":
        return evidence["wrong_profiles"], states
    if condition == "zero_image":
        return torch.zeros_like(profiles), states
    if condition == "wrong_state":
        return profiles, torch.roll(states, shifts=1, dims=0)
    if condition == "zero_state":
        return profiles, torch.zeros_like(states)
    raise ValueError(f"unknown evidence condition: {condition}")


def _fill_cluster_ids(
    predicted: list[int],
    record: dict[str, Any],
    *,
    top_k: int,
) -> list[int]:
    output: list[int] = []
    fallback = [
        *predicted,
        *record["source_selected_cluster_ids"],
        *range(len(record["keepers"])),
    ]
    for value in fallback:
        value = int(value)
        if value >= 0 and value not in output:
            output.append(value)
        if len(output) >= min(int(top_k), len(record["keepers"])):
            break
    return output


@torch.no_grad()
def _selection_from_heads(
    cluster_head: ClusterCoveragePointerProbe,
    representative_head: RepresentativeQualityProbe,
    cluster_features: torch.Tensor,
    representative_features: torch.Tensor,
    *,
    candidate_valid: torch.Tensor,
    membership: torch.Tensor,
    cluster_valid: torch.Tensor,
    scores: torch.Tensor,
    records: list[dict[str, Any]],
    top_k: int,
) -> list[list[int]]:
    cluster_head.eval()
    representative_head.eval()
    _logits, selections = cluster_head(
        cluster_features,
        candidate_valid,
        membership,
        cluster_valid,
        scores,
    )
    representative_logits = representative_head(
        representative_features,
        candidate_valid,
        scores,
    )
    rows: list[list[int]] = []
    for batch_index, record in enumerate(records):
        cluster_ids = [
            int(value)
            for value in selections[batch_index].tolist()
            if int(value) >= 0
        ]
        cluster_ids = _fill_cluster_ids(cluster_ids, record, top_k=top_k)
        rows.append(
            _representative_ids(
                cluster_ids,
                record,
                representative_logits[batch_index],
                learned=True,
            )
        )
    return rows


def _rows_from_decisions(
    cluster_selections: torch.Tensor,
    representative_logits: torch.Tensor,
    records: list[dict[str, Any]],
    *,
    top_k: int,
    learned_cluster: bool,
    learned_representative: bool,
) -> list[list[int]]:
    rows: list[list[int]] = []
    for batch_index, record in enumerate(records):
        if learned_cluster:
            cluster_ids = _fill_cluster_ids(
                [
                    int(value)
                    for value in cluster_selections[batch_index].tolist()
                    if int(value) >= 0
                ],
                record,
                top_k=top_k,
            )
        else:
            cluster_ids = list(range(min(int(top_k), len(record["keepers"]))))
        rows.append(
            _representative_ids(
                cluster_ids,
                record,
                representative_logits[batch_index],
                learned=learned_representative,
            )
        )
    return rows


def _raw_source_ids(
    score: torch.Tensor,
    valid: torch.Tensor,
    *,
    top_k: int,
) -> list[int]:
    ids = torch.nonzero(valid.bool(), as_tuple=False).flatten().tolist()
    ids.sort(key=lambda index: float(score[index]), reverse=True)
    return [int(value) for value in ids[: int(top_k)]]


def _source_nms_ids(record: dict[str, Any], *, top_k: int) -> list[int]:
    clusters = list(range(min(int(top_k), len(record["keepers"]))))
    return _representative_ids(
        clusters,
        record,
        torch.zeros(max(len(record["keepers"]), 1)),
        learned=False,
    )


def _new_mode_counts(names: Iterable[str]) -> dict[str, dict[str, int]]:
    return {name: _new_counts() for name in names}


def _finish_mode_counts(rows: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {name: _finish_counts(value) for name, value in rows.items()}


def _metric_gain(row: dict[str, Any], source: dict[str, Any]) -> dict[str, float]:
    return {
        "recall_050_points": 100.0
        * (float(row["recall_050"]) - float(source["recall_050"])),
        "recall_070_points": 100.0
        * (float(row["recall_070"]) - float(source["recall_070"])),
        "f1_050_points": 100.0
        * (float(row["f1_050"]) - float(source["f1_050"])),
        "f1_070_points": 100.0
        * (float(row["f1_070"]) - float(source["f1_070"])),
    }


def _control_delta(
    correct: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, float]:
    return {
        "recall_050_points": 100.0
        * (float(correct["recall_050"]) - float(control["recall_050"])),
        "recall_070_points": 100.0
        * (float(correct["recall_070"]) - float(control["recall_070"])),
    }


def _evaluation_mode_names() -> list[str]:
    names = ["raw_source_top4", "source_nms", "hierarchy_teacher"]
    for arm in ARM_NAMES:
        names.extend(
            (
                f"{arm}_hierarchical",
                f"{arm}_cluster_only",
                f"{arm}_representative_only",
            )
        )
    names.extend(
        (
            "decoupled_p34_cluster_p2_representative",
            "inverse_p2_cluster_p34_representative",
            "p2_strip_wrong_image_hierarchical",
            "p2_strip_zero_image_hierarchical",
            "p34_context_wrong_image_hierarchical",
            "p34_context_zero_image_hierarchical",
            "joint_wrong_image_hierarchical",
            "joint_zero_image_hierarchical",
            "joint_wrong_state_hierarchical",
            "joint_zero_state_hierarchical",
            "decoupled_wrong_image",
            "decoupled_zero_image",
        )
    )
    return names


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    arms: nn.ModuleDict,
    cache: dict[str, Any],
    hierarchy: dict[str, Any],
    hard_targets: dict[str, Any],
    scores: torch.Tensor,
    loader: DataLoader,
    positions: list[int],
    *,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    row_indices: torch.Tensor,
    p2_offsets: torch.Tensor,
    context_offsets: torch.Tensor,
    input_h: int,
    input_w: int,
    range_temperature: float,
    top_k: int,
) -> dict[str, Any]:
    arms.eval()
    counts = _new_mode_counts(_evaluation_mode_names())
    oracle_hits = {0.5: 0, 0.7: 0}
    hierarchy_oracle_hits = {0.5: 0, 0.7: 0}
    oracle_gt = 0
    parity_max = 0.0
    offset = 0
    expected_paths = _paths_for_positions(cache, positions)
    progress = tqdm(loader, desc="multiscale evidence eval", ncols=90)
    for images, _targets, metas in progress:
        batch = int(images.shape[0])
        batch_positions = positions[offset : offset + batch]
        _assert_batch_paths(metas, expected_paths[offset : offset + batch])
        index = torch.tensor(batch_positions, dtype=torch.long)
        images = _channels_last_images(
            images,
            device=device,
            channels_last=channels_last,
        )
        curves = cache["stage"]["pred_x_rows"].index_select(0, index)
        evidence = extract_frozen_sources(
            model,
            images,
            cached_curves=curves,
            row_indices=row_indices,
            p2_offsets=p2_offsets,
            context_offsets=context_offsets,
            input_h=input_h,
            input_w=input_w,
            range_temperature=range_temperature,
            amp_dtype=amp_dtype,
            include_wrong_image=True,
        )
        parity_max = max(
            parity_max,
            float(evidence["prediction_parity_max_abs_px"]),
        )
        base = cache["features"].index_select(0, index).to(
            device=device,
            dtype=torch.float32,
        )
        valid = cache["candidate_valid"].index_select(0, index).to(device)
        membership = hierarchy["membership"].index_select(0, index).to(device)
        cluster_valid = hierarchy["cluster_valid"].index_select(0, index).to(device)
        source = scores.index_select(0, index).to(device)
        records = [hierarchy["records"][value] for value in batch_positions]
        teacher_sequences = hard_targets["sequence"].index_select(0, index)

        descriptors: dict[str, dict[str, torch.Tensor]] = {}
        arms_by_condition = {
            "correct": ARM_NAMES,
            "wrong_image": ("p2_strip", "p34_context", "joint"),
            "zero_image": ("p2_strip", "p34_context", "joint"),
            "wrong_state": ("joint",),
            "zero_state": ("joint",),
        }
        for condition, condition_arms in arms_by_condition.items():
            profiles, states = _descriptor_inputs(evidence, condition)
            descriptors[condition] = {
                name: arms[name].descriptor(
                    base,
                    states,
                    profiles,
                    evidence["row_weights"],
                )
                for name in condition_arms
            }

        selected: dict[str, list[list[int]]] = {}
        correct = descriptors["correct"]
        correct_cluster_selections: dict[str, torch.Tensor] = {}
        correct_representative_logits: dict[str, torch.Tensor] = {}
        for arm in ARM_NAMES:
            _cluster_logits, cluster_ids = arms[arm].cluster(
                correct[arm],
                valid,
                membership,
                cluster_valid,
                source,
            )
            representative_logits = arms[arm].representative(
                correct[arm],
                valid,
                source,
            )
            correct_cluster_selections[arm] = cluster_ids
            correct_representative_logits[arm] = representative_logits
            selected[f"{arm}_hierarchical"] = _rows_from_decisions(
                cluster_ids,
                representative_logits,
                records,
                top_k=top_k,
                learned_cluster=True,
                learned_representative=True,
            )
            selected[f"{arm}_cluster_only"] = _rows_from_decisions(
                cluster_ids,
                representative_logits,
                records,
                top_k=top_k,
                learned_cluster=True,
                learned_representative=False,
            )
            selected[f"{arm}_representative_only"] = _rows_from_decisions(
                cluster_ids,
                representative_logits,
                records,
                top_k=top_k,
                learned_cluster=False,
                learned_representative=True,
            )

        selected["decoupled_p34_cluster_p2_representative"] = _rows_from_decisions(
            correct_cluster_selections["p34_context"],
            correct_representative_logits["p2_strip"],
            records,
            top_k=top_k,
            learned_cluster=True,
            learned_representative=True,
        )
        selected["inverse_p2_cluster_p34_representative"] = _rows_from_decisions(
            correct_cluster_selections["p2_strip"],
            correct_representative_logits["p34_context"],
            records,
            top_k=top_k,
            learned_cluster=True,
            learned_representative=True,
        )
        control_specs = {
            "p2_strip_wrong_image_hierarchical": ("p2_strip", "wrong_image"),
            "p2_strip_zero_image_hierarchical": ("p2_strip", "zero_image"),
            "p34_context_wrong_image_hierarchical": ("p34_context", "wrong_image"),
            "p34_context_zero_image_hierarchical": ("p34_context", "zero_image"),
            "joint_wrong_image_hierarchical": ("joint", "wrong_image"),
            "joint_zero_image_hierarchical": ("joint", "zero_image"),
            "joint_wrong_state_hierarchical": ("joint", "wrong_state"),
            "joint_zero_state_hierarchical": ("joint", "zero_state"),
        }
        for mode, (arm, condition) in control_specs.items():
            descriptor = descriptors[condition][arm]
            selected[mode] = _selection_from_heads(
                arms[arm].cluster,
                arms[arm].representative,
                descriptor,
                descriptor,
                candidate_valid=valid,
                membership=membership,
                cluster_valid=cluster_valid,
                scores=source,
                records=records,
                top_k=top_k,
            )
        for condition, mode in (
            ("wrong_image", "decoupled_wrong_image"),
            ("zero_image", "decoupled_zero_image"),
        ):
            selected[mode] = _selection_from_heads(
                arms["p34_context"].cluster,
                arms["p2_strip"].representative,
                descriptors[condition]["p34_context"],
                descriptors[condition]["p2_strip"],
                candidate_valid=valid,
                membership=membership,
                cluster_valid=cluster_valid,
                scores=source,
                records=records,
                top_k=top_k,
            )

        for local_index, global_index in enumerate(batch_positions):
            official = cache["official_iou"][global_index]
            candidate_valid_row = cache["candidate_valid"][global_index]
            source_row = scores[global_index]
            record = hierarchy["records"][global_index]
            raw_ids = _raw_source_ids(
                source_row,
                candidate_valid_row,
                top_k=top_k,
            )
            nms_ids = _source_nms_ids(record, top_k=top_k)
            teacher_clusters = [
                int(value)
                for value in teacher_sequences[local_index].tolist()
                if int(value) >= 0
            ]
            teacher_ids: list[int] = []
            representative_target = record["representative_targets"]
            for cluster_index in teacher_clusters:
                members = record["members"][cluster_index]
                teacher_ids.append(
                    max(
                        members,
                        key=lambda candidate: float(
                            representative_target[int(candidate)]
                        ),
                    )
                )
            _update_counts(counts["raw_source_top4"], official, raw_ids)
            _update_counts(counts["source_nms"], official, nms_ids)
            _update_counts(counts["hierarchy_teacher"], official, teacher_ids)
            for mode, rows in selected.items():
                _update_counts(counts[mode], official, rows[local_index])
            oracle_gt += int(official.shape[0])
            hierarchy_iou = cluster_iou_matrix(official, record)
            for threshold in (0.5, 0.7):
                oracle_hits[threshold] += int(
                    cardinality_oracle_assignment(
                        official,
                        threshold=threshold,
                        top_k=top_k,
                        candidate_valid=candidate_valid_row,
                    ).hit_count
                )
                hierarchy_oracle_hits[threshold] += int(
                    cardinality_oracle_assignment(
                        hierarchy_iou,
                        threshold=threshold,
                        top_k=top_k,
                        candidate_valid=torch.ones(
                            int(hierarchy_iou.shape[1]),
                            dtype=torch.bool,
                        ),
                    ).hit_count
                )
        offset += batch
    if offset != len(positions):
        raise ValueError(f"evaluated {offset} images, expected {len(positions)}")
    return {
        "images": len(positions),
        "prediction_cache_parity_max_abs_px": parity_max,
        "modes": _finish_mode_counts(counts),
        "candidate_oracle_top4": {
            f"{threshold:.2f}": {
                "gt_lanes": int(oracle_gt),
                "tp": int(oracle_hits[threshold]),
                "recall": float(oracle_hits[threshold]) / float(max(oracle_gt, 1)),
            }
            for threshold in (0.5, 0.7)
        },
        "nms_partition_oracle_top4": {
            f"{threshold:.2f}": {
                "gt_lanes": int(oracle_gt),
                "tp": int(hierarchy_oracle_hits[threshold]),
                "recall": float(hierarchy_oracle_hits[threshold])
                / float(max(oracle_gt, 1)),
            }
            for threshold in (0.5, 0.7)
        },
    }


def summarize_evidence_decision(
    train_eval: dict[str, Any],
    val_eval: dict[str, Any],
) -> dict[str, Any]:
    def summarize(split: dict[str, Any]) -> dict[str, Any]:
        modes = split["modes"]
        source = modes["source_nms"]
        gains = {
            name: _metric_gain(row, source)
            for name, row in modes.items()
            if name != "source_nms"
        }
        controls = {
            "p2_strip_correct_minus_wrong": _control_delta(
                modes["p2_strip_hierarchical"],
                modes["p2_strip_wrong_image_hierarchical"],
            ),
            "p2_strip_correct_minus_zero": _control_delta(
                modes["p2_strip_hierarchical"],
                modes["p2_strip_zero_image_hierarchical"],
            ),
            "p34_correct_minus_wrong": _control_delta(
                modes["p34_context_hierarchical"],
                modes["p34_context_wrong_image_hierarchical"],
            ),
            "p34_correct_minus_zero": _control_delta(
                modes["p34_context_hierarchical"],
                modes["p34_context_zero_image_hierarchical"],
            ),
            "joint_correct_minus_wrong_image": _control_delta(
                modes["joint_hierarchical"],
                modes["joint_wrong_image_hierarchical"],
            ),
            "joint_correct_minus_zero_image": _control_delta(
                modes["joint_hierarchical"],
                modes["joint_zero_image_hierarchical"],
            ),
            "joint_correct_minus_wrong_state": _control_delta(
                modes["joint_hierarchical"],
                modes["joint_wrong_state_hierarchical"],
            ),
            "joint_correct_minus_zero_state": _control_delta(
                modes["joint_hierarchical"],
                modes["joint_zero_state_hierarchical"],
            ),
            "decoupled_correct_minus_wrong_image": _control_delta(
                modes["decoupled_p34_cluster_p2_representative"],
                modes["decoupled_wrong_image"],
            ),
            "decoupled_correct_minus_zero_image": _control_delta(
                modes["decoupled_p34_cluster_p2_representative"],
                modes["decoupled_zero_image"],
            ),
        }
        structural_contrasts = {
            "p2_strip_minus_center": _control_delta(
                modes["p2_strip_hierarchical"],
                modes["p2_center_hierarchical"],
            ),
            "p34_cluster_minus_p2_cluster": _control_delta(
                modes["p34_context_cluster_only"],
                modes["p2_strip_cluster_only"],
            ),
            "p2_representative_minus_p34_representative": _control_delta(
                modes["p2_strip_representative_only"],
                modes["p34_context_representative_only"],
            ),
            "decoupled_minus_inverse": _control_delta(
                modes["decoupled_p34_cluster_p2_representative"],
                modes["inverse_p2_cluster_p34_representative"],
            ),
            "joint_minus_state": _control_delta(
                modes["joint_hierarchical"],
                modes["state_hierarchical"],
            ),
        }
        return {
            "gains_over_source_nms": gains,
            "controls": controls,
            "structural_contrasts": structural_contrasts,
        }

    train = summarize(train_eval)
    val = summarize(val_eval)

    def stable_gain(mode: str, threshold: str, minimum: float = 0.5) -> bool:
        key = f"recall_{threshold}_points"
        return bool(
            train["gains_over_source_nms"][mode][key] >= float(minimum)
            and val["gains_over_source_nms"][mode][key] >= float(minimum)
        )

    def grounded(control_name: str, minimum: float = 0.5) -> bool:
        return bool(
            train["controls"][control_name]["recall_050_points"] >= float(minimum)
            and val["controls"][control_name]["recall_050_points"] >= float(minimum)
        )

    state_signal = stable_gain("state_hierarchical", "050")
    p2_signal = bool(
        stable_gain("p2_strip_hierarchical", "070")
        and grounded("p2_strip_correct_minus_wrong")
        and grounded("p2_strip_correct_minus_zero")
    )
    context_signal = bool(
        stable_gain("p34_context_cluster_only", "050")
        and grounded("p34_correct_minus_wrong")
        and grounded("p34_correct_minus_zero")
    )
    decoupled_signal = bool(
        stable_gain("decoupled_p34_cluster_p2_representative", "050")
        and stable_gain("decoupled_p34_cluster_p2_representative", "070")
        and grounded("decoupled_correct_minus_wrong_image")
        and grounded("decoupled_correct_minus_zero_image")
    )
    oracle_050 = float(val_eval["candidate_oracle_top4"]["0.50"]["recall"])
    partition_050 = float(
        val_eval["nms_partition_oracle_top4"]["0.50"]["recall"]
    )
    partition_loss_points = 100.0 * (oracle_050 - partition_050)
    if partition_loss_points >= 3.0:
        diagnosis = "nms_partition_merges_or_discards_structurally_useful_candidates"
        recommendation = (
            "Redesign duplicate partitioning/assignment before another selector. "
            "The fixed NMS clusters destroy at least three recall points of the "
            "candidate oracle."
        )
    elif decoupled_signal:
        diagnosis = "decoupled_semantic_geometry_contract_is_supported"
        recommendation = (
            "Integrate P3/P4 only into cluster/existence selection and keep "
            "row-preserving P2 neighborhood evidence in representative/geometry."
        )
    elif state_signal:
        diagnosis = "decoder_state_is_informative_but_production_pooling_is_lossy"
        recommendation = (
            "Redesign the selector interface to consume the ordered row-state "
            "sequence; do not redesign the whole decoder yet."
        )
    elif p2_signal:
        diagnosis = "local_p2_evidence_exists_but_is_not_transported_to_selection"
        recommendation = (
            "Add row-preserving curve-neighborhood acquisition before the final "
            "representative/quality decision."
        )
    elif context_signal:
        diagnosis = "semantic_context_is_missing_from_cluster_existence_selection"
        recommendation = (
            "Add an isolated P3/P4 semantic selector branch without feeding low-"
            "resolution context back into geometry rows."
        )
    elif oracle_050 >= 0.85:
        diagnosis = "candidate_set_is_rich_but_frozen_decoder_and_fpn_readouts_are_insufficient"
        recommendation = (
            "Stop selector/loss patching. The next experiment must change joint "
            "evidence acquisition or candidate assignment; this frozen contract "
            "cannot recover the oracle gap."
        )
    else:
        diagnosis = "candidate_generation_is_the_primary_bottleneck"
        recommendation = (
            "Improve decoder proposal diversity/geometry before further selection work."
        )
    return {
        "train": train,
        "validation": val,
        "gates": {
            "state_sequence_stable": state_signal,
            "p2_image_grounded_stable": p2_signal,
            "p34_context_grounded_stable": context_signal,
            "decoupled_contract_stable": decoupled_signal,
            "minimum_stable_gain_points": 0.5,
            "minimum_correct_over_control_points": 0.5,
            "nms_partition_loss_050_points": partition_loss_points,
            "nms_partition_structural_failure": partition_loss_points >= 3.0,
        },
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "limits": (
            "This is a frozen-checkpoint sufficiency test. A negative result "
            "rejects the present 10k representation contract, not every jointly "
            "trained decoder or every later checkpoint."
        ),
    }


def _hierarchy_statistics(hierarchy: dict[str, Any]) -> dict[str, Any]:
    valid = hierarchy["cluster_valid"].bool()
    membership = hierarchy["membership"].bool()
    sizes = membership.sum(dim=-1)[valid]
    return {
        "images": int(valid.shape[0]),
        "valid_clusters": int(valid.sum()),
        "mean_clusters_per_image": float(valid.sum()) / float(max(int(valid.shape[0]), 1)),
        "mean_candidates_per_cluster": float(sizes.float().mean())
        if int(sizes.numel())
        else None,
        "multi_candidate_cluster_fraction": float((sizes >= 2).float().mean())
        if int(sizes.numel())
        else None,
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    model_cfg = cfg["model"]
    input_h = int(model_cfg["input_h"])
    input_w = int(model_cfg["input_w"])
    total_rows = int(model_cfg["num_rows"])
    curve_samples = min(int(args.curve_samples), total_rows)
    row_indices = torch.linspace(0, total_rows - 1, curve_samples).round().long().to(device)
    p2_offsets = torch.tensor(args.p2_offsets_px, dtype=torch.float32, device=device)
    context_offsets = torch.tensor(
        args.context_offsets_px,
        dtype=torch.float32,
        device=device,
    )
    offset_contract = torch.stack((p2_offsets, context_offsets, context_offsets))

    train_cache = _load_cache(args.train_cache)
    val_cache = _load_cache(args.val_cache)
    _validate_cache(train_cache, "train")
    _validate_cache(val_cache, "val")
    if int(train_cache["features"].shape[1]) != int(val_cache["features"].shape[1]):
        raise ValueError("train/validation candidate counts differ")
    train_paths = set(_paths_for_positions(train_cache, list(range(len(train_cache["features"])))))
    val_paths = set(_paths_for_positions(val_cache, list(range(len(val_cache["features"])))))
    overlap = train_paths & val_paths
    if overlap:
        raise ValueError(f"train/validation cache overlap: {sorted(overlap)[0]}")

    model = build_model(cfg)
    checkpoint_iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    if model.structured_query_head is None:
        raise ValueError("checkpoint has no structured query head")
    selector = model.structured_query_head.set_selection_head
    if selector is None or not selector.unified_score:
        raise ValueError("checkpoint has no unified set-selection head")
    selector_cpu = deepcopy(selector).float().cpu()
    # The selector is not needed in the frozen feature pass; removing it keeps
    # the diagnostic memory bounded while leaving decoder rows/curves intact.
    model.structured_query_head.set_selection_head = None
    model.structured_query_head.intermediate_supervision = False
    model.requires_grad_(False)
    model = model.to(device).eval()
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False) and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    train_scores = source_scores(
        deepcopy(selector_cpu),
        train_cache,
        device=device,
        batch_size=64,
    )
    val_scores = source_scores(
        deepcopy(selector_cpu),
        val_cache,
        device=device,
        batch_size=64,
    )
    train_hierarchy = build_hierarchical_supervision(
        train_cache,
        train_scores,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        nms_distance=args.nms_distance,
        nms_min_overlap_points=args.nms_min_overlap_points,
        top_k=args.top_k,
        positive_iou=args.representative_positive_iou,
    )
    val_hierarchy = build_hierarchical_supervision(
        val_cache,
        val_scores,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        nms_distance=args.nms_distance,
        nms_min_overlap_points=args.nms_min_overlap_points,
        top_k=args.top_k,
        positive_iou=args.representative_positive_iou,
    )
    train_targets = build_hard_decision_targets(
        train_cache,
        train_hierarchy,
        top_k=args.top_k,
        temperature=args.coverage_temperature,
        min_gain=args.coverage_min_gain,
        weight_050=args.coverage_weight_050,
        weight_070=args.coverage_weight_070,
        weight_iou=args.coverage_weight_iou,
    )
    val_targets = build_hard_decision_targets(
        val_cache,
        val_hierarchy,
        top_k=args.top_k,
        temperature=args.coverage_temperature,
        min_gain=args.coverage_min_gain,
        weight_050=args.coverage_weight_050,
        weight_070=args.coverage_weight_070,
        weight_iou=args.coverage_weight_iou,
    )

    full_positions = list(range(int(train_cache["features"].shape[0])))
    train_loader = _loader_for_cache_positions(
        cfg,
        train_cache,
        split="train",
        positions=full_positions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    preview_images, _preview_targets, preview_metas = next(iter(train_loader))
    preview_positions = full_positions[: int(preview_images.shape[0])]
    _assert_batch_paths(preview_metas, _paths_for_positions(train_cache, preview_positions))
    preview_images = _channels_last_images(
        preview_images,
        device=device,
        channels_last=channels_last,
    )
    preview_index = torch.tensor(preview_positions, dtype=torch.long)
    preview = extract_frozen_sources(
        model,
        preview_images,
        cached_curves=train_cache["stage"]["pred_x_rows"].index_select(
            0,
            preview_index,
        ),
        row_indices=row_indices,
        p2_offsets=p2_offsets,
        context_offsets=context_offsets,
        input_h=input_h,
        input_w=input_w,
        range_temperature=args.range_temperature,
        amp_dtype=amp_dtype,
        include_wrong_image=False,
    )
    base_dim = int(train_cache["features"].shape[-1])
    row_state_dim = int(preview["row_states"].shape[-1])
    feature_dim = int(preview["profiles"].shape[-1])
    if int(preview["profiles"].shape[3]) != len(SCALE_NAMES):
        raise ValueError("preview did not expose P2/P3/P4")
    preview_parity = float(preview["prediction_parity_max_abs_px"])
    if preview_parity > 1.0:
        raise ValueError(
            "checkpoint/cache prediction parity exceeded 1px "
            f"(max_abs={preview_parity:.6f}px); refusing the probe"
        )
    del preview, preview_images
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _seed_everything(args.seed)
    masks = evidence_masks(len(args.p2_offsets_px))
    template_arm = EvidenceSelectionArm(
        base_dim=base_dim,
        row_state_dim=row_state_dim,
        feature_dim=feature_dim,
        rows=curve_samples,
        offsets_px=offset_contract.cpu(),
        evidence_mask=masks["state"],
        sequence_dim=args.sequence_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        top_k=args.top_k,
    )
    arm_rows: dict[str, EvidenceSelectionArm] = {}
    for name in ARM_NAMES:
        arm = deepcopy(template_arm)
        arm.descriptor.evidence_mask.copy_(masks[name])
        arm_rows[name] = arm
    arms = nn.ModuleDict(arm_rows).to(device)
    parameter_counts = {
        name: sum(parameter.numel() for parameter in arms[name].parameters())
        for name in ARM_NAMES
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"equal-capacity arm contract violated: {parameter_counts}")
    optimizer = torch.optim.AdamW(
        arms.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    totals = {
        name: {"coverage": 0.0, "representative": 0.0, "eligible": 0}
        for name in ARM_NAMES
    }
    parity_max = 0.0
    step = 0
    progress = tqdm(total=int(args.train_steps), desc="evidence sufficiency train", ncols=96)
    arms.train()
    while step < int(args.train_steps):
        offset = 0
        expected = _paths_for_positions(train_cache, full_positions)
        for images, _batch_targets, metas in train_loader:
            if step >= int(args.train_steps):
                break
            batch = int(images.shape[0])
            positions = full_positions[offset : offset + batch]
            _assert_batch_paths(metas, expected[offset : offset + batch])
            index = torch.tensor(positions, dtype=torch.long)
            images = _channels_last_images(
                images,
                device=device,
                channels_last=channels_last,
            )
            evidence = extract_frozen_sources(
                model,
                images,
                cached_curves=train_cache["stage"]["pred_x_rows"].index_select(
                    0,
                    index,
                ),
                row_indices=row_indices,
                p2_offsets=p2_offsets,
                context_offsets=context_offsets,
                input_h=input_h,
                input_w=input_w,
                range_temperature=args.range_temperature,
                amp_dtype=amp_dtype,
                include_wrong_image=False,
            )
            parity_max = max(
                parity_max,
                float(evidence["prediction_parity_max_abs_px"]),
            )
            base = train_cache["features"].index_select(0, index).to(
                device=device,
                dtype=torch.float32,
            )
            valid = train_cache["candidate_valid"].index_select(0, index).to(device)
            membership = train_hierarchy["membership"].index_select(0, index).to(device)
            cluster_valid = train_hierarchy["cluster_valid"].index_select(0, index).to(device)
            source = train_scores.index_select(0, index).to(device)
            target_distribution = train_targets["distribution"].index_select(
                0,
                index,
            ).to(device)
            target_sequence = train_targets["sequence"].index_select(0, index).to(device)
            target_active = train_targets["active"].index_select(0, index).to(device)
            representative_targets = train_hierarchy["representative_targets"].index_select(
                0,
                index,
            ).to(device)

            losses: list[torch.Tensor] = []
            running_message: list[str] = []
            for name in ARM_NAMES:
                descriptor = arms[name].descriptor(
                    base,
                    evidence["row_states"],
                    evidence["profiles"],
                    evidence["row_weights"],
                )
                cluster_logits, _selection = arms[name].cluster(
                    descriptor,
                    valid,
                    membership,
                    cluster_valid,
                    source,
                    teacher_sequence=target_sequence,
                )
                coverage = coverage_pointer_loss(
                    cluster_logits,
                    target_distribution,
                    target_active,
                )
                representative_logits = arms[name].representative(
                    descriptor,
                    valid,
                    source,
                )
                representative, statistics = representative_listwise_loss(
                    representative_logits,
                    representative_targets,
                    membership,
                    cluster_valid,
                    temperature=args.representative_temperature,
                    positive_iou=args.representative_positive_iou,
                    min_spread=args.representative_min_spread,
                )
                losses.append(coverage + representative)
                totals[name]["coverage"] += float(coverage.detach())
                totals[name]["representative"] += float(representative.detach())
                totals[name]["eligible"] += int(statistics["eligible_clusters"])
                running_message.append(
                    f"{name}={float((coverage + representative).detach()):.3f}"
                )
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(arms.parameters(), max_norm=5.0)
            optimizer.step()
            step += 1
            offset += batch
            progress.update(1)
            if int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
                progress.write(f"step {step:04d}: " + " ".join(running_message))
    progress.close()
    if parity_max > 1.0:
        raise ValueError(
            "training prediction/cache parity exceeded 1px "
            f"(max_abs={parity_max:.6f}px)"
        )

    train_eval_count = min(int(args.train_eval_images), len(full_positions))
    train_eval_positions = uniformly_spaced_indices(len(full_positions), train_eval_count)
    val_all_positions = list(range(int(val_cache["features"].shape[0])))
    val_eval_count = min(int(args.val_eval_images), len(val_all_positions))
    val_eval_positions = uniformly_spaced_indices(len(val_all_positions), val_eval_count)
    train_eval_loader = _loader_for_cache_positions(
        cfg,
        train_cache,
        split="train",
        positions=train_eval_positions,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    val_eval_loader = _loader_for_cache_positions(
        cfg,
        val_cache,
        split="val",
        positions=val_eval_positions,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    train_eval = evaluate_split(
        model,
        arms,
        train_cache,
        train_hierarchy,
        train_targets,
        train_scores,
        train_eval_loader,
        train_eval_positions,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        row_indices=row_indices,
        p2_offsets=p2_offsets,
        context_offsets=context_offsets,
        input_h=input_h,
        input_w=input_w,
        range_temperature=args.range_temperature,
        top_k=args.top_k,
    )
    val_eval = evaluate_split(
        model,
        arms,
        val_cache,
        val_hierarchy,
        val_targets,
        val_scores,
        val_eval_loader,
        val_eval_positions,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        row_indices=row_indices,
        p2_offsets=p2_offsets,
        context_offsets=context_offsets,
        input_h=input_h,
        input_w=input_w,
        range_temperature=args.range_temperature,
        top_k=args.top_k,
    )
    decision = summarize_evidence_decision(train_eval, val_eval)

    payload = {
        "diagnostic_only": True,
        "warning": (
            "The detector, decoder, candidate curves, FPN, NMS partition, and "
            "source scores are frozen. Only equal-capacity diagnostic readouts "
            "are trained; these are not benchmark results."
        ),
        "question": (
            "Is the ceiling caused by lossy decoder-state pooling, missing local "
            "P2 neighborhood evidence, missing P3/P4 semantic context, or an "
            "incompatible unified selection contract?"
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "feature_contract": {
            "base_descriptor_dim": base_dim,
            "row_state_dim": row_state_dim,
            "fpn_feature_dim": feature_dim,
            "curve_samples": curve_samples,
            "scales": list(SCALE_NAMES),
            "p2_offsets_px": [float(value) for value in args.p2_offsets_px],
            "context_offsets_px": [float(value) for value in args.context_offsets_px],
            "p3_p4_source": (
                "trained top-down tensors feeding P2; unused random "
                "pyramid_outputs convolutions are excluded"
            ),
            "row_axis_preserved_before_sequence_convolution": True,
            "arm_masks": {
                name: masks[name].int().tolist() for name in ARM_NAMES
            },
            "equal_parameter_count": len(set(parameter_counts.values())) == 1,
            "parameter_counts": parameter_counts,
        },
        "protocol": {
            "train_cache_images": int(train_cache["features"].shape[0]),
            "val_cache_images": int(val_cache["features"].shape[0]),
            "train_steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "train_val_paths_disjoint": True,
            "validation_not_used_for_checkpoint_selection": True,
            "wrong_image_control": (
                "roll image feature maps within the batch, then sample them at "
                "the original image's frozen candidate curves"
            ),
            "wrong_state_control": (
                "roll ordered decoder row states while preserving base "
                "descriptors and correct image profiles"
            ),
            "source_nms_distance_px": float(args.nms_distance),
            "top_k": int(args.top_k),
        },
        "training_losses": {
            name: {
                "mean_coverage": values["coverage"] / float(max(args.train_steps, 1)),
                "mean_representative": values["representative"]
                / float(max(args.train_steps, 1)),
                "eligible_cluster_observations": int(values["eligible"]),
            }
            for name, values in totals.items()
        },
        "hierarchy": {
            "train": _hierarchy_statistics(train_hierarchy),
            "validation": _hierarchy_statistics(val_hierarchy),
            "train_coverage_targets": train_targets["statistics"],
            "validation_coverage_targets": val_targets["statistics"],
        },
        "train_evaluation": train_eval,
        "validation_evaluation": val_eval,
        "decision": decision,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_path = Path(args.save_probe)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arms": arms.state_dict(),
            "source_checkpoint": args.checkpoint,
            "source_iteration": checkpoint_iteration,
            "arm_masks": {name: masks[name] for name in ARM_NAMES},
            "p2_offsets_px": [float(value) for value in args.p2_offsets_px],
            "context_offsets_px": [float(value) for value in args.context_offsets_px],
            "seed": int(args.seed),
            "train_steps": int(args.train_steps),
        },
        save_path,
    )
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_path}")
    print(f"probe_checkpoint: {save_path}")


if __name__ == "__main__":
    main()
