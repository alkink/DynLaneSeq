from __future__ import annotations

import argparse
import copy
import hashlib
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
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.losses.loss_s0 import (
    build_four_slot_cluster_targets,
    four_slot_collision_loss,
    four_slot_factorized_permutation_loss,
)
from dynlaneseq_eg.modeling.common import fixed_row_fractions, sort_range_norm
from dynlaneseq_eg.modeling.four_slot_selection import (
    decode_unique_real_slot_routes,
)
from dynlaneseq_eg.tools.audit_v8_route_support_reference_policies import (
    _hard_min_slots,
)
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.train import seed_everything


ARMS = ("descriptor", "row_tokens", "p2_evidence", "combined")
CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the correct global GT proposal support is "
            "observable from the production 819D descriptor, full proposal "
            "row tokens, proposal-aligned seven-offset P2 evidence, or their "
            "combination. All arms use the same four-slot set probe."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--train-images", type=int, default=4096)
    parser.add_argument("--early-stop-images", type=int, default=512)
    parser.add_argument("--val-images", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--row-samples", type=int, default=24)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=8849)
    parser.add_argument("--split-seed", type=int, default=1907)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--row-layers", type=int, default=2)
    parser.add_argument("--candidate-layers", type=int, default=2)
    parser.add_argument("--slot-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--collision-weight", type=float, default=0.01)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(3407, 5419),
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fixed_rademacher_projection(
    input_dimension: int,
    output_dimension: int,
    seed: int,
) -> torch.Tensor:
    if int(input_dimension) < 1 or int(output_dimension) < 1:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (int(input_dimension), int(output_dimension)),
        generator=generator,
        dtype=torch.int64,
    )
    return (signs.float() * 2.0 - 1.0) / math.sqrt(
        float(output_dimension)
    )


def _autocast_context(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and str(amp_dtype) != "none"
    dtype = {
        "none": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(amp_dtype)]
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(args.config))
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.feature_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(int(args.num_workers) > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _cache_signature(
    args: argparse.Namespace,
    *,
    split: str,
    images: int,
) -> dict[str, Any]:
    config = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    return {
        "version": CACHE_VERSION,
        "split": str(split),
        "images": int(images),
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "amp_dtype": str(args.amp_dtype),
        "row_samples": int(args.row_samples),
        "feature_dim": int(args.feature_dim),
        "projection_seed": int(args.projection_seed),
    }


def _pad_target_rows(
    rows: list[torch.Tensor],
    *,
    slots: int,
    candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.zeros(
        (len(rows), int(slots), int(candidates) + 1),
        dtype=torch.float16,
    )
    active = torch.zeros((len(rows), int(slots)), dtype=torch.bool)
    for image_index, value in enumerate(rows):
        count = min(int(value.shape[0]), int(slots))
        if count:
            padded[image_index, :count] = value[:count].detach().cpu().half()
            active[image_index, :count] = True
    return padded, active


def _target_rows_for_ids(
    cache: dict[str, Any],
    ids: list[int],
) -> list[torch.Tensor]:
    padded = cache["target_rows"][ids]
    active = cache["target_active"][ids]
    return [
        padded[row, active[row]].float()
        for row in range(int(padded.shape[0]))
    ]


@torch.no_grad()
def collect_feature_cache(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model: nn.Module,
    *,
    split: str,
    images_requested: int,
    cache_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    signature = _cache_signature(
        args,
        split=split,
        images=images_requested,
    )
    if bool(args.reuse_cache) and cache_path.is_file():
        payload = _load_torch(cache_path)
        if payload.get("signature") != signature:
            raise ValueError(
                f"cache signature mismatch for {cache_path}; do not reuse it"
            )
        return payload

    loader = build_dataloader(cfg, split=split, training=False)
    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy="uniform",
        max_batches=math.ceil(
            int(images_requested) / int(args.feature_batch_size)
        ),
        num_workers=int(args.num_workers),
    )
    selector = model.structured_query_head.set_selection_head
    geometry = selector.slot_owned_geometry
    if geometry is None:
        raise ValueError("global-support probe requires V9 slot-owned geometry")

    feature_rows: dict[str, list[torch.Tensor]] = {
        arm: [] for arm in ARMS
    }
    visible_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    target_active: list[torch.Tensor] = []
    baseline_active: list[torch.Tensor] = []
    baseline_route: list[torch.Tensor] = []
    image_ids: list[str] = []
    row_ids_cpu: torch.Tensor | None = None
    projections: dict[str, torch.Tensor] = {}
    collected = 0
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})
    model.eval()

    for images, targets, metas in tqdm(
        loader,
        desc=f"V9 {split} observability cache",
        ncols=96,
    ):
        if collected >= int(images_requested):
            break
        images = images.to(device, non_blocking=True)
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, _args, kwargs):
            captured.update(kwargs)

        handle = geometry.register_forward_pre_hook(capture, with_kwargs=True)
        with _autocast_context(device, str(args.amp_dtype)):
            outputs = model(images)
        handle.remove()
        if "row_value_features" not in captured:
            raise RuntimeError("V9 geometry hook did not capture P2 row features")

        descriptor = selector._proposal_features(outputs).detach().float()
        row_tokens = outputs["structured_row_tokens"].detach().float()
        local_p2 = geometry._sample_local_evidence(
            captured["row_value_features"],
            outputs["pred_x_rows"],
        ).detach().float()
        batch, candidates, rows, channels = row_tokens.shape
        offsets = int(local_p2.shape[3])
        if row_ids_cpu is None:
            row_ids_cpu = torch.linspace(
                0,
                rows - 1,
                min(int(args.row_samples), rows),
            ).round().long().unique(sorted=True)
            projections = {
                "descriptor": fixed_rademacher_projection(
                    int(descriptor.shape[-1]),
                    int(args.feature_dim),
                    int(args.projection_seed),
                ),
                "row_tokens": fixed_rademacher_projection(
                    channels,
                    int(args.feature_dim),
                    int(args.projection_seed) + 1,
                ),
                "p2_evidence": fixed_rademacher_projection(
                    offsets * int(local_p2.shape[-1]),
                    int(args.feature_dim),
                    int(args.projection_seed) + 2,
                ),
                "combined": fixed_rademacher_projection(
                    2 * int(args.feature_dim),
                    int(args.feature_dim),
                    int(args.projection_seed) + 3,
                ),
            }
        assert row_ids_cpu is not None
        row_ids = row_ids_cpu.to(device)
        descriptor_feature = torch.matmul(
            descriptor,
            projections["descriptor"].to(device),
        )
        row_feature = torch.matmul(
            row_tokens.index_select(2, row_ids),
            projections["row_tokens"].to(device),
        )
        p2_sampled = local_p2.index_select(2, row_ids).flatten(3)
        p2_feature = torch.matmul(
            p2_sampled,
            projections["p2_evidence"].to(device),
        )
        combined = torch.matmul(
            torch.cat((row_feature, p2_feature), dim=-1),
            projections["combined"].to(device),
        )

        ranges = sort_range_norm(outputs["range_norm"].detach().float())
        y = fixed_row_fractions(
            rows,
            device=device,
            dtype=torch.float32,
        ).index_select(0, row_ids)
        visible = (
            (y.view(1, 1, -1) >= ranges[..., :1])
            & (y.view(1, 1, -1) <= ranges[..., 1:])
        )
        candidate_valid = outputs[
            "selection_slot_candidate_valid"
        ].detach().bool()
        target_data = build_four_slot_cluster_targets(
            outputs,
            targets,
            num_slots=int(outputs["selection_slot_real_route_logits"].shape[1]),
            input_h=int(model_cfg.get("input_h", 288)),
            line_width=float(loss_cfg.get("four_slot_line_width", 30.0)),
            min_valid_rows=int(loss_cfg.get("four_slot_min_valid_rows", 5)),
            representable_min=float(
                loss_cfg.get("four_slot_representable_min", 0.0)
            ),
            cluster_min=float(loss_cfg.get("four_slot_cluster_min", 0.0)),
            cluster_delta=float(
                loss_cfg.get("four_slot_cluster_delta", 0.10)
            ),
            temperature=float(
                loss_cfg.get("four_slot_cluster_temperature", 0.03)
            ),
            target_mode=str(loss_cfg.get("four_slot_target_mode", "all_gt")),
        )
        padded, active = _pad_target_rows(
            target_data["rows"],
            slots=int(outputs["selection_slot_real_route_logits"].shape[1]),
            candidates=candidates,
        )
        keep = min(int(batch), int(images_requested) - collected)
        descriptor_sequence = descriptor_feature.unsqueeze(2).expand(
            -1,
            -1,
            int(row_ids.numel()),
            -1,
        )
        values = {
            "descriptor": descriptor_sequence,
            "row_tokens": row_feature,
            "p2_evidence": p2_feature,
            "combined": combined,
        }
        for arm, value in values.items():
            feature_rows[arm].append(
                value[:keep].detach().to(device="cpu", dtype=torch.float16)
            )
        visible_rows.append(visible[:keep].detach().cpu())
        valid_rows.append(candidate_valid[:keep].detach().cpu())
        target_rows.append(padded[:keep])
        target_active.append(active[:keep])
        baseline_active.append(
            outputs["selection_slot_active_logits"][:keep]
            .detach()
            .to(device="cpu", dtype=torch.float16)
        )
        baseline_route.append(
            outputs["selection_slot_real_route_logits"][:keep]
            .detach()
            .to(device="cpu", dtype=torch.float16)
        )
        for local_index in range(keep):
            image_ids.append(str(metas[local_index].get("image_path", collected)))
            collected += 1

    if collected != int(images_requested):
        raise RuntimeError(
            f"requested {images_requested} {split} images, collected {collected}"
        )
    assert row_ids_cpu is not None
    payload = {
        "signature": signature,
        "metadata": {
            "sampled_dataset_indices": sampled_indices[:collected],
            "image_ids": image_ids,
            "augmentation_enabled": False,
            "row_ids": row_ids_cpu.tolist(),
            "arms": list(ARMS),
            "feature_dim": int(args.feature_dim),
        },
        "features": {
            arm: torch.cat(rows_value, dim=0)
            for arm, rows_value in feature_rows.items()
        },
        "visible": torch.cat(visible_rows, dim=0),
        "candidate_valid": torch.cat(valid_rows, dim=0),
        "target_rows": torch.cat(target_rows, dim=0),
        "target_active": torch.cat(target_active, dim=0),
        "baseline_active_logits": torch.cat(baseline_active, dim=0),
        "baseline_route_logits": torch.cat(baseline_route, dim=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def feature_moments(
    cache: dict[str, Any],
    arm: str,
    indices: list[int],
    *,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature = cache["features"][arm]
    dimension = int(feature.shape[-1])
    total = torch.zeros(dimension, dtype=torch.float64)
    square = torch.zeros(dimension, dtype=torch.float64)
    count = 0
    for start in range(0, len(indices), int(chunk_size)):
        ids = indices[start : start + int(chunk_size)]
        value = feature[ids].float()
        mask = (
            cache["visible"][ids].bool()
            & cache["candidate_valid"][ids].bool().unsqueeze(-1)
        )
        selected = value[mask]
        if not int(selected.numel()):
            continue
        total += selected.double().sum(dim=0)
        square += selected.double().square().sum(dim=0)
        count += int(selected.shape[0])
    if count < 2:
        raise ValueError("not enough valid features for normalization")
    mean = total / float(count)
    variance = square / float(count) - mean.square()
    return mean.float(), variance.clamp_min(1e-6).sqrt().float()


class GlobalSupportSetProbe(nn.Module):
    """Common-capacity row encoder and four-slot global support selector."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        row_count: int,
        hidden_dim: int,
        row_layers: int,
        candidate_layers: int,
        slot_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        num_slots: int = 4,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads):
            raise ValueError("probe hidden dimension must divide attention heads")
        self.register_buffer("feature_mean", mean.float())
        self.register_buffer("feature_std", std.float().clamp_min(1e-3))
        self.input_projection = nn.Linear(int(mean.numel()), int(hidden_dim))
        self.row_position = nn.Parameter(
            torch.zeros(1, int(row_count), int(hidden_dim))
        )
        row_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(
            row_layer,
            num_layers=int(row_layers),
            enable_nested_tensor=False,
        )
        self.row_pool = nn.Linear(int(hidden_dim), 1)
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(
            candidate_layer,
            num_layers=int(candidate_layers),
            enable_nested_tensor=False,
        )
        slot_layer = nn.TransformerDecoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            slot_layer,
            num_layers=int(slot_layers),
        )
        self.slot_tokens = nn.Embedding(int(num_slots), int(hidden_dim))
        self.slot_norm = nn.LayerNorm(int(hidden_dim))
        self.candidate_norm = nn.LayerNorm(int(hidden_dim))
        self.slot_query = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.candidate_key = nn.Linear(
            int(hidden_dim),
            int(hidden_dim),
            bias=False,
        )
        self.active = nn.Linear(int(hidden_dim), 1)
        nn.init.trunc_normal_(self.row_position, std=0.02)
        nn.init.normal_(self.slot_tokens.weight, std=0.02)
        nn.init.zeros_(self.active.weight)
        nn.init.constant_(self.active.bias, math.log(4.0))

    def forward(
        self,
        features: torch.Tensor,
        visible: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, rows, dimension = features.shape
        if int(dimension) != int(self.feature_mean.numel()):
            raise ValueError("probe feature dimension mismatch")
        normalized = (features.float() - self.feature_mean) / self.feature_std
        hidden = self.input_projection(normalized)
        hidden = hidden + self.row_position.unsqueeze(1)
        hidden = hidden.reshape(batch * candidates, rows, -1)
        keep = visible.bool().reshape(batch * candidates, rows)
        safe_keep = keep.clone()
        empty = ~safe_keep.any(dim=-1)
        if bool(empty.any()):
            safe_keep[empty, 0] = True
        hidden = self.row_encoder(hidden, src_key_padding_mask=~safe_keep)
        pool_logit = self.row_pool(hidden).squeeze(-1).masked_fill(
            ~safe_keep,
            -1.0e4,
        )
        pooled = (
            hidden * torch.softmax(pool_logit, dim=-1).unsqueeze(-1)
        ).sum(dim=1)
        memory = pooled.reshape(batch, candidates, -1)
        attention_valid = candidate_valid.bool().clone()
        attention_valid[:, 0] |= ~attention_valid.any(dim=-1)
        memory = self.candidate_encoder(
            memory,
            src_key_padding_mask=~attention_valid,
        )
        slots = self.slot_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
        slots = self.slot_decoder(
            slots,
            memory,
            memory_key_padding_mask=~attention_valid,
        )
        slots = self.slot_norm(slots)
        memory = self.candidate_norm(memory)
        route = torch.einsum(
            "bsd,bnd->bsn",
            self.slot_query(slots),
            self.candidate_key(memory),
        ) / math.sqrt(float(memory.shape[-1]))
        route = route.masked_fill(~candidate_valid[:, None, :].bool(), -1.0e4)
        active = self.active(slots).squeeze(-1)
        return active, route


def _model_for_arm(
    args: argparse.Namespace,
    moments: tuple[torch.Tensor, torch.Tensor],
    row_count: int,
) -> GlobalSupportSetProbe:
    return GlobalSupportSetProbe(
        *moments,
        row_count=int(row_count),
        hidden_dim=int(args.hidden_dim),
        row_layers=int(args.row_layers),
        candidate_layers=int(args.candidate_layers),
        slot_layers=int(args.slot_layers),
        num_heads=int(args.num_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    )


def _forward_cache(
    model: GlobalSupportSetProbe,
    cache: dict[str, Any],
    arm: str,
    ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        cache["features"][arm][ids].to(device),
        cache["visible"][ids].to(device),
        cache["candidate_valid"][ids].to(device),
    )


def _metric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.5)),
    }


@torch.no_grad()
def evaluate_logits(
    cache: dict[str, Any],
    indices: list[int],
    active_logits: torch.Tensor,
    route_logits: torch.Tensor,
) -> dict[str, Any]:
    if int(active_logits.shape[0]) != len(indices):
        raise ValueError("evaluation logits and index count differ")
    support_mass: list[float] = []
    ranks: list[int] = []
    support_hit = 0
    exact_id = 0
    assigned_gt = 0
    count_error: list[float] = []
    count_exact = 0
    route_probability = torch.softmax(route_logits.float(), dim=-1)
    valid = cache["candidate_valid"][indices].bool()
    decoded = decode_unique_real_slot_routes(route_logits.float(), valid)[
        "indices"
    ]
    target_rows = _target_rows_for_ids(cache, indices)
    for local_index, target in enumerate(target_rows):
        gt_count = int(target.shape[0])
        predicted_count = int((active_logits[local_index] >= 0.0).sum())
        error = abs(predicted_count - gt_count)
        count_error.append(float(error))
        count_exact += int(error == 0)
        if gt_count == 0:
            continue
        slots_for_gt = _hard_min_slots(
            active_logits[local_index],
            route_logits[local_index],
            target[..., :-1],
        )
        for gt, slot in enumerate(slots_for_gt):
            row = target[gt, :-1].float()
            support = row > 0.0
            probability = route_probability[local_index, int(slot)]
            mass = float(probability[support].sum())
            target_id = int(row.argmax())
            order = probability.argsort(descending=True)
            rank = int((order == target_id).nonzero(as_tuple=False)[0]) + 1
            current = int(decoded[local_index, int(slot)])
            support_mass.append(mass)
            ranks.append(rank)
            support_hit += int(current >= 0 and bool(support[current]))
            exact_id += int(current == target_id)
            assigned_gt += 1
    denominator = max(assigned_gt, 1)
    return {
        "images": len(indices),
        "assigned_gt": assigned_gt,
        "mean_target_support_mass": float(sum(support_mass)) / denominator,
        "median_target_support_mass": _metric_summary(support_mass)["median"],
        "route_in_support_fraction": float(support_hit) / denominator,
        "decoded_target_id_fraction": float(exact_id) / denominator,
        "target_id_rank_top1_fraction": float(sum(rank <= 1 for rank in ranks))
        / denominator,
        "target_id_rank_top2_fraction": float(sum(rank <= 2 for rank in ranks))
        / denominator,
        "target_id_rank_top4_fraction": float(sum(rank <= 4 for rank in ranks))
        / denominator,
        "cardinality_exact_fraction": float(count_exact) / max(len(indices), 1),
        "cardinality_mae": float(sum(count_error)) / max(len(indices), 1),
        "selection_score": (
            float(sum(support_mass)) / denominator
            + float(support_hit) / denominator
        )
        / 2.0,
    }


@torch.no_grad()
def evaluate_probe(
    model: GlobalSupportSetProbe,
    cache: dict[str, Any],
    arm: str,
    indices: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    active_rows: list[torch.Tensor] = []
    route_rows: list[torch.Tensor] = []
    for start in range(0, len(indices), int(batch_size)):
        ids = indices[start : start + int(batch_size)]
        active, route = _forward_cache(model, cache, arm, ids, device)
        active_rows.append(active.detach().cpu())
        route_rows.append(route.detach().cpu())
    return evaluate_logits(
        cache,
        indices,
        torch.cat(active_rows, dim=0),
        torch.cat(route_rows, dim=0),
    )


def _state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_probe(
    args: argparse.Namespace,
    train_cache: dict[str, Any],
    val_cache: dict[str, Any],
    *,
    arm: str,
    seed: int,
    fit_indices: list[int],
    early_indices: list[int],
    moments: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(int(seed))
    model = _model_for_arm(
        args,
        moments,
        int(train_cache["features"][arm].shape[2]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 101)
    best_score = -float("inf")
    best_epoch = 0
    best_state = _state_to_cpu(model)
    stale = 0
    trajectory: list[dict[str, Any]] = []
    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        order = torch.randperm(len(fit_indices), generator=generator).tolist()
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(order), int(args.train_batch_size)):
            positions = order[start : start + int(args.train_batch_size)]
            ids = [fit_indices[position] for position in positions]
            active, route = _forward_cache(
                model,
                train_cache,
                arm,
                ids,
                device,
            )
            targets = _target_rows_for_ids(train_cache, ids)
            permutation = four_slot_factorized_permutation_loss(
                active,
                route,
                targets,
                assignment_mode="hard_min",
            )
            collision = four_slot_collision_loss(route, has_dustbin=False)
            loss = permutation + float(args.collision_weight) * collision
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
        early = evaluate_probe(
            model,
            train_cache,
            arm,
            early_indices,
            batch_size=int(args.train_batch_size),
            device=device,
        )
        score = float(early["selection_score"])
        trajectory.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(batches, 1),
                "early_stop": early,
            }
        )
        if score > best_score + 1.0e-4:
            best_score = score
            best_epoch = epoch
            best_state = _state_to_cpu(model)
            stale = 0
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    model.load_state_dict(best_state)
    fit = evaluate_probe(
        model,
        train_cache,
        arm,
        fit_indices,
        batch_size=int(args.train_batch_size),
        device=device,
    )
    early = evaluate_probe(
        model,
        train_cache,
        arm,
        early_indices,
        batch_size=int(args.train_batch_size),
        device=device,
    )
    val_indices = list(range(int(val_cache["candidate_valid"].shape[0])))
    validation = evaluate_probe(
        model,
        val_cache,
        arm,
        val_indices,
        batch_size=int(args.train_batch_size),
        device=device,
    )
    return {
        "arm": arm,
        "seed": int(seed),
        "parameter_count": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
        "best_epoch": int(best_epoch),
        "epochs_run": len(trajectory),
        "trajectory": trajectory,
        "fit": fit,
        "early_stop": early,
        "validation": validation,
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "mean_target_support_mass",
        "route_in_support_fraction",
        "decoded_target_id_fraction",
        "target_id_rank_top1_fraction",
        "target_id_rank_top2_fraction",
        "target_id_rank_top4_fraction",
        "cardinality_exact_fraction",
        "cardinality_mae",
        "selection_score",
    )
    result: dict[str, Any] = {
        "seeds": [int(run["seed"]) for run in runs],
        "parameter_count": int(runs[0]["parameter_count"]),
        "runs": runs,
    }
    for split in ("fit", "early_stop", "validation"):
        result[split] = {}
        for metric in metrics:
            values = torch.tensor(
                [float(run[split][metric]) for run in runs],
                dtype=torch.float64,
            )
            result[split][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
                "values": values.tolist(),
            }
    return result


def make_decision(
    baseline: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    descriptor = summaries["descriptor"]

    def value(arm: str, metric: str) -> float:
        return float(summaries[arm]["validation"][metric]["mean"])

    descriptor_valid = (
        value("descriptor", "mean_target_support_mass")
        >= float(baseline["mean_target_support_mass"]) - 0.05
        and value("descriptor", "route_in_support_fraction")
        >= float(baseline["route_in_support_fraction"]) - 0.05
    )
    del descriptor
    evidence_arms = ("row_tokens", "p2_evidence", "combined")
    best_arm = max(
        evidence_arms,
        key=lambda arm: value(arm, "selection_score"),
    )
    support_mass_gain = value(
        best_arm,
        "mean_target_support_mass",
    ) - value("descriptor", "mean_target_support_mass")
    support_hit_gain = value(
        best_arm,
        "route_in_support_fraction",
    ) - value("descriptor", "route_in_support_fraction")
    top1_gain = value(
        best_arm,
        "target_id_rank_top1_fraction",
    ) - value("descriptor", "target_id_rank_top1_fraction")
    train_val_gap = (
        float(
            summaries[best_arm]["fit"]["route_in_support_fraction"]["mean"]
        )
        - value(best_arm, "route_in_support_fraction")
    )
    material_gain = support_mass_gain >= 0.10 and support_hit_gain >= 0.10
    strong_observability = (
        value(best_arm, "mean_target_support_mass") >= 0.60
        and value(best_arm, "route_in_support_fraction") >= 0.65
        and train_val_gap <= 0.15
    )
    passed = bool(descriptor_valid and material_gain and strong_observability)
    if passed:
        diagnosis = "global_support_is_observable_in_richer_candidate_evidence"
        action = (
            f"Feed {best_arm} evidence into the global router before slot "
            "assignment and supervise aggregate target-support mass."
        )
    elif descriptor_valid and (
        support_mass_gain >= 0.05 or support_hit_gain >= 0.05
    ):
        diagnosis = "richer_evidence_contains_partial_global_support_signal"
        action = (
            "Run one bounded architecture gate that exposes the best evidence "
            "arm to global routing; do not authorize long training."
        )
    elif descriptor_valid:
        diagnosis = "global_support_signal_is_not_recovered_by_frozen_evidence"
        action = (
            "Do not add another frozen adapter. Final-slot supervision must "
            "reshape the proposal/P2 representation or slots must query live "
            "image evidence directly."
        )
    else:
        diagnosis = "probe_protocol_did_not_reproduce_the_descriptor_control"
        action = (
            "Treat the audit as inconclusive; increase the development split "
            "or debug optimization before comparing feature sources."
        )
    return {
        "passed": passed,
        "diagnosis": diagnosis,
        "action": action,
        "best_arm": best_arm,
        "descriptor_control_valid": descriptor_valid,
        "support_mass_gain_over_descriptor": support_mass_gain,
        "support_hit_gain_over_descriptor": support_hit_gain,
        "target_id_top1_gain_over_descriptor": top1_gain,
        "best_arm_train_val_support_hit_gap": train_val_gap,
        "thresholds": {
            "descriptor_baseline_tolerance": 0.05,
            "material_support_mass_gain": 0.10,
            "material_support_hit_gain": 0.10,
            "strong_support_mass": 0.60,
            "strong_support_hit": 0.65,
            "max_train_val_support_hit_gap": 0.15,
        },
    }


def main() -> None:
    args = parse_args()
    if int(args.train_images) <= int(args.early_stop_images):
        raise ValueError("train-images must exceed early-stop-images")
    if int(args.val_images) < 1:
        raise ValueError("val-images must be positive")
    device = torch.device(args.device)
    seed_everything(int(args.seeds[0]))
    cfg = _prepare_config(args)
    model = build_model(cfg).to(device)
    iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()
    cache_dir = Path(args.cache_dir).expanduser()
    train_cache = collect_feature_cache(
        args,
        cfg,
        model,
        split="train",
        images_requested=int(args.train_images),
        cache_path=cache_dir / "train_features.pt",
        device=device,
    )
    val_cache = collect_feature_cache(
        args,
        cfg,
        model,
        split="val",
        images_requested=int(args.val_images),
        cache_path=cache_dir / "val_features.pt",
        device=device,
    )
    if bool(args.cache_only):
        print(
            json.dumps(
                {
                    "cache_only": True,
                    "train_cache": str(cache_dir / "train_features.pt"),
                    "val_cache": str(cache_dir / "val_features.pt"),
                },
                indent=2,
            )
        )
        return

    total_train = int(train_cache["candidate_valid"].shape[0])
    split_generator = torch.Generator(device="cpu")
    split_generator.manual_seed(int(args.split_seed))
    split_order = torch.randperm(
        total_train,
        generator=split_generator,
    ).tolist()
    early_indices = split_order[: int(args.early_stop_images)]
    fit_indices = split_order[int(args.early_stop_images) :]
    val_indices = list(range(int(val_cache["candidate_valid"].shape[0])))
    baseline = evaluate_logits(
        val_cache,
        val_indices,
        val_cache["baseline_active_logits"].float(),
        val_cache["baseline_route_logits"].float(),
    )
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        print(f"training global-support arm={arm}", flush=True)
        moments = feature_moments(train_cache, arm, fit_indices)
        runs = [
            train_probe(
                args,
                train_cache,
                val_cache,
                arm=arm,
                seed=int(seed),
                fit_indices=fit_indices,
                early_indices=early_indices,
                moments=moments,
                device=device,
            )
            for seed in args.seeds
        ]
        summaries[arm] = _aggregate_runs(runs)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "validation": summaries[arm]["validation"],
                },
                indent=2,
            ),
            flush=True,
        )
    decision = make_decision(baseline, summaries)
    payload = {
        "experiment": "V9 global target-support observability probe",
        "iteration": int(iteration),
        "test_set_used": False,
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "cache_contract": {
            "train_images": int(args.train_images),
            "early_stop_images": int(args.early_stop_images),
            "val_images": int(args.val_images),
            "sample_strategy": "uniform",
            "augmentation_enabled": False,
            "row_samples": int(args.row_samples),
            "feature_dim": int(args.feature_dim),
            "projection": "fixed_rademacher",
            "split_seed": int(args.split_seed),
        },
        "probe_contract": {
            "common_architecture": "row_encoder+candidate_encoder+four_slot_decoder",
            "hidden_dim": int(args.hidden_dim),
            "row_layers": int(args.row_layers),
            "candidate_layers": int(args.candidate_layers),
            "slot_layers": int(args.slot_layers),
            "num_heads": int(args.num_heads),
            "ff_dim": int(args.ff_dim),
            "seeds": [int(seed) for seed in args.seeds],
        },
        "baseline": baseline,
        "arms": summaries,
        "decision": decision,
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision}, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
