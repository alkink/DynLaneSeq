from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    official_proposal_gt_iou_matrix,
    resolve_list_path,
    sha256_file,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader
from dynlaneseq_eg.tools.probe_official_set_selection import (
    SetAwareQualityProbe,
    official_unique_quality_targets,
    selection_features,
    training_index_schedule,
)
from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    _frozen_outputs,
    pairwise_quality_ranking_loss,
    quality_focal_loss,
)


CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Early structural probe: freeze a V5 detector and compare a "
            "32-proposal set scorer against four learned object slots over "
            "the exact same cached proposal memory."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--train-cache-images", type=int, default=4096)
    parser.add_argument("--val-cache-images", type=int, default=256)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Refuse detector inference when either frozen cache is missing.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--curve-samples", type=int, default=20)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--proposal-layers", type=int, default=2)
    parser.add_argument(
        "--matched-proposal-layers",
        type=int,
        default=5,
        help=(
            "Depth of the capacity-matched 32-query control. Five layers "
            "is approximately parameter matched to the default slot router."
        ),
    )
    parser.add_argument("--slot-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--representable-min", type=float, default=0.50)
    parser.add_argument("--cluster-min", type=float, default=0.30)
    parser.add_argument("--cluster-delta", type=float, default=0.05)
    parser.add_argument("--cluster-temperature", type=float, default=0.03)
    parser.add_argument("--permutation-temperature", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=0.10)
    parser.add_argument("--rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-target-margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--strong-gain-050-points", type=float, default=3.0)
    parser.add_argument("--strong-gain-075-points", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--save-probes", required=True)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _amp_context(device: torch.device, name: str):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(name)
    if dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.cache_batch_size
    )
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    cfg.setdefault("model", {}).setdefault("structured_query", {})[
        "intermediate_supervision"
    ] = False
    return cfg


def _feature_tensor(
    outputs: dict[str, torch.Tensor],
    *,
    input_w: int,
    curve_samples: int,
) -> torch.Tensor:
    base = selection_features(
        outputs,
        input_w=int(input_w),
        curve_samples=int(curve_samples),
    )
    ownership = outputs.get("ownership_state")
    if not isinstance(ownership, torch.Tensor):
        raise ValueError(
            "four-slot probe requires retained V5 ownership_state tensors"
        )
    exist = outputs["exist_logits"].detach().float()
    foreground_margin = (exist[..., :1] - exist[..., 1:2]).detach()
    return torch.cat(
        (base, ownership.detach().float(), foreground_margin), dim=-1
    )


def _stage(
    outputs: dict[str, torch.Tensor], batch_index: int
) -> dict[str, torch.Tensor]:
    return {
        name: outputs[name][batch_index].detach().float().cpu()
        for name in (
            "pred_x_rows",
            "range_norm",
            "exist_logits",
            "quality_logits",
        )
        if isinstance(outputs.get(name), torch.Tensor)
    }


def _signature(
    *,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    checkpoint_sha256: str,
    split: str,
    sample_count: int,
) -> dict[str, Any]:
    list_path = resolve_list_path(cfg, split)
    return {
        "cache_version": CACHE_VERSION,
        "resolved_config_sha256": _sha256_json(cfg),
        "config_path": str(Path(args.config)),
        "checkpoint_path": str(Path(args.checkpoint)),
        "checkpoint_sha256": str(checkpoint_sha256),
        "split": str(split),
        "list_path": str(list_path.resolve()),
        "list_sha256": sha256_file(list_path),
        "sample_strategy": "uniform",
        "sample_count": int(sample_count),
        "curve_samples": int(args.curve_samples),
        "line_width": float(args.line_width),
        "min_valid_rows": int(args.min_valid_rows),
        "row_visibility_thresh": float(args.row_visibility_thresh),
        "feature_contract": "geometry_selection_plus_v5_ownership_state_v1",
    }


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def _collect_cache(
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    args: argparse.Namespace,
    split: str,
    sample_count: int,
    signature: dict[str, Any],
    output_path: Path,
    device: torch.device,
    channels_last: bool,
) -> dict[str, Any]:
    loader_cfg = json.loads(json.dumps(cfg))
    loader_cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.cache_batch_size
    )
    loader_cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        loader_cfg["dataloader"]["persistent_workers"] = False
    base_loader = build_dataloader(loader_cfg, split=split, training=False)
    loader, dataset_indices = select_diagnostic_loader(
        base_loader,
        strategy="uniform",
        max_batches=math.ceil(int(sample_count) / int(args.cache_batch_size)),
        num_workers=int(args.num_workers),
    )
    if len(dataset_indices) != int(sample_count):
        raise ValueError(
            f"requested {sample_count} {split} images, got {len(dataset_indices)}"
        )
    features: list[torch.Tensor] = []
    unique_targets: list[torch.Tensor] = []
    candidate_valid_rows: list[torch.Tensor] = []
    official_rows: list[torch.Tensor] = []
    stage_rows: dict[str, list[torch.Tensor]] = {
        "pred_x_rows": [],
        "range_norm": [],
        "exist_logits": [],
        "quality_logits": [],
    }
    image_paths: list[str] = []
    input_w = int(cfg.get("model", {}).get("input_w", 800))
    for images, _targets, metas in tqdm(
        loader, desc=f"V5 slot cache ({split})", ncols=88
    ):
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        with _amp_context(device, args.amp_dtype):
            outputs = _frozen_outputs(model, images)
        batch_features = _feature_tensor(
            outputs,
            input_w=input_w,
            curve_samples=int(args.curve_samples),
        )
        for batch_index, meta in enumerate(metas):
            stage = _stage(outputs, batch_index)
            record = {"stages": {"main": stage}, "meta": meta}
            official_iou, candidate_valid = official_proposal_gt_iou_matrix(
                record,
                "main",
                line_width=float(args.line_width),
                min_valid_rows=int(args.min_valid_rows),
                row_visibility_thresh=float(args.row_visibility_thresh),
            )
            features.append(batch_features[batch_index].detach().half().cpu())
            unique_targets.append(
                official_unique_quality_targets(
                    official_iou, candidate_valid
                ).float()
            )
            candidate_valid_rows.append(candidate_valid.bool())
            official_rows.append(official_iou.float())
            for name, tensor in stage.items():
                stage_rows[name].append(tensor)
            image_paths.append(str(meta.get("image_path", "")))
    if not features:
        raise ValueError(f"no cache records for split={split}")
    cache = {
        "metadata": {
            **signature,
            "dataset_indices": list(dataset_indices),
            "image_paths": image_paths,
            "num_images": len(features),
            "num_candidates": int(features[0].shape[0]),
            "feature_dim": int(features[0].shape[-1]),
        },
        "features": torch.stack(features),
        "unique_targets": torch.stack(unique_targets),
        "candidate_valid": torch.stack(candidate_valid_rows),
        "official_iou": official_rows,
        "stage": {
            name: torch.stack(rows) for name, rows in stage_rows.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


def _load_or_collect_cache(
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    args: argparse.Namespace,
    split: str,
    sample_count: int,
    signature: dict[str, Any],
    output_path: Path,
    device: torch.device,
    channels_last: bool,
) -> dict[str, Any]:
    if args.reuse_cache and output_path.exists():
        cache = _torch_load(output_path)
        metadata = cache.get("metadata", {})
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in signature.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"cache signature mismatch for {output_path}: {mismatches}"
            )
        return cache
    if args.cache_only:
        raise FileNotFoundError(
            f"required frozen cache is missing: {output_path}"
        )
    return _collect_cache(
        model,
        cfg,
        args=args,
        split=split,
        sample_count=sample_count,
        signature=signature,
        output_path=output_path,
        device=device,
        channels_last=channels_last,
    )


class FourSlotRouter(nn.Module):
    """Four object slots route over a frozen 32-proposal memory."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        num_slots: int,
        proposal_layers: int,
        slot_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.input_projection = nn.Linear(int(input_dim), self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.proposal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(proposal_layers),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(slot_layers)
        )
        self.slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.slot_norm = nn.LayerNorm(self.hidden_dim)
        self.candidate_norm = nn.LayerNorm(self.hidden_dim)
        self.slot_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.candidate_key = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.dustbin = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.slot_tokens.weight, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.input_projection(self.input_norm(features))
        memory = self.proposal_encoder(
            memory, src_key_padding_mask=~candidate_valid.bool()
        )
        slots = self.slot_tokens.weight.unsqueeze(0).expand(
            features.shape[0], -1, -1
        )
        slots = self.slot_decoder(
            slots,
            memory,
            memory_key_padding_mask=~candidate_valid.bool(),
        )
        slots = self.slot_norm(slots)
        candidates = self.candidate_norm(memory)
        logits = torch.einsum(
            "bsd,bnd->bsn",
            self.slot_query(slots),
            self.candidate_key(candidates),
        ) / math.sqrt(float(self.hidden_dim))
        logits = logits.masked_fill(~candidate_valid[:, None, :], -1.0e4)
        dustbin = self.dustbin(slots)
        return torch.cat((logits, dustbin), dim=-1)


def _jointly_representable_targets(
    official_iou: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    num_slots: int,
    representable_min: float,
    cluster_min: float,
    cluster_delta: float,
    temperature: float,
) -> torch.Tensor:
    gt_count, candidate_count = official_iou.shape
    output = official_iou.new_zeros((0, candidate_count + 1))
    valid_ids = [
        index
        for index in range(int(candidate_count))
        if bool(candidate_valid[index])
    ]
    if int(gt_count) == 0 or not valid_ids:
        return output
    gt_ids, local_candidate_ids = linear_sum_assignment(
        1.0 - official_iou[:, valid_ids].detach().cpu().numpy()
    )
    jointly_representable = [
        int(gt_index)
        for gt_index, local_index in zip(gt_ids, local_candidate_ids)
        if float(official_iou[int(gt_index), valid_ids[int(local_index)]])
        >= float(representable_min)
    ]
    if len(jointly_representable) > int(num_slots):
        jointly_representable.sort(
            key=lambda gt: float(official_iou[gt, valid_ids].max()),
            reverse=True,
        )
        jointly_representable = jointly_representable[: int(num_slots)]
    rows = []
    for gt_index in jointly_representable:
        quality = official_iou[gt_index]
        best = float(quality[candidate_valid].max())
        cutoff = max(float(cluster_min), best - float(cluster_delta))
        support = candidate_valid & (quality >= cutoff)
        if not bool(support.any()):
            continue
        target = quality.new_zeros(candidate_count + 1)
        target[:candidate_count][support] = torch.softmax(
            quality[support] / max(float(temperature), 1.0e-4), dim=0
        )
        rows.append(target)
    if not rows:
        return output
    return torch.stack(rows)


def _permutation_marginal_slot_loss(
    route_logits: torch.Tensor,
    target_rows: list[torch.Tensor],
    *,
    permutation_temperature: float,
) -> torch.Tensor:
    batch, slots, classes = route_logits.shape
    dustbin_index = classes - 1
    log_probability = F.log_softmax(route_logits.float(), dim=-1)
    losses = []
    temperature = max(float(permutation_temperature), 1.0e-4)
    for batch_index in range(batch):
        targets = target_rows[batch_index].to(
            device=route_logits.device, dtype=torch.float32
        )
        gt_count = int(targets.shape[0])
        dustbin_cost = -log_probability[batch_index, :, dustbin_index]
        if gt_count == 0:
            losses.append(dustbin_cost.sum())
            continue
        candidate_cost = -torch.einsum(
            "sc,gc->sg", log_probability[batch_index], targets
        )
        costs = []
        for assigned_slots in permutations(range(slots), gt_count):
            assigned = set(int(value) for value in assigned_slots)
            cost = candidate_cost.new_zeros(())
            for gt_index, slot_index in enumerate(assigned_slots):
                cost = cost + candidate_cost[int(slot_index), gt_index]
            for slot_index in range(slots):
                if slot_index not in assigned:
                    cost = cost + dustbin_cost[slot_index]
            costs.append(cost)
        stacked = torch.stack(costs)
        marginal = -temperature * torch.logsumexp(
            -stacked / temperature, dim=0
        ) + temperature * math.log(float(len(costs)))
        losses.append(marginal)
    return torch.stack(losses).mean() / float(max(slots, 1))


def _slot_collision_loss(route_logits: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(route_logits.float(), dim=-1)[..., :-1]
    gram = torch.einsum("bsn,btn->bst", probability, probability)
    slots = int(probability.shape[1])
    mask = torch.triu(
        torch.ones((slots, slots), device=gram.device, dtype=torch.bool),
        diagonal=1,
    )
    return gram[:, mask].mean() if bool(mask.any()) else gram.sum() * 0.0


def _slot_target_batch(
    cache: dict[str, Any],
    indices: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> list[torch.Tensor]:
    return [
        _jointly_representable_targets(
            cache["official_iou"][int(index)],
            cache["candidate_valid"][int(index)].bool(),
            num_slots=int(args.num_slots),
            representable_min=float(args.representable_min),
            cluster_min=float(args.cluster_min),
            cluster_delta=float(args.cluster_delta),
            temperature=float(args.cluster_temperature),
        )
        for index in indices.tolist()
    ]


def _aggregate_proposal_targets(
    target_rows: list[torch.Tensor],
    *,
    candidate_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Remove only the GT axis while preserving the slot arm's soft support.

    Both diagnostic arms therefore see the same representable GT clusters.
    The 32-query arm must collapse those clusters into one scalar per proposal;
    the four-slot arm is allowed to retain the object/GT axis.
    """

    batch_targets = []
    for rows in target_rows:
        if int(rows.shape[0]) == 0:
            batch_targets.append(
                torch.zeros(
                    candidate_count, device=device, dtype=torch.float32
                )
            )
        else:
            batch_targets.append(
                rows[:, :candidate_count]
                .to(device=device, dtype=torch.float32)
                .amax(dim=0)
            )
    return torch.stack(batch_targets)


def _decode_slots(
    logits: torch.Tensor, candidate_valid: torch.Tensor
) -> tuple[list[int], torch.Tensor]:
    slots, classes = logits.shape
    candidates = classes - 1
    matrix = logits.detach().float().cpu().new_full(
        (slots, candidates + slots), -1.0e9
    )
    matrix[:, :candidates] = logits[:, :candidates].detach().float().cpu()
    invalid_ids = torch.nonzero(
        ~candidate_valid.detach().bool().cpu(), as_tuple=False
    ).flatten()
    if int(invalid_ids.numel()):
        matrix[:, invalid_ids] = -1.0e9
    dustbin = logits[:, candidates].detach().float().cpu()
    for slot_index in range(slots):
        matrix[slot_index, candidates + slot_index] = dustbin[slot_index]
    row_ids, column_ids = linear_sum_assignment(-matrix.numpy())
    selected = [
        int(column)
        for _row, column in zip(row_ids, column_ids)
        if int(column) < candidates
    ]
    probabilities = torch.softmax(logits.detach().float().cpu(), dim=-1)
    candidate_scores = probabilities[:, :candidates].amax(dim=0)
    return selected, candidate_scores


def _topk(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    top_k: int,
    positive: torch.Tensor | None = None,
) -> list[int]:
    ids = [
        index
        for index in range(int(scores.shape[0]))
        if bool(candidate_valid[index])
        and (positive is None or bool(positive[index]))
    ]
    ids.sort(key=lambda index: float(scores[index]), reverse=True)
    return ids[: int(top_k)]


def _new_counts() -> dict[str, Any]:
    return {"gt": 0, "pred": 0, "tp_050": 0, "tp_075": 0}


def _update_counts(
    counts: dict[str, Any],
    official_iou: torch.Tensor,
    selected: list[int],
) -> None:
    counts["gt"] += int(official_iou.shape[0])
    counts["pred"] += len(selected)
    counts["tp_050"] += int(
        evaluator_hungarian_assignment(
            official_iou, selected, threshold=0.50
        ).hit_count
    )
    counts["tp_075"] += int(
        evaluator_hungarian_assignment(
            official_iou, selected, threshold=0.75
        ).hit_count
    )


def _finish_counts(counts: dict[str, Any]) -> dict[str, Any]:
    output = dict(counts)
    for suffix in ("050", "075"):
        tp = int(counts[f"tp_{suffix}"])
        precision = tp / float(max(int(counts["pred"]), 1))
        recall = tp / float(max(int(counts["gt"]), 1))
        output[f"precision_{suffix}"] = precision
        output[f"recall_{suffix}"] = recall
        output[f"f1_{suffix}"] = (
            2.0 * precision * recall / max(precision + recall, 1.0e-12)
        )
    return output


@torch.no_grad()
def _evaluate(
    cache: dict[str, Any],
    proposal_probe: nn.Module,
    matched_proposal_probe: nn.Module,
    slot_probe: FourSlotRouter,
    *,
    device: torch.device,
    batch_size: int,
    num_slots: int,
) -> dict[str, Any]:
    proposal_probe.eval()
    matched_proposal_probe.eval()
    slot_probe.eval()
    proposal_logits = []
    matched_proposal_logits = []
    slot_logits = []
    for start in range(0, int(cache["features"].shape[0]), int(batch_size)):
        stop = start + int(batch_size)
        features = cache["features"][start:stop].to(
            device=device, dtype=torch.float32
        )
        valid = cache["candidate_valid"][start:stop].to(device)
        proposal_logits.append(proposal_probe(features).float().cpu())
        matched_proposal_logits.append(
            matched_proposal_probe(features).float().cpu()
        )
        slot_logits.append(slot_probe(features, valid).float().cpu())
    learned_scores = torch.sigmoid(torch.cat(proposal_logits))
    matched_learned_scores = torch.sigmoid(
        torch.cat(matched_proposal_logits)
    )
    routes = torch.cat(slot_logits)
    exist_logits = cache["stage"]["exist_logits"].float()
    current_scores = torch.softmax(exist_logits, dim=-1)[..., 0]
    strategies = {
        "current_v5_direct": _new_counts(),
        "current_v5_top4": _new_counts(),
        "learned_32_direct": _new_counts(),
        "learned_32_top4": _new_counts(),
        "learned_32_parameter_matched_direct": _new_counts(),
        "learned_32_parameter_matched_top4": _new_counts(),
        "learned_4_slots": _new_counts(),
    }
    oracle = {"gt": 0, "tp_050": 0, "tp_075": 0}
    duplicate_slot_assignments = 0
    for image_index, official_iou in enumerate(cache["official_iou"]):
        valid = cache["candidate_valid"][image_index].bool()
        current = current_scores[image_index]
        learned = learned_scores[image_index]
        matched_learned = matched_learned_scores[image_index]
        current_positive = exist_logits[image_index, :, 0] > exist_logits[
            image_index, :, 1
        ]
        learned_positive = learned > 0.5
        matched_learned_positive = matched_learned > 0.5
        selections = {
            "current_v5_direct": _topk(
                current,
                valid,
                top_k=num_slots,
                positive=current_positive,
            ),
            "current_v5_top4": _topk(
                current, valid, top_k=num_slots
            ),
            "learned_32_direct": _topk(
                learned,
                valid,
                top_k=num_slots,
                positive=learned_positive,
            ),
            "learned_32_top4": _topk(
                learned, valid, top_k=num_slots
            ),
            "learned_32_parameter_matched_direct": _topk(
                matched_learned,
                valid,
                top_k=num_slots,
                positive=matched_learned_positive,
            ),
            "learned_32_parameter_matched_top4": _topk(
                matched_learned, valid, top_k=num_slots
            ),
        }
        slot_selected, _slot_scores = _decode_slots(
            routes[image_index], valid
        )
        duplicate_slot_assignments += len(slot_selected) - len(
            set(slot_selected)
        )
        selections["learned_4_slots"] = slot_selected
        for name, selected in selections.items():
            _update_counts(strategies[name], official_iou, selected)
        oracle["gt"] += int(official_iou.shape[0])
        for threshold, suffix in ((0.50, "050"), (0.75, "075")):
            oracle[f"tp_{suffix}"] += int(
                cardinality_oracle_assignment(
                    official_iou,
                    threshold=threshold,
                    top_k=num_slots,
                    candidate_valid=valid,
                ).hit_count
            )
    finished = {
        name: _finish_counts(counts) for name, counts in strategies.items()
    }
    for suffix in ("050", "075"):
        oracle[f"recall_{suffix}"] = oracle[f"tp_{suffix}"] / float(
            max(int(oracle["gt"]), 1)
        )
    current_tp = int(finished["current_v5_direct"]["tp_050"])
    oracle_gap = int(oracle["tp_050"]) - current_tp
    for row in finished.values():
        row["oracle_gap_closure_050"] = (
            (int(row["tp_050"]) - current_tp) / float(max(oracle_gap, 1))
        )
    return {
        "strategies": finished,
        "oracle_top4": oracle,
        "slot_duplicate_candidate_assignments": int(
            duplicate_slot_assignments
        ),
    }


def main() -> None:
    args = parse_args()
    positive = (
        "cache_batch_size",
        "train_cache_images",
        "val_cache_images",
        "train_steps",
        "probe_batch_size",
        "hidden_dim",
        "proposal_layers",
        "matched_proposal_layers",
        "slot_layers",
        "num_heads",
        "ff_dim",
        "num_slots",
    )
    for name in positive:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(args.num_slots) > 4:
        raise ValueError("CULane early slot probe is limited to four slots")
    _set_seed(args.seed)
    cfg = _prepare_config(args)
    device = torch.device(args.device)
    model = build_model(cfg)
    iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if model.structured_query_head is None:
        raise ValueError("slot probe requires a structured query head")
    model.structured_query_head.intermediate_supervision = False
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    checkpoint_sha = sha256_file(args.checkpoint)
    train_signature = _signature(
        cfg=cfg,
        args=args,
        checkpoint_sha256=checkpoint_sha,
        split="train",
        sample_count=int(args.train_cache_images),
    )
    val_signature = _signature(
        cfg=cfg,
        args=args,
        checkpoint_sha256=checkpoint_sha,
        split="val",
        sample_count=int(args.val_cache_images),
    )
    train_cache = _load_or_collect_cache(
        model,
        cfg,
        args=args,
        split="train",
        sample_count=int(args.train_cache_images),
        signature=train_signature,
        output_path=Path(args.train_cache),
        device=device,
        channels_last=channels_last,
    )
    val_cache = _load_or_collect_cache(
        model,
        cfg,
        args=args,
        split="val",
        sample_count=int(args.val_cache_images),
        signature=val_signature,
        output_path=Path(args.val_cache),
        device=device,
        channels_last=channels_last,
    )
    overlap = set(train_cache["metadata"]["image_paths"]) & set(
        val_cache["metadata"]["image_paths"]
    )
    if overlap:
        raise ValueError(f"train/validation cache overlap: {sorted(overlap)[0]}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _set_seed(args.seed)
    feature_dim = int(train_cache["features"].shape[-1])
    proposal_probe = SetAwareQualityProbe(
        feature_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.proposal_layers),
        num_heads=int(args.num_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    matched_proposal_probe = SetAwareQualityProbe(
        feature_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.matched_proposal_layers),
        num_heads=int(args.num_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    slot_probe = FourSlotRouter(
        feature_dim,
        hidden_dim=int(args.hidden_dim),
        num_slots=int(args.num_slots),
        proposal_layers=int(args.proposal_layers),
        slot_layers=int(args.slot_layers),
        num_heads=int(args.num_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(proposal_probe.parameters())
        + list(matched_proposal_probe.parameters())
        + list(slot_probe.parameters()),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    schedule = training_index_schedule(
        num_examples=int(train_cache["features"].shape[0]),
        batch_size=int(args.probe_batch_size),
        steps=int(args.train_steps),
        seed=int(args.seed),
    )
    totals = {
        "proposal": 0.0,
        "proposal_rank": 0.0,
        "matched_proposal": 0.0,
        "matched_proposal_rank": 0.0,
        "slot": 0.0,
        "collision": 0.0,
    }
    running = dict(totals)
    proposal_probe.train()
    matched_proposal_probe.train()
    slot_probe.train()
    for step_index in tqdm(
        range(int(args.train_steps)), desc="V5 32-query vs 4-slot probe", ncols=90
    ):
        indices = schedule[step_index]
        features = train_cache["features"][indices].to(
            device=device, dtype=torch.float32
        )
        valid = train_cache["candidate_valid"][indices].to(device)
        proposal_logits = proposal_probe(features)
        matched_proposal_logits = matched_proposal_probe(features)
        route_logits = slot_probe(features, valid)
        target_rows = _slot_target_batch(train_cache, indices, args=args)
        proposal_targets = _aggregate_proposal_targets(
            target_rows,
            candidate_count=int(proposal_logits.shape[1]),
            device=device,
        )
        proposal_loss = quality_focal_loss(
            proposal_logits, proposal_targets, beta=2.0
        )
        rank_loss = pairwise_quality_ranking_loss(
            proposal_logits,
            proposal_targets,
            target_margin=float(args.rank_target_margin),
        )
        matched_proposal_loss = quality_focal_loss(
            matched_proposal_logits, proposal_targets, beta=2.0
        )
        matched_rank_loss = pairwise_quality_ranking_loss(
            matched_proposal_logits,
            proposal_targets,
            target_margin=float(args.rank_target_margin),
        )
        slot_loss = _permutation_marginal_slot_loss(
            route_logits,
            target_rows,
            permutation_temperature=float(args.permutation_temperature),
        )
        collision = _slot_collision_loss(route_logits)
        total = (
            proposal_loss
            + float(args.rank_loss_weight) * rank_loss
            + matched_proposal_loss
            + float(args.rank_loss_weight) * matched_rank_loss
            + slot_loss
            + float(args.collision_weight) * collision
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        # Clip each independent diagnostic arm separately.  A joint norm
        # would make one architecture's gradient magnitude change the other
        # arms' effective learning rate and weaken the paired comparison.
        torch.nn.utils.clip_grad_norm_(proposal_probe.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(
            matched_proposal_probe.parameters(), max_norm=5.0
        )
        torch.nn.utils.clip_grad_norm_(slot_probe.parameters(), max_norm=5.0)
        optimizer.step()
        values = {
            "proposal": float(proposal_loss.detach()),
            "proposal_rank": float(rank_loss.detach()),
            "matched_proposal": float(matched_proposal_loss.detach()),
            "matched_proposal_rank": float(matched_rank_loss.detach()),
            "slot": float(slot_loss.detach()),
            "collision": float(collision.detach()),
        }
        for name, value in values.items():
            totals[name] += value
            running[name] += value
        step = step_index + 1
        if int(args.log_interval) and step % int(args.log_interval) == 0:
            denominator = float(args.log_interval)
            tqdm.write(
                f"step {step:05d}/{int(args.train_steps):05d} "
                f"proposal={running['proposal']/denominator:.4f} "
                f"matched={running['matched_proposal']/denominator:.4f} "
                f"slot={running['slot']/denominator:.4f} "
                f"collision={running['collision']/denominator:.4f}"
            )
            running = {name: 0.0 for name in running}
    evaluation = _evaluate(
        val_cache,
        proposal_probe,
        matched_proposal_probe,
        slot_probe,
        device=device,
        batch_size=int(args.probe_batch_size),
        num_slots=int(args.num_slots),
    )
    rows = evaluation["strategies"]
    baseline_names = (
        "current_v5_direct",
        "current_v5_top4",
        "learned_32_direct",
        "learned_32_top4",
        "learned_32_parameter_matched_direct",
        "learned_32_parameter_matched_top4",
    )
    baseline_name = max(
        baseline_names,
        key=lambda name: float(rows[name]["f1_050"]),
    )
    baseline = rows[baseline_name]
    slot = rows["learned_4_slots"]
    gain_050 = 100.0 * (float(slot["f1_050"]) - float(baseline["f1_050"]))
    gain_075 = 100.0 * (float(slot["f1_075"]) - float(baseline["f1_075"]))
    strong = bool(
        gain_050 >= float(args.strong_gain_050_points)
        and gain_075 >= float(args.strong_gain_075_points)
        and int(evaluation["slot_duplicate_candidate_assignments"]) == 0
    )
    if strong:
        recommendation = "four_final_slots_have_strong_early_support"
    elif gain_050 >= 1.0:
        recommendation = "four_slots_promising_but_not_decisive"
    else:
        recommendation = "frozen_slot_probe_negative_or_inconclusive"
    result = {
        "diagnostic_only": True,
        "warning": (
            "This probe freezes the detector and uses train-split official "
            "IoU only for diagnostic heads. A negative result cannot rule "
            "out a from-scratch slot decoder whose geometry co-adapts."
        ),
        "question": (
            "Are four learned final object slots easier to learn than "
            "directly scoring 32 final proposal identities when proposal "
            "memory and geometry are held exactly fixed?"
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": int(iteration),
        "checkpoint_sha256": checkpoint_sha,
        "feature_contract": {
            "feature_dim": int(feature_dim),
            "same_frozen_proposal_memory_for_both_arms": True,
            "includes_v5_ownership_state": True,
            "geometry_trainable": False,
            "backbone_trainable": False,
        },
        "training": {
            "steps": int(args.train_steps),
            "batch_size": int(args.probe_batch_size),
            "seed": int(args.seed),
            "mean_losses": {
                name: value / float(args.train_steps)
                for name, value in totals.items()
            },
        },
        "models": {
            "proposal_set_scorer_parameters": sum(
                parameter.numel() for parameter in proposal_probe.parameters()
            ),
            "parameter_matched_proposal_set_scorer_parameters": sum(
                parameter.numel()
                for parameter in matched_proposal_probe.parameters()
            ),
            "four_slot_router_parameters": sum(
                parameter.numel() for parameter in slot_probe.parameters()
            ),
            "num_proposals": int(train_cache["features"].shape[1]),
            "num_final_slots": int(args.num_slots),
            "proposal_layers": int(args.proposal_layers),
            "matched_proposal_layers": int(args.matched_proposal_layers),
            "slot_layers": int(args.slot_layers),
            "slot_output": "global unique proposal assignment plus learned dustbin",
            "slot_geometry_refinement": False,
        },
        "target": {
            "representable_min": float(args.representable_min),
            "cluster_min": float(args.cluster_min),
            "cluster_delta": float(args.cluster_delta),
            "cluster_temperature": float(args.cluster_temperature),
            "permutation_marginalized": True,
            "same_soft_cluster_support_for_both_learned_arms": True,
            "proposal_arm_collapses_gt_axis_with_candidatewise_max": True,
        },
        "evaluation": evaluation,
        "decision": {
            "baseline_is_best_32_query_strategy": True,
            "best_32_query_baseline": baseline_name,
            "gain_f1_050_points": gain_050,
            "gain_f1_075_points": gain_075,
            "strong_gain_050_points": float(args.strong_gain_050_points),
            "strong_gain_075_points": float(args.strong_gain_075_points),
            "strong_early_slot_signal": strong,
            "recommendation": recommendation,
        },
        "caches": {
            "train": {"path": args.train_cache, **train_cache["metadata"]},
            "val": {"path": args.val_cache, **val_cache["metadata"]},
        },
    }
    save_path = Path(args.save_probes)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "proposal_set_scorer": proposal_probe.state_dict(),
            "parameter_matched_proposal_set_scorer": (
                matched_proposal_probe.state_dict()
            ),
            "four_slot_router": slot_probe.state_dict(),
            "feature_dim": int(feature_dim),
            "args": vars(args),
        },
        save_path,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {output}")
    print(f"probe_checkpoint: {save_path}")


if __name__ == "__main__":
    main()
