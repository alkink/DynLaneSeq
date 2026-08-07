from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.culane_metric import culane_metric, interp, load_culane_img_data
from dynlaneseq_eg.evaluation.proposal_recall import collect_prediction_stages
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm


CACHE_VERSION = 6
STAGE_TENSOR_FIELDS = (
    "pred_x_rows",
    "exist_logits",
    "quality_logits",
    "selection_logits",
    "selection_pointer_logits",
    "selection_pointer_indices",
    "selection_pointer_scores",
    "selection_slot_logits",
    "selection_slot_candidate_valid",
    "selection_slot_raw_indices",
    "selection_slot_raw_collision_count",
    "selection_slot_route_entropy",
    "range_norm",
    "row_visibility_logits",
)


def override_eval_list(cfg: dict[str, Any], split: str, list_path: str | Path | None) -> dict[str, Any]:
    if not list_path:
        return cfg
    out = deepcopy(cfg)
    dataset = dict(out.get("dataset", {}))
    lists = dict(dataset.get("lists", {}))
    lists[str(split)] = str(Path(list_path).expanduser().resolve())
    dataset["lists"] = lists
    out["dataset"] = dataset
    return out


def resolve_list_path(cfg: dict[str, Any], split: str) -> Path:
    dataset = cfg.get("dataset", {})
    root = Path(dataset.get("root", "dataset"))
    rel = dataset.get("lists", {}).get(split)
    if rel is None:
        raise KeyError(f"No dataset list configured for split={split}")
    path = Path(rel)
    return path if path.is_absolute() else root / path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(
    cache_dir: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    list_path: str | Path,
    split: str,
    max_batches: int,
    eval_batch_size: int | None,
    sample_strategy: str,
) -> Path:
    checkpoint = Path(checkpoint_path)
    stat = checkpoint.stat()
    payload = "|".join(
        [
            str(Path(config_path).resolve()),
            str(checkpoint.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(Path(list_path).resolve()),
            sha256_file(list_path),
            str(split),
            str(max_batches),
            str(eval_batch_size),
            str(sample_strategy),
            str(CACHE_VERSION),
        ]
    )
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{checkpoint.stem}_{key}.pt"


def _resolve_dataset_root(cfg: dict[str, Any], config_path: str | Path) -> tuple[Path, Path]:
    config_file = Path(config_path).expanduser().resolve()
    project_root = next(
        (
            parent
            for parent in config_file.parents
            if (parent / "pyproject.toml").exists() and (parent / "dynlaneseq_eg").is_dir()
        ),
        config_file.parent,
    )
    dataset_root = Path(cfg.get("dataset", {}).get("root", "dataset")).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    return project_root, dataset_root.resolve()


def _absolute_data_path(value: str | Path, project_root: Path, dataset_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        if "dataset" in path.parts:
            dataset_index = path.parts.index("dataset")
            return (dataset_root / Path(*path.parts[dataset_index + 1 :])).resolve()
        return path.resolve()
    project_candidate = project_root / path
    if project_candidate.exists() or path.parts[:1] == ("dataset",):
        return project_candidate.resolve()
    return (dataset_root / path).resolve()


def _plain_meta(meta: dict[str, Any], project_root: Path, dataset_root: Path) -> dict[str, Any]:
    keys = (
        "image_path",
        "anno_path",
        "orig_h",
        "orig_w",
        "input_h",
        "input_w",
        "scale_x",
        "scale_y",
        "crop_x",
        "crop_y",
    )
    out: dict[str, Any] = {}
    for key in keys:
        value = meta.get(key)
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        if value is not None:
            if key in {"image_path", "anno_path"}:
                value = str(_absolute_data_path(value, project_root, dataset_root))
            out[key] = value
    return out


def _upgrade_cache_paths(cache: dict[str, Any], project_root: Path, dataset_root: Path) -> bool:
    changed = False
    for record in cache.get("records", []):
        meta = record.get("meta", {})
        for key in ("image_path", "anno_path"):
            value = meta.get(key)
            if value:
                resolved = str(_absolute_data_path(value, project_root, dataset_root))
                if resolved != str(value):
                    meta[key] = resolved
                    changed = True
    if changed:
        cache.setdefault("metadata", {}).pop("official_iou_cache", None)
        for record in cache.get("records", []):
            for stage in record.get("stages", {}).values():
                stage.pop("official_iou", None)
                stage.pop("official_candidate_valid", None)
    cache.setdefault("metadata", {})["project_root"] = str(project_root)
    cache["metadata"]["dataset_root"] = str(dataset_root)
    return changed


def _cpu_stage(stage: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: stage[key].detach().to(device="cpu")
        for key in STAGE_TENSOR_FIELDS
        if isinstance(stage.get(key), torch.Tensor)
    }


@torch.no_grad()
def load_or_collect_cache(
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str = "val",
    list_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    device: str | torch.device = "cuda",
    cache_dir: str | Path = "outputs/diagnostic_cache",
    reuse_cache: bool = False,
    require_cache: bool = False,
    max_batches: int = 0,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    sample_strategy: str = "sequential",
    desc: str = "candidate cache",
) -> dict[str, Any]:
    cfg = override_eval_list(load_config(config_path), split, list_path)
    if dataset_root:
        cfg.setdefault("dataset", {})["root"] = str(Path(dataset_root).expanduser())
    dataloader_cfg = cfg.setdefault("dataloader", {})
    if eval_batch_size is not None:
        dataloader_cfg["eval_batch_size"] = int(eval_batch_size)
    if num_workers is not None:
        dataloader_cfg["num_workers"] = int(num_workers)
        if int(num_workers) == 0:
            dataloader_cfg["persistent_workers"] = False
    project_root, dataset_root = _resolve_dataset_root(cfg, config_path)
    resolved_list = resolve_list_path(cfg, split).resolve()
    if not resolved_list.exists():
        raise FileNotFoundError(f"Evaluation list does not exist: {resolved_list}")
    effective_eval_batch_size = int(dataloader_cfg.get("eval_batch_size", 1))
    cache_path = _cache_path(
        cache_dir,
        config_path,
        checkpoint_path,
        resolved_list,
        split,
        max_batches,
        effective_eval_batch_size,
        sample_strategy,
    )
    if (reuse_cache or require_cache) and cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only was added.
            cached = torch.load(cache_path, map_location="cpu")
        cached.setdefault("metadata", {})["cache_path"] = str(cache_path.resolve())
        if _upgrade_cache_paths(cached, project_root, dataset_root):
            torch.save(cached, cache_path)
        return cached
    if require_cache:
        raise FileNotFoundError(
            "Required diagnostic cache is missing; refusing to run model "
            f"inference: {cache_path}"
        )

    torch_device = torch.device(device)
    model = build_model(cfg).to(torch_device)
    load_checkpoint(checkpoint_path, model, strict=False)
    supports_inference_only = bool(getattr(model, "supports_inference_only", False))
    if supports_inference_only and hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    model.eval()
    loader = build_dataloader(cfg, split=split, training=False)
    from dynlaneseq_eg.tools.diagnostic_sampling import select_diagnostic_loader

    loader, sampled_indices = select_diagnostic_loader(
        loader,
        strategy=sample_strategy,
        max_batches=max_batches,
        num_workers=int(dataloader_cfg.get("num_workers", 0)),
    )
    pass_targets = bool(getattr(model, "oracle_coarse_enabled", False))
    records: list[dict[str, Any]] = []

    for batch_idx, (images, targets, metas) in enumerate(tqdm(loader, ncols=80, desc=desc)):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(torch_device, non_blocking=True)
        if pass_targets:
            outputs = model(images, targets=targets)
        elif supports_inference_only:
            outputs = model(images, inference_only=True)
        else:
            outputs = model(images)
        stages = collect_prediction_stages(outputs)
        cpu_stages = {name: _cpu_stage(stage) for name, stage in stages.items()}
        for bi, (target, meta) in enumerate(zip(targets, metas)):
            image_id = str(meta.get("image_path", f"batch{batch_idx:06d}_{bi}"))
            record_stages = {
                name: {key: value[bi].clone() for key, value in stage.items()}
                for name, stage in cpu_stages.items()
            }
            records.append(
                {
                    "image_id": image_id,
                    "meta": _plain_meta(meta, project_root, dataset_root),
                    "target": {
                        "x_rows": target["x_rows"].detach().cpu(),
                        "valid_mask": target["valid_mask"].detach().cpu().bool(),
                    },
                    "stages": record_stages,
                }
            )

    model_cfg = cfg.get("model", {})
    payload = {
        "cache_version": CACHE_VERSION,
        "metadata": {
            "config": str(Path(config_path)),
            "checkpoint": str(Path(checkpoint_path)),
            "split": str(split),
            "list_path": str(resolved_list),
            "list_sha256": sha256_file(resolved_list),
            "max_batches": int(max_batches),
            "eval_batch_size": effective_eval_batch_size,
            "num_workers": int(dataloader_cfg.get("num_workers", 0)),
            "sample_strategy": str(sample_strategy),
            "sampled_dataset_indices": sampled_indices,
            "input_w": int(model_cfg.get("input_w", 800)),
            "input_h": int(model_cfg.get("input_h", 288)),
            "postprocess": deepcopy(cfg.get("postprocess", {})),
            "num_records": len(records),
            "cache_path": str(cache_path.resolve()),
            "project_root": str(project_root),
            "dataset_root": str(dataset_root),
        },
        "records": records,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def stage_scores(
    stage: dict[str, torch.Tensor],
    quality_power: float = 0.0,
    score_mode: str = "exist",
) -> torch.Tensor:
    pred_x = stage["pred_x_rows"]
    mode = str(score_mode).strip().lower()
    if mode in {"selection", "set_selection", "set"}:
        selection = stage.get("selection_logits")
        if selection is None:
            raise ValueError("selection score mode requires selection_logits")
        score = torch.sigmoid(selection.float())
    elif mode in {"pointer", "sequential_pointer", "pointer_stop"}:
        indices = stage.get("selection_pointer_indices")
        pointer_scores = stage.get("selection_pointer_scores")
        if not isinstance(indices, torch.Tensor):
            raise ValueError("pointer score mode requires selection_pointer_indices")
        score = pred_x.new_zeros((pred_x.shape[0],), dtype=torch.float32)
        safe = indices.clamp(min=0, max=max(int(pred_x.shape[0]) - 1, 0))
        valid = indices >= 0
        values = (
            pointer_scores.float()
            if isinstance(pointer_scores, torch.Tensor)
            else torch.ones_like(indices, dtype=torch.float32)
        )
        score.scatter_reduce_(
            0,
            safe,
            torch.where(valid, values, torch.zeros_like(values)),
            reduce="amax",
            include_self=True,
        )
    else:
        logits = stage.get("exist_logits")
        if logits is None:
            score = pred_x.new_ones((pred_x.shape[0],), dtype=torch.float32)
        else:
            score = torch.softmax(logits.float(), dim=-1)[..., 0]
    quality = stage.get("quality_logits")
    if quality_power > 0 and quality is not None:
        score = score * torch.sigmoid(quality.float()).clamp_min(1e-6).pow(float(quality_power))
    return score


def candidate_row_masks(
    stage: dict[str, torch.Tensor],
    input_h: int,
    input_w: int,
    min_valid_rows: int = 5,
    row_visibility_thresh: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_x = stage["pred_x_rows"].float().clamp(0.0, float(input_w - 1))
    n, rows = pred_x.shape
    y_rows = fixed_y_rows(rows, input_h, device=pred_x.device, dtype=pred_x.dtype)
    ranges = stage.get("range_norm")
    if ranges is None:
        masks = torch.ones((n, rows), dtype=torch.bool, device=pred_x.device)
    else:
        ranges = sort_range_norm(ranges.float())
        y_min = ranges[:, 0:1] * float(input_h)
        y_max = ranges[:, 1:2] * float(input_h)
        masks = (y_rows.view(1, -1) >= y_min) & (y_rows.view(1, -1) <= y_max)
    visibility = stage.get("row_visibility_logits")
    if row_visibility_thresh > 0 and visibility is not None:
        masks &= torch.sigmoid(visibility.float()) >= float(row_visibility_thresh)
    masks &= torch.isfinite(pred_x)
    candidate_valid = masks.sum(dim=-1) >= int(min_valid_rows)
    return pred_x, masks, candidate_valid


def proposal_gt_iou_matrix(
    stage: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    input_h: int = 288,
    input_w: int = 800,
    line_width: float = 30.0,
    min_valid_rows: int = 5,
    row_visibility_thresh: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_x, pred_masks, candidate_valid = candidate_row_masks(
        stage,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
    )
    gt_x_all = target["x_rows"].float()
    gt_masks_all = target["valid_mask"].bool() & torch.isfinite(gt_x_all)
    valid_gt = gt_masks_all.sum(dim=-1) >= int(min_valid_rows)
    gt_x = gt_x_all[valid_gt]
    gt_masks = gt_masks_all[valid_gt]
    if gt_x.numel() == 0 or pred_x.numel() == 0:
        return pred_x.new_zeros((gt_x.shape[0], pred_x.shape[0])), valid_gt, candidate_valid

    gt = gt_x[:, None, :]
    pred = pred_x[None, :, :]
    gt_valid = gt_masks[:, None, :]
    pred_valid = pred_masks[None, :, :]
    both = gt_valid & pred_valid
    either = gt_valid | pred_valid
    radius = float(line_width) * 0.5
    overlap = (2.0 * radius - (gt - pred).abs()).clamp(min=0.0)
    overlap = torch.where(both, overlap, torch.zeros_like(overlap))
    union = torch.where(
        both,
        2.0 * float(line_width) - overlap,
        torch.where(either, torch.full_like(overlap, float(line_width)), torch.zeros_like(overlap)),
    )
    iou = overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1e-6)
    iou[:, ~candidate_valid] = 0.0
    return iou, valid_gt, candidate_valid


@dataclass(frozen=True)
class OracleSelection:
    proposal_ids: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    hit_count: int
    iou_sum: float


def evaluator_hungarian_assignment(
    iou_matrix: torch.Tensor,
    proposal_ids: Iterable[int],
    threshold: float,
) -> OracleSelection:
    """Replicate CULane evaluation: maximize total IoU, then threshold matched pairs."""
    ids = tuple(int(index) for index in proposal_ids)
    if iou_matrix.shape[0] == 0 or not ids:
        return OracleSelection((), (), 0, 0.0)
    selected = iou_matrix[:, list(ids)].detach().cpu().numpy()
    gt_indices, local_proposal_indices = linear_sum_assignment(1.0 - selected)
    pairs: list[tuple[int, int]] = []
    iou_sum = 0.0
    for gt_idx, local_idx in zip(gt_indices.tolist(), local_proposal_indices.tolist()):
        value = float(selected[gt_idx, local_idx])
        if value > float(threshold):
            proposal_idx = ids[local_idx]
            pairs.append((gt_idx, proposal_idx))
            iou_sum += value
    return OracleSelection(
        tuple(proposal_idx for _gt_idx, proposal_idx in pairs),
        tuple(pairs),
        len(pairs),
        iou_sum,
    )


def cardinality_oracle_assignment(
    iou_matrix: torch.Tensor,
    threshold: float,
    top_k: int,
    candidate_valid: torch.Tensor | None = None,
) -> OracleSelection:
    """Maximum-cardinality GT/proposal matching using Hungarian assignment.

    Qualified edges receive a reward larger than the maximum possible sum of
    all IoU tie-breakers.  This makes the optimization lexicographic: maximize
    the number of matches at ``threshold`` first, then maximize their IoU.
    Running ordinary maximum-IoU Hungarian matching and thresholding afterward
    is not equivalent and can discard an otherwise valid extra match.
    """
    gt_count, proposal_count = iou_matrix.shape
    if gt_count == 0 or proposal_count == 0 or top_k <= 0:
        return OracleSelection((), (), 0, 0.0)
    if candidate_valid is None:
        candidate_valid = torch.ones(proposal_count, dtype=torch.bool, device=iou_matrix.device)

    valid_ids = [idx for idx in range(proposal_count) if bool(candidate_valid[idx])]
    if not valid_ids:
        return OracleSelection((), (), 0, 0.0)
    selected = iou_matrix[:, valid_ids].detach().cpu().numpy()
    assignment_size = min(int(selected.shape[0]), int(selected.shape[1]))
    # CULane's evaluator counts a TP only when IoU is strictly greater than
    # the threshold, so the diagnostic oracle must use the same boundary.
    qualified = selected > float(threshold)
    cardinality_bonus = float(assignment_size + 1)
    reward = qualified.astype(np.float64) * cardinality_bonus + selected.astype(np.float64)
    gt_indices, local_proposal_indices = linear_sum_assignment(-reward)
    pairs = [
        (int(gt_idx), int(valid_ids[local_idx]))
        for gt_idx, local_idx in zip(gt_indices.tolist(), local_proposal_indices.tolist())
        if bool(qualified[gt_idx, local_idx])
    ]

    # Top-K caps deployable lane count.  Cardinality is already maximal; when
    # more than K valid pairs exist, retain the best-localized K pairs.
    if len(pairs) > top_k:
        pairs.sort(key=lambda pair: float(iou_matrix[pair[0], pair[1]]), reverse=True)
        pairs = pairs[:top_k]

    iou_sum = sum(float(iou_matrix[gt, prop]) for gt, prop in pairs)
    proposal_ids = tuple(prop for _gt, prop in pairs)
    return OracleSelection(proposal_ids, tuple(pairs), len(pairs), iou_sum)


def _lane_distance(lane_a: list[tuple[float, float]], lane_b: list[tuple[float, float]]) -> tuple[float, int]:
    xa_by_y = {round(y, 4): x for x, y in lane_a}
    diffs = [abs(xa_by_y[round(y, 4)] - x) for x, y in lane_b if round(y, 4) in xa_by_y]
    if not diffs:
        return float("inf"), 0
    return float(sum(diffs) / len(diffs)), len(diffs)


def stage_lane_points(
    stage: dict[str, torch.Tensor],
    input_h: int,
    input_w: int,
    min_valid_rows: int,
    row_visibility_thresh: float,
) -> tuple[list[list[tuple[float, float]]], torch.Tensor]:
    pred_x, masks, candidate_valid = candidate_row_masks(
        stage,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
    )
    y_rows = fixed_y_rows(pred_x.shape[-1], input_h, device=pred_x.device, dtype=pred_x.dtype)
    lanes: list[list[tuple[float, float]]] = []
    for proposal_idx in range(pred_x.shape[0]):
        mask = masks[proposal_idx]
        lanes.append([(float(x), float(y)) for x, y in zip(pred_x[proposal_idx, mask], y_rows[mask])])
    return lanes, candidate_valid


def trace_postprocess(
    stage: dict[str, torch.Tensor],
    input_h: int = 288,
    input_w: int = 800,
    score_thresh: float = 0.5,
    quality_power: float = 0.0,
    min_valid_rows: int = 5,
    nms_distance_thresh_px: float = 20.0,
    nms_min_overlap_points: int = 5,
    top_k: int = 4,
    row_visibility_thresh: float = 0.0,
    allowed_ids: Iterable[int] | None = None,
    score_override: dict[int, float] | None = None,
    score_mode: str = "exist",
) -> dict[str, Any]:
    pred_x, masks, candidate_valid = candidate_row_masks(
        stage,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
    )
    scores = stage_scores(
        stage,
        quality_power=quality_power,
        score_mode=score_mode,
    )
    if score_override:
        scores = scores.clone()
        for proposal_idx, value in score_override.items():
            scores[int(proposal_idx)] = float(value)
    allowed = set(int(i) for i in allowed_ids) if allowed_ids is not None else None
    status = ["not_allowed" if allowed is not None else "pending" for _ in range(pred_x.shape[0])]
    suppressed_by: dict[int, int] = {}
    eligible: list[int] = []
    for proposal_idx in range(pred_x.shape[0]):
        if allowed is not None and proposal_idx not in allowed:
            continue
        if not bool(candidate_valid[proposal_idx]):
            status[proposal_idx] = "invalid_range"
        elif float(scores[proposal_idx]) < float(score_thresh):
            status[proposal_idx] = "below_threshold"
        else:
            status[proposal_idx] = "threshold_pass"
            eligible.append(proposal_idx)
    eligible.sort(key=lambda idx: float(scores[idx]), reverse=True)

    nms_kept: list[int] = []
    if not eligible:
        pass
    elif nms_distance_thresh_px <= 0:
        nms_kept = eligible.copy()
    else:
        # Precompute distances purely in tensors to bypass Python loop overhead completely
        E = len(eligible)
        el_pred = pred_x[eligible]
        el_masks = masks[eligible]
        
        # (E, 1, R) & (1, E, R) -> (E, E, R)
        omasks = el_masks.unsqueeze(1) & el_masks.unsqueeze(0)
        overlaps = omasks.sum(dim=-1)
        diffs = (el_pred.unsqueeze(1) - el_pred.unsqueeze(0)).abs()
        diffs = torch.where(omasks, diffs, diffs.new_zeros((1,)))
        distances = diffs.sum(dim=-1) / overlaps.clamp_min(1)
        
        is_close = (overlaps >= int(nms_min_overlap_points)) & (distances < float(nms_distance_thresh_px))
        is_close_np = is_close.cpu().numpy()
        
        nms_kept_e_idx: list[int] = []
        for e_idx, proposal_idx in enumerate(eligible):
            keeper = None
            for kept_e, kept_idx in zip(nms_kept_e_idx, nms_kept):
                if is_close_np[e_idx, kept_e]:
                    keeper = kept_idx
                    break
            
            if keeper is None:
                nms_kept.append(proposal_idx)
                nms_kept_e_idx.append(e_idx)
            else:
                status[proposal_idx] = "nms_removed"
                suppressed_by[proposal_idx] = keeper

    selected = nms_kept[: int(top_k)] if top_k > 0 else nms_kept
    selected_set = set(selected)
    for proposal_idx in nms_kept:
        status[proposal_idx] = "selected" if proposal_idx in selected_set else "topk_removed"
    return {
        "selected_ids": selected,
        "nms_kept_ids": nms_kept,
        "eligible_ids": eligible,
        "status": status,
        "suppressed_by": suppressed_by,
        "scores": scores.detach().cpu(),
        "candidate_valid": candidate_valid.detach().cpu(),
    }


def recall_from_ids(iou_matrix: torch.Tensor, proposal_ids: Iterable[int], threshold: float) -> tuple[int, int, torch.Tensor]:
    ids = list(int(i) for i in proposal_ids)
    gt_count = int(iou_matrix.shape[0])
    if gt_count == 0:
        return 0, 0, iou_matrix.new_zeros((0,))
    if not ids:
        best = iou_matrix.new_zeros((gt_count,))
    else:
        best = iou_matrix[:, ids].max(dim=1).values
    # Recall must be duplicate-safe: one selected proposal cannot recover two
    # GT lanes.  Keep per-GT best IoU for localization summaries, but use the
    # same one-to-one Hungarian counting rule as CULane for hits.
    assignment = evaluator_hungarian_assignment(iou_matrix, ids, threshold=float(threshold))
    return int(assignment.hit_count), gt_count, best


def unique_candidate_labels(
    iou_matrix: torch.Tensor,
    threshold: float,
    candidate_valid: torch.Tensor,
) -> tuple[list[str], OracleSelection]:
    # Use Hungarian matching (O(GT^3)) instead of the exponential-DP
    # cardinality_oracle_assignment (O(2^GT * N)), which hangs on full val sets
    # with >=10 GT lanes per image.
    valid_ids = [idx for idx in range(int(iou_matrix.shape[1])) if bool(candidate_valid[idx])]
    assignment = evaluator_hungarian_assignment(iou_matrix, valid_ids, threshold=threshold)
    primary = set(assignment.proposal_ids)
    best_iou = iou_matrix.max(dim=0).values if iou_matrix.shape[0] else iou_matrix.new_zeros((iou_matrix.shape[1],))
    labels: list[str] = []
    for proposal_idx in range(iou_matrix.shape[1]):
        if not bool(candidate_valid[proposal_idx]):
            labels.append("invalid_range")
        elif proposal_idx in primary:
            labels.append("unique_tp")
        elif float(best_iou[proposal_idx]) >= float(threshold):
            labels.append("duplicate")
        else:
            labels.append("background")
    return labels, assignment


def _raster_lane_mask(
    lane: list[tuple[float, float]],
    image_h: int,
    image_w: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((int(image_h), int(image_w)), dtype=np.uint8)
    points = interp(lane, n=5)
    if len(points) >= 2:
        points = np.rint(points).astype(np.int32)
        for p1, p2 in zip(points[:-1], points[1:]):
            cv2.line(mask, tuple(p1), tuple(p2), color=1, thickness=int(width))
    return mask


def official_proposal_gt_iou_matrix(
    record: dict[str, Any],
    stage_name: str,
    line_width: float = 30.0,
    min_valid_rows: int = 5,
    row_visibility_thresh: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    stage = record["stages"][stage_name]
    lanes, candidate_valid = stage_lane_points(
        stage,
        input_h=int(record["meta"].get("input_h", 288)),
        input_w=int(record["meta"].get("input_w", 800)),
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
    )
    pred_lanes = lanes_to_original(lanes, record["meta"])
    anno_path = record["meta"].get("anno_path")
    gt_lanes = load_culane_img_data(anno_path) if anno_path else []
    image_h = int(record["meta"].get("orig_h", 590))
    image_w = int(record["meta"].get("orig_w", 1640))
    width = int(round(line_width))
    matrix = torch.zeros((len(gt_lanes), len(pred_lanes)), dtype=torch.float32)
    if not gt_lanes or not pred_lanes:
        return matrix, candidate_valid

    gt_masks = [_raster_lane_mask(lane, image_h, image_w, width) for lane in gt_lanes]
    gt_counts = [int(mask.sum()) for mask in gt_masks]
    for proposal_idx, lane in enumerate(pred_lanes):
        if not bool(candidate_valid[proposal_idx]):
            continue
        pred_mask = _raster_lane_mask(lane, image_h, image_w, width)
        pred_count = int(pred_mask.sum())
        if pred_count == 0:
            continue
        for gt_idx, gt_mask in enumerate(gt_masks):
            intersection = int(cv2.countNonZero(cv2.bitwise_and(pred_mask, gt_mask)))
            union = pred_count + gt_counts[gt_idx] - intersection
            matrix[gt_idx, proposal_idx] = 0.0 if union <= 0 else float(intersection) / float(union)
    return matrix, candidate_valid


def ensure_official_iou_cache(
    cache: dict[str, Any],
    line_width: float = 30.0,
    min_valid_rows: int = 5,
    row_visibility_thresh: float = 0.0,
    workers: int = 0,
) -> dict[str, Any]:
    signature = {
        "line_width": float(line_width),
        "min_valid_rows": int(min_valid_rows),
        "row_visibility_thresh": float(row_visibility_thresh),
    }
    if cache.get("metadata", {}).get("official_iou_cache") == signature and all(
        "official_iou" in stage
        for record in cache.get("records", [])
        for stage in record.get("stages", {}).values()
    ):
        return cache
    records = cache.get("records", [])

    def compute(record: dict[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            stage_name: official_proposal_gt_iou_matrix(
                record,
                stage_name,
                line_width=line_width,
                min_valid_rows=min_valid_rows,
                row_visibility_thresh=row_visibility_thresh,
            )
            for stage_name in record.get("stages", {})
        }

    worker_count = max(0, int(workers))
    if worker_count > 1:
        # Avoid multiplying the Python worker count by OpenCV's own thread
        # pool (for example 12 x 12 runnable threads on a 12-core host).
        previous_cv_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                computed = executor.map(compute, records)
                iterator = tqdm(
                    zip(records, computed),
                    total=len(records),
                    ncols=80,
                    desc="official IoU cache",
                )
                for record, stage_results in iterator:
                    for stage_name, (matrix, candidate_valid) in stage_results.items():
                        stage = record["stages"][stage_name]
                        stage["official_iou"] = matrix
                        stage["official_candidate_valid"] = candidate_valid.cpu()
        finally:
            cv2.setNumThreads(previous_cv_threads)
    else:
        for record in tqdm(records, ncols=80, desc="official IoU cache"):
            for stage_name, (matrix, candidate_valid) in compute(record).items():
                stage = record["stages"][stage_name]
                stage["official_iou"] = matrix
                stage["official_candidate_valid"] = candidate_valid.cpu()
    cache["metadata"]["official_iou_cache"] = signature
    cache_path = Path(cache["metadata"].get("cache_path", ""))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    return cache


def diagnostic_iou_matrix(
    record: dict[str, Any],
    stage_name: str,
    use_official: bool,
    input_h: int = 288,
    input_w: int = 800,
    line_width: float = 30.0,
    min_valid_rows: int = 5,
    row_visibility_thresh: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stage = record["stages"][stage_name]
    if use_official:
        matrix = stage.get("official_iou")
        candidate_valid = stage.get("official_candidate_valid")
        if matrix is None or candidate_valid is None:
            raise RuntimeError("Official IoU cache was not prepared")
        valid_gt = torch.ones(matrix.shape[0], dtype=torch.bool)
        return matrix.float(), valid_gt, candidate_valid.bool()
    return proposal_gt_iou_matrix(
        stage,
        record["target"],
        input_h=input_h,
        input_w=input_w,
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
    )


def lanes_to_original(
    lanes: Iterable[list[tuple[float, float]]],
    meta: dict[str, Any],
) -> list[list[tuple[float, float]]]:
    sx = float(meta.get("scale_x", 1.0))
    sy = float(meta.get("scale_y", 1.0))
    crop_x = float(meta.get("crop_x", 0.0))
    crop_y = float(meta.get("crop_y", 0.0))
    return [
        [(round(x / sx + crop_x, 3), round(y / sy + crop_y, 3)) for x, y in lane]
        for lane in lanes
    ]


def exact_official_counts(
    records: list[dict[str, Any]],
    stage_name: str,
    selections: dict[str, list[int]],
    iou_threshold: float = 0.5,
    width: int = 30,
) -> dict[str, int | float]:
    tp = fp = fn = 0
    for record in records:
        stage = record["stages"].get(stage_name)
        if stage is None:
            continue
        selected = selections.get(record["image_id"], [])
        official_iou = stage.get("official_iou")
        if official_iou is not None:
            # Use Hungarian matching for exact_official_counts too to prevent hangs
            assignment = evaluator_hungarian_assignment(
                official_iou,
                selected,
                threshold=float(iou_threshold),
            )
            image_tp = assignment.hit_count
            tp += image_tp
            fp += max(0, len(selected) - image_tp)
            fn += max(0, int(official_iou.shape[0]) - image_tp)
            continue
        lanes, _ = stage_lane_points(
            stage,
            input_h=int(record["meta"].get("input_h", 288)),
            input_w=int(record["meta"].get("input_w", 800)),
            min_valid_rows=2,
            row_visibility_thresh=0.0,
        )
        pred = lanes_to_original([lanes[idx] for idx in selected], record["meta"])
        anno_path = record["meta"].get("anno_path")
        anno = load_culane_img_data(anno_path) if anno_path else []
        result = culane_metric(pred, anno, width=width, iou_thresholds=(iou_threshold,), official=True)
        image_tp, image_fp, image_fn = result[float(iou_threshold)]
        tp += int(image_tp)
        fp += int(image_fp)
        fn += int(image_fn)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


def metadata_for_json(cache: dict[str, Any], **extra: Any) -> dict[str, Any]:
    out = deepcopy(cache["metadata"])
    out.update(extra)
    return out
