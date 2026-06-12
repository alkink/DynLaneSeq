from __future__ import annotations

from itertools import chain
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import CULaneDataset, lane_collate
from .losses import HungarianMatcherS0, S0Criterion, S1Criterion, S2Criterion, S3Criterion, S4Criterion
from .losses.loss_s0 import LossConfig
from .losses.loss_s1 import S1LossConfig
from .losses.loss_s2 import S2LossConfig
from .losses.loss_s4 import S4LossConfig
from .losses.matcher_s0 import MatcherConfig
from .modeling import DynLaneSeqS0, DynLaneSeqS1, DynLaneSeqS2, DynLaneSeqS3, DynLaneSeqS4


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    name = cfg.get("model", {}).get("name", "DynLaneSeqS0")
    table = {
        "DynLaneSeqS0": DynLaneSeqS0,
        "DynLaneSeqS1": DynLaneSeqS1,
        "DynLaneSeqS2": DynLaneSeqS2,
        "DynLaneSeqS3": DynLaneSeqS3,
        "DynLaneSeqS4": DynLaneSeqS4,
    }
    if name not in table:
        raise ValueError(f"Unsupported model.name: {name}")
    return table[name](cfg)


def build_matcher(cfg: dict[str, Any]) -> HungarianMatcherS0:
    m = cfg.get("matcher", {})
    model = cfg.get("model", {})
    return HungarianMatcherS0(
        MatcherConfig(
            lambda_obj=float(m.get("lambda_obj", 2.0)),
            lambda_point=float(m.get("lambda_point", 5.0)),
            lambda_range=float(m.get("lambda_range", 1.0)),
            lambda_line_iou=float(m.get("lambda_line_iou", 0.0)),
            line_iou_radius=float(m.get("line_iou_radius", cfg.get("loss", {}).get("line_iou_radius", 7.5))),
            input_w=int(model.get("input_w", 800)),
            input_h=int(model.get("input_h", 288)),
            assignment=str(m.get("assignment", "hungarian")),
            num_groups=int(m.get("num_groups", 1)),
        )
    )


def build_criterion(cfg: dict[str, Any]) -> torch.nn.Module:
    loss = cfg.get("loss", {})
    model = cfg.get("model", {})
    base_kwargs = dict(
        w_exist=float(loss.get("w_exist", 2.0)),
        w_point=float(loss.get("w_point", 5.0)),
        w_range=float(loss.get("w_range", 1.0)),
        w_smooth=float(loss.get("w_smooth", 0.0)),
        smooth_l1_beta=float(loss.get("smooth_l1_beta", 0.01)),
        input_w=int(model.get("input_w", 800)),
        input_h=int(model.get("input_h", 288)),
        no_lane_weight=float(loss.get("no_lane_weight", 1.0)),
        exist_loss_type=str(loss.get("exist_loss_type", "ce")),
        focal_alpha=float(loss.get("focal_alpha", 0.25)),
        focal_gamma=float(loss.get("focal_gamma", 2.0)),
        w_line_iou=float(loss.get("w_line_iou", 0.0)),
        line_iou_radius=float(loss.get("line_iou_radius", 15.0)),
        w_seg=float(loss.get("w_seg", 0.0)),
        seg_pos_weight=float(loss.get("seg_pos_weight", 1.0)),
        seg_extra_weights=dict(loss.get("seg_extra_weights", {})),
        w_quality=float(loss.get("w_quality", 0.0)),
        w_centerline=float(loss.get("w_centerline", 0.0)),
        centerline_sigma_bins=float(loss.get("centerline_sigma_bins", 1.5)),
        centerline_pos_weight=float(loss.get("centerline_pos_weight", 1.0)),
        w_dynamic_proposal_heatmap=float(loss.get("w_dynamic_proposal_heatmap", 0.0)),
        w_dynamic_proposal_x=float(loss.get("w_dynamic_proposal_x", 0.0)),
        w_dynamic_proposal_range=float(loss.get("w_dynamic_proposal_range", 0.0)),
        dynamic_proposal_sigma_bins=float(loss.get("dynamic_proposal_sigma_bins", 1.5)),
        dynamic_proposal_seed_radius_bins=int(loss.get("dynamic_proposal_seed_radius_bins", 2)),
        dynamic_proposal_heatmap_pos_weight=float(loss.get("dynamic_proposal_heatmap_pos_weight", 1.0)),
        lambda_geometry_draft=float(loss.get("lambda_geometry_draft", 0.0)),
    )
    if "smoothness_contiguous" in getattr(LossConfig, "__dataclass_fields__", {}):
        base_kwargs["smoothness_contiguous"] = bool(loss.get("smoothness_contiguous", True))
    name = model.get("name", "DynLaneSeqS0")
    if name == "DynLaneSeqS0":
        return S0Criterion(LossConfig(**base_kwargs, lambda_coarse=float(loss.get("lambda_coarse", 0.0))))
    if name == "DynLaneSeqS1":
        return S1Criterion(
            S1LossConfig(
                **base_kwargs,
                lambda_coarse=float(loss.get("lambda_coarse", 0.0)),
                w_token=float(loss.get("w_token", 0.5)),
                token_label_smoothing=float(loss.get("token_label_smoothing", 0.0)),
                w_visibility=float(loss.get("w_visibility", 0.0)),
                visibility_pos_weight=float(loss.get("visibility_pos_weight", 1.0)),
            )
        )
    if name in {"DynLaneSeqS2", "DynLaneSeqS3"}:
        cls = S2Criterion if name == "DynLaneSeqS2" else S3Criterion
        s2_loss_cfg = S2LossConfig(
            **base_kwargs,
            w_token=float(loss.get("w_token", 0.5)),
            token_label_smoothing=float(loss.get("token_label_smoothing", 0.0)),
            w_visibility=float(loss.get("w_visibility", 0.0)),
            visibility_pos_weight=float(loss.get("visibility_pos_weight", 1.0)),
            lambda_coarse=float(loss.get("lambda_coarse", 0.5)),
            w_active_offset_reg=float(loss.get("w_active_offset_reg", 0.0)),
            w_active_offset_ce=float(loss.get("w_active_offset_ce", 0.0)),
            active_offset_max=float(loss.get("active_offset_max", 32.0)),
            active_offset_label_smoothing=float(loss.get("active_offset_label_smoothing", 0.0)),
            cascade_matching=bool(loss.get("cascade_matching", False)),
        )
        if name == "DynLaneSeqS3" and bool(loss.get("cascade_matching", False)):
            return cls(s2_loss_cfg, matcher=build_matcher(cfg))
        return cls(s2_loss_cfg)
    if name == "DynLaneSeqS4":
        return S4Criterion(
            S4LossConfig(
                **base_kwargs,
                w_token=float(loss.get("w_token", 0.5)),
                token_label_smoothing=float(loss.get("token_label_smoothing", 0.0)),
                w_visibility=float(loss.get("w_visibility", 0.0)),
                visibility_pos_weight=float(loss.get("visibility_pos_weight", 1.0)),
                lambda_stage1=float(loss.get("lambda_stage1", 0.5)),
                lambda_coarse=float(loss.get("lambda_coarse", 0.25)),
            )
        )
    raise ValueError(f"Unsupported criterion for model.name: {name}")


def dataset_cfg_for_split(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    out = dict(cfg.get("dataset", {}))
    out["input_w"] = int(cfg.get("model", {}).get("input_w", 800))
    out["input_h"] = int(cfg.get("model", {}).get("input_h", 288))
    out["num_rows"] = int(cfg.get("model", {}).get("num_rows", 72))
    out["x_bins"] = int(cfg.get("model", {}).get("x_bins", 200))
    out["token_ignore_index"] = int(cfg.get("model", {}).get("token_ignore_index", -100))
    out["augmentation"] = cfg.get("augmentation", {})
    if split == "train" and cfg.get("dataset", {}).get("mode") == "overfit":
        out["num_samples"] = int(cfg.get("dataset", {}).get("num_samples", 10))
    return out


def build_dataset(cfg: dict[str, Any], split: str = "train", training: bool = False) -> CULaneDataset:
    return CULaneDataset(dataset_cfg_for_split(cfg, split), split=split, training=training)


def build_dataloader(cfg: dict[str, Any], split: str = "train", training: bool = False) -> DataLoader:
    dl_cfg = cfg.get("dataloader", {})
    train_cfg = cfg.get("training", {})
    dataset = build_dataset(cfg, split=split, training=training)
    num_workers = int(dl_cfg.get("num_workers", 2))
    kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 2) if training else dl_cfg.get("eval_batch_size", 1)),
        "shuffle": bool(training and dl_cfg.get("shuffle", True)),
        "num_workers": num_workers,
        "pin_memory": bool(dl_cfg.get("pin_memory", True)),
        "collate_fn": lane_collate,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(dl_cfg.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(dl_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def build_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    base_lr = float(opt_cfg.get("base_lr", 1e-4))
    backbone_lr = float(opt_cfg.get("backbone_lr", 1e-5))
    row_decoder_lr = opt_cfg.get("row_decoder_lr")
    row_decoder_lr = float(row_decoder_lr) if row_decoder_lr is not None else None
    evidence_lr = opt_cfg.get("evidence_lr")
    evidence_lr = float(evidence_lr) if evidence_lr is not None else None
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    decay = []
    no_decay = []
    backbone_decay = []
    backbone_no_decay = []
    row_decay = []
    row_no_decay = []
    evidence_decay = []
    evidence_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower()
        is_backbone = ".backbone." in name or name.startswith("encoder.backbone")
        is_row_decoder = row_decoder_lr is not None and (name.startswith("row_decoder.") or name.startswith("row_embedding."))
        is_evidence = evidence_lr is not None and (
            name.startswith("adapter.")
            or name.startswith("bridge.")
            or name.startswith("multi_scale_sampler.")
            or name.startswith("offset_fusion.")
            or name.startswith("active_corridor.")
            or name.startswith("active_corridor_sampler.")
            or name.startswith("quality_calibrator.")
            or name.startswith("s0_geometry_refiner.")
            or name.startswith("encoder.dynamic_proposal.")
            or name.startswith("structured_query_head.")
            or name.startswith("encoder.ms_proj.")
            or "evidence" in name
            or "dynamic_proposal" in name
            or "structured_query" in name
        )
        if is_evidence and is_no_decay:
            evidence_no_decay.append(param)
        elif is_evidence:
            evidence_decay.append(param)
        elif is_row_decoder and is_no_decay:
            row_no_decay.append(param)
        elif is_row_decoder:
            row_decay.append(param)
        elif is_backbone and is_no_decay:
            backbone_no_decay.append(param)
        elif is_backbone:
            backbone_decay.append(param)
        elif is_no_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": wd, "name": "backbone_decay"},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_no_decay"},
        {"params": row_decay, "lr": row_decoder_lr or base_lr, "weight_decay": wd, "name": "row_decoder_decay"},
        {"params": row_no_decay, "lr": row_decoder_lr or base_lr, "weight_decay": 0.0, "name": "row_decoder_no_decay"},
        {"params": evidence_decay, "lr": evidence_lr or base_lr, "weight_decay": wd, "name": "evidence_decay"},
        {"params": evidence_no_decay, "lr": evidence_lr or base_lr, "weight_decay": 0.0, "name": "evidence_no_decay"},
        {"params": decay, "lr": base_lr, "weight_decay": wd, "name": "model_decay"},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "name": "model_no_decay"},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    return torch.optim.AdamW(groups, betas=tuple(opt_cfg.get("betas", [0.9, 0.999])))


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer, total_iters: int):
    sched_cfg = cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "none")).lower()
    if name in {"", "none", "constant"}:
        return None
    total_iters = int(sched_cfg.get("total_iters", total_iters))

    if name == "cosine":
        warmup_iters = int(sched_cfg.get("warmup_iters", 0))
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

        def lr_lambda(step: int) -> float:
            step = int(step)
            if warmup_iters > 0 and step < warmup_iters:
                return max(1, step + 1) / float(warmup_iters)
            progress = (step - warmup_iters) / float(max(1, total_iters - warmup_iters))
            progress = min(max(progress, 0.0), 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if name == "multistep":
        milestones = [int(x) for x in sched_cfg.get("milestones", [])]
        gamma = float(sched_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    raise ValueError(f"Unsupported scheduler.name: {name}")


def all_params(groups):
    return chain.from_iterable(group["params"] for group in groups)
