from __future__ import annotations

from itertools import chain
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import CULaneDataset, TuSimpleDataset, lane_collate
from .data.resume_safe import GlobalIterationBatchSampler
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
            lambda_obj_start=(
                None
                if m.get("lambda_obj_start") is None
                else float(m.get("lambda_obj_start"))
            ),
            lambda_obj_end=(
                None
                if m.get("lambda_obj_end") is None
                else float(m.get("lambda_obj_end"))
            ),
            lambda_obj_ramp_start_iter=int(
                m.get("lambda_obj_ramp_start_iter", 0)
            ),
            lambda_obj_ramp_end_iter=int(
                m.get("lambda_obj_ramp_end_iter", 0)
            ),
            lambda_point=float(m.get("lambda_point", 5.0)),
            lambda_range=float(m.get("lambda_range", 1.0)),
            lambda_line_iou=float(m.get("lambda_line_iou", 0.0)),
            line_iou_radius=float(m.get("line_iou_radius", cfg.get("loss", {}).get("line_iou_radius", 7.5))),
            input_w=int(model.get("input_w", 800)),
            input_h=int(model.get("input_h", 288)),
            assignment=str(m.get("assignment", "hungarian")),
            num_groups=int(m.get("num_groups", 1)),
            object_cost_type=str(m.get("object_cost_type", "neg_log_probability")),
            cost_type=str(m.get("cost_type", "composite")),
            range_aware_line_width=float(m.get("range_aware_line_width", 30.0)),
            range_aware_min_valid_rows=int(
                m.get("range_aware_min_valid_rows", 5)
            ),
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
        exist_target_mode=str(loss.get("exist_target_mode", "binary")),
        exist_quality_floor=float(loss.get("exist_quality_floor", 0.5)),
        exist_quality_beta=float(loss.get("exist_quality_beta", 2.0)),
        exist_quality_line_width=float(
            loss.get("exist_quality_line_width", 30.0)
        ),
        exist_quality_min_valid_rows=int(
            loss.get("exist_quality_min_valid_rows", 5)
        ),
        focal_alpha=float(loss.get("focal_alpha", 0.25)),
        focal_gamma=float(loss.get("focal_gamma", 2.0)),
        w_line_iou=float(loss.get("w_line_iou", 0.0)),
        line_iou_radius=float(loss.get("line_iou_radius", 15.0)),
        w_seg=float(loss.get("w_seg", 0.0)),
        seg_pos_weight=float(loss.get("seg_pos_weight", 1.0)),
        seg_extra_weights=dict(loss.get("seg_extra_weights", {})),
        w_quality=float(loss.get("w_quality", 0.0)),
        w_cardinality=float(loss.get("w_cardinality", 0.0)),
        w_score_margin=float(loss.get("w_score_margin", 0.0)),
        score_margin=float(loss.get("score_margin", 0.5)),
        score_margin_topk_negatives=int(
            loss.get("score_margin_topk_negatives", 8)
        ),
        w_set_selection=float(loss.get("w_set_selection", 0.0)),
        set_selection_line_width=float(
            loss.get("set_selection_line_width", 30.0)
        ),
        set_selection_focal_beta=float(
            loss.get("set_selection_focal_beta", 2.0)
        ),
        set_selection_rank_weight=float(
            loss.get("set_selection_rank_weight", 0.25)
        ),
        set_selection_target_margin=float(
            loss.get("set_selection_target_margin", 0.10)
        ),
        set_selection_min_valid_rows=int(
            loss.get("set_selection_min_valid_rows", 5)
        ),
        set_selection_share_matcher_assignment=bool(
            loss.get("set_selection_share_matcher_assignment", False)
        ),
        set_selection_negative_weight=float(
            loss.get("set_selection_negative_weight", 1.0)
        ),
        set_selection_positive_floor=float(
            loss.get("set_selection_positive_floor", 0.0)
        ),
        set_selection_coverage_weight=float(
            loss.get("set_selection_coverage_weight", 0.0)
        ),
        set_selection_duplicate_weight=float(
            loss.get("set_selection_duplicate_weight", 0.0)
        ),
        set_selection_winner_weight=float(
            loss.get("set_selection_winner_weight", 0.0)
        ),
        set_selection_count_weight=float(
            loss.get("set_selection_count_weight", 0.0)
        ),
        set_selection_duplicate_quality_min=float(
            loss.get("set_selection_duplicate_quality_min", 0.30)
        ),
        set_selection_winner_quality_min=float(
            loss.get("set_selection_winner_quality_min", 0.30)
        ),
        w_pointer_selection=float(loss.get("w_pointer_selection", 0.0)),
        pointer_quality_weight=float(
            loss.get("pointer_quality_weight", 0.5)
        ),
        pointer_cluster_listwise_weight=float(
            loss.get("pointer_cluster_listwise_weight", 0.0)
        ),
        pointer_cluster_listwise_logit_temperature=float(
            loss.get("pointer_cluster_listwise_logit_temperature", 1.0)
        ),
        pointer_stop_weight=float(loss.get("pointer_stop_weight", 1.0)),
        pointer_unary_target_mode=str(
            loss.get("pointer_unary_target_mode", "max_quality")
        ),
        w_four_slot_selection=float(
            loss.get("w_four_slot_selection", 0.0)
        ),
        four_slot_line_width=float(
            loss.get("four_slot_line_width", 30.0)
        ),
        four_slot_min_valid_rows=int(
            loss.get("four_slot_min_valid_rows", 5)
        ),
        four_slot_representable_min=float(
            loss.get("four_slot_representable_min", 0.50)
        ),
        four_slot_cluster_min=float(
            loss.get("four_slot_cluster_min", 0.30)
        ),
        four_slot_cluster_delta=float(
            loss.get("four_slot_cluster_delta", 0.05)
        ),
        four_slot_cluster_temperature=float(
            loss.get("four_slot_cluster_temperature", 0.03)
        ),
        four_slot_target_mode=str(
            loss.get("four_slot_target_mode", "joint_threshold")
        ),
        four_slot_permutation_temperature=float(
            loss.get("four_slot_permutation_temperature", 1.0)
        ),
        four_slot_assignment_mode=str(
            loss.get("four_slot_assignment_mode", "marginal")
        ),
        four_slot_collision_weight=float(
            loss.get("four_slot_collision_weight", 0.10)
        ),
        w_four_slot_geometry=float(
            loss.get("w_four_slot_geometry", 0.0)
        ),
        four_slot_geometry_point_weight=float(
            loss.get("four_slot_geometry_point_weight", 5.0)
        ),
        four_slot_geometry_line_iou_weight=float(
            loss.get("four_slot_geometry_line_iou_weight", 2.0)
        ),
        four_slot_geometry_dfl_weight=float(
            loss.get("four_slot_geometry_dfl_weight", 1.0)
        ),
        four_slot_geometry_range_weight=float(
            loss.get("four_slot_geometry_range_weight", 0.0)
        ),
        four_slot_geometry_match_min_quality=float(
            loss.get("four_slot_geometry_match_min_quality", 0.20)
        ),
        four_slot_geometry_match_all_slots=bool(
            loss.get("four_slot_geometry_match_all_slots", False)
        ),
        w_four_slot_unified=float(
            loss.get("w_four_slot_unified", 0.0)
        ),
        four_slot_unified_active_weight=float(
            loss.get("four_slot_unified_active_weight", 1.0)
        ),
        four_slot_unified_attention_weight=float(
            loss.get("four_slot_unified_attention_weight", 1.0)
        ),
        four_slot_unified_point_weight=float(
            loss.get("four_slot_unified_point_weight", 5.0)
        ),
        four_slot_unified_range_weight=float(
            loss.get("four_slot_unified_range_weight", 1.0)
        ),
        four_slot_unified_line_iou_weight=float(
            loss.get("four_slot_unified_line_iou_weight", 2.0)
        ),
        four_slot_unified_dfl_weight=float(
            loss.get("four_slot_unified_dfl_weight", 1.0)
        ),
        four_slot_unified_aux_geometry_weight=float(
            loss.get("four_slot_unified_aux_geometry_weight", 0.25)
        ),
        w_four_slot_visual_first=float(
            loss.get("w_four_slot_visual_first", 0.0)
        ),
        four_slot_visual_first_first_pass_weight=float(
            loss.get("four_slot_visual_first_first_pass_weight", 0.5)
        ),
        four_slot_visual_first_final_pass_weight=float(
            loss.get("four_slot_visual_first_final_pass_weight", 1.0)
        ),
        four_slot_visual_first_proposal_weight=float(
            loss.get("four_slot_visual_first_proposal_weight", 1.0)
        ),
        w_four_slot_visual_precision=float(
            loss.get("w_four_slot_visual_precision", 0.0)
        ),
        four_slot_visual_precision_point_weight=float(
            loss.get("four_slot_visual_precision_point_weight", 5.0)
        ),
        four_slot_visual_precision_range_weight=float(
            loss.get("four_slot_visual_precision_range_weight", 1.0)
        ),
        four_slot_visual_precision_line_iou_weight=float(
            loss.get("four_slot_visual_precision_line_iou_weight", 2.0)
        ),
        four_slot_visual_precision_dfl_weight=float(
            loss.get("four_slot_visual_precision_dfl_weight", 1.0)
        ),
        w_centerline=float(loss.get("w_centerline", 0.0)),
        w_row_dfl=float(loss.get("w_row_dfl", 0.0)),
        row_dfl_warmup_iters=int(loss.get("row_dfl_warmup_iters", 0)),
        centerline_sigma_bins=float(loss.get("centerline_sigma_bins", 1.5)),
        centerline_pos_weight=float(loss.get("centerline_pos_weight", 1.0)),
        w_dynamic_proposal_heatmap=float(loss.get("w_dynamic_proposal_heatmap", 0.0)),
        w_dynamic_proposal_x=float(loss.get("w_dynamic_proposal_x", 0.0)),
        w_dynamic_proposal_range=float(loss.get("w_dynamic_proposal_range", 0.0)),
        dynamic_proposal_sigma_bins=float(loss.get("dynamic_proposal_sigma_bins", 1.5)),
        dynamic_proposal_seed_radius_bins=int(loss.get("dynamic_proposal_seed_radius_bins", 2)),
        dynamic_proposal_heatmap_pos_weight=float(loss.get("dynamic_proposal_heatmap_pos_weight", 1.0)),
        lambda_geometry_draft=float(loss.get("lambda_geometry_draft", 0.0)),
        lambda_intermediate=float(loss.get("lambda_intermediate", 0.0)),
        w_intermediate_exist=(
            None
            if loss.get("w_intermediate_exist") is None
            else float(loss.get("w_intermediate_exist"))
        ),
        intermediate_layer_weights=tuple(float(v) for v in loss.get("intermediate_layer_weights", [])),
        lambda_training_auxiliary=float(
            loss.get("lambda_training_auxiliary", 0.0)
        ),
        geometry_reduction=str(loss.get("geometry_reduction", "global_rows")),
    )
    if "smoothness_contiguous" in getattr(LossConfig, "__dataclass_fields__", {}):
        base_kwargs["smoothness_contiguous"] = bool(loss.get("smoothness_contiguous", True))
    name = model.get("name", "DynLaneSeqS0")
    if name == "DynLaneSeqS0":
        loss_cfg = LossConfig(**base_kwargs, lambda_coarse=float(loss.get("lambda_coarse", 0.0)))
        aux_matcher = build_matcher(cfg) if loss_cfg.lambda_intermediate > 0.0 else None
        return S0Criterion(loss_cfg, matcher=aux_matcher)
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


def build_dataset(cfg: dict[str, Any], split: str = "train", training: bool = False):
    dataset_cfg = dataset_cfg_for_split(cfg, split)
    name = str(dataset_cfg.get("name", "CULane")).strip().lower()
    if name == "culane":
        dataset_cls = CULaneDataset
    elif name in {"tusimple", "tu_simple"}:
        dataset_cls = TuSimpleDataset
    else:
        raise ValueError(f"Unsupported dataset.name: {dataset_cfg.get('name')!r}")
    return dataset_cls(dataset_cfg, split=split, training=training)


def build_dataloader(
    cfg: dict[str, Any],
    split: str = "train",
    training: bool = False,
    *,
    start_iteration: int = 0,
) -> DataLoader:
    dl_cfg = cfg.get("dataloader", {})
    train_cfg = cfg.get("training", {})
    dataset = build_dataset(cfg, split=split, training=training)
    num_workers = int(dl_cfg.get("num_workers", 2))
    batch_size = int(
        train_cfg.get("batch_size", 2)
        if training
        else dl_cfg.get("eval_batch_size", 1)
    )
    resume_safe = bool(training and dl_cfg.get("resume_safe", False))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(dl_cfg.get("pin_memory", True)),
        "collate_fn": lane_collate,
    }
    if resume_safe:
        if "seed" not in train_cfg:
            raise ValueError("dataloader.resume_safe requires training.seed")
        kwargs["batch_sampler"] = GlobalIterationBatchSampler(
            dataset,
            batch_size=batch_size,
            base_seed=int(train_cfg["seed"]),
            start_iteration=int(start_iteration),
            gradient_accumulation_steps=max(
                int(train_cfg.get("gradient_accumulation_steps", 1)), 1
            ),
        )
    else:
        kwargs.update(
            {
                "batch_size": batch_size,
                "shuffle": bool(training and dl_cfg.get("shuffle", True)),
                "drop_last": False,
            }
        )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(dl_cfg.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(dl_cfg.get("prefetch_factor", 2))
    if "seed" in train_cfg:
        generator = torch.Generator()
        generator.manual_seed(int(train_cfg["seed"]) + (0 if training else 100_000))
        kwargs["generator"] = generator
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
    custom_specs = []
    custom_group_names: set[str] = set()
    for raw_spec in opt_cfg.get("parameter_groups", []):
        if not isinstance(raw_spec, dict):
            raise ValueError("optimizer.parameter_groups entries must be mappings")
        group_name = str(raw_spec.get("name", "")).strip()
        if not group_name:
            raise ValueError("optimizer.parameter_groups entry has an empty name")
        if group_name in custom_group_names:
            raise ValueError(f"duplicate optimizer parameter-group name: {group_name}")
        prefixes = tuple(str(value) for value in raw_spec.get("prefixes", []))
        if not prefixes or any(not prefix for prefix in prefixes):
            raise ValueError(
                f"optimizer parameter group {group_name!r} needs non-empty prefixes"
            )
        group_lr = float(raw_spec.get("lr", base_lr))
        if not math.isfinite(group_lr) or group_lr < 0.0:
            raise ValueError(
                f"optimizer parameter group {group_name!r} has invalid LR {group_lr}"
            )
        custom_group_names.add(group_name)
        custom_specs.append(
            {
                "name": group_name,
                "prefixes": prefixes,
                "lr": group_lr,
                "decay": [],
                "no_decay": [],
                "matched_names": [],
            }
        )
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
        custom_matches = [
            spec for spec in custom_specs if name.startswith(spec["prefixes"])
        ]
        if len(custom_matches) > 1:
            raise ValueError(
                f"parameter {name!r} matches multiple custom optimizer groups: "
                + ", ".join(spec["name"] for spec in custom_matches)
            )
        if custom_matches:
            spec = custom_matches[0]
            bucket = "no_decay" if is_no_decay else "decay"
            spec[bucket].append(param)
            spec["matched_names"].append(name)
            continue
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
    custom_groups = []
    for spec in custom_specs:
        if not spec["matched_names"]:
            raise ValueError(
                f"custom optimizer group {spec['name']!r} matched no parameters"
            )
        custom_groups.extend(
            [
                {
                    "params": spec["decay"],
                    "lr": spec["lr"],
                    "weight_decay": wd,
                    "name": f"{spec['name']}_decay",
                },
                {
                    "params": spec["no_decay"],
                    "lr": spec["lr"],
                    "weight_decay": 0.0,
                    "name": f"{spec['name']}_no_decay",
                },
            ]
        )
    groups = custom_groups + [
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
