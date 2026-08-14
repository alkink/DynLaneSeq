from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch.nn import functional as F

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.engine.frozen_training import (
    freeze_except_parameter_prefixes,
    set_frozen_detector_eval,
)
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.audit_v19_counterfactual_fidelity_contract import (
    FORBIDDEN_FORWARD_ARGUMENTS,
    _difference,
    _writer_kwargs,
)
from dynlaneseq_eg.tools.train import seed_everything


MODULE_PREFIX = (
    "structured_query_head.set_selection_head.slot_owned_safe_replacement"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V20 exact KEEP parity and frozen V7/V19 isolation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-iteration", type=int, default=233000)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(
        Path(args.dataset_root).expanduser().resolve()
    )
    cfg["dataset"].setdefault("lists", {})["train"] = str(
        Path(args.list_path).expanduser().resolve()
    )
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def main() -> None:
    args = parse_args()
    cfg = _config(args)
    seed_everything(int(cfg.get("training", {}).get("seed", 3407)))
    device = torch.device(args.device)
    model = build_model(cfg)
    iteration = int(load_checkpoint(args.checkpoint, model, strict=False))
    model = model.to(device)
    train_cfg = cfg.get("training", {})
    prefixes = tuple(train_cfg.get("trainable_parameter_prefixes", ()))
    module_prefixes = tuple(train_cfg.get("trainable_module_prefixes", ()))
    freeze_stats = freeze_except_parameter_prefixes(model, prefixes)
    set_frozen_detector_eval(model, module_prefixes)
    selector = model.structured_query_head.set_selection_head
    v19 = selector.counterfactual_fidelity
    v20 = selector.slot_owned_safe_replacement
    if v19 is None or v20 is None:
        raise ValueError("V20 Gate 0 requires both V19 and V20")
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    only_v20_trainable = bool(named) and all(
        name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + ".")
        for name, _parameter in named
    )
    frozen_state_before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not (name == MODULE_PREFIX or name.startswith(MODULE_PREFIX + "."))
    }

    loader = build_dataloader(cfg, split="train", training=False)
    images, _targets, metas = next(iter(loader))
    images = images.to(device, non_blocking=True)

    # Same-loaded-model V7 reference avoids attributing independent-kernel
    # noise to V20.  The immutable V19 module is restored before V20 forward.
    selector.counterfactual_fidelity = None
    selector.slot_owned_safe_replacement = None
    with torch.no_grad():
        v7_outputs = model(images)
    selector.counterfactual_fidelity = v19
    selector.slot_owned_safe_replacement = v20
    outputs = model(images)

    parity_names = (
        "selection_slot_real_route_logits",
        "selection_slot_geometry_route_indices",
        "selection_slot_indices",
        "selection_slot_scores",
        "selection_slot_active_logits",
        "selection_slot_active",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    )
    parity = {
        name: _difference(v7_outputs[name], outputs[name])
        for name in parity_names
    }
    internal = {
        "route": _difference(
            outputs["selection_slot_v20_v7_geometry_route_indices"],
            outputs["selection_slot_geometry_route_indices"],
        ),
        "indices": _difference(
            outputs["selection_slot_v20_v7_indices"],
            outputs["selection_slot_indices"],
        ),
        "scores": _difference(
            outputs["selection_slot_v20_v7_scores"],
            outputs["selection_slot_scores"],
        ),
        "activity": _difference(
            outputs["selection_slot_v20_v7_active"],
            outputs["selection_slot_active"],
        ),
    }
    with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
        left_paths = write_culane_predictions(
            v7_outputs, metas, left, **_writer_kwargs(cfg)
        )
        right_paths = write_culane_predictions(
            outputs, metas, right, **_writer_kwargs(cfg)
        )
        left_bytes = {path.relative_to(left): path.read_bytes() for path in left_paths}
        right_bytes = {path.relative_to(right): path.read_bytes() for path in right_paths}
        writer_exact = left_bytes == right_bytes

    head_kwargs = {
        "candidate_state": outputs["selection_slot_v19_candidate_state"],
        "p50": outputs["selection_slot_v19_p50"],
        "p75": outputs["selection_slot_v19_p75"],
        "expected_iou": outputs["selection_slot_v19_expected_iou"],
        "legacy_route_logits": outputs[
            "selection_slot_v19_v7_real_route_logits"
        ],
        "counterfactual_x": outputs[
            "selection_slot_v19_counterfactual_x_rows"
        ],
        "counterfactual_range": outputs[
            "selection_slot_v19_counterfactual_range_norm"
        ],
        "counterfactual_valid": outputs[
            "selection_slot_v19_counterfactual_valid"
        ],
        "source_route": outputs[
            "selection_slot_v20_v7_geometry_route_indices"
        ],
        "source_active": outputs["selection_slot_v20_v7_active"],
    }
    treatment = v20(**head_kwargs, force_context_mode="treatment")
    control = v20(**head_kwargs, force_context_mode="masked")
    context_hidden_difference = _difference(
        treatment["action_hidden"], control["action_hidden"]
    )
    context_deployment_equal = torch.equal(
        treatment["selected_route"], control["selected_route"]
    )

    action_valid = treatment["action_valid"]
    policy_target = torch.zeros(
        (action_valid.shape[0], 1 + action_valid[0].numel()),
        device=device,
    )
    policy_target[:, 0] = 1.0
    policy_logits = torch.cat(
        (
            treatment["policy_logits"].new_zeros((action_valid.shape[0], 1)),
            treatment["policy_logits"].reshape(action_valid.shape[0], -1),
        ),
        dim=-1,
    )
    synthetic_loss = -(
        policy_target * F.log_softmax(policy_logits.float(), dim=-1)
    ).sum(dim=-1).mean()
    synthetic_loss = synthetic_loss + F.cross_entropy(
        treatment["delta50_logits"][action_valid].float(),
        torch.ones(int(action_valid.sum()), dtype=torch.long, device=device),
    )
    gradients = torch.autograd.grad(
        synthetic_loss,
        [parameter for _name, parameter in named],
        allow_unused=True,
    )
    gradient_finite = all(
        value is None or bool(torch.isfinite(value).all()) for value in gradients
    )
    output_gradient = sum(
        float(value.detach().float().square().sum())
        for (name, _parameter), value in zip(named, gradients)
        if value is not None and "_output." in name
    ) ** 0.5

    frozen_state_after = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in frozen_state_before
    }
    frozen_exact = all(
        torch.equal(value, frozen_state_after[name])
        for name, value in frozen_state_before.items()
    )
    zero_outputs = {
        name: float(value.abs().max().detach().cpu())
        for name, value in {
            "policy": outputs["selection_slot_v20_policy_logits"],
            "delta50": outputs["selection_slot_v20_delta50_logits"],
            "delta75": outputs["selection_slot_v20_delta75_logits"],
            "duplicate": outputs["selection_slot_v20_duplicate_logits"],
            "abandon": outputs["selection_slot_v20_abandon_logits"],
            "delta_iou": outputs["selection_slot_v20_delta_iou"],
        }.items()
    }
    augmentation = cfg.get("augmentation", {})
    augmentation_off = all(
        augmentation.get(name) == value
        for name, value in {
            "horizontal_flip_prob": 0.0,
            "color_jitter": False,
            "channel_shuffle_prob": 0.0,
            "hue_saturation_prob": 0.0,
            "blur_prob": 0.0,
            "affine_prob": 0.0,
            "affine_translate_x": 0.0,
            "affine_translate_y": 0.0,
            "affine_rotate_deg": 0.0,
            "affine_scale_min": 1.0,
            "affine_scale_max": 1.0,
            "random_shadow_prob": 0.0,
        }.items()
    )
    target_free = not bool(
        FORBIDDEN_FORWARD_ARGUMENTS.intersection(
            inspect.signature(v20.forward).parameters
        )
    )
    finite = all(
        bool(torch.isfinite(value).all())
        for name, value in outputs.items()
        if name.startswith("selection_slot_v20_")
        and isinstance(value, torch.Tensor)
        and value.dtype.is_floating_point
    )
    passed = all(
        (
            iteration == int(args.expected_iteration),
            only_v20_trainable,
            frozen_exact,
            writer_exact,
            all(value == 0.0 for value in parity.values()),
            all(value == 0.0 for value in internal.values()),
            all(value == 0.0 for value in zero_outputs.values()),
            int(outputs["selection_slot_v20_edit_count"].sum()) == 0,
            context_hidden_difference > 0.0,
            context_deployment_equal,
            augmentation_off,
            target_free,
            finite,
            gradient_finite,
            output_gradient > 0.0,
        )
    )
    report = {
        "version": "v20_slot_owned_safe_replacement_gate0",
        "passed": bool(passed),
        "cache_authorized": bool(passed),
        "training_authorized": bool(passed),
        "long_training_authorized": False,
        "iteration": iteration,
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "list_path": str(Path(args.list_path).resolve()),
        "list_sha256": sha256_file(args.list_path),
        "only_v20_trainable": only_v20_trainable,
        "freeze_stats": freeze_stats,
        "frozen_v7_v19_state_and_buffers_exact": frozen_exact,
        "augmentation_exactly_disabled": augmentation_off,
        "target_free_inference": target_free,
        "tensors_finite": finite,
        "zero_step": {
            "public_v7_parity": parity,
            "same_forward_internal_v7_parity": internal,
            "writer_bytes_exact": writer_exact,
            "zero_output_heads": zero_outputs,
            "edit_count": int(outputs["selection_slot_v20_edit_count"].sum()),
            "treatment_masked_hidden_max_difference": context_hidden_difference,
            "treatment_masked_deployment_equal": context_deployment_equal,
        },
        "gradient": {
            "synthetic_loss": float(synthetic_loss.detach().cpu()),
            "output_head_norm": output_gradient,
            "finite": gradient_finite,
        },
        "contract": {
            "max_active_edits": 1,
            "default_action": "KEEP",
            "exact_official_raster_cached_targets": True,
            "main_control": "same head with other-lane context masked",
            "cross_clip_wrong_context": "inference-only causal audit",
        },
        "test_set_used": False,
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

