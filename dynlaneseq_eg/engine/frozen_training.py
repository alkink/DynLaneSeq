from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def normalize_prefixes(values: Iterable[object]) -> tuple[str, ...]:
    """Normalize dotted module/parameter prefixes and reject empty entries."""

    prefixes = tuple(str(value).strip().rstrip(".") for value in values)
    if any(not value for value in prefixes):
        raise ValueError("frozen-training prefixes must not be empty")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("frozen-training prefixes must be unique")
    return prefixes


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def freeze_except_parameter_prefixes(
    model: nn.Module,
    prefixes: Iterable[object],
) -> dict[str, object]:
    """Freeze a detector while leaving only explicitly named score paths live."""

    normalized = normalize_prefixes(prefixes)
    selected: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = _matches(name, normalized)
        parameter.requires_grad_(trainable)
        (selected if trainable else frozen).append(name)
    if not selected:
        raise ValueError(
            "training.trainable_parameter_prefixes matched no model parameters: "
            + ", ".join(normalized)
        )
    return {
        "prefixes": list(normalized),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_tensor_count": len(selected),
        "frozen_tensor_count": len(frozen),
        "trainable_names": selected,
    }


def set_frozen_detector_eval(
    model: nn.Module,
    trainable_module_prefixes: Iterable[object],
) -> tuple[str, ...]:
    """Keep a frozen detector deterministic while training selected subtrees.

    ``model.train()`` would otherwise reactivate BatchNorm updates and dropout
    throughout the frozen detector.  That changes its candidate geometry even
    when every detector parameter has ``requires_grad=False``.  This helper
    first places the complete model in evaluation mode and then re-enables
    training mode only for the explicitly selected score-only modules.
    """

    normalized = normalize_prefixes(trainable_module_prefixes)
    model.eval()
    matched: list[str] = []
    for name, module in model.named_modules():
        if name and _matches(name, normalized):
            module.train(True)
            matched.append(name)
    missing = [
        prefix
        for prefix in normalized
        if not any(name == prefix or name.startswith(prefix + ".") for name in matched)
    ]
    if missing:
        raise ValueError(
            "training.trainable_module_prefixes matched no modules: "
            + ", ".join(missing)
        )
    return tuple(matched)
