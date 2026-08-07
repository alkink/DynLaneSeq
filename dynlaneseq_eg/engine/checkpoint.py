from __future__ import annotations

import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import torch


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only`` was added.
        return torch.load(path, map_location="cpu")


def _normalize_prefixes(values) -> tuple[str, ...]:
    prefixes = tuple(str(value).strip().rstrip(".") for value in values or ())
    if any(not value for value in prefixes):
        raise ValueError("checkpoint model-state prefixes must not be empty")
    return prefixes


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _resolve_base_checkpoint(
    value: str | Path,
    *,
    delta_checkpoint: Path,
) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            (
                Path.cwd() / raw,
                delta_checkpoint.parent / raw,
            )
        )
    else:
        # A compact selector may be copied from /workspace to a local clone.
        # Preserve portability when the stored path contains a project-relative
        # ``outputs/...`` suffix.
        parts = raw.parts
        if "outputs" in parts:
            output_index = parts.index("outputs")
            candidates.append(Path.cwd().joinpath(*parts[output_index:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "compact checkpoint base model is unavailable; checked: " + rendered
    )


def _materialize_model_state(
    path: str | Path,
    *,
    payload: dict[str, Any] | None = None,
    visited: set[Path] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    seen = set() if visited is None else visited
    if checkpoint_path in seen:
        raise ValueError(f"cyclic compact checkpoint dependency: {checkpoint_path}")
    seen.add(checkpoint_path)
    loaded = _torch_load(checkpoint_path) if payload is None else payload
    if not isinstance(loaded, dict):
        raise TypeError(f"checkpoint payload must be a mapping: {checkpoint_path}")
    model_state = loaded.get("model")
    if model_state is None and loaded and all(
        isinstance(value, torch.Tensor) for value in loaded.values()
    ):
        # Preserve compatibility with bare state_dict files accepted by the
        # historical ``load_compatible_model_weights`` path.
        return dict(loaded), {"model": loaded, "model_state_mode": "full"}
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    mode = str(loaded.get("model_state_mode", "full")).strip().lower()
    if mode == "full":
        return dict(model_state), loaded
    if mode != "delta":
        raise ValueError(f"unsupported checkpoint model_state_mode={mode!r}")
    base_value = loaded.get("base_checkpoint")
    if not base_value:
        raise ValueError(f"delta checkpoint has no base_checkpoint: {checkpoint_path}")
    base_path = _resolve_base_checkpoint(
        base_value,
        delta_checkpoint=checkpoint_path,
    )
    base_state, _base_payload = _materialize_model_state(
        base_path,
        visited=seen,
    )
    base_state.update(model_state)
    return base_state, loaded


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically and never leave a corrupt final path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception as exc:
        partial_bytes = temporary.stat().st_size if temporary.exists() else 0
        try:
            free_bytes = shutil.disk_usage(path.parent).free
        except OSError:
            free_bytes = -1
        temporary.unlink(missing_ok=True)
        free_text = "unknown" if free_bytes < 0 else f"{free_bytes / (1024 ** 3):.2f} GiB"
        raise RuntimeError(
            f"checkpoint write failed for {path}; partial={partial_bytes / (1024 ** 2):.1f} MiB, "
            f"filesystem_free={free_text}. Check df -h, df -i, quota, and filesystem health."
        ) from exc


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError("checkpoint rng_state must be a mapping")
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA RNG state count does not match visible devices: "
                f"{len(cuda_state)} vs {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def remap_optimizer_state_by_parameter(
    source_optimizer: torch.optim.Optimizer,
    target_optimizer: torch.optim.Optimizer,
    *,
    allow_target_superset: bool = False,
) -> dict[str, int]:
    """Move per-parameter optimizer state across a new group topology.

    By default both optimizers must reference the exact same Parameter
    objects.  ``allow_target_superset`` additionally supports a newly added
    module: old parameters retain AdamW moments while target-only parameters
    start with empty optimizer state. Group hyperparameters intentionally come
    from ``target_optimizer``.
    """

    source_parameters = {
        parameter
        for group in source_optimizer.param_groups
        for parameter in group["params"]
    }
    target_parameters = {
        parameter
        for group in target_optimizer.param_groups
        for parameter in group["params"]
    }
    parameter_sets_valid = (
        source_parameters <= target_parameters
        if allow_target_superset
        else source_parameters == target_parameters
    )
    if not parameter_sets_valid:
        raise ValueError(
            "cannot remap optimizer state: source and target parameter sets differ"
        )

    target_optimizer.state.clear()
    migrated = 0
    for parameter, state in source_optimizer.state.items():
        if parameter not in target_parameters:
            raise ValueError("source optimizer state contains an unknown parameter")
        target_optimizer.state[parameter] = state
        migrated += 1
    stats = {
        "parameters": len(target_parameters),
        "state_entries": migrated,
        "source_groups": len(source_optimizer.param_groups),
        "target_groups": len(target_optimizer.param_groups),
    }
    if source_parameters != target_parameters:
        stats["new_parameters"] = len(target_parameters - source_parameters)
    return stats


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scaler=None,
    iteration: int = 0,
    cfg: dict[str, Any] | None = None,
    scheduler=None,
    model_state_prefixes=(),
    base_checkpoint: str | Path | None = None,
    include_rng_state: bool = False,
) -> None:
    path = Path(path)
    prefixes = _normalize_prefixes(model_state_prefixes)
    full_state = model.state_dict()
    if prefixes:
        if not base_checkpoint:
            raise ValueError(
                "partial checkpoint model state requires a base_checkpoint"
            )
        model_state = {
            name: value
            for name, value in full_state.items()
            if _matches_prefix(name, prefixes)
        }
        if not model_state:
            raise ValueError(
                "checkpoint model-state prefixes matched no tensors: "
                + ", ".join(prefixes)
            )
        state_mode = "delta"
    else:
        model_state = full_state
        state_mode = "full"
    payload = {
        "model": model_state,
        "model_state_mode": state_mode,
        "iteration": iteration,
        "cfg": cfg or {},
    }
    if prefixes:
        payload["model_state_prefixes"] = list(prefixes)
        payload["base_checkpoint"] = str(base_checkpoint)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if include_rng_state:
        payload["rng_state"] = _capture_rng_state()
    _atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scaler=None,
    strict: bool = False,
    scheduler=None,
    restore_rng_state: bool = False,
) -> int:
    model_state, payload = _materialize_model_state(path)
    model.load_state_dict(model_state, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng_state and "rng_state" in payload:
        _restore_rng_state(payload["rng_state"])
    return int(payload.get("iteration", 0))


def load_compatible_model_weights(path: str | Path, model) -> dict[str, int]:
    source, _payload = _materialize_model_state(path)
    for key, value in list(source.items()):
        if key.startswith("heads.exist."):
            source.setdefault("exist_head." + key[len("heads.exist.") :], value)
        elif key.startswith("heads.range."):
            source.setdefault("range_head." + key[len("heads.range.") :], value)
    target = model.state_dict()
    compatible = {}
    skipped = 0
    for key, value in source.items():
        if key not in target:
            skipped += 1
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            skipped += 1
            continue
        compatible[key] = value
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "skipped": skipped,
        "missing": len(target) - len(compatible),
    }
