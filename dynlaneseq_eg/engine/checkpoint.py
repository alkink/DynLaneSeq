from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scaler=None,
    iteration: int = 0,
    cfg: dict[str, Any] | None = None,
    scheduler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "iteration": iteration, "cfg": cfg or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scaler=None, strict: bool = False, scheduler=None) -> int:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("iteration", 0))


def load_compatible_model_weights(path: str | Path, model) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu")
    source = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    source = dict(source)
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
