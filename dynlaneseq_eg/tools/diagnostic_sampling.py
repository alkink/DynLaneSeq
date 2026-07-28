from __future__ import annotations

from torch.utils.data import DataLoader, Subset


def uniformly_spaced_indices(length: int, count: int) -> list[int]:
    """Choose deterministic indices spanning the complete ordered dataset."""

    length = int(length)
    count = int(count)
    if length < 0:
        raise ValueError("length must be non-negative")
    if count <= 0 or count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    indices = [
        round(position * (length - 1) / float(count - 1))
        for position in range(count)
    ]
    if len(set(indices)) != len(indices):
        raise RuntimeError("uniform diagnostic indices unexpectedly repeated")
    return indices


def select_diagnostic_loader(
    loader: DataLoader,
    *,
    strategy: str,
    max_batches: int,
    num_workers: int,
) -> tuple[DataLoader, list[int]]:
    """Rebuild an evaluation loader over a deterministic diagnostic subset."""

    strategy = str(strategy).strip().lower()
    if strategy not in {"sequential", "uniform"}:
        raise ValueError(f"Unsupported diagnostic sampling strategy: {strategy!r}")
    batch_size = int(loader.batch_size or 1)
    sample_count = (
        len(loader.dataset)
        if int(max_batches) <= 0
        else min(len(loader.dataset), int(max_batches) * batch_size)
    )
    if strategy == "uniform":
        indices = uniformly_spaced_indices(len(loader.dataset), sample_count)
    else:
        indices = list(range(sample_count))
    worker_count = int(num_workers)
    kwargs = {
        "dataset": Subset(loader.dataset, indices),
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": worker_count,
        "collate_fn": loader.collate_fn,
        "pin_memory": bool(loader.pin_memory),
        "drop_last": False,
        "persistent_workers": bool(
            worker_count > 0 and getattr(loader, "persistent_workers", False)
        ),
    }
    if worker_count > 0 and getattr(loader, "prefetch_factor", None) is not None:
        kwargs["prefetch_factor"] = int(loader.prefetch_factor)
    return DataLoader(**kwargs), indices
