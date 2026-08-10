from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


_UINT64_MASK = (1 << 64) - 1
_TORCH_SEED_MASK = (1 << 63) - 1


def _splitmix64(value: int) -> int:
    """Return a stable 64-bit mix without depending on Python's hash seed."""

    value = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def deterministic_seed(base_seed: int, *components: int) -> int:
    """Mix integer coordinates into a deterministic non-negative seed."""

    state = int(base_seed) & _UINT64_MASK
    for component in components:
        state = _splitmix64(state ^ (int(component) & _UINT64_MASK))
    return state & _TORCH_SEED_MASK


@dataclass(frozen=True)
class SeededSampleIndex:
    """Dataset index plus augmentation identity for one sample presentation.

    Passing the seed through the sampler, instead of deriving randomness in a
    worker process, makes augmentation independent of worker count, prefetch
    depth, scheduling, and process restarts.
    """

    index: int
    augmentation_seed: int
    epoch: int
    position: int


def unpack_seeded_sample_index(index: int | SeededSampleIndex) -> tuple[int, int | None]:
    if isinstance(index, SeededSampleIndex):
        return int(index.index), int(index.augmentation_seed)
    return int(index), None


class GlobalIterationBatchSampler(Sampler):
    """Deterministic shuffled batches addressed by optimizer iteration.

    The first micro-batch is derived from ``start_iteration * grad_accum``.
    Epoch permutations and per-presentation augmentation seeds are pure
    functions of the training seed and this global position. No mutable
    sampler or worker RNG state is required in a checkpoint.

    One ``__iter__`` call yields the remainder of the current logical epoch.
    The training loop may request another iterator and will then receive the
    next full epoch, matching ordinary finite DataLoader behavior.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        batch_size: int,
        base_seed: int,
        start_iteration: int = 0,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.dataset_size = int(len(dataset))
        self.batch_size = int(batch_size)
        self.base_seed = int(base_seed)
        self.start_iteration = int(start_iteration)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.dataset_size <= 0:
            raise ValueError("resume-safe sampling requires a non-empty dataset")
        if self.batch_size <= 0:
            raise ValueError("resume-safe sampling requires batch_size > 0")
        if self.start_iteration < 0:
            raise ValueError("resume-safe sampling requires start_iteration >= 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "resume-safe sampling requires gradient_accumulation_steps > 0"
            )
        self.batches_per_epoch = int(
            math.ceil(self.dataset_size / float(self.batch_size))
        )
        self._next_microbatch = (
            self.start_iteration * self.gradient_accumulation_steps
        )

    @property
    def next_microbatch(self) -> int:
        return int(self._next_microbatch)

    @property
    def logical_epoch(self) -> int:
        return int(self._next_microbatch // self.batches_per_epoch)

    @property
    def batch_in_epoch(self) -> int:
        return int(self._next_microbatch % self.batches_per_epoch)

    def _permutation(self, epoch: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(
            deterministic_seed(self.base_seed, 0x45504F4348, int(epoch))
        )
        return torch.randperm(self.dataset_size, generator=generator)

    def __iter__(self) -> Iterator[list[SeededSampleIndex]]:
        epoch = self.logical_epoch
        first_batch = self.batch_in_epoch
        permutation = self._permutation(epoch)
        for batch_index in range(first_batch, self.batches_per_epoch):
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, self.dataset_size)
            batch: list[SeededSampleIndex] = []
            for position in range(start, stop):
                sample_index = int(permutation[position])
                augmentation_seed = deterministic_seed(
                    self.base_seed,
                    0x4155474D454E54,
                    epoch,
                    position,
                    sample_index,
                ) & 0xFFFFFFFF
                batch.append(
                    SeededSampleIndex(
                        index=sample_index,
                        augmentation_seed=augmentation_seed,
                        epoch=epoch,
                        position=position,
                    )
                )
            # Advance before yielding so the state always describes the next
            # batch already handed to the DataLoader prefetch queue.
            self._next_microbatch += 1
            yield batch

    def __len__(self) -> int:
        # A stable full-epoch length keeps iteration->epoch reporting identical
        # to the legacy DataLoader, even when the first iterator starts midway.
        return self.batches_per_epoch

    def contract(self) -> dict[str, int | bool]:
        return {
            "enabled": True,
            "base_seed": self.base_seed,
            "start_iteration": self.start_iteration,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "start_microbatch": (
                self.start_iteration * self.gradient_accumulation_steps
            ),
            "batches_per_epoch": self.batches_per_epoch,
            "start_epoch": (
                self.start_iteration
                * self.gradient_accumulation_steps
                // self.batches_per_epoch
            ),
            "start_batch_in_epoch": (
                self.start_iteration
                * self.gradient_accumulation_steps
                % self.batches_per_epoch
            ),
            "per_sample_augmentation_seed": True,
        }
