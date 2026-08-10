from __future__ import annotations

import copy

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from dynlaneseq_eg.data.resume_safe import (
    GlobalIterationBatchSampler,
    SeededSampleIndex,
)
from dynlaneseq_eg.data.transforms import LaneTransforms, TransformConfig


class _SizedDataset(Dataset):
    def __init__(self, size: int):
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: SeededSampleIndex) -> torch.Tensor:
        rng = np.random.RandomState(item.augmentation_seed)
        return torch.tensor(
            [
                float(item.index),
                float(item.epoch),
                float(item.position),
                float(rng.uniform(-1.0, 1.0)),
            ],
            dtype=torch.float64,
        )


def _take_sampler_batches(
    sampler: GlobalIterationBatchSampler,
    count: int,
) -> list[list[SeededSampleIndex]]:
    output: list[list[SeededSampleIndex]] = []
    while len(output) < count:
        for batch in sampler:
            output.append(batch)
            if len(output) == count:
                break
    return output


def _batch_identity(batch: list[SeededSampleIndex]) -> list[tuple[int, int, int, int]]:
    return [
        (item.index, item.augmentation_seed, item.epoch, item.position)
        for item in batch
    ]


def test_global_iteration_sampler_resume_matches_uninterrupted_stream() -> None:
    dataset = _SizedDataset(23)
    accumulation = 3
    split_iteration = 7
    tail_iterations = 5

    uninterrupted = GlobalIterationBatchSampler(
        dataset,
        batch_size=4,
        base_seed=3407,
        start_iteration=0,
        gradient_accumulation_steps=accumulation,
    )
    all_batches = _take_sampler_batches(
        uninterrupted,
        (split_iteration + tail_iterations) * accumulation,
    )
    resumed = GlobalIterationBatchSampler(
        dataset,
        batch_size=4,
        base_seed=3407,
        start_iteration=split_iteration,
        gradient_accumulation_steps=accumulation,
    )
    resumed_batches = _take_sampler_batches(
        resumed,
        tail_iterations * accumulation,
    )

    expected = all_batches[split_iteration * accumulation :]
    assert [_batch_identity(batch) for batch in resumed_batches] == [
        _batch_identity(batch) for batch in expected
    ]


def test_resume_safe_samples_do_not_depend_on_worker_count() -> None:
    dataset = _SizedDataset(32)

    def collect(num_workers: int) -> list[torch.Tensor]:
        sampler = GlobalIterationBatchSampler(
            dataset,
            batch_size=4,
            base_seed=3407,
            start_iteration=3,
            gradient_accumulation_steps=2,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers else None,
        )
        return [batch.clone() for batch in loader]

    single_process = collect(0)
    worker_processes = collect(2)
    assert len(single_process) == len(worker_processes)
    for expected, actual in zip(single_process, worker_processes):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_sample_local_rng_repeats_complete_lane_transform() -> None:
    pixels = np.arange(96 * 48 * 3, dtype=np.uint32).reshape(48, 96, 3)
    image = Image.fromarray((pixels % 256).astype(np.uint8), mode="RGB")
    lanes = [[(10.0, 15.0), (30.0, 30.0), (45.0, 42.0)]]
    transforms = LaneTransforms(
        TransformConfig(
            input_w=96,
            input_h=48,
            horizontal_flip_prob=0.5,
            color_jitter=True,
            channel_shuffle_prob=1.0,
            hue_saturation_prob=1.0,
            blur_prob=1.0,
            affine_prob=1.0,
            affine_translate_x=0.1,
            affine_translate_y=0.1,
            affine_rotate_deg=10.0,
            affine_scale_min=0.8,
            affine_scale_max=1.2,
            random_shadow_prob=1.0,
            random_shadow_roi_start_y=0.0,
        )
    )

    first = transforms(
        image,
        lanes,
        training=True,
        rng=np.random.RandomState(12345),
    )
    repeated = transforms(
        image,
        lanes,
        training=True,
        rng=np.random.RandomState(12345),
    )
    different = transforms(
        image,
        lanes,
        training=True,
        rng=np.random.RandomState(12346),
    )

    torch.testing.assert_close(first[0], repeated[0], rtol=0.0, atol=0.0)
    assert first[1:] == repeated[1:]
    assert not torch.equal(first[0], different[0])


def test_sample_local_random_state_preserves_legacy_transform_math() -> None:
    pixels = np.arange(64 * 32 * 3, dtype=np.uint32).reshape(32, 64, 3)
    image = Image.fromarray((pixels % 256).astype(np.uint8), mode="RGB")
    lanes = [[(8.0, 10.0), (20.0, 18.0), (32.0, 28.0)]]
    transforms = LaneTransforms(
        TransformConfig(
            input_w=64,
            input_h=32,
            horizontal_flip_prob=0.5,
            color_jitter=True,
            channel_shuffle_prob=0.5,
            hue_saturation_prob=0.7,
            blur_prob=0.2,
            affine_prob=0.7,
            affine_translate_x=0.1,
            affine_translate_y=0.1,
            affine_rotate_deg=10.0,
            affine_scale_min=0.8,
            affine_scale_max=1.2,
        )
    )

    np.random.seed(3407)
    legacy = transforms(image, lanes, training=True)
    sample_local = transforms(
        image,
        lanes,
        training=True,
        rng=np.random.RandomState(3407),
    )

    torch.testing.assert_close(legacy[0], sample_local[0], rtol=0.0, atol=0.0)
    assert legacy[1:] == sample_local[1:]


def _train_deterministic_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    start_iteration: int,
    iterations: int,
) -> None:
    dataset = _SizedDataset(20)
    accumulation = 2
    sampler = GlobalIterationBatchSampler(
        dataset,
        batch_size=2,
        base_seed=91,
        start_iteration=start_iteration,
        gradient_accumulation_steps=accumulation,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    loader_iterator = iter(loader)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(accumulation):
            try:
                batch = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(loader)
                batch = next(loader_iterator)
            features = torch.stack(
                (batch[:, 0] / 20.0, batch[:, 3]),
                dim=1,
            )
            target = (0.3 * batch[:, 0] - 0.7 * batch[:, 3]).unsqueeze(1)
            prediction = model(features)
            loss = torch.nn.functional.mse_loss(prediction, target)
            (loss / accumulation).backward()
        optimizer.step()


def test_model_update_is_exact_across_resume_boundary() -> None:
    torch.manual_seed(17)
    initial = torch.nn.Linear(2, 1, dtype=torch.float64)
    initial_state = copy.deepcopy(initial.state_dict())

    continuous = torch.nn.Linear(2, 1, dtype=torch.float64)
    continuous.load_state_dict(initial_state)
    continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=1e-3)
    _train_deterministic_steps(
        continuous,
        continuous_optimizer,
        start_iteration=0,
        iterations=8,
    )

    first_process = torch.nn.Linear(2, 1, dtype=torch.float64)
    first_process.load_state_dict(initial_state)
    first_optimizer = torch.optim.AdamW(first_process.parameters(), lr=1e-3)
    _train_deterministic_steps(
        first_process,
        first_optimizer,
        start_iteration=0,
        iterations=3,
    )
    checkpoint_model = copy.deepcopy(first_process.state_dict())
    checkpoint_optimizer = copy.deepcopy(first_optimizer.state_dict())

    resumed = torch.nn.Linear(2, 1, dtype=torch.float64)
    resumed.load_state_dict(checkpoint_model)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(checkpoint_optimizer)
    _train_deterministic_steps(
        resumed,
        resumed_optimizer,
        start_iteration=3,
        iterations=5,
    )

    for continuous_parameter, resumed_parameter in zip(
        continuous.parameters(), resumed.parameters()
    ):
        torch.testing.assert_close(
            resumed_parameter,
            continuous_parameter,
            rtol=0.0,
            atol=0.0,
        )
