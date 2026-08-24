from __future__ import annotations

import numpy as np
import torch

from dynlaneseq_eg.modeling.v35_raw_rgb_scorer import (
    CandidateRibbonScorer,
    matched_arm_input,
    v35_pair_loss,
)
from dynlaneseq_eg.tools.audit_v34_temporal_candidate_observability import (
    TemporalSample,
)
from dynlaneseq_eg.tools.audit_v35_raw_rgb_temporal_observability import (
    _geometry_features,
    _sample_ribbon,
    build_learning_pairs,
)


def test_matched_arms_keep_identical_shape_and_single_repeats_target() -> None:
    ribbons = torch.arange(2 * 18 * 4 * 3, dtype=torch.float32).view(2, 18, 4, 3)
    geometry_only = matched_arm_input(ribbons, "G")
    single = matched_arm_input(ribbons, "S")
    temporal = matched_arm_input(ribbons, "T")
    assert geometry_only.shape == single.shape == temporal.shape == ribbons.shape
    assert torch.count_nonzero(geometry_only) == 0
    assert torch.equal(single[:, :6], ribbons[:, 6:12])
    assert torch.equal(single[:, 6:12], ribbons[:, 6:12])
    assert torch.equal(single[:, 12:18], ribbons[:, 6:12])
    assert torch.equal(temporal, ribbons)


def test_candidate_scorer_and_pair_loss_backpropagate() -> None:
    torch.manual_seed(3)
    model = CandidateRibbonScorer()
    ribbons = torch.randn(4, 18, 64, 25)
    geometry = torch.randn(4, 4, 64)
    scores = model(ribbons, geometry, arm="T")
    total, diagnostics = v35_pair_loss(
        scores[:2],
        scores[2:],
        torch.tensor([0.9, 0.8]),
        torch.tensor([0.2, 0.4]),
    )
    total.backward()
    assert scores.shape == (4,)
    assert float(diagnostics["loss_total"]) > 0.0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_raw_ribbon_sampling_masks_invalid_rows() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(30, dtype=np.uint8)[None]
    x = np.asarray([[10.0, 12.0], [20.0, 22.0]], dtype=np.float32)
    y = np.asarray([[5.0, 7.0], [8.0, 9.0]], dtype=np.float32)
    valid = np.asarray([[True, False], [True, True]])
    sampled = _sample_ribbon(image, x, y, valid, radius=2.0, samples=5)
    assert sampled.shape == (2, 3, 2, 5)
    assert np.count_nonzero(sampled[0, :, 1]) == 0
    assert np.count_nonzero(sampled[1, 0, 0]) > 0


def test_learning_pairs_use_oracle_good_and_current_wrong() -> None:
    image_id = "driver/clip/00030.jpg"
    stage = {
        "official_iou": torch.tensor([[0.20, 0.90, 0.45]]),
        "official_candidate_valid": torch.tensor([True, True, True]),
        "selection_slot_candidate_valid": torch.tensor([True, True, True]),
        "selection_slot_indices": torch.tensor([0]),
        "selection_slot_official_iou": torch.tensor([[0.20]]),
        "selection_slot_active": torch.tensor([True]),
        "selection_slot_official_candidate_valid": torch.tensor([True]),
    }
    records = {image_id: {"stages": {"main": stage}}}
    sample = TemporalSample(
        fold="a",
        clip="driver/clip",
        target=image_id,
        previous="driver/clip/00000.jpg",
        following="driver/clip/00060.jpg",
        wrong_context="other/clip/00030.jpg",
    )
    evaluation_row = {
        "fold": "a",
        "clip": "driver/clip",
        "image_id": image_id,
        "slot_id": 0,
        "gt_id": 0,
        "good_candidate": 1,
        "wrong_candidate": 0,
        "good_iou": 0.90,
        "wrong_iou": 0.20,
        "thresholds": [0.50, 0.75],
    }
    learning, evaluation = build_learning_pairs(
        records, {image_id: sample}, [evaluation_row]
    )
    assert len(learning) == len(evaluation) == 1
    assert learning[0].good_candidate == 1
    assert learning[0].wrong_candidate == 0
    assert learning[0].thresholds == (0.50, 0.75)
    assert evaluation[0].evaluation_pair is True


def test_geometry_features_are_finite_and_masked() -> None:
    x = np.asarray([[10.0, 11.0, 13.0, 16.0]], dtype=np.float32)
    masks = np.asarray([[True, True, False, True]])
    geometry = _geometry_features(
        x, masks, np.asarray([0, 1, 2, 3]), input_w=100
    )
    assert geometry.shape == (1, 4, 4)
    assert np.isfinite(geometry).all()
    assert np.count_nonzero(geometry[0, :3, 2]) == 0
    assert geometry[0, 3, 2] == 0
