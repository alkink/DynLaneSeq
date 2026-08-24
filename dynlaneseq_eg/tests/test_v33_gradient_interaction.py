from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import V25LossWeights
from dynlaneseq_eg.tools.audit_v33_primary_aux_gradient_interaction import (
    _apply_normalized_direction,
    _restore_parameters,
    gradient_interaction_metrics,
    split_primary_auxiliary_weights,
)


def test_split_weights_isolates_primary_and_auxiliary_objectives() -> None:
    weights = V25LossWeights(
        existence=2.0,
        row_distribution=5.0,
        point=1.0,
        strip_iou=2.0,
        range=1.0,
        quality50=0.0,
        quality75=0.0,
        smoothness=0.25,
        order=0.25,
        duplicate=0.25,
        visibility=0.0,
        proposal_coverage=1.0,
    )
    primary, auxiliary = split_primary_auxiliary_weights(weights)
    assert primary.proposal_coverage == 0.0
    assert primary.row_distribution == 5.0
    assert auxiliary.proposal_coverage == 1.0
    for name in (
        "existence",
        "row_distribution",
        "point",
        "strip_iou",
        "range",
        "quality50",
        "quality75",
        "smoothness",
        "order",
        "duplicate",
        "visibility",
        "tail_emphasis",
    ):
        assert getattr(auxiliary, name) == 0.0, name


def test_gradient_metrics_detect_aligned_and_opposed_directions() -> None:
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    named = [
        ("detector.backbone.level0.weight", first),
        ("detector.fpn.lateral.0.weight", second),
    ]
    primary = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]))
    aligned = gradient_interaction_metrics(named, primary, primary)
    assert abs(float(aligned["all_shared"]["cosine"]) - 1.0) < 1.0e-7
    assert float(aligned["all_shared"]["negative_tensor_fraction"]) == 0.0

    auxiliary = (-primary[0], -primary[1])
    opposed = gradient_interaction_metrics(named, primary, auxiliary)
    assert abs(float(opposed["all_shared"]["cosine"]) + 1.0) < 1.0e-7
    assert float(opposed["all_shared"]["negative_tensor_fraction"]) == 1.0
    assert float(opposed["all_shared"]["negative_dot_energy_fraction"]) == 1.0


def test_normalized_virtual_direction_has_fixed_relative_norm_and_restores() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameters = [parameter]
    originals = [parameter.detach().clone()]
    gradient = (torch.tensor([3.0, 4.0]),)
    scale = _apply_normalized_direction(
        parameters,
        originals,
        gradient,
        relative_step=0.10,
        parameter_norm=5.0,
    )
    assert abs(scale - 0.10) < 1.0e-7
    assert torch.allclose(parameter, torch.tensor([2.7, 3.6]))
    assert abs(float(torch.linalg.vector_norm(parameter - originals[0])) - 0.5) < 1.0e-6
    _restore_parameters(parameters, originals)
    assert torch.equal(parameter, originals[0])
