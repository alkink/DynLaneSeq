from __future__ import annotations

import pytest
import torch
from torch import nn

from dynlaneseq_eg.engine.checkpoint import remap_optimizer_state_by_parameter
from dynlaneseq_eg.factory import build_optimizer


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(4, 4)
        self.encoder.fpn = nn.Linear(4, 4)
        self.encoder.proj = nn.Linear(4, 4)
        self.structured_query_head = nn.Module()
        self.structured_query_head.layers = nn.ModuleList(
            [nn.Linear(4, 4), nn.Linear(4, 4)]
        )
        self.structured_query_head.row_x = nn.Linear(4, 4)


def _parameters_by_group(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, set[str]]:
    names = {parameter: name for name, parameter in model.named_parameters()}
    return {
        str(group["name"]): {names[parameter] for parameter in group["params"]}
        for group in optimizer.param_groups
    }


def test_custom_optimizer_parameter_groups_override_legacy_groups() -> None:
    model = _ToyModel()
    cfg = {
        "optimizer": {
            "base_lr": 1e-5,
            "backbone_lr": 1e-6,
            "evidence_lr": 1e-5,
            "parameter_groups": [
                {
                    "name": "visual_neck",
                    "lr": 2e-6,
                    "prefixes": ["encoder.fpn.", "encoder.proj."],
                },
                {
                    "name": "decoder_l1",
                    "lr": 2e-6,
                    "prefixes": ["structured_query_head.layers.0."],
                },
                {
                    "name": "prediction_heads",
                    "lr": 1e-5,
                    "prefixes": ["structured_query_head.row_x."],
                },
            ],
        }
    }

    optimizer = build_optimizer(cfg, model)
    groups = _parameters_by_group(model, optimizer)

    assert groups["visual_neck_decay"] == {
        "encoder.fpn.weight",
        "encoder.proj.weight",
    }
    assert groups["decoder_l1_decay"] == {
        "structured_query_head.layers.0.weight"
    }
    assert groups["prediction_heads_decay"] == {
        "structured_query_head.row_x.weight"
    }
    grouped_parameters = {
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert grouped_parameters == set(model.parameters())


def test_custom_optimizer_parameter_groups_reject_overlap() -> None:
    model = _ToyModel()
    cfg = {
        "optimizer": {
            "parameter_groups": [
                {
                    "name": "structured",
                    "lr": 1e-5,
                    "prefixes": ["structured_query_head."],
                },
                {
                    "name": "layer_one",
                    "lr": 2e-6,
                    "prefixes": ["structured_query_head.layers.0."],
                },
            ]
        }
    }

    with pytest.raises(ValueError, match="multiple custom optimizer groups"):
        build_optimizer(cfg, model)


def test_optimizer_state_remap_preserves_moments_and_new_group_lrs() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    source = torch.optim.AdamW(model.parameters(), lr=1e-4)
    source.zero_grad(set_to_none=True)
    model(torch.ones(2, 4)).sum().backward()
    source.step()

    parameters = list(model.parameters())
    expected_exp_avg = {
        parameter: source.state[parameter]["exp_avg"].clone()
        for parameter in parameters
    }
    target = torch.optim.AdamW(
        [
            {"params": parameters[:2], "lr": 2e-6, "name": "slow"},
            {"params": parameters[2:], "lr": 1e-5, "name": "fast"},
        ]
    )

    stats = remap_optimizer_state_by_parameter(source, target)

    assert stats == {
        "parameters": len(parameters),
        "state_entries": len(parameters),
        "source_groups": 1,
        "target_groups": 2,
    }
    assert target.param_groups[0]["lr"] == 2e-6
    assert target.param_groups[1]["lr"] == 1e-5
    for parameter in parameters:
        torch.testing.assert_close(
            target.state[parameter]["exp_avg"],
            expected_exp_avg[parameter],
        )
