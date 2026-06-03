from __future__ import annotations

import warnings

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.act(out + identity)


class ResNet34Backbone(nn.Module):
    """ResNet-34 feature extractor returning C2-C5 at PRD strides."""

    out_channels = {"c2": 64, "c3": 128, "c4": 256, "c5": 512}

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        self._init_weights()
        if pretrained:
            self._try_load_torchvision()

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _try_load_torchvision(self) -> None:
        try:
            from torchvision.models import ResNet34_Weights, resnet34

            tv = resnet34(weights=ResNet34_Weights.DEFAULT)
            self.stem[0].load_state_dict(tv.conv1.state_dict())
            self.stem[1].load_state_dict(tv.bn1.state_dict())
            self.layer1.load_state_dict(tv.layer1.state_dict(), strict=False)
            self.layer2.load_state_dict(tv.layer2.state_dict(), strict=False)
            self.layer3.load_state_dict(tv.layer3.state_dict(), strict=False)
            self.layer4.load_state_dict(tv.layer4.state_dict(), strict=False)
        except Exception as exc:  # pragma: no cover - depends on local deps/network cache.
            warnings.warn(f"Could not load torchvision ResNet-34 weights; using random init. Reason: {exc}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}

