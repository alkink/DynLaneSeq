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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        expanded_channels = out_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, expanded_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(expanded_channels)
        self.act = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != expanded_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.act(out + identity)


class ResNetBackbone(nn.Module):
    """ResNet-18/34/50/101 feature extractor returning C2-C5."""

    out_channels = {"c2": 64, "c3": 128, "c4": 256, "c5": 512}
    _depth_to_spec = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
    }

    def __init__(self, depth: int = 34, pretrained: bool = True, require_pretrained: bool = False):
        super().__init__()
        depth = int(depth)
        if depth not in self._depth_to_spec:
            raise ValueError(f"Unsupported ResNet depth: {depth}. Supported: {sorted(self._depth_to_spec)}")
        self.depth = depth
        self.require_pretrained = bool(require_pretrained)
        block, blocks = self._depth_to_spec[depth]
        self.out_channels = {
            "c2": 64 * block.expansion,
            "c3": 128 * block.expansion,
            "c4": 256 * block.expansion,
            "c5": 512 * block.expansion,
        }
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64, 64, blocks=blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64 * block.expansion, 128, blocks=blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128 * block.expansion, 256, blocks=blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256 * block.expansion, 512, blocks=blocks[3], stride=2)
        self._init_weights()
        if pretrained:
            self._try_load_torchvision()

    @staticmethod
    def _make_layer(
        block: type[nn.Module],
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [block(in_channels, out_channels, stride=stride)]
        expanded_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(expanded_channels, out_channels, stride=1))
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
            if self.depth == 18:
                from torchvision.models import ResNet18_Weights, resnet18

                tv = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif self.depth == 34:
                from torchvision.models import ResNet34_Weights, resnet34

                tv = resnet34(weights=ResNet34_Weights.DEFAULT)
            elif self.depth == 50:
                from torchvision.models import ResNet50_Weights, resnet50

                tv = resnet50(weights=ResNet50_Weights.DEFAULT)
            elif self.depth == 101:
                from torchvision.models import ResNet101_Weights, resnet101

                tv = resnet101(weights=ResNet101_Weights.DEFAULT)
            else:  # Constructor validation makes this unreachable.
                raise ValueError(f"Unsupported ResNet depth: {self.depth}")
            self.stem[0].load_state_dict(tv.conv1.state_dict())
            self.stem[1].load_state_dict(tv.bn1.state_dict())
            self.layer1.load_state_dict(tv.layer1.state_dict(), strict=False)
            self.layer2.load_state_dict(tv.layer2.state_dict(), strict=False)
            self.layer3.load_state_dict(tv.layer3.state_dict(), strict=False)
            self.layer4.load_state_dict(tv.layer4.state_dict(), strict=False)
        except Exception as exc:  # pragma: no cover - depends on local deps/network cache.
            msg = f"Could not load torchvision ResNet-{self.depth} weights; using random init. Reason: {exc}"
            if self.require_pretrained:
                raise RuntimeError(msg) from exc
            warnings.warn(msg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


class ResNet18Backbone(ResNetBackbone):
    def __init__(self, pretrained: bool = True, require_pretrained: bool = False):
        super().__init__(depth=18, pretrained=pretrained, require_pretrained=require_pretrained)


class ResNet34Backbone(ResNetBackbone):
    def __init__(self, pretrained: bool = True, require_pretrained: bool = False):
        super().__init__(depth=34, pretrained=pretrained, require_pretrained=require_pretrained)


class ResNet101Backbone(ResNetBackbone):
    def __init__(self, pretrained: bool = True, require_pretrained: bool = False):
        super().__init__(depth=101, pretrained=pretrained, require_pretrained=require_pretrained)
