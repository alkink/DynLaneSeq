from __future__ import annotations

import os
from pathlib import Path
import warnings

import torch
from torch import nn


_DLA34_FILENAME = "dla34-ba72cf86.pth"
_DLA34_URLS = (
    "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth",
    "https://huggingface.co/datasets/ckevar/fairmot_models/resolve/main/dla34-ba72cf86.pth",
)


class DLABasicBlock(nn.Module):
    """Basic residual block used by the canonical DLA-34 architecture."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is None:
            residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DLARoot(nn.Module):
    """Fuse the branches collected by one hierarchical DLA tree."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, residual: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.residual = bool(residual)

    def forward(self, *branches: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.conv(torch.cat(branches, dim=1)))
        if self.residual:
            out = out + branches[0]
        return self.relu(out)


class DLATree(nn.Module):
    """Recursive iterative/deep aggregation tree from the original DLA-34."""

    def __init__(
        self,
        levels: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        level_root: bool = False,
        root_dim: int = 0,
        root_kernel_size: int = 1,
        dilation: int = 1,
        root_residual: bool = False,
    ):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        if levels == 1:
            self.tree1 = DLABasicBlock(in_channels, out_channels, stride=stride, dilation=dilation)
            self.tree2 = DLABasicBlock(out_channels, out_channels, stride=1, dilation=dilation)
            self.root = DLARoot(root_dim, out_channels, root_kernel_size, root_residual)
        else:
            self.tree1 = DLATree(
                levels - 1,
                in_channels,
                out_channels,
                stride=stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = DLATree(
                levels - 1,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )

        self.level_root = bool(level_root)
        self.levels = int(levels)
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=0.1),
            )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        children: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            return self.root(x2, x1, *children)
        children.append(x1)
        return self.tree2(x1, children=children)


class DLA34Backbone(nn.Module):
    """ImageNet-pretrained DLA-34 returning an FPN-compatible C2-C5 pyramid.

    The returned levels match the existing ResNet-34 contract exactly:
    C2/C3/C4/C5 have strides 4/8/16/32 and channels 64/128/256/512.
    The implementation follows the official DLA architecture and the wrappers
    used by CLRNet and CondLSTR, while leaving DynLaneSeq's FPN and head intact.
    """

    levels = (1, 1, 1, 2, 2, 1)
    channels = (16, 32, 64, 128, 256, 512)
    out_channels = {"c2": 64, "c3": 128, "c4": 256, "c5": 512}

    def __init__(
        self,
        pretrained: bool = True,
        require_pretrained: bool = False,
        weights_path: str | Path | None = None,
    ):
        super().__init__()
        self.require_pretrained = bool(require_pretrained)
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, self.channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.channels[0], momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(self.channels[0], self.channels[0], self.levels[0])
        self.level1 = self._make_conv_level(
            self.channels[0], self.channels[1], self.levels[1], stride=2
        )
        self.level2 = DLATree(self.levels[2], self.channels[1], self.channels[2], stride=2)
        self.level3 = DLATree(
            self.levels[3], self.channels[2], self.channels[3], stride=2, level_root=True
        )
        self.level4 = DLATree(
            self.levels[4], self.channels[3], self.channels[4], stride=2, level_root=True
        )
        self.level5 = DLATree(
            self.levels[5], self.channels[4], self.channels[5], stride=2, level_root=True
        )
        self._init_weights()
        if pretrained:
            try:
                self._load_pretrained(weights_path)
            except Exception as exc:  # pragma: no cover - depends on cache/network.
                msg = f"Could not load ImageNet DLA-34 weights; using random init. Reason: {exc}"
                if self.require_pretrained:
                    raise RuntimeError(msg) from exc
                warnings.warn(msg)

    @staticmethod
    def _make_conv_level(
        in_channels: int,
        out_channels: int,
        convs: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        modules: list[nn.Module] = []
        for index in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride if index == 0 else 1,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels, momentum=0.1),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        return nn.Sequential(*modules)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility.
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"DLA-34 checkpoint must contain a state dict, got {type(state)!r}")
        return {
            (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
            for key, value in state.items()
        }

    def _load_pretrained(self, weights_path: str | Path | None) -> None:
        configured_path = weights_path or os.environ.get("DYNLANESEQ_DLA34_WEIGHTS", "")
        if configured_path:
            path = Path(configured_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"DLA-34 pretrained weights not found: {path}")
            state = self._load_state_dict_file(path)
        else:
            state = None
            errors: list[str] = []
            for url in _DLA34_URLS:
                try:
                    state = torch.hub.load_state_dict_from_url(
                        url,
                        map_location="cpu",
                        progress=True,
                        check_hash=True,
                        file_name=_DLA34_FILENAME,
                    )
                    break
                except Exception as exc:  # pragma: no cover - network dependent.
                    errors.append(f"{url}: {exc}")
            if state is None:
                raise RuntimeError("; ".join(errors))
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {
                (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
                for key, value in state.items()
            }

        incompatible = self.load_state_dict(state, strict=False)
        bad_missing = [key for key in incompatible.missing_keys if not key.endswith("num_batches_tracked")]
        bad_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("fc.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Incompatible DLA-34 pretrained weights: "
                f"missing={bad_missing}, unexpected={bad_unexpected}"
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        c2 = self.level2(x)
        c3 = self.level3(c2)
        c4 = self.level4(c3)
        c5 = self.level5(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
