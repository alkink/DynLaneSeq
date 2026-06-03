from __future__ import annotations

from .loss_s2 import S2Criterion, S2LossConfig


class S3Criterion(S2Criterion):
    def __init__(self, cfg: S2LossConfig | None = None):
        super().__init__(cfg)

