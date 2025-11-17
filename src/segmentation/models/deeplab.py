from __future__ import annotations

from typing import Dict

import segmentation_models_pytorch as smp
import torch
from torch import nn


class DeepLabWrapper(nn.Module):
    """segmentation_models_pytorch DeepLabV3 wrapper."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 21,
        pretrained: bool = True,
        encoder_weights: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        weights = encoder_weights
        if weights is None:
            weights = "imagenet" if pretrained else None
        self.model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.model(x)
        return {"main": logits, "aux": None}

    def train(self, mode: bool = True) -> "DeepLabWrapper":
        super().train(mode)
        if self.freeze_encoder:
            self.model.encoder.eval()
        if mode:
            self._set_bn_eval()
        return self

    def _set_bn_eval(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
