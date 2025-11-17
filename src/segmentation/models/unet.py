from __future__ import annotations

from typing import Dict, Sequence

import segmentation_models_pytorch as smp
import torch
from torch import nn


class ResNetUNet(nn.Module):
    """Thin wrapper around segmentation_models_pytorch Unet."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        num_classes: int = 21,
        pretrained: bool = True,
        decoder_channels: Sequence[int] | None = None,
        encoder_weights: str | None = None,
        encoder_freeze: bool = False,
    ) -> None:
        super().__init__()
        weights = encoder_weights
        if weights is None:
            weights = "imagenet" if pretrained else None
        decoder_kwargs = {}
        if decoder_channels is not None:
            decoder_kwargs["decoder_channels"] = tuple(decoder_channels)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
            **decoder_kwargs,
        )
        if encoder_freeze:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        self.encoder_freeze = encoder_freeze

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.model(x)
        return {"main": logits}
