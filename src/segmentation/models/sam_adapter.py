from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "segment-anything is required for SAMAdapter. Install via requirements.txt"
    ) from exc


class SAMAdapter(nn.Module):
    """Lightweight decoder on top of SAM's frozen image encoder."""

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        num_classes: int = 21,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        if freeze_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        self.resize = ResizeLongestSide(self.sam.image_encoder.img_size)
        enc_dim = getattr(self.sam.image_encoder, "embed_dim", None)
        if enc_dim is None and hasattr(self.sam.image_encoder, "neck"):
            enc_dim = getattr(self.sam.image_encoder.neck[-1], "out_channels", None)  # type: ignore[index]
        if enc_dim is None:
            enc_dim = getattr(self.sam.image_encoder, "out_channels", 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(enc_dim, enc_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_dim // 2, enc_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_dim // 4, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = next(self.sam.parameters()).device
        resized = self._prepare_images(x).to(device)
        features = self.sam.image_encoder(resized)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"main": logits}

    def _prepare_images(self, x: torch.Tensor) -> torch.Tensor:
        batch = []
        for img in x:
            img_np = img.detach().cpu().numpy()
            img_np = (img_np * 255.0).transpose(1, 2, 0)
            resized = self.resize.apply_image(img_np)
            resized = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            batch.append(resized)
        batch_tensor = torch.stack(batch, dim=0)
        return batch_tensor
