from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sam2.build_sam import build_sam2
except ImportError as exc:
    raise ImportError(
        "SAM-2 is required for SAM2Adapter. Install it via `pip install -e external/sam2`."
    ) from exc


class SAM2Adapter(nn.Module):
    """Improved decoder head on top of a SAM-2 image encoder."""

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        num_classes: int = 21,
        freeze_encoder: bool = True,
        backbone_device: Optional[str] = None,
        fine_tune_last_blocks: bool = True,
    ) -> None:
        super().__init__()

        # -----------------------------
        # Load SAM2 backbone
        # -----------------------------
        cfg_path = model_cfg
        if not cfg_path.startswith("configs/"):
            cfg_path = f"configs/{cfg_path.lstrip('./')}"
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

        if backbone_device is None:
            backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone_device = torch.device(backbone_device)

        self.sam2 = build_sam2(
            config_file=cfg_path,
            ckpt_path=str(ckpt_path),
            device=str(self.backbone_device),
            apply_postprocessing=False,
        )

        # -----------------------------
        # Encoder freezing / fine-tuning
        # -----------------------------
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for name, param in self.sam2.named_parameters():
                param.requires_grad = False

        # Partial fine-tuning (optional)
        elif fine_tune_last_blocks:
            for name, param in self.sam2.named_parameters():
                if any(k in name for k in ["blocks[-1]", "neck", "head"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # -----------------------------
        # Decoder definition (deeper head)
        # -----------------------------
        self.image_size = getattr(self.sam2, "image_size", 1024)
        hidden_dim = getattr(self.sam2, "hidden_dim", 256)

        # 3-stage upsampling decoder (learnable)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 2, stride=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 2, stride=2),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim // 4, num_classes, 1)
        )

        # -----------------------------
        # Normalization buffers
        # -----------------------------
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    # --------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig_size = x.shape[-2:]
        decoder_device = x.device

        # Resize + normalize
        inputs = self._prepare_images(x.to(self.backbone_device))

        # Backbone forward
        with torch.set_grad_enabled(not self.freeze_encoder):
            backbone_out = self.sam2.forward_image(inputs)
            _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)

            feat_h, feat_w = feat_sizes[-1]
            feat = vision_feats[-1].permute(1, 2, 0).reshape(x.size(0), -1, feat_h, feat_w)

            if self.freeze_encoder:
                feat = feat.detach()

        # Gradient device sync
        if not self.freeze_encoder and self.backbone_device != decoder_device:
            raise RuntimeError(
                "SAM2 encoder gradients require encoder and decoder on the same device. "
                "Either freeze encoder or match backbone_device to trainer device."
            )

        # Decode
        feat = feat.to(decoder_device, non_blocking=True)
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
        return {"main": logits}

    # --------------------------------------------------------------------------
    def train(self, mode: bool = True) -> "SAM2Adapter":
        super().train(mode)
        if self.freeze_encoder:
            self.sam2.eval()
        else:
            self.sam2.train(mode)
        return self

    # --------------------------------------------------------------------------
    def _prepare_images(self, x: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        mean = self.mean.to(resized.device)
        std = self.std.to(resized.device)
        return (resized - mean) / std

    # --------------------------------------------------------------------------
    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self.sam2.to(self.backbone_device)
        return module