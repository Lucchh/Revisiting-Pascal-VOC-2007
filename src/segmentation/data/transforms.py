from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as T

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:  # pragma: no cover
    A = None
    ToTensorV2 = None


@dataclass
class TransformParams:
    image_size: int = 256
    color_jitter: bool = True
    random_flip: bool = True
    random_crop: bool = True
    backend: str = "torchvision"
    enabled: bool = True


class SegmentationAugmenter:
    """Applies identical spatial transforms to image and mask."""

    def __init__(self, params: TransformParams, training: bool = True) -> None:
        self.params = params
        self.training = training
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            img, mask = self._apply_training_transforms(img, mask)
        else:
            img, mask = self._apply_eval_transforms(img, mask)
        img_tensor, mask_tensor = self._to_tensor(img, mask)
        img_tensor = self.normalize(img_tensor)
        return img_tensor, mask_tensor

    def _apply_training_transforms(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.params.random_crop:
            img, mask = _random_resized_crop(img, mask, self.params.image_size)
        else:
            img = img.resize((self.params.image_size, self.params.image_size), Image.BILINEAR)
            mask = mask.resize((self.params.image_size, self.params.image_size), Image.NEAREST)

        if self.params.random_flip and random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle, fill=0)

        if self.params.color_jitter:
            jitter_range = 0.3
            hue_range = 0.05
            img = F.adjust_brightness(img, 1 + (random.random() * 2 - 1) * jitter_range)
            img = F.adjust_contrast(img, 1 + (random.random() * 2 - 1) * jitter_range)
            img = F.adjust_saturation(img, 1 + (random.random() * 2 - 1) * jitter_range)
            img = F.adjust_hue(img, (random.random() * 2 - 1) * hue_range)
        return img, mask

    def _apply_eval_transforms(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        img = img.resize((self.params.image_size, self.params.image_size), Image.BILINEAR)
        mask = mask.resize((self.params.image_size, self.params.image_size), Image.NEAREST)
        return img, mask

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        return arr.permute(2, 0, 1)

    def _to_tensor(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img_tensor = self._pil_to_tensor(img)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_tensor, mask_tensor


def _random_resized_crop(img: Image.Image, mask: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
    scale = random.uniform(0.5, 1.0)
    width, height = img.size
    new_w, new_h = int(width * scale), int(height * scale)
    if new_w < 1 or new_h < 1:
        return img, mask
    left = random.randint(0, width - new_w)
    top = random.randint(0, height - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    mask = mask.crop((left, top, left + new_w, top + new_h))
    img = img.resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.NEAREST)
    return img, mask


class AlbumentationsAugmenter:
    """Albumentations-based pipeline mirroring the stronger reference augmentations."""

    def __init__(self, params: TransformParams, training: bool = True) -> None:
        if A is None or ToTensorV2 is None:
            raise ImportError("albumentations is not installed. Install it or switch backend to torchvision.")
        self.training = training
        if training and params.enabled:
            self.pipeline = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Affine(scale=(0.7, 1.3), translate_percent=(0.0, 0.1), rotate=(-10, 10), p=1.0),
                    A.PadIfNeeded(min_height=params.image_size, min_width=params.image_size, border_mode=0, p=1.0),
                    A.RandomCrop(height=params.image_size, width=params.image_size, p=1.0),
                    A.GaussNoise(p=0.2),
                    A.Perspective(p=0.5),
                    A.OneOf(
                        [
                            A.CLAHE(p=1.0),
                            A.RandomBrightnessContrast(p=1.0),
                            A.RandomGamma(p=1.0),
                        ],
                        p=0.9,
                    ),
                    A.Resize(height=params.image_size, width=params.image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.pipeline = A.Compose(
                [
                    A.Resize(height=params.image_size, width=params.image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np = np.array(img)
        mask_np = np.array(mask)
        transformed = self.pipeline(image=img_np, mask=mask_np)
        img_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"].long()
        return img_tensor, mask_tensor
