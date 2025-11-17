from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass
class DataModule:
    train_loader: DataLoader
    val_loader: DataLoader


def _build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    image_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    def mask_transform(mask):
        mask = TF.resize(mask, (image_size, image_size), interpolation=InterpolationMode.NEAREST)
        mask_np = np.array(mask, dtype=np.int64, copy=True)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        invalid = (mask_tensor < 0) | (mask_tensor >= len(VOC_CLASSES))
        mask_tensor[invalid] = 255
        return mask_tensor.contiguous()

    return image_tf, transforms.Lambda(mask_transform)


def create_datasets(
    root: str,
    image_size: int,
    train_split: str,
    val_split: str,
) -> Tuple[VOCSegmentation, VOCSegmentation]:
    img_tf, mask_tf = _build_transforms(image_size)
    train_dataset = VOCSegmentation(
        root=root,
        year="2007",
        image_set=train_split,
        download=False,
        transform=img_tf,
        target_transform=mask_tf,
    )
    val_dataset = VOCSegmentation(
        root=root,
        year="2007",
        image_set=val_split,
        download=False,
        transform=img_tf,
        target_transform=mask_tf,
    )
    return train_dataset, val_dataset


def build_dataloaders(
    root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train_split: str,
    val_split: str,
    augmentations: Dict,
) -> DataModule:
    train_dataset, val_dataset = create_datasets(root, image_size, train_split, val_split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return DataModule(train_loader=train_loader, val_loader=val_loader)


def show_sample(img: torch.Tensor, mask: torch.Tensor) -> None:
    img = img.permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy().copy()
    mask[mask > len(VOC_CLASSES) - 1] = 0
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Input Image")
    plt.subplot(1, 2, 2)
    seg_plot = plt.imshow(mask, cmap="tab20", vmin=0, vmax=len(VOC_CLASSES) - 1)
    plt.axis("off")
    plt.title("Segmentation Mask")
    cbar = plt.colorbar(seg_plot, ticks=range(len(VOC_CLASSES)))
    cbar.ax.set_yticklabels([f"{i} – {name}" for i, name in enumerate(VOC_CLASSES)])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = Path("../VOCtrainval_06-Nov-2007")
    train_dataset, _ = create_datasets(str(root), 256, "train", "val")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, masks = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)
    show_sample(images[0], masks[0])
    print("Classes in mask:", np.unique(masks[0].numpy()))
