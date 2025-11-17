"""Quick dataset sanity check utility for Pascal VOC 2007 segmentation."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

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


def make_dataloaders(root: Path, size: int = 256, batch_size: int = 4):
    transform_img = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    transform_target = transforms.Compose(
        [transforms.Resize((size, size)), transforms.PILToTensor()]
    )
    train = VOCSegmentation(
        root=str(root),
        year="2007",
        image_set="train",
        download=False,
        transform=transform_img,
        target_transform=transform_target,
    )
    val = VOCSegmentation(
        root=str(root),
        year="2007",
        image_set="val",
        download=False,
        transform=transform_img,
        target_transform=transform_target,
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
    )


def show_sample(img: torch.Tensor, mask: torch.Tensor) -> None:
    img = img.permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy().copy()
    mask[mask > 20] = 0
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    seg_map = plt.imshow(mask, cmap="tab20", vmin=0, vmax=20)
    plt.title("Mask")
    plt.axis("off")
    cbar = plt.colorbar(seg_map, ticks=range(21))
    cbar.ax.set_yticklabels([f"{i} – {name}" for i, name in enumerate(VOC_CLASSES)])
    plt.tight_layout()
    plt.show()


def main() -> None:
    root = Path("./VOCtrainval_06-Nov-2007")
    train_loader, val_loader = make_dataloaders(root)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    images, masks = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)
    idx = random.randint(0, images.size(0) - 1)
    show_sample(images[idx], masks[idx])
    print("Classes in mask:", np.unique(masks[idx].numpy()))


if __name__ == "__main__":
    main()
