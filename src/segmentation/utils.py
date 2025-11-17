from __future__ import annotations

import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data.voc import VOC_CLASSES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_visualizations(images: torch.Tensor, preds: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = min(len(images), 4)
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(preds_np[i], cmap="tab20", vmin=0, vmax=len(VOC_CLASSES) - 1)
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(targets_np[i], cmap="tab20", vmin=0, vmax=len(VOC_CLASSES) - 1)
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def topk_examples(metric_map: List[float], k: int = 3) -> List[int]:
    args = np.argsort(metric_map)
    best = args[-k:][::-1]
    worst = args[:k]
    return list(best), list(worst)
