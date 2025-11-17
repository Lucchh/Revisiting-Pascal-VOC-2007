from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np
import torch
from scipy import ndimage


@dataclass
class MetricState:
    num_classes: int
    confusion_matrix: torch.Tensor = field(init=False)
    dice_scores: Dict[int, list] = field(default_factory=dict)
    hd95_scores: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)
        self.dice_scores = {i: [] for i in range(self.num_classes)}

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1)
        self._update_confusion(preds, targets)
        self._update_dice(preds, targets)
        self._update_hd95(preds, targets)

    def _update_confusion(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        with torch.no_grad():
            preds = preds.view(-1)
            targets = targets.view(-1)
            mask = (targets >= 0) & (targets < self.num_classes)
            indices = self.num_classes * targets[mask] + preds[mask]
            cm = torch.bincount(indices, minlength=self.num_classes ** 2)
            cm = cm.reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += cm.cpu()

    def _update_dice(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        for cls in range(self.num_classes):
            pred_cls = preds == cls
            target_cls = targets == cls
            intersection = (pred_cls & target_cls).sum().item()
            union = pred_cls.sum().item() + target_cls.sum().item()
            if union == 0:
                continue
            dice = (2 * intersection) / union
            self.dice_scores[cls].append(dice)

    def _update_hd95(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        for cls in np.unique(targets_np):
            if cls < 0 or cls >= self.num_classes:
                continue
            pred_mask = preds_np == cls
            target_mask = targets_np == cls
            hd = hausdorff_distance_95(pred_mask, target_mask)
            if np.isfinite(hd):
                self.hd95_scores.append(hd)

    def compute(self) -> Dict[str, torch.Tensor | float]:
        cm = self.confusion_matrix
        eps = 1e-6
        tp = cm.diag()
        per_class_iou = tp / (cm.sum(1) + cm.sum(0) - tp + eps)
        per_class_acc = tp / (cm.sum(1) + eps)
        miou = per_class_iou.mean().item()
        pixel_acc = tp.sum().item() / (cm.sum().item() + eps)
        mean_dice = np.mean([np.mean(v) for v in self.dice_scores.values() if v])
        hd95 = float(np.mean(self.hd95_scores)) if self.hd95_scores else float("nan")
        return {
            "miou": miou,
            "pixel_acc": pixel_acc,
            "per_class_iou": per_class_iou,
            "per_class_acc": per_class_acc,
            "mean_dice": mean_dice,
            "hd95": hd95,
        }


def hausdorff_distance_95(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    if pred_mask.sum() == 0 and target_mask.sum() == 0:
        return 0.0
    if pred_mask.sum() == 0 or target_mask.sum() == 0:
        return float("inf")
    pred_dist = ndimage.distance_transform_edt(~pred_mask)
    target_dist = ndimage.distance_transform_edt(~target_mask)
    surface_pred = pred_dist[target_mask]
    surface_target = target_dist[pred_mask]
    if surface_pred.size == 0 or surface_target.size == 0:
        return float("inf")
    distances = np.concatenate([surface_pred, surface_target], axis=0)
    return float(np.percentile(distances, 95))


def summarize_metrics(states: Iterable[MetricState]) -> Dict[str, float]:
    aggregated = {}
    for state in states:
        result = state.compute()
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                value = value.mean().item()
            aggregated.setdefault(key, []).append(value)
    return {k: float(np.nanmean(v)) for k, v in aggregated.items()}
