#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from segmentation.config import ExperimentConfig
from segmentation.data.voc import VOC_CLASSES
from segmentation.engine import Trainer, build_model
from segmentation.utils import save_visualizations, set_seed, topk_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create qualitative visualizations")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--k", type=int, default=3, help="Number of best/worst samples")
    return parser.parse_args()


def sample_mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        denom = target_mask.sum() + pred_mask.sum() - (pred_mask & target_mask).sum()
        if denom == 0:
            continue
        iou = (pred_mask & target_mask).sum().item() / denom.item()
        ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    model_cfg = next((m for m in cfg.models if m.name == args.model_name), None)
    if model_cfg is None:
        raise ValueError(f"Model {args.model_name} not found")
    model = build_model(model_cfg, num_classes=len(VOC_CLASSES))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(trainer.device)
    model.eval()

    samples = []
    with torch.inference_mode():
        for images, masks in trainer.dataloaders.val_loader:
            images = images.to(trainer.device)
            outputs = model(images)
            preds = torch.argmax(outputs["main"], dim=1).cpu()
            for i in range(images.size(0)):
                score = sample_mean_iou(preds[i], masks[i], len(VOC_CLASSES))
                samples.append((score, images[i].cpu(), preds[i], masks[i]))

    scores = [s[0] for s in samples]
    best_idx, worst_idx = topk_examples(scores, k=args.k)
    vis_dir = Path(cfg.output_dir) / "qualitative"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in [("best", best_idx), ("worst", worst_idx)]:
        for j, idx in enumerate(indices):
            _, img, pred, target = samples[idx]
            save_visualizations(
                torch.stack([img]),
                torch.stack([pred]),
                torch.stack([target]),
                vis_dir / f"{args.model_name}_{name}_{j}.png",
            )


if __name__ == "__main__":
    main()
