#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from segmentation.config import ExperimentConfig
from segmentation.data.voc import VOC_CLASSES
from segmentation.engine import Trainer, build_model
from segmentation.metrics import MetricState
from segmentation.utils import save_visualizations, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained segmentation checkpoints")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--visualize", action="store_true", help="Save qualitative outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    model_cfg = next((m for m in cfg.models if m.name == args.model_name), None)
    if model_cfg is None:
        raise ValueError(f"Model {args.model_name} not found in config")
    model = build_model(model_cfg, num_classes=len(VOC_CLASSES))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(trainer.device)
    model.eval()

    meter = MetricState(num_classes=len(VOC_CLASSES))
    visuals = []
    with torch.inference_mode():
        for images, masks in trainer.dataloaders.val_loader:
            images = images.to(trainer.device)
            outputs = model(images)
            logits = outputs["main"].cpu()
            meter.update(logits, masks)
            if args.visualize and len(visuals) < cfg.num_visualizations:
                preds = torch.argmax(logits, dim=1)
                visuals.append((images.cpu(), preds, masks))
    metrics = meter.compute()
    for key, value in metrics.items():
        if hasattr(value, "tolist"):
            print(f"{key}: {value.tolist()}")
        else:
            print(f"{key}: {value}")

    if args.visualize and visuals:
        vis_dir = Path(cfg.output_dir) / "visuals"
        for idx, pack in enumerate(visuals):
            images, preds, masks = pack
            save_visualizations(images, preds.numpy(), masks.numpy(), vis_dir / f"val_{idx}.png")


if __name__ == "__main__":
    main()
