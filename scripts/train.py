#!/usr/bin/env python
from __future__ import annotations

import argparse

from pathlib import Path

from segmentation.config import ExperimentConfig
from segmentation.engine import Trainer
from segmentation.utils import set_seed


def str2bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Boolean argument cannot be None")
    lowered = value.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train segmentation models on Pascal VOC 2007")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to YAML config")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional list of model names from the config to train (default: all models).",
    )
    parser.add_argument("--max-epochs", type=int, help="Override the max_epochs value from the config.")
    parser.add_argument("--experiment-name", type=str, help="Override experiment name.")
    parser.add_argument("--output-dir", type=str, help="Override output directory.")
    parser.add_argument("--seed", type=int, help="Set random seed.")
    parser.add_argument("--voc-root", type=str, help="Path to VOC dataset root.")
    parser.add_argument("--image-size", type=int, help="Input resolution for training.")
    parser.add_argument("--batch-size", type=int, help="Batch size.")
    parser.add_argument("--num-workers", type=int, help="Number of dataloader workers.")
    parser.add_argument("--train-split", type=str, help="Dataset split for training.")
    parser.add_argument("--val-split", type=str, help="Dataset split for validation.")
    parser.add_argument("--optimizer-name", type=str, help="Optimizer type (adamw/sgd).")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Weight decay.")
    parser.add_argument("--scheduler-name", type=str, help="LR scheduler type.")
    parser.add_argument("--min-lr", type=float, help="Minimum learning rate for schedulers.")
    parser.add_argument("--warmup-epochs", type=int, help="Warmup epochs for schedulers.")
    parser.add_argument("--loss-primary", type=str, help="Primary loss name.")
    parser.add_argument("--loss-aux", type=str, help="Auxiliary loss name (or 'none').")
    parser.add_argument("--loss-aux-weight", type=float, help="Auxiliary loss weight.")
    parser.add_argument("--augmentations-enabled", type=str2bool, help="Toggle all augmentations.")
    parser.add_argument("--color-jitter", type=str2bool, help="Toggle color jitter augmentation.")
    parser.add_argument("--random-flip", type=str2bool, help="Toggle horizontal flip augmentation.")
    parser.add_argument("--random-crop", type=str2bool, help="Toggle random crop augmentation.")
    parser.add_argument("--mixup", type=str2bool, help="Toggle mixup augmentation.")
    parser.add_argument("--val-interval", type=int, help="Validation frequency in epochs.")
    parser.add_argument("--checkpoint-interval", type=int, help="Checkpoint frequency in epochs.")
    parser.add_argument("--num-visualizations", type=int, help="Number of samples to visualize.")
    parser.add_argument("--metrics", nargs="+", help="Metrics to compute during evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.experiment_name:
        cfg.experiment_name = args.experiment_name
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.voc_root:
        cfg.voc_root = Path(args.voc_root)
    if args.image_size:
        cfg.image_size = args.image_size
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.num_workers:
        cfg.num_workers = args.num_workers
    if args.train_split:
        cfg.train_split = args.train_split
    if args.val_split:
        cfg.val_split = args.val_split
    if args.max_epochs:
        cfg.max_epochs = args.max_epochs
    if args.optimizer_name:
        cfg.optimizer.name = args.optimizer_name
    if args.lr:
        cfg.optimizer.lr = args.lr
    if args.weight_decay:
        cfg.optimizer.weight_decay = args.weight_decay
    if args.scheduler_name:
        cfg.lr_scheduler.name = args.scheduler_name
    if args.min_lr:
        cfg.lr_scheduler.min_lr = args.min_lr
    if args.warmup_epochs is not None:
        cfg.lr_scheduler.warmup_epochs = args.warmup_epochs
    if args.loss_primary:
        cfg.loss.primary = args.loss_primary
    if args.loss_aux:
        cfg.loss.aux = None if args.loss_aux.lower() == "none" else args.loss_aux
    if args.loss_aux_weight is not None:
        cfg.loss.aux_weight = args.loss_aux_weight
    if args.augmentations_enabled is not None:
        cfg.augmentations.enabled = args.augmentations_enabled
    if args.color_jitter is not None:
        cfg.augmentations.color_jitter = args.color_jitter
    if args.random_flip is not None:
        cfg.augmentations.random_flip = args.random_flip
    if args.random_crop is not None:
        cfg.augmentations.random_crop = args.random_crop
    if args.mixup is not None:
        cfg.augmentations.mixup = args.mixup
    if args.val_interval:
        cfg.val_interval = args.val_interval
    if args.checkpoint_interval:
        cfg.checkpoint_interval = args.checkpoint_interval
    if args.num_visualizations:
        cfg.num_visualizations = args.num_visualizations
    if args.metrics:
        cfg.metrics = args.metrics
    if args.models:
        selected = {name.lower() for name in args.models}
        filtered = [m for m in cfg.models if m.name.lower() in selected]
        if not filtered:
            raise ValueError(f"No models from {args.models} found in config.")
        cfg.models = filtered
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train_all()


if __name__ == "__main__":
    main()
