#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from segmentation.config import ExperimentConfig, ModelConfig
from segmentation.engine import Trainer
from segmentation.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation suites")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument(
        "--study",
        type=str,
        choices=["backbone", "augmentation", "loss"],
        default="backbone",
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="Override epochs for speed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = ExperimentConfig.from_yaml(args.config)
    scenarios = build_scenarios(base_cfg, args.study)
    for scenario_name, cfg in scenarios.items():
        print(f"\n=== Running ablation: {scenario_name} ===")
        cfg.experiment_name = scenario_name
        cfg.output_dir = Path(f"runs/{scenario_name}")
        if args.max_epochs:
            cfg.max_epochs = args.max_epochs
        set_seed(cfg.seed)
        trainer = Trainer(cfg)
        trainer.train_all()


def build_scenarios(base_cfg: ExperimentConfig, study: str):
    scenarios = {}
    if study == "backbone":
        for encoder in ["resnet18", "resnet50"]:
            cfg = copy.deepcopy(base_cfg)
            cfg.models = [
                ModelConfig(name=f"unet_{encoder}", type="unet", encoder=encoder, pretrained=True)
            ]
            scenarios[f"ablation_backbone_{encoder}"] = cfg
    elif study == "augmentation":
        for enabled in [False, True]:
            cfg = copy.deepcopy(base_cfg)
            cfg.augmentations.enabled = enabled
            cfg.augmentations.color_jitter = enabled
            cfg.augmentations.random_flip = enabled
            cfg.augmentations.random_crop = enabled
            flag = "with_aug" if enabled else "no_aug"
            cfg.models = [cfg.models[0]]
            scenarios[f"ablation_aug_{flag}"] = cfg
    elif study == "loss":
        for primary, aux in [("cross_entropy", None), ("cross_entropy", "dice")]:
            cfg = copy.deepcopy(base_cfg)
            cfg.loss.primary = primary
            cfg.loss.aux = aux
            tag = "dice" if aux else "ce"
            cfg.models = [cfg.models[0]]
            scenarios[f"ablation_loss_{tag}"] = cfg
    return scenarios


if __name__ == "__main__":
    main()
