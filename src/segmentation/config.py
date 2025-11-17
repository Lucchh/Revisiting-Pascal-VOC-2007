from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 1e-2


@dataclass
class LRSchedulerConfig:
    name: str = "cosine"
    min_lr: float = 1e-6
    warmup_epochs: int = 5


@dataclass
class LossConfig:
    primary: str = "cross_entropy"
    aux: str | None = "dice"
    aux_weight: float = 0.3


@dataclass
class AugmentationConfig:
    enabled: bool = True
    color_jitter: bool = True
    random_flip: bool = True
    random_crop: bool = True
    mixup: bool = False
    backend: str = "torchvision"


@dataclass
class ModelConfig:
    name: str
    type: str
    encoder: str | None = None
    backbone: str | None = None
    pretrained: bool = True
    checkpoint: str | None = None
    encoder_adapter: str | None = None
    model_cfg: str | None = None
    freeze_encoder: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiment_name: str
    output_dir: Path
    seed: int = 42
    voc_root: Path = Path("./VOCtrainval_06-Nov-2007")
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    train_split: str = "train"
    val_split: str = "val"
    max_epochs: int = 50
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    models: List[ModelConfig] = field(default_factory=list)
    metrics: List[str] = field(default_factory=lambda: ["miou", "mean_dice", "pixel_acc"])
    val_interval: int = 1
    checkpoint_interval: int = 5
    num_visualizations: int = 4

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw["output_dir"] = Path(raw["output_dir"])
        raw["voc_root"] = Path(raw.get("voc_root", "./VOCtrainval_06-Nov-2007"))
        raw["optimizer"] = OptimizerConfig(**raw.get("optimizer", {}))
        raw["lr_scheduler"] = LRSchedulerConfig(**raw.get("lr_scheduler", {}))
        raw["loss"] = LossConfig(**raw.get("loss", {}))
        raw["augmentations"] = AugmentationConfig(**raw.get("augmentations", {}))
        raw_models = raw.get("models", [])
        raw["models"] = [ModelConfig(**cfg) for cfg in raw_models]
        return ExperimentConfig(**raw)
