from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import ExperimentConfig, ModelConfig
from .data.voc import VOC_CLASSES, build_dataloaders
from .metrics import MetricState
from .models.deeplab import DeepLabWrapper
from .models.unet import ResNetUNet

IGNORE_INDEX = 255


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloaders = build_dataloaders(
            root=str(config.voc_root),
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            train_split=config.train_split,
            val_split=config.val_split,
            augmentations=config.augmentations.__dict__,
        )
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    def train_all(self) -> None:
        for model_cfg in self.cfg.models:
            self.train_single_model(model_cfg)

    def train_single_model(self, model_cfg: ModelConfig) -> None:
        model = build_model(model_cfg, num_classes=len(VOC_CLASSES)).to(self.device)
        optimizer = build_optimizer(model, self.cfg)
        scheduler = build_scheduler(optimizer, self.cfg)
        history = []
        best_miou = 0.0
        best_path: Path | None = None
        best_epoch: int | None = None

        for epoch in range(1, self.cfg.max_epochs + 1):
            train_metrics, train_loss = self._run_epoch(
                model, self.dataloaders.train_loader, optimizer, train=True
            )
            log_entry = {
                "epoch": epoch,
                "mode": "train",
                "loss": train_loss,
                **train_metrics,
            }
            history.append(log_entry)
            self._print_epoch_summary(model_cfg.name, epoch, "train", train_loss, train_metrics)

            if epoch % self.cfg.val_interval == 0:
                val_metrics, val_loss = self._run_epoch(
                    model, self.dataloaders.val_loader, optimizer=None, train=False
                )
                history.append(
                    {
                        "epoch": epoch,
                        "mode": "val",
                        "loss": val_loss,
                        **val_metrics,
                    }
                )
                miou = val_metrics.get("miou", 0.0)
                self._print_epoch_summary(model_cfg.name, epoch, "val", val_loss, val_metrics)
                if miou > best_miou:
                    best_miou = miou
                    best_path = self._save_checkpoint(model, model_cfg.name, epoch, best=True)
                    best_epoch = epoch
            if scheduler is not None:
                scheduler.step()

            if epoch % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(model, model_cfg.name, epoch)

        metrics_path = self.cfg.output_dir / f"metrics_{model_cfg.name}.csv"
        pd.DataFrame(history).to_csv(metrics_path, index=False)
        if best_path:
            summary = {
                "best_epoch": best_epoch,
                "best_miou": best_miou,
                "checkpoint": str(best_path),
            }
            with open(self.cfg.output_dir / f"summary_{model_cfg.name}.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    def _run_epoch(
        self,
        model: nn.Module,
        dataloader,
        optimizer: optim.Optimizer | None,
        train: bool,
    ) -> Tuple[Dict[str, float], float]:
        mode = "train" if train else "val"
        model.train(train)
        total_loss = 0.0
        meter = MetricState(num_classes=len(VOC_CLASSES))
        progress = tqdm(dataloader, desc=f"{mode}", leave=False)
        for images, masks in progress:
            images = images.to(self.device)
            if masks.ndim == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            masks = masks.to(self.device)
            with torch.set_grad_enabled(train):
                outputs = model(images)
                loss = compute_loss(outputs, masks, self.cfg.loss)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * images.size(0)
            logits = outputs["main"].detach().cpu()
            meter.update(logits, masks.cpu())
            progress.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(dataloader.dataset)
        metrics = meter.compute()
        return metrics, avg_loss

    def _save_checkpoint(self, model: nn.Module, model_name: str, epoch: int, best: bool = False) -> Path:
        tag = "best" if best else f"epoch_{epoch}"
        ckpt_path = self.cfg.output_dir / f"{model_name}_{tag}.pt"
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path)
        return ckpt_path

    def _print_epoch_summary(
        self,
        model_name: str,
        epoch: int,
        mode: str,
        loss: float,
        metrics: Dict[str, float],
    ) -> None:
        metric_str = ", ".join(
            f"{key}: {value:.4f}" for key, value in metrics.items() if isinstance(value, (int, float))
        )
        print(f"[{model_name}] Epoch {epoch:03d} | {mode.upper()} | Loss: {loss:.4f} | {metric_str}")


def build_model(cfg: ModelConfig, num_classes: int) -> nn.Module:
    model_type = cfg.type.lower()
    if model_type == "unet":
        extra = cfg.extra or {}
        decoder_channels = extra.get("decoder_channels")
        if decoder_channels is not None:
            decoder_channels = tuple(decoder_channels)
        encoder_freeze = extra.get("encoder_freeze", False)
        encoder_weights = extra.get("encoder_weights")
        return ResNetUNet(
            encoder_name=cfg.encoder or "resnet18",
            num_classes=num_classes,
            pretrained=cfg.pretrained,
            decoder_channels=decoder_channels,
            encoder_weights=encoder_weights,
            encoder_freeze=encoder_freeze,
        )
    if model_type == "deeplabv3":
        extra = cfg.extra or {}
        encoder_weights = extra.get("encoder_weights")
        freeze_backbone = extra.get("freeze_backbone", extra.get("freeze_encoder", False))
        return DeepLabWrapper(
            backbone=cfg.backbone or "resnet50",
            num_classes=num_classes,
            pretrained=cfg.pretrained,
            encoder_weights=encoder_weights,
            freeze_encoder=freeze_backbone,
        )
    if model_type == "sam":
        from .models.sam_adapter import SAMAdapter  # lazy import
        return SAMAdapter(checkpoint=cfg.checkpoint or "", num_classes=num_classes)
    if model_type == "sam2":
        from .models.sam2_adapter import SAM2Adapter  # lazy import
        extra = cfg.extra or {}
        backbone_device = extra.get("backbone_device")
        return SAM2Adapter(
            model_cfg=cfg.model_cfg or "configs/sam2.1/sam2.1_hiera_b+.yaml",
            checkpoint=cfg.checkpoint or "",
            num_classes=num_classes,
            freeze_encoder=getattr(cfg, "freeze_encoder", True),
            backbone_device=backbone_device,
        )
    raise ValueError(f"Unknown model type: {cfg.type}")


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> optim.Optimizer:
    optim_name = cfg.optimizer.name.lower()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optim_name == "adamw":
        return optim.AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    if optim_name == "sgd":
        return optim.SGD(params, lr=cfg.optimizer.lr, momentum=0.9, weight_decay=cfg.optimizer.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")


def build_scheduler(optimizer: optim.Optimizer, cfg: ExperimentConfig):
    sched_name = cfg.lr_scheduler.name.lower()
    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr_scheduler.min_lr)
    return None


def compute_loss(outputs: Dict[str, torch.Tensor | None], targets: torch.Tensor, loss_cfg) -> torch.Tensor:
    logits = outputs["main"]
    ce = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
    total = ce
    if loss_cfg.aux == "dice":
        total += loss_cfg.aux_weight * dice_loss(logits, targets, logits.shape[1])
    if outputs.get("aux") is not None:
        aux_logits = outputs["aux"]
        total += 0.5 * F.cross_entropy(aux_logits, targets, ignore_index=IGNORE_INDEX)
    return total


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    targets = targets.clone()
    targets[targets == IGNORE_INDEX] = 0
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1 - dice.mean()
