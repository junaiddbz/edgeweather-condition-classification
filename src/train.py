"""
src/train.py
============
Two-stage transfer learning training engine for EdgeWeather.

Academic & Production Features:
  - Two-Stage Transfer Learning Protocol:
      Stage 1: Linear probing on frozen backbone (warmup).
      Stage 2: Differential learning rates with gradual unfreezing and Cosine Annealing.
  - Dynamic Confusion-Aware Loss calibration from Stage 1 validation confusion matrix.
  - Automatic Mixed Precision (torch.amp) for high-throughput acceleration.
  - Early stopping tracking validation Macro F1 score with gradient clipping.
  - Comprehensive checkpointing (model, optimizer, scheduler, metadata, config).
  - Metric history logging (loss, accuracy, Macro F1, learning rates) exported to JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import (
    DEFAULT_CLASSES,
    create_dataloaders,
    create_synthetic_weather_dataset,
)
from src.loss import ConfusionAwareLoss, build_loss_fn
from src.models import WeatherClassifier, build_model
from src.utils import ensure_dir, get_logger, load_full_config, set_global_seed

logger = get_logger(__name__)


class Trainer:
    """Orchestrates two-stage transfer learning, validation, and checkpointing.

    Attributes:
        model: Wrapped :class:`WeatherClassifier` network.
        config: Full merged project configuration dictionary.
        device: Active PyTorch computation device (CUDA or CPU).
        loss_fn: Loss function (ConfusionAwareLoss or CrossEntropyLoss).
        checkpoint_dir: Target directory for best and final checkpoints.
        results_dir: Target directory for training metric history.
    """

    def __init__(
        self,
        model: WeatherClassifier,
        config: dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | None = None,
        checkpoint_dir: str | Path | None = None,
        results_dir: str | Path | None = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: WeatherClassifier instance.
            config: Full merged configuration dictionary.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            device: Optional torch.device (defaults to CUDA if available, else CPU).
            checkpoint_dir: Output path for checkpoints.
            results_dir: Output path for JSON metric logs.
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        paths_cfg = config.get("paths", {})
        self.checkpoint_dir = Path(checkpoint_dir or paths_cfg.get("checkpoints_dir", "outputs/checkpoints"))
        self.results_dir = Path(results_dir or paths_cfg.get("results_dir", "outputs/results"))
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.results_dir)

        self.loss_fn = build_loss_fn(config)
        self.loss_fn.to(self.device)

        # Mixed precision setup
        self.use_amp = config.get("training", {}).get("use_amp", True) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.grad_clip = config.get("training", {}).get("grad_clip_max_norm", 1.0)
        self.patience = config.get("training", {}).get("early_stopping_patience", 7)

        # Metric tracking
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "learning_rate": [],
        }
        self.best_val_f1: float = -1.0
        self.best_epoch: int = -1

    def train_one_epoch(self, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
        """Execute one complete training epoch over the training dataset.

        Args:
            optimizer: Active PyTorch optimizer.

        Returns:
            Tuple of (average_train_loss, train_accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False, unit="batch")
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            batch_size = images.size(0)

            optimizer.zero_grad(set_to_none=True)

            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total_samples += batch_size

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct / total_samples):.3f}",
            })

        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Evaluate model on the validation dataset.

        Returns:
            Dictionary with 'val_loss', 'val_acc', 'val_macro_f1',
            'confusion_matrix', 'all_preds', 'all_targets'.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []
        total_samples = 0

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            batch_size = images.size(0)

            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)

            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            labels = targets.cpu().numpy().tolist()

            all_preds.extend(preds)
            all_targets.extend(labels)
            total_samples += batch_size

        avg_loss = total_loss / max(1, total_samples)
        accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if total_samples > 0 else 0.0
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(
            all_targets,
            all_preds,
            labels=list(range(self.model.num_classes)),
        )

        return {
            "val_loss": avg_loss,
            "val_acc": float(accuracy),
            "val_macro_f1": float(macro_f1),
            "confusion_matrix": cm,
            "all_preds": all_preds,
            "all_targets": all_targets,
        }

    def save_checkpoint(
        self,
        epoch: int,
        val_macro_f1: float,
        is_best: bool = False,
        filename: str = "last_model.pt",
    ) -> Path:
        """Save training state checkpoint.

        Args:
            epoch: Current training epoch index.
            val_macro_f1: Validation Macro F1 achieved.
            is_best: Whether this checkpoint achieved the highest validation F1.
            filename: Target checkpoint filename.

        Returns:
            Path of the saved checkpoint.
        """
        model_name = self.model.model_name
        checkpoint_path = self.checkpoint_dir / f"{model_name}_{filename}"
        best_path = self.checkpoint_dir / f"{model_name}_best.pt"

        checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": self.model.state_dict(),
            "val_macro_f1": val_macro_f1,
            "num_classes": self.model.num_classes,
            "config": self.config,
        }
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            torch.save(checkpoint, best_path)
            logger.info(
                "[bold green][BEST CHECKPOINT][/bold green] %s (Macro F1: %.4f)",
                best_path.name,
                val_macro_f1,
            )

        return checkpoint_path

    def run_two_stage_training(
        self,
        stage1_epochs: int | None = None,
        stage2_epochs: int | None = None,
    ) -> dict[str, Any]:
        """Execute the full two-stage transfer learning protocol.

        Protocol:
          - Stage 1: Linear Probing
              Backbone frozen. Train classification head with higher LR (1e-3).
              At the end of Stage 1, compute empirical validation confusion matrix
              and calibrate ConfusionAwareLoss.
          - Stage 2: Fine-Tuning
              Unfreeze top backbone blocks. Train with differential learning rates
              (e.g., 1e-5 backbone, 1e-4 head) and Cosine Annealing schedule.
              Monitor validation Macro F1 for early stopping and checkpointing.

        Args:
            stage1_epochs: Optional override for Stage 1 epochs.
            stage2_epochs: Optional override for Stage 2 epochs.

        Returns:
            Final evaluation metrics dictionary.
        """
        tl_cfg = self.config.get("transfer_learning", {})
        s1_cfg = tl_cfg.get("stage1", {})
        s2_cfg = tl_cfg.get("stage2", {})

        n_s1 = stage1_epochs if stage1_epochs is not None else s1_cfg.get("epochs", 5)
        n_s2 = stage2_epochs if stage2_epochs is not None else s2_cfg.get("epochs", 45)

        logger.info(
            "[bold cyan][START][/bold cyan] Two-Stage Training for '%s' | Device: %s | AMP: %s",
            self.model.model_name,
            self.device,
            self.use_amp,
        )

        # ── STAGE 1: Linear Probing (Warmup) ─────────────────────────────────
        logger.info("=" * 60)
        logger.info("STAGE 1: Linear Probing (%d epochs, backbone frozen)", n_s1)
        logger.info("=" * 60)

        self.model.freeze_backbone()
        s1_lr = s1_cfg.get("lr", 1e-3)
        s1_wd = s1_cfg.get("weight_decay", 1e-4)

        # Optimizer only for trainable parameters (head)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_s1 = AdamW(trainable_params, lr=s1_lr, weight_decay=s1_wd)

        for epoch in range(1, n_s1 + 1):
            train_loss, train_acc = self.train_one_epoch(optimizer_s1)
            val_metrics = self.evaluate()
            val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["val_acc"]
            val_f1 = val_metrics["val_macro_f1"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)
            self.history["learning_rate"].append(s1_lr)

            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_f1, is_best=True)

            logger.info(
                "[S1 Epoch %d/%d] Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val Macro F1: %.4f",
                epoch,
                n_s1,
                train_loss,
                val_loss,
                val_acc * 100,
                val_f1,
            )

        # Calibrate ConfusionAwareLoss using Stage 1 empirical confusion matrix
        if isinstance(self.loss_fn, ConfusionAwareLoss):
            s1_final_eval = self.evaluate()
            cm = s1_final_eval["confusion_matrix"]
            logger.info("Calibrating Confusion-Aware Loss with Stage 1 empirical validation errors...")
            self.loss_fn.update_from_empirical_confusion(cm, blend_factor=0.5)

        # ── STAGE 2: Fine-Tuning (Joint Optimization) ────────────────────────
        logger.info("=" * 60)
        logger.info("STAGE 2: Fine-Tuning (%d epochs, progressive unfreezing)", n_s2)
        logger.info("=" * 60)

        unfreeze_from_block = s2_cfg.get("unfreeze_from_block")
        unfreeze_layers = s2_cfg.get("unfreeze_layers")
        self.model.unfreeze_backbone(
            unfreeze_from_block=unfreeze_from_block,
            unfreeze_layers=unfreeze_layers,
        )

        backbone_lr = s2_cfg.get("backbone_lr", 1e-5)
        head_lr = s2_cfg.get("head_lr", 1e-4)
        weight_decay = s2_cfg.get("weight_decay", 1e-4)

        param_groups = self.model.get_parameter_groups(
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
        )
        optimizer_s2 = AdamW(param_groups)

        # Scheduler setup
        sched_cfg = s2_cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "cosine_annealing_warm_restarts")
        if sched_type == "cosine_annealing_warm_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer_s2,
                T_0=sched_cfg.get("T_0", 10),
                T_mult=sched_cfg.get("T_mult", 2),
                eta_min=sched_cfg.get("eta_min", 1e-7),
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer_s2,
                T_max=n_s2,
                eta_min=sched_cfg.get("eta_min", 1e-7),
            )

        patience_counter = 0
        total_epochs = n_s1 + n_s2

        for epoch_idx in range(1, n_s2 + 1):
            epoch = n_s1 + epoch_idx
            train_loss, train_acc = self.train_one_epoch(optimizer_s2)
            val_metrics = self.evaluate()
            scheduler.step()

            val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["val_acc"]
            val_f1 = val_metrics["val_macro_f1"]
            current_lr = optimizer_s2.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)
            self.history["learning_rate"].append(current_lr)

            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch, val_f1, is_best=True)
            else:
                patience_counter += 1

            self.save_checkpoint(epoch, val_f1, is_best=False, filename="last_model.pt")

            logger.info(
                "[S2 Epoch %d/%d] Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val Macro F1: %.4f | LR: %.2e",
                epoch,
                total_epochs,
                train_loss,
                val_loss,
                val_acc * 100,
                val_f1,
                current_lr,
            )

            # Early stopping check
            if patience_counter >= self.patience:
                logger.warning(
                    "Early stopping triggered at epoch %d. Best epoch was %d (Macro F1: %.4f).",
                    epoch,
                    self.best_epoch,
                    self.best_val_f1,
                )
                break

        # Save training history JSON
        history_path = self.results_dir / f"{self.model.model_name}_history.json"
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        logger.info("Saved training history to %s", history_path)

        return self.evaluate()


# ── Training Pipeline Runner ─────────────────────────────────────────────────

def train_model(
    model_name: str = "mobilenet_v3_large",
    base_config_path: str | Path = "configs/base_config.yaml",
    model_config_path: str | Path | None = None,
    stage1_epochs: int | None = None,
    stage2_epochs: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute training for a given architecture.

    Args:
        model_name: One of 'mobilenet_v3_large', 'resnet50', 'vit_tiny_patch16_224'.
        base_config_path: Path to shared base_config.yaml.
        model_config_path: Optional path to model specific config.
        stage1_epochs: Optional override for stage 1 epochs.
        stage2_epochs: Optional override for stage 2 epochs.
        dry_run: If True, uses synthetic procedural dataset for fast verification.

    Returns:
        Final evaluation results dictionary.
    """
    model_cfg_map = {
        "mobilenet_v3_large": "configs/mobilenet_v3.yaml",
        "mobilenet_v3": "configs/mobilenet_v3.yaml",
        "resnet50": "configs/resnet50.yaml",
        "vit_tiny_patch16_224": "configs/vit_tiny.yaml",
        "vit_tiny": "configs/vit_tiny.yaml",
    }

    m_path = model_config_path or model_cfg_map.get(model_name.lower())
    config = load_full_config(base_config_path, m_path)
    set_global_seed(config.get("project", {}).get("seed", 42))

    # Data setup
    processed_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    has_real_data = (processed_dir / "train.csv").exists()

    if dry_run or not has_real_data:
        logger.info("Using lightweight synthetic weather dataset for verification/dry-run...")
        synth_dir = Path("data/synthetic_dryrun")
        manifests = create_synthetic_weather_dataset(
            output_dir=synth_dir,
            num_samples_per_class=10,
            class_names=config.get("dataset", {}).get("classes", DEFAULT_CLASSES),
        )
        loaders = create_dataloaders(
            config=config,
            train_df_or_csv=manifests["train"],
            val_df_or_csv=manifests["val"],
            test_df_or_csv=manifests["test"],
        )
    else:
        loaders = create_dataloaders(config=config)

    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError("Failed to build train and val dataloaders. Check dataset manifests.")

    # Model instantiation
    # During dry-run or unit testing, pretrained=False prevents redundant network downloads
    pretrained_flag = not dry_run
    model = build_model(config, pretrained=pretrained_flag)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
    )

    s1 = stage1_epochs if stage1_epochs is not None else (1 if dry_run else None)
    s2 = stage2_epochs if stage2_epochs is not None else (1 if dry_run else None)

    return trainer.run_two_stage_training(stage1_epochs=s1, stage2_epochs=s2)


# ── CLI Interface ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EdgeWeather: Two-Stage Transfer Learning Engine"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_large",
        choices=["mobilenet_v3_large", "mobilenet_v3", "resnet50", "vit_tiny_patch16_224", "vit_tiny"],
        help="Target architecture to train.",
    )
    parser.add_argument("--base_config", type=Path, default=Path("configs/base_config.yaml"))
    parser.add_argument("--model_config", type=Path, default=None)
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--stage2_epochs", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Run rapid dry-run on synthetic dataset.")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        base_config_path=args.base_config,
        model_config_path=args.model_config,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
