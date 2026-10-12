"""
tests/test_train.py
===================
Unit tests for the two-stage transfer learning Trainer and training pipeline.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.dataset import DEFAULT_CLASSES, create_dataloaders, create_synthetic_weather_dataset
from src.models import build_model
from src.train import Trainer


@pytest.fixture(scope="module")
def train_setup():
    """Create a temporary synthetic dataset and DataLoaders for testing the training engine."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifests = create_synthetic_weather_dataset(
            output_dir=tmp_path,
            num_samples_per_class=6,
            image_size=(224, 224),
            class_names=DEFAULT_CLASSES,
        )
        config = {
            "project": {"seed": 42},
            "dataset": {"classes": DEFAULT_CLASSES, "num_classes": len(DEFAULT_CLASSES), "image_size": 224},
            "dataloader": {"batch_size": 4, "num_workers": 0, "pin_memory": False},
            "augmentation": {},
            "paths": {
                "checkpoints_dir": str(tmp_path / "checkpoints"),
                "results_dir": str(tmp_path / "results"),
            },
            "training": {
                "use_amp": False,  # CPU unit testing
                "grad_clip_max_norm": 1.0,
                "early_stopping_patience": 3,
            },
            "loss": {
                "label_smoothing": 0.1,
                "confusion_aware": {"enabled": True, "lambda_weight": 0.3},
            },
            "transfer_learning": {
                "stage1": {"epochs": 1, "lr": 1e-3, "weight_decay": 1e-4},
                "stage2": {
                    "epochs": 1,
                    "backbone_lr": 1e-5,
                    "head_lr": 1e-4,
                    "weight_decay": 1e-4,
                    "unfreeze_from_block": 14,
                    "scheduler": {"type": "cosine_annealing_warm_restarts", "T_0": 5, "T_mult": 1, "eta_min": 1e-7},
                },
            },
        }

        loaders = create_dataloaders(
            config=config,
            train_df_or_csv=manifests["train"],
            val_df_or_csv=manifests["val"],
            test_df_or_csv=manifests["test"],
        )

        yield {
            "root": tmp_path,
            "config": config,
            "loaders": loaders,
        }


def test_train_one_epoch_and_evaluate(train_setup):
    """Verify single epoch training and evaluation execution."""
    config = train_setup["config"]
    loaders = train_setup["loaders"]
    k = len(DEFAULT_CLASSES)
    model = build_model("mobilenet_v3_large", num_classes=k, pretrained=False)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=torch.device("cpu"),
        checkpoint_dir=train_setup["root"] / "checkpoints",
        results_dir=train_setup["root"] / "results",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loss, train_acc = trainer.train_one_epoch(optimizer)

    assert isinstance(train_loss, float)
    assert train_loss > 0.0
    assert 0.0 <= train_acc <= 1.0

    eval_metrics = trainer.evaluate()
    assert "val_loss" in eval_metrics
    assert "val_acc" in eval_metrics
    assert "val_macro_f1" in eval_metrics
    assert "confusion_matrix" in eval_metrics
    assert eval_metrics["confusion_matrix"].shape == (k, k)


def test_checkpoint_saving(train_setup):
    """Verify checkpoint persistence."""
    config = train_setup["config"]
    loaders = train_setup["loaders"]
    k = len(DEFAULT_CLASSES)
    model = build_model("mobilenet_v3_large", num_classes=k, pretrained=False)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=torch.device("cpu"),
        checkpoint_dir=train_setup["root"] / "checkpoints",
        results_dir=train_setup["root"] / "results",
    )

    ckpt_path = trainer.save_checkpoint(epoch=1, val_macro_f1=0.75, is_best=True)
    assert ckpt_path.exists()

    loaded = torch.save and torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in loaded
    assert loaded["epoch"] == 1
    assert loaded["val_macro_f1"] == 0.75


def test_two_stage_training_end_to_end(train_setup):
    """Verify full two-stage training loop execution (Stage 1 + Stage 2)."""
    config = train_setup["config"]
    loaders = train_setup["loaders"]
    k = len(DEFAULT_CLASSES)
    model = build_model("mobilenet_v3_large", num_classes=k, pretrained=False)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=torch.device("cpu"),
        checkpoint_dir=train_setup["root"] / "checkpoints",
        results_dir=train_setup["root"] / "results",
    )

    final_metrics = trainer.run_two_stage_training(stage1_epochs=1, stage2_epochs=1)
    assert "val_macro_f1" in final_metrics

    # History JSON exists
    history_file = train_setup["root"] / "results" / "mobilenet_v3_large_history.json"
    assert history_file.exists()
    assert len(trainer.history["train_loss"]) == 2  # 1 epoch stage 1 + 1 epoch stage 2
