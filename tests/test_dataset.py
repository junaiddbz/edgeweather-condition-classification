"""
tests/test_dataset.py
=====================
Unit tests for WeatherDataset, Albumentations transforms, and DataLoaders.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import (
    DEFAULT_CLASSES,
    WeatherDataset,
    create_dataloaders,
    create_synthetic_weather_dataset,
    get_eval_transforms,
    get_train_transforms,
)


@pytest.fixture(scope="module")
def synthetic_data_dir():
    """Create a temporary synthetic dataset for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifests = create_synthetic_weather_dataset(
            output_dir=tmp_path,
            num_samples_per_class=4,
            image_size=(224, 224),
            class_names=DEFAULT_CLASSES,
        )
        yield {
            "root": tmp_path,
            "manifests": manifests,
        }


def test_synthetic_dataset_creation(synthetic_data_dir):
    """Verify that synthetic dataset generator produces valid CSV manifests and images."""
    manifests = synthetic_data_dir["manifests"]
    for split in ["train", "val", "test"]:
        csv_path = manifests[split]
        assert csv_path.exists(), f"{split}.csv missing"
        df = pd.read_csv(csv_path)
        assert "filepath" in df.columns
        assert "label" in df.columns
        assert len(df) > 0
        # Check image files physically exist
        for img_path in df["filepath"]:
            assert Path(img_path).exists(), f"Image file missing: {img_path}"


def test_train_transforms():
    """Verify train transforms produce tensor of correct dimensions and normalization."""
    transforms = get_train_transforms(image_size=224)
    dummy_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    augmented = transforms(image=dummy_img)

    tensor = augmented["image"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transforms():
    """Verify deterministic eval transforms produce exact target dimension."""
    transforms = get_eval_transforms(image_size=224)
    dummy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    augmented = transforms(image=dummy_img)

    tensor = augmented["image"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_weather_dataset_getitem(synthetic_data_dir):
    """Verify sample retrieval and tensor dictionary format."""
    manifests = synthetic_data_dir["manifests"]
    dataset = WeatherDataset(manifests["train"], class_names=DEFAULT_CLASSES)

    assert len(dataset) > 0
    sample = dataset[0]
    assert "image" in sample
    assert "label" in sample
    assert "label_name" in sample
    assert "filepath" in sample

    assert sample["image"].shape == (3, 224, 224)
    assert sample["label"].dtype == torch.long
    assert 0 <= sample["label"].item() < len(DEFAULT_CLASSES)
    assert sample["label_name"] in DEFAULT_CLASSES


def test_create_dataloaders(synthetic_data_dir):
    """Verify DataLoader factory instantiates iterable loaders."""
    manifests = synthetic_data_dir["manifests"]
    config = {
        "dataset": {"classes": DEFAULT_CLASSES, "image_size": 224},
        "dataloader": {"batch_size": 4, "num_workers": 0, "pin_memory": False},
        "augmentation": {},
        "paths": {"processed_dir": str(synthetic_data_dir["root"])},
    }

    loaders = create_dataloaders(
        config=config,
        train_df_or_csv=manifests["train"],
        val_df_or_csv=manifests["val"],
        test_df_or_csv=manifests["test"],
    )

    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    # Fetch a batch
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape[0] == 4
    assert batch["image"].shape[1:] == (3, 224, 224)
    assert batch["label"].shape[0] == 4
