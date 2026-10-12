"""
tests/test_evaluate.py
======================
Unit tests for the scientific evaluation suite: ECE, reliability diagrams,
confusion matrix plotting, and failure analysis.
"""

from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import DEFAULT_CLASSES
from src.evaluate import (
    compute_ece,
    evaluate_model,
    extract_top_failure_cases,
    plot_confusion_matrix,
    plot_reliability_diagram,
)
from src.models import build_model


def test_compute_ece_perfect_calibration():
    """Verify that perfectly calibrated predictions yield an ECE near 0."""
    # 100 samples with 100% confidence that are all correct
    probs = np.zeros((100, 3))
    probs[:, 0] = 1.0
    targets = np.zeros(100, dtype=int)

    ece, stats = compute_ece(probs, targets, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5)
    assert "bin_accuracies" in stats
    assert "bin_confidences" in stats


def test_compute_ece_overconfidence():
    """Verify that confident incorrect predictions yield a positive ECE."""
    # 100 samples with 100% confidence that are all WRONG
    probs = np.zeros((100, 3))
    probs[:, 0] = 1.0
    targets = np.ones(100, dtype=int)  # true is class 1

    ece, stats = compute_ece(probs, targets, n_bins=10)
    assert ece > 0.9  # Confident 1.0, Acc 0.0 -> Gap = 1.0


def test_plot_reliability_diagram():
    """Verify reliability diagram figure generation and persistence."""
    probs = np.random.dirichlet(np.ones(6), size=50)
    targets = np.random.randint(0, 6, size=50)

    ece, stats = compute_ece(probs, targets, n_bins=10)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "reliability.png"
        fig = plot_reliability_diagram(stats, ece=ece, model_name="TestModel", save_path=save_path)

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    plt.close("all")


def test_plot_confusion_matrix():
    """Verify confusion matrix heatmap figure generation and persistence."""
    cm = np.array([
        [10, 2, 0, 0, 1, 0],
        [1, 12, 0, 0, 0, 0],
        [0, 0, 15, 1, 0, 0],
        [0, 1, 0, 14, 0, 0],
        [2, 0, 0, 0, 11, 1],
        [0, 0, 0, 0, 2, 13],
    ])

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "cm.png"
        fig = plot_confusion_matrix(cm, class_names=DEFAULT_CLASSES, normalize=True, save_path=save_path)

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    plt.close("all")


def test_extract_top_failure_cases():
    """Verify extraction of worst-case misclassifications."""
    probs = np.array([
        [0.9, 0.1],  # Pred 0, True 0 -> Correct
        [0.85, 0.15], # Pred 0, True 1 -> Error (conf 0.85)
        [0.95, 0.05], # Pred 0, True 1 -> Error (conf 0.95)
    ])
    targets = np.array([0, 1, 1])
    paths = ["img0.jpg", "img1.jpg", "img2.jpg"]

    failures = extract_top_failure_cases(probs, targets, paths, class_names=["c0", "c1"], top_k=2)

    assert len(failures) == 2
    # img2 should be first because confidence 0.95 > 0.85
    assert failures[0]["filepath"] == "img2.jpg"
    assert failures[0]["confidence"] == pytest.approx(0.95)
    assert failures[1]["filepath"] == "img1.jpg"


def test_evaluate_model():
    """Verify evaluate_model function over synthetic tensor DataLoader."""
    k = len(DEFAULT_CLASSES)
    model = build_model("mobilenet_v3_large", num_classes=k, pretrained=False)

    # Create dummy DataLoader
    dummy_images = torch.randn(8, 3, 224, 224)
    dummy_labels = torch.randint(0, k, (8,))

    # Custom dict batch structure matching WeatherDataset
    class DummyLoader:
        def __iter__(self):
            yield {
                "image": dummy_images,
                "label": dummy_labels,
                "filepath": [f"dummy_{i}.jpg" for i in range(8)],
            }

    results = evaluate_model(
        model=model,
        dataloader=DummyLoader(),
        class_names=DEFAULT_CLASSES,
        device=torch.device("cpu"),
    )

    assert "accuracy" in results
    assert "macro_f1" in results
    assert "ece" in results
    assert "confusion_matrix" in results
    assert results["confusion_matrix"].shape == (k, k)
