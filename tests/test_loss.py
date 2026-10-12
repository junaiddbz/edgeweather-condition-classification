"""
tests/test_loss.py
==================
Unit tests for LabelSmoothingCrossEntropy and ConfusionAwareLoss.
"""

import numpy as np
import torch

from src.loss import (
    DEFAULT_CLASSES,
    ConfusionAwareLoss,
    build_domain_confusion_cost_matrix,
    build_loss_fn,
)


def test_domain_confusion_cost_matrix():
    """Verify properties of the domain-informed confusion cost matrix."""
    k = len(DEFAULT_CLASSES)
    matrix = build_domain_confusion_cost_matrix(DEFAULT_CLASSES)
    assert matrix.shape == (k, k)

    # Diagonal must be strictly 0
    assert torch.all(torch.diag(matrix) == 0.0)

    # High-risk pairs (predicting clear/rainbow when true is fog or rain)
    rainbow_idx = DEFAULT_CLASSES.index("rainbow")
    fog_idx = DEFAULT_CLASSES.index("fog")
    assert matrix[rainbow_idx, fog_idx] >= 2.5

    # Low-risk pairs (frost <-> rime)
    frost_idx = DEFAULT_CLASSES.index("frost")
    rime_idx = DEFAULT_CLASSES.index("rime")
    assert matrix[frost_idx, rime_idx] <= 0.6


def test_confusion_aware_loss_forward_and_backward():
    """Verify forward and backward passes of ConfusionAwareLoss."""
    num_classes = len(DEFAULT_CLASSES)
    loss_fn = ConfusionAwareLoss(num_classes=num_classes, lambda_weight=0.3)

    logits = torch.randn(4, num_classes, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Test backward pass
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_confusion_penalty_ordering():
    """Verify that a high-cost misclassification yields higher loss than low-cost."""
    num_classes = len(DEFAULT_CLASSES)
    loss_fn = ConfusionAwareLoss(num_classes=num_classes, lambda_weight=1.0, label_smoothing=0.0)

    fog_idx = DEFAULT_CLASSES.index("fog")
    rainbow_idx = DEFAULT_CLASSES.index("rainbow")
    rain_idx = DEFAULT_CLASSES.index("rain")

    target = torch.tensor([fog_idx], dtype=torch.long)

    # Case A: Model misclassifies as 'rainbow' (high penalty)
    logits_high = torch.zeros(1, num_classes)
    logits_high[0, rainbow_idx] = 10.0
    loss_high = loss_fn(logits_high, target).item()

    # Case B: Standard penalty
    logits_std = torch.zeros(1, num_classes)
    logits_std[0, rain_idx] = 10.0
    loss_std = loss_fn(logits_std, target).item()

    assert loss_high >= loss_std


def test_empirical_calibration_update():
    """Verify dynamic calibration of the confusion cost matrix."""
    loss_fn = ConfusionAwareLoss(num_classes=6)
    initial_matrix = loss_fn.cost_matrix.clone()

    empirical_cm = np.array(
        [
            [50, 2, 0, 1, 0, 5],
            [3, 45, 1, 0, 8, 2],
            [0, 1, 55, 4, 0, 0],
            [2, 0, 3, 52, 0, 1],
            [0, 7, 0, 0, 48, 3],
            [6, 1, 0, 2, 4, 45],
        ],
        dtype=np.float32,
    )

    loss_fn.update_from_empirical_confusion(empirical_cm, blend_factor=0.5)

    assert not torch.allclose(loss_fn.cost_matrix, initial_matrix)
    assert torch.all(torch.diag(loss_fn.cost_matrix) == 0.0)
    assert not torch.isnan(loss_fn.cost_matrix).any()


def test_build_loss_fn_factory():
    """Verify build_loss_fn handles config toggles."""
    config_ce = {"loss": {"label_smoothing": 0.1, "confusion_aware": {"enabled": False}}}
    loss_ce = build_loss_fn(config_ce)
    assert isinstance(loss_ce, torch.nn.CrossEntropyLoss)

    config_ca = {
        "loss": {"label_smoothing": 0.1, "confusion_aware": {"enabled": True, "lambda_weight": 0.4}}
    }
    loss_ca = build_loss_fn(config_ca)
    assert isinstance(loss_ca, ConfusionAwareLoss)
    assert loss_ca.lambda_weight == 0.4
