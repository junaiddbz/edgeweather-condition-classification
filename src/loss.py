"""
src/loss.py
===========
Loss functions for EdgeWeather: Label Smoothing & Novel Confusion-Aware Loss.

Academic Contribution:
  Standard Cross-Entropy penalizes all misclassifications uniformly. In real-world
  atmospheric sensing (autonomous driving, UAV navigation), asymmetric misclassification
  costs exist:
    - Predicting 'shine' (clear sky) when ground truth is 'fog' or 'rain' (zero-visibility)
      is safety-critical.
    - Predicting 'cloudy' when ground truth is 'fog' is a low-consequence perceptual error.

  We formulate Confusion-Aware Hierarchical Loss as:
    L_total = L_CE(logits, y) + lambda * sum_{i != j} C_{ij} * P(pred=i | true=j)
  where C in R^{K x K} is a domain-informed confusion cost matrix.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CLASSES: list[str] = [
    "dew",
    "fog",
    "frost",
    "glaze",
    "hail",
    "lightning",
    "rain",
    "rainbow",
    "rime",
    "sandstorm",
    "snow",
]


def build_domain_confusion_cost_matrix(
    class_names: Sequence[str] = DEFAULT_CLASSES,
    high_cost_multiplier: float = 3.0,
    low_cost_multiplier: float = 0.5,
) -> torch.Tensor:
    """Construct a domain-informed asymmetric confusion cost matrix.

    Encodes domain semantics for adverse weather sensing:
      - Zero cost on diagonal (correct predictions: C_{ii} = 0).
      - Default cost = 1.0 for standard misclassifications.
      - Low cost (~0.5) for perceptually adjacent classes (e.g. cloudy <-> fog).
      - High cost (~3.0) for safety-critical false negatives (e.g. predicting clear 'shine'
        when conditions are 'fog', 'rain', or 'snow').

    Args:
        class_names: Ordered list of class names.
        high_cost_multiplier: Cost penalty for high-risk safety errors.
        low_cost_multiplier: Cost penalty for visually adjacent errors.

    Returns:
        FloatTensor of shape (K, K) where C[i, j] is the penalty for predicting i
        when ground truth is j.
    """
    k = len(class_names)
    name_to_idx = {name.lower(): idx for idx, name in enumerate(class_names)}
    cost_matrix = torch.ones((k, k), dtype=torch.float32)

    # Diagonal is zero: no penalty for correct predictions
    cost_matrix.fill_diagonal_(0.0)

    # 1. Low-consequence perceptual adjacencies (rime <-> frost, rain <-> snow)
    low_risk_pairs = [
        ("cloudy", "fog"),
        ("fog", "cloudy"),
        ("rain", "snow"),
        ("snow", "rain"),
        ("rime", "frost"),
        ("frost", "rime"),
        ("glaze", "frost"),
        ("frost", "glaze"),
    ]
    for pred, true in low_risk_pairs:
        if pred in name_to_idx and true in name_to_idx:
            cost_matrix[name_to_idx[pred], name_to_idx[true]] = low_cost_multiplier

    # 2. High-consequence safety risks: False reassurance
    hazards = ["fog", "rain", "snow", "hail", "glaze", "sandstorm", "lightning"]
    reassuring = ["shine", "sunrise", "rainbow", "dew"]
    for true in hazards:
        for pred in reassuring:
            if pred in name_to_idx and true in name_to_idx:
                cost_matrix[name_to_idx[pred], name_to_idx[true]] = high_cost_multiplier

    logger.debug(
        "Constructed domain confusion cost matrix (%dx%d) with high_cost=%.1f, low_cost=%.1f",
        k,
        k,
        high_cost_multiplier,
        low_cost_multiplier,
    )
    return cost_matrix


class ConfusionAwareLoss(nn.Module):
    """Novel Confusion-Aware Loss incorporating asymmetric misclassification penalties.

    Formula:
        L_total = L_CE_smooth(y_pred, y_true) + lambda_weight * L_confusion
    where:
        L_confusion = (1 / N) * sum_{n=1}^N sum_{i != y_n} C_{i, y_n} * p_{n, i}
    and p_{n, i} is the Softmax probability assigned to incorrect class i for sample n.

    Attributes:
        label_smoothing: Smoothing factor epsilon in standard CE.
        lambda_weight: Regularization coefficient balancing CE and confusion penalty.
        cost_matrix: (K, K) tensor of misclassification costs.
    """

    def __init__(
        self,
        cost_matrix: torch.Tensor | np.ndarray | None = None,
        num_classes: int = 6,
        label_smoothing: float = 0.1,
        lambda_weight: float = 0.3,
        class_names: Sequence[str] | None = None,
    ) -> None:
        """Initialize ConfusionAwareLoss.

        Args:
            cost_matrix: Optional pre-computed cost tensor of shape (K, K).
            num_classes: Number of classes K (used if cost_matrix is not supplied).
            label_smoothing: Label smoothing rate for the base CE loss.
            lambda_weight: Weight lambda for the confusion penalty term.
            class_names: Optional class name list to generate domain-informed cost matrix.
        """
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.lambda_weight = lambda_weight

        if cost_matrix is not None:
            if isinstance(cost_matrix, np.ndarray):
                cost_tensor = torch.from_numpy(cost_matrix).float()
            else:
                cost_tensor = cost_matrix.clone().float()
        else:
            names = class_names or DEFAULT_CLASSES[:num_classes]
            cost_tensor = build_domain_confusion_cost_matrix(names)

        # Register cost matrix as a non-trainable persistent buffer
        self.register_buffer("cost_matrix", cost_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the total confusion-aware loss.

        Args:
            logits: Unnormalized network predictions of shape (B, K).
            targets: Ground-truth class index tensor of shape (B,).

        Returns:
            Scalar loss tensor with full autograd graph.
        """
        # Base Label-Smoothed Cross-Entropy
        ce_loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        if self.lambda_weight <= 0.0 or self.cost_matrix is None:
            return ce_loss

        # Compute predicted class probabilities via Softmax
        probs = F.softmax(logits, dim=-1)  # (B, K)

        # Extract column C[:, y_n] for each sample in the batch -> shape (B, K)
        # cost_matrix[i, j] is the cost of predicting i when true class is j.
        batch_costs = self.cost_matrix[:, targets].transpose(0, 1)  # (B, K)

        # Penalty is element-wise product of predicted probabilities and cost weights
        confusion_penalty = torch.sum(probs * batch_costs, dim=-1).mean()

        total_loss = ce_loss + self.lambda_weight * confusion_penalty
        return total_loss

    def update_from_empirical_confusion(
        self,
        empirical_confusion_matrix: np.ndarray | torch.Tensor,
        normalize: bool = True,
        blend_factor: float = 0.5,
    ) -> None:
        """Calibrate cost matrix from empirical validation confusion counts.

        Allows dynamic adaptation after Stage 1 linear probing:
        misclassification pairs that the model frequently confuses empirically
        get elevated costs for Stage 2 fine-tuning.

        Args:
            empirical_confusion_matrix: (K, K) matrix where entry (i, j) is the
                count of true class i predicted as class j.
            normalize: Whether to normalize by row sums (class marginals).
            blend_factor: Weight for empirical matrix (1-blend keeps prior domain cost).
        """
        if isinstance(empirical_confusion_matrix, np.ndarray):
            cm = torch.from_numpy(empirical_confusion_matrix).float().to(self.cost_matrix.device)
        else:
            cm = empirical_confusion_matrix.clone().float().to(self.cost_matrix.device)

        # Transpose if given as (true, pred) to match our convention (pred, true)
        cm_pred_true = cm.transpose(0, 1)

        # Zero out diagonal
        cm_pred_true.fill_diagonal_(0.0)

        if normalize:
            col_sums = cm_pred_true.sum(dim=0, keepdim=True).clamp(min=1.0)
            empirical_costs = cm_pred_true / col_sums
            # Scale to mean 1.0 off-diagonal
            off_diag_mask = ~torch.eye(self.num_classes, dtype=torch.bool, device=cm.device)
            mean_val = empirical_costs[off_diag_mask].mean().clamp(min=1e-5)
            empirical_costs = empirical_costs / mean_val
        else:
            empirical_costs = cm_pred_true

        # Blend with existing prior
        updated = (1.0 - blend_factor) * self.cost_matrix + blend_factor * empirical_costs
        updated.fill_diagonal_(0.0)
        self.cost_matrix.copy_(updated)
        logger.info("Updated confusion cost matrix with empirical validation counts (blend=%.2f).", blend_factor)


# ── Loss Factory ─────────────────────────────────────────────────────────────

def build_loss_fn(
    config: dict[str, Any],
    num_classes: int | None = None,
    class_names: Sequence[str] | None = None,
) -> nn.Module:
    """Build loss function based on project YAML configuration.

    Supports:
      - Standard Cross-Entropy with Label Smoothing (ablation baseline).
      - ConfusionAwareLoss (novel academic contribution).

    Args:
        config: Full configuration dictionary.
        num_classes: Optional number of target classes.
        class_names: Optional sequence of class names.

    Returns:
        Initialized :class:`nn.Module` loss function.
    """
    loss_cfg = config.get("loss", {})
    dataset_cfg = config.get("dataset", {})

    classes = class_names or dataset_cfg.get("classes", DEFAULT_CLASSES)
    n_classes = num_classes or dataset_cfg.get("num_classes", len(classes))
    smoothing = loss_cfg.get("label_smoothing", 0.1)

    conf_aware_cfg = loss_cfg.get("confusion_aware", {})
    is_conf_aware = conf_aware_cfg.get("enabled", True)
    lambda_weight = conf_aware_cfg.get("lambda_weight", 0.3)

    if is_conf_aware:
        logger.info(
            "Building ConfusionAwareLoss | lambda=%.2f | smoothing=%.2f | classes=%d",
            lambda_weight,
            smoothing,
            n_classes,
        )
        return ConfusionAwareLoss(
            num_classes=n_classes,
            label_smoothing=smoothing,
            lambda_weight=lambda_weight,
            class_names=classes,
        )
    else:
        logger.info("Building standard CrossEntropyLoss | smoothing=%.2f", smoothing)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
