"""
src/models.py
=============
Model factory supporting MobileNetV3-Large, ResNet-50, and ViT-Tiny.

Academic & Production Features:
  - Unified wrapper interface (:class:`WeatherClassifier`) across all three architectures.
  - Multi-stage transfer learning control: Stage 1 linear probing (frozen backbone)
    and Stage 2 fine-tuning (progressive unfreezing with differential learning rates).
  - Penultimate feature extractor (:meth:`extract_features`) for t-SNE / UMAP / FID analysis.
  - Target layer hooks for Grad-CAM and Grad-CAM++ explainability.
  - Parameter grouping utility for weight-decay filtering (no decay on biases/norms).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import timm
import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    mobilenet_v3_large,
    resnet50,
)

from src.utils import get_logger

logger = get_logger(__name__)

SUPPORTED_MODELS = ("mobilenet_v3_large", "resnet50", "vit_tiny_patch16_224")


class WeatherClassifier(nn.Module):
    """Unified wrapper for weather condition classification backbones.

    Wraps MobileNetV3-Large, ResNet-50, or ViT-Tiny to expose a uniform API
    for feature extraction, Grad-CAM layer targeting, progressive unfreezing,
    and differential optimizer parameter groups.

    Attributes:
        model_name: Canonical model architecture identifier.
        num_classes: Number of target weather classes.
        dropout_rate: Dropout probability in the classification head.
        backbone: Feature extraction network.
        head: Multi-class classification head.
        feature_dim: Dimensionality of the penultimate feature vector.
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v3_large",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize the WeatherClassifier wrapper.

        Args:
            model_name: One of 'mobilenet_v3_large', 'resnet50', 'vit_tiny_patch16_224'.
            num_classes: Number of output prediction classes.
            pretrained: Whether to load ImageNet pre-trained weights.
            dropout_rate: Dropout rate for the newly initialized head.
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if self.model_name == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = mobilenet_v3_large(weights=weights)
            # MobileNet features output has 960 channels -> classifier input is 960
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = 960
            self.head = nn.Sequential(
                nn.Linear(960, 1280),
                nn.Hardswish(),
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(1280, num_classes),
            )

        elif self.model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = resnet50(weights=weights)
            # ResNet-50 features: stem + 4 residual stages
            self.backbone = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = 2048
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(2048, num_classes),
            )

        elif self.model_name in ("vit_tiny_patch16_224", "vit_tiny"):
            self.model_name = "vit_tiny_patch16_224"
            # Use timm for vision transformer
            base = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout_rate,
            )
            self.feature_dim = base.embed_dim  # 192 for vit_tiny
            self.backbone = base
            self.pool = None
            self.head = None  # Built into timm vit model directly
        else:
            raise ValueError(
                f"Unsupported model: '{model_name}'. Choose from: {SUPPORTED_MODELS}"
            )

        logger.info(
            "Initialized '%s' | num_classes=%d | feature_dim=%d | pretrained=%s",
            self.model_name,
            self.num_classes,
            self.feature_dim,
            pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing class prediction logits.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        if self.model_name == "vit_tiny_patch16_224":
            return self.backbone(x)

        feats = self.backbone(x)
        pooled = self.pool(feats)
        flattened = torch.flatten(pooled, 1)
        return self.head(flattened)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embeddings.

        Used for t-SNE / UMAP manifold visualization, domain shift FID
        quantification, and representation space clustering metrics.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Feature embedding tensor of shape (B, feature_dim).
        """
        if self.model_name == "vit_tiny_patch16_224":
            # Extract [CLS] token representation from the final transformer block
            feats = self.backbone.forward_features(x)
            if hasattr(self.backbone, "forward_head"):
                # timm >= 0.9 forward_head(feats, pre_logits=True)
                return self.backbone.forward_head(feats, pre_logits=True)
            elif isinstance(feats, torch.Tensor) and feats.ndim == 3:
                return feats[:, 0]  # CLS token
            elif isinstance(feats, torch.Tensor) and feats.ndim == 2:
                return feats
            return self.backbone.head_drop(self.backbone.fc_norm(feats[:, 0]))

        feats = self.backbone(x)
        pooled = self.pool(feats)
        return torch.flatten(pooled, 1)

    def get_cam_target_layer(self) -> nn.Module:
        """Return the optimal convolutional layer for Grad-CAM / Grad-CAM++ hooks.

        Returns:
            Target :class:`nn.Module` layer.

        Raises:
            ValueError: If called on a Transformer backbone (ViT uses Attention Rollout).
        """
        if self.model_name == "mobilenet_v3_large":
            # Last Conv2dNormActivation block in features
            return self.backbone[-1]
        elif self.model_name == "resnet50":
            # Last bottleneck block in layer4 (which is backbone[7])
            return self.backbone[7][-1]
        else:
            raise ValueError(
                f"Model '{self.model_name}' is a Vision Transformer. "
                "Use Attention Rollout instead of Grad-CAM."
            )

    # ── Two-Stage Transfer Learning Protocol ─────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for Stage 1 Linear Probing.

        Only the newly attached classification head remains trainable.
        """
        if self.model_name == "vit_tiny_patch16_224":
            for name, param in self.backbone.named_parameters():
                if "head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = True

        logger.info("Stage 1: Backbone FROZEN. Only classification head is trainable.")

    def unfreeze_backbone(
        self,
        unfreeze_from_block: int | None = None,
        unfreeze_layers: Sequence[str] | None = None,
    ) -> None:
        """Progressively unfreeze top layers for Stage 2 Fine-Tuning.

        Args:
            unfreeze_from_block: Integer block threshold (for MobileNetV3 or ViT).
            unfreeze_layers: Layer names to unfreeze (for ResNet-50, e.g. ['layer3', 'layer4']).
        """
        if self.model_name == "mobilenet_v3_large":
            threshold = unfreeze_from_block if unfreeze_from_block is not None else 14
            for idx, child in enumerate(self.backbone):
                requires_grad = idx >= threshold
                for param in child.parameters():
                    param.requires_grad = requires_grad
            # Head always trainable
            for param in self.head.parameters():
                param.requires_grad = True
            logger.info("Stage 2: Unfroze MobileNetV3 features[%d:].", threshold)

        elif self.model_name == "resnet50":
            layers_to_unfreeze = set(unfreeze_layers or ["layer3", "layer4"])
            # self.backbone has: 0:conv1, 1:bn1, 2:relu, 3:maxpool, 4:layer1, 5:layer2, 6:layer3, 7:layer4
            layer_map = {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}
            for layer_name, idx in layer_map.items():
                req_grad = layer_name in layers_to_unfreeze
                for param in self.backbone[idx].parameters():
                    param.requires_grad = req_grad
            for param in self.head.parameters():
                param.requires_grad = True
            logger.info("Stage 2: Unfroze ResNet-50 layers: %s.", list(layers_to_unfreeze))

        elif self.model_name == "vit_tiny_patch16_224":
            threshold = unfreeze_from_block if unfreeze_from_block is not None else 8
            # In timm ViT, blocks are self.backbone.blocks
            for idx, block in enumerate(self.backbone.blocks):
                req_grad = idx >= threshold
                for param in block.parameters():
                    param.requires_grad = req_grad
            # Norm and head are trainable
            if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "head") and self.backbone.head is not None:
                for param in self.backbone.head.parameters():
                    param.requires_grad = True
            logger.info("Stage 2: Unfroze ViT-Tiny blocks[%d:].", threshold)

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """Construct differential parameter groups with weight decay filtering.

        Standard academic best practice:
          - Applies differential learning rates (lower for backbone, higher for head).
          - Disables weight decay for 1D parameters (biases, BatchNorm / LayerNorm weights)
            to prevent underfitting and destabilizing normalizations.

        Args:
            backbone_lr: Learning rate for unfrozen backbone parameters.
            head_lr: Learning rate for classifier head parameters.
            weight_decay: Weight decay penalty for multi-dimensional weights.

        Returns:
            List of parameter group dictionaries compatible with PyTorch optimizers.
        """
        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Determine whether parameter is 1D (bias or normalization scale/shift)
            no_decay = param.ndim <= 1 or name.endswith(".bias")

            # Check if parameter belongs to the head
            is_head = (
                "head" in name
                or (self.head is not None and any(p is param for p in self.head.parameters()))
            )

            if is_head:
                if no_decay:
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)
            else:
                if no_decay:
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)

        groups: list[dict[str, Any]] = []
        if backbone_decay:
            groups.append({"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay})
        if backbone_no_decay:
            groups.append({"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0})
        if head_decay:
            groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
        if head_no_decay:
            groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})

        logger.debug(
            "Created %d parameter groups (decay/no-decay split across backbone & head).",
            len(groups),
        )
        return groups

    def count_parameters(self) -> dict[str, Any]:
        """Compute parameter metrics and estimated memory footprint.

        Returns:
            Dictionary with 'total_params', 'trainable_params', 'frozen_params',
            and 'size_mb'.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        size_mb = (total * 4) / (1024 * 1024)  # FP32: 4 bytes per param

        return {
            "model_name": self.model_name,
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "trainable_pct": (trainable / total * 100) if total > 0 else 0.0,
            "size_mb": round(size_mb, 2),
        }


# ── Model Factory ────────────────────────────────────────────────────────────

def build_model(
    config_or_name: dict[str, Any] | str,
    num_classes: int | None = None,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
) -> WeatherClassifier:
    """Build and return an initialized WeatherClassifier instance.

    Args:
        config_or_name: Either a loaded YAML configuration dictionary or a model name string.
        num_classes: Number of target classes (defaults to config value or 6).
        pretrained: Whether to load pre-trained ImageNet weights.
        dropout_rate: Dropout rate for classifier head.

    Returns:
        Configured :class:`WeatherClassifier` module.
    """
    if isinstance(config_or_name, dict):
        cfg = config_or_name
        model_cfg = cfg.get("model", {})
        dataset_cfg = cfg.get("dataset", {})

        model_name = model_cfg.get("name", "mobilenet_v3_large")
        n_classes = num_classes or dataset_cfg.get("num_classes", len(dataset_cfg.get("classes", [1] * 6)))
        is_pretrained = model_cfg.get("pretrained", pretrained)
        drop = model_cfg.get("dropout_rate", dropout_rate)
    else:
        model_name = str(config_or_name)
        n_classes = num_classes or 6
        is_pretrained = pretrained
        drop = dropout_rate

    return WeatherClassifier(
        model_name=model_name,
        num_classes=n_classes,
        pretrained=is_pretrained,
        dropout_rate=drop,
    )
