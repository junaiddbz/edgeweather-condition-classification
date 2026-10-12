"""
tests/test_explainability.py
============================
Unit tests for Grad-CAM, Grad-CAM++, Attention Rollout, heatmap overlays,
and 4-panel comparative XAI visualization.
"""

from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.dataset import DEFAULT_CLASSES
from src.explainability import (
    AttentionRolloutViT,
    GradCAM,
    generate_comparative_xai_grid,
    overlay_heatmap_on_image,
)
from src.models import build_model


def test_gradcam_mobilenet():
    """Verify Grad-CAM on MobileNetV3 produces valid normalized heatmap."""
    model = build_model("mobilenet_v3_large", num_classes=6, pretrained=False)
    cam = GradCAM(model, variant="gradcam")

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap, pred_class, conf = cam.generate_heatmap(dummy_input)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert 0 <= pred_class < 6
    assert 0.0 <= conf <= 1.0
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    cam.remove_hooks()


def test_gradcam_plus_plus_resnet():
    """Verify Grad-CAM++ variant on ResNet-50."""
    model = build_model("resnet50", num_classes=6, pretrained=False)
    cam = GradCAM(model, variant="gradcam++")

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap, pred_class, conf = cam.generate_heatmap(dummy_input)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert 0 <= pred_class < 6
    assert 0.0 <= conf <= 1.0
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    cam.remove_hooks()


def test_attention_rollout_vit():
    """Verify Attention Rollout on Vision Transformer (ViT-Tiny)."""
    model = build_model("vit_tiny_patch16_224", num_classes=6, pretrained=False)
    rollout = AttentionRolloutViT(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap, pred_class, conf = rollout.generate_rollout(dummy_input)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (14, 14)  # 14x14 patch grid for 224/16
    assert 0 <= pred_class < 6
    assert 0.0 <= conf <= 1.0
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_overlay_heatmap_on_image():
    """Verify heatmap overlay blending on RGB images."""
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    dummy_heat = np.random.rand(14, 14).astype(np.float32)

    blended = overlay_heatmap_on_image(dummy_img, dummy_heat, alpha=0.5)

    assert isinstance(blended, np.ndarray)
    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8


def test_comparative_xai_grid_generation():
    """Verify 4-panel comparative XAI grid generation across all three backbones."""
    m_mobile = build_model("mobilenet_v3_large", num_classes=6, pretrained=False)
    m_res = build_model("resnet50", num_classes=6, pretrained=False)
    m_vit = build_model("vit_tiny_patch16_224", num_classes=6, pretrained=False)

    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "xai_grid.png"
        fig, meta = generate_comparative_xai_grid(
            image_input=dummy_img,
            mobilenet_model=m_mobile,
            resnet_model=m_res,
            vit_model=m_vit,
            class_names=DEFAULT_CLASSES,
            save_path=save_path,
        )

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        assert "mobilenet" in meta
        assert "resnet" in meta
        assert "vit" in meta
    plt.close("all")
