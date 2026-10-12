"""
tests/test_models.py
====================
Unit tests for WeatherClassifier model factory, 2-stage transfer learning hooks,
forward pass tensor dimensions, and feature extractors.
"""

import pytest
import torch

from src.models import SUPPORTED_MODELS, build_model


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_forward_shape(model_name: str):
    """Verify that all three architectures output correct logits shape (B, num_classes)."""
    num_classes = 6
    batch_size = 2
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_feature_extraction_shape(model_name: str):
    """Verify penultimate feature extraction shape across backbones."""
    batch_size = 2
    model = build_model(model_name, num_classes=6, pretrained=False)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        features = model.extract_features(dummy_input)

    expected_dims = {
        "mobilenet_v3_large": 960,
        "resnet50": 2048,
        "vit_tiny_patch16_224": 192,
    }
    assert features.shape == (batch_size, expected_dims[model_name])
    assert not torch.isnan(features).any()


def test_transfer_learning_stages_mobilenet():
    """Verify backbone freezing and progressive unfreezing for MobileNetV3."""
    model = build_model("mobilenet_v3_large", num_classes=6, pretrained=False)

    # Initial state
    counts_init = model.count_parameters()
    assert counts_init["trainable_params"] == counts_init["total_params"]

    # Stage 1: Freeze backbone
    model.freeze_backbone()
    counts_stage1 = model.count_parameters()
    assert counts_stage1["trainable_params"] < counts_init["total_params"]
    # Only head is trainable
    for p in model.head.parameters():
        assert p.requires_grad
    for p in model.backbone.parameters():
        assert not p.requires_grad

    # Stage 2: Unfreeze top layers
    model.unfreeze_backbone(unfreeze_from_block=14)
    counts_stage2 = model.count_parameters()
    assert counts_stage2["trainable_params"] > counts_stage1["trainable_params"]


def test_transfer_learning_stages_resnet():
    """Verify backbone freezing and progressive unfreezing for ResNet-50."""
    model = build_model("resnet50", num_classes=6, pretrained=False)

    model.freeze_backbone()
    counts_stage1 = model.count_parameters()
    assert counts_stage1["trainable_params"] < counts_stage1["total_params"]

    # Unfreeze layer4
    model.unfreeze_backbone(unfreeze_layers=["layer4"])
    counts_stage2 = model.count_parameters()
    assert counts_stage2["trainable_params"] > counts_stage1["trainable_params"]


def test_parameter_groups_differentiation():
    """Verify differential parameter groups and weight decay separation."""
    model = build_model("mobilenet_v3_large", num_classes=6, pretrained=False)
    model.unfreeze_backbone(unfreeze_from_block=14)

    groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-4)
    assert len(groups) >= 2

    lrs = {g["lr"] for g in groups}
    assert 1e-5 in lrs
    assert 1e-4 in lrs

    weight_decays = {g["weight_decay"] for g in groups}
    assert 0.0 in weight_decays
    assert 1e-4 in weight_decays


def test_cam_target_layers():
    """Verify CAM target layers are accessible for CNNs and rejected for ViT."""
    mobilenet = build_model("mobilenet_v3_large", pretrained=False)
    layer_mobile = mobilenet.get_cam_target_layer()
    assert isinstance(layer_mobile, torch.nn.Module)

    resnet = build_model("resnet50", pretrained=False)
    layer_res = resnet.get_cam_target_layer()
    assert isinstance(layer_res, torch.nn.Module)

    vit = build_model("vit_tiny_patch16_224", pretrained=False)
    with pytest.raises(ValueError, match="Attention Rollout"):
        vit.get_cam_target_layer()
