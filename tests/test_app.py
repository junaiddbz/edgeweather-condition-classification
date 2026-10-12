"""
tests/test_app.py
=================
Unit tests for EdgeWeather Streamlit web application helpers and telemetry.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from app.streamlit_app import (
    DEFAULT_CLASSES,
    SAFETY_ADVISORIES,
    get_preloaded_sample_paths,
    load_onnx_session,
    load_pytorch_model,
    preprocess_image_for_model,
)


def test_safety_advisories_coverage() -> None:
    """Ensure every weather class has an advisory with required metadata."""
    for cls in DEFAULT_CLASSES:
        assert cls in SAFETY_ADVISORIES
        adv = SAFETY_ADVISORIES[cls]
        assert "level" in adv
        assert "icon" in adv
        assert "title" in adv
        assert "desc" in adv


def test_preprocess_image_for_model() -> None:
    """Validate preprocessing converts PIL to RGB array and normalized tensor."""
    dummy_pil = Image.new("RGB", (300, 200), color=(120, 150, 180))
    rgb_arr, tensor = preprocess_image_for_model(dummy_pil, target_size=(224, 224))

    assert isinstance(rgb_arr, np.ndarray)
    assert rgb_arr.shape == (200, 300, 3)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_load_pytorch_model_caching() -> None:
    """Test PyTorch model loading without checkpoint for inference."""
    model = load_pytorch_model("mobilenet_v3_large")
    assert model is not None
    assert not model.training  # Model is set to eval mode


def test_load_onnx_session_fallback_on_missing() -> None:
    """Ensure missing ONNX paths return None gracefully."""
    session = load_onnx_session("non_existent_model.onnx")
    assert session is None


def test_get_preloaded_sample_paths() -> None:
    """Test scanning preloaded synthetic samples."""
    samples = get_preloaded_sample_paths()
    assert isinstance(samples, dict)
    for cls_name, path in samples.items():
        assert cls_name in DEFAULT_CLASSES
        assert Path(path).exists()
