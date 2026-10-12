"""
tests/test_app.py
=================
Unit tests for EdgeWeather Streamlit web application helpers and telemetry.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.streamlit_app import (
    ARCH_SPECS,
    DEFAULT_CLASSES,
    get_preloaded_sample_paths,
    load_onnx_session,
    load_pytorch_model,
    preprocess_image_for_model,
)


def test_arch_specs_coverage() -> None:
    """Ensure ARCH_SPECS contains specifications for all benchmark models."""
    assert len(ARCH_SPECS) == 3
    for spec in ARCH_SPECS:
        assert "name" in spec
        assert "key" in spec
        assert "params_m" in spec
        assert "latency_ms" in spec
        assert spec["params_m"] > 0
        assert spec["latency_ms"] > 0


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
