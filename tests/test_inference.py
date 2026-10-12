"""
tests/test_inference.py
=======================
Unit tests for ONNX export, numerical parity verification, INT8 quantization,
and latency benchmarking profilers.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.benchmark import (
    benchmark_onnx,
    benchmark_pytorch,
    plot_latency_comparison,
)
from src.export_onnx import (
    export_to_onnx,
    quantize_onnx_model,
    validate_onnx_parity,
)
from src.models import build_model


@pytest.fixture(scope="module")
def onnx_artifacts():
    """Fixture providing exported and quantized MobileNetV3 models in a temp dir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model = build_model("mobilenet_v3_large", num_classes=6, pretrained=False)
        model.eval()

        onnx_fp32 = tmp_path / "mobilenet_v3_fp32.onnx"
        export_to_onnx(
            model=model,
            output_path=onnx_fp32,
            input_shape=(1, 3, 224, 224),
            opset_version=17,
            dynamic_batch=True,
        )

        onnx_int8, quant_stats = quantize_onnx_model(onnx_fp32)

        yield {
            "root": tmp_path,
            "model": model,
            "fp32_path": onnx_fp32,
            "int8_path": onnx_int8,
            "quant_stats": quant_stats,
        }


def test_export_to_onnx(onnx_artifacts):
    """Verify that export_to_onnx produces a valid, readable ONNX graph."""
    fp32_path = onnx_artifacts["fp32_path"]
    assert fp32_path.exists()
    assert fp32_path.stat().st_size > 1_000_000  # > 1 MB


def test_validate_onnx_parity(onnx_artifacts):
    """Verify bit-exact output parity between PyTorch and ONNX Runtime."""
    model = onnx_artifacts["model"]
    fp32_path = onnx_artifacts["fp32_path"]

    parity = validate_onnx_parity(model, fp32_path, input_shape=(1, 3, 224, 224), tolerance=1e-3)
    assert parity["passed"]
    assert parity["max_abs_diff"] < 1e-3


def test_quantize_onnx_model(onnx_artifacts):
    """Verify that dynamic INT8 quantization shrinks model size significantly."""
    int8_path = onnx_artifacts["int8_path"]
    stats = onnx_artifacts["quant_stats"]

    assert int8_path.exists()
    assert stats["reduction_pct"] > 15.0  # Demonstrates meaningful compression
    assert stats["int8_size_mb"] < stats["fp32_size_mb"]


def test_benchmark_pytorch(onnx_artifacts):
    """Verify PyTorch latency profiler metrics."""
    model = onnx_artifacts["model"]
    stats = benchmark_pytorch(model, warmup_iters=5, measure_iters=15, device="cpu")

    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "fps" in stats
    assert stats["p50_ms"] > 0.0
    assert stats["fps"] > 0.0


def test_benchmark_onnx(onnx_artifacts):
    """Verify ONNX Runtime latency profiler metrics."""
    fp32_path = onnx_artifacts["fp32_path"]
    stats = benchmark_onnx(fp32_path, warmup_iters=5, measure_iters=15)

    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "fps" in stats
    assert stats["p50_ms"] > 0.0
    assert stats["fps"] > 0.0


def test_plot_latency_comparison():
    """Verify latency bar chart figure generation."""
    data = {
        "model": ["MobileNetV3", "MobileNetV3", "ResNet50"],
        "engine": ["PyTorch FP32", "ONNX INT8", "PyTorch FP32"],
        "p50_ms": [18.5, 7.2, 45.1],
        "fps": [54.0, 138.8, 22.1],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "latency_plot.png"
        fig = plot_latency_comparison(df, save_path=save_path)

        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    plt.close("all")
