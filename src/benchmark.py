"""
src/benchmark.py
================
Edge inference latency and throughput benchmarking profiler for EdgeWeather.

Academic & Industry Features:
  - Multi-engine Profiling: PyTorch FP32 vs. ONNX Runtime FP32 vs. ONNX Runtime INT8.
  - Rigorous Latency Statistics: Warm-up cycles, mean, std, median (p50), p95, p99 (tail latency).
  - Throughput Quantification: Frames Per Second (FPS).
  - Memory Footprint Tracking: On-disk model sizes in Megabytes (MB).
  - Artifact Generation:
      - CSV summary table (outputs/results/benchmark_results.csv)
      - Publication-quality comparative bar chart (outputs/figures/latency_comparison.png)
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

from src.models import build_model
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


# ── PyTorch Engine Profiler ──────────────────────────────────────────────────


def benchmark_pytorch(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    warmup_iters: int = 50,
    measure_iters: int = 300,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Profile latency and throughput of a PyTorch model.

    Args:
        model: Evaluated PyTorch model.
        input_shape: Tensor dimension (B, C, H, W).
        warmup_iters: Discarded initial iterations to stabilize CPU caches.
        measure_iters: Number of benchmark evaluation runs.
        device: Computation device ('cpu' or 'cuda').

    Returns:
        Dictionary containing p50, p95, p99, mean, std (in milliseconds) and FPS.
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=dev)
    is_cuda = dev.type == "cuda"

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()

    # Measurement
    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(measure_iters):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "fps": round(fps, 1),
    }


# ── ONNX Runtime Profiler ────────────────────────────────────────────────────


def benchmark_onnx(
    onnx_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    warmup_iters: int = 50,
    measure_iters: int = 300,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> dict[str, float]:
    """Profile latency and throughput of an ONNX / Quantized INT8 model via ONNX Runtime.

    Args:
        onnx_path: Filepath to .onnx model.
        input_shape: Input dimension.
        warmup_iters: Warmup cycles.
        measure_iters: Benchmark cycles.
        providers: ONNX Runtime execution provider list.

    Returns:
        Dictionary containing p50, p95, p99, mean, std (in milliseconds) and FPS.
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), opts, providers=list(providers))

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_iters):
        _ = session.run([output_name], {input_name: dummy_input})

    # Measurement
    latencies_ms: list[float] = []
    for _ in range(measure_iters):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "fps": round(fps, 1),
    }


# ── Latency Plotting ─────────────────────────────────────────────────────────


def plot_latency_comparison(
    df_results: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate comparative horizontal bar chart of median latency (p50) and FPS.

    Args:
        df_results: DataFrame with columns 'engine', 'p50_ms', 'fps', 'model'.
        save_path: Optional destination path to save PNG figure.

    Returns:
        Generated :class:`matplotlib.figure.Figure`.
    """
    fig_height = max(5.0, len(df_results) * 0.65)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_height))

    # Engine label formatting
    labels = [f"{row['model']}\n({row['engine']})" for _, row in df_results.iterrows()]
    y_pos = np.arange(len(labels))

    colors = [
        "#2b5c8f"
        if "INT8" in eng
        else "#4a90e2"
        if "ONNX" in eng
        else "#8b5cf6"
        if "CUDA" in eng or "GPU" in eng
        else "#10b981"
        for eng in df_results["engine"]
    ]

    # Panel 1: Median Latency (Lower is better)
    bars1 = ax1.barh(y_pos, df_results["p50_ms"], color=colors, edgecolor="black", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Latency p50 (ms) — Lower is Better", fontsize=11, fontweight="bold")
    ax1.set_title("Edge Inference Latency (CPU)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    for bar, val in zip(bars1, df_results["p50_ms"], strict=False):
        ax1.text(
            val + max(df_results["p50_ms"]) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} ms",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Panel 2: FPS Throughput (Higher is better)
    bars2 = ax2.barh(y_pos, df_results["fps"], color=colors, edgecolor="black", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Throughput (FPS) — Higher is Better", fontsize=11, fontweight="bold")
    ax2.set_title("Inference Throughput (FPS)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    for bar, val in zip(bars2, df_results["fps"], strict=False):
        ax2.text(
            val + max(df_results["fps"]) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} FPS",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        ensure_dir(out_p.parent)
        fig.savefig(out_p, dpi=300)
        logger.info("Saved latency comparison figure to %s", out_p)

    return fig


# ── Comprehensive Benchmark Pipeline ─────────────────────────────────────────


def run_benchmarks(
    onnx_dir: str | Path = "outputs/onnx",
    checkpoints_dir: str | Path = "outputs/checkpoints",
    output_dir: str | Path = "outputs/results",
    figures_dir: str | Path = "outputs/figures",
    warmup: int = 20,
    iterations: int = 100,
) -> pd.DataFrame:
    """Run cross-engine benchmarks across PyTorch FP32, ONNX FP32, and ONNX INT8.

    Args:
        onnx_dir: Directory containing exported .onnx models.
        checkpoints_dir: Directory containing .pt checkpoints.
        output_dir: Directory to save benchmark_results.csv.
        figures_dir: Directory to save latency_comparison.png.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.

    Returns:
        Structured :class:`pandas.DataFrame` of benchmark results.
    """
    records: list[dict[str, Any]] = []
    onnx_p = Path(onnx_dir)
    ckpt_p = Path(checkpoints_dir)

    models_to_test = ["mobilenet_v3_large", "resnet50", "vit_tiny_patch16_224"]

    for model_name in models_to_test:
        logger.info("=" * 60)
        logger.info("Benchmarking architecture: %s", model_name)
        logger.info("=" * 60)

        # 1. PyTorch FP32
        ckpt_candidate = ckpt_p / f"{model_name}_best.pt"
        data = None
        n_classes = 11
        if ckpt_candidate.exists():
            data = torch.load(ckpt_candidate, map_location="cpu", weights_only=False)
            if "config" in data:
                n_classes = data["config"].get("dataset", {}).get("num_classes", 11)
            elif "num_classes" in data:
                n_classes = data["num_classes"]

        model = build_model(model_name, num_classes=n_classes, pretrained=False)
        if data is not None and "model_state_dict" in data:
            model.load_state_dict(data["model_state_dict"])

        pt_stats = benchmark_pytorch(
            model, warmup_iters=warmup, measure_iters=iterations, device="cpu"
        )
        pt_size_mb = model.count_parameters()["size_mb"]
        records.append(
            {
                "model": model_name,
                "engine": "PyTorch FP32 (CPU)",
                "size_mb": pt_size_mb,
                **pt_stats,
            }
        )

        # 1b. PyTorch CUDA (GPU) if available
        if torch.cuda.is_available():
            try:
                model_gpu = build_model(model_name, num_classes=n_classes, pretrained=False).to(
                    "cuda"
                )
                if data is not None and "model_state_dict" in data:
                    model_gpu.load_state_dict(data["model_state_dict"])
                cuda_stats = benchmark_pytorch(
                    model_gpu, warmup_iters=warmup, measure_iters=iterations, device="cuda"
                )
                records.append(
                    {
                        "model": model_name,
                        "engine": "PyTorch CUDA (GPU)",
                        "size_mb": pt_size_mb,
                        **cuda_stats,
                    }
                )
                del model_gpu
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning("Failed to benchmark %s on CUDA: %s", model_name, e)

        # 2. ONNX FP32
        onnx_fp32 = onnx_p / f"{model_name}_fp32.onnx"
        if onnx_fp32.exists():
            onnx_stats = benchmark_onnx(onnx_fp32, warmup_iters=warmup, measure_iters=iterations)
            onnx_size_mb = round(onnx_fp32.stat().st_size / (1024 * 1024), 2)
            records.append(
                {
                    "model": model_name,
                    "engine": "ONNX Runtime FP32",
                    "size_mb": onnx_size_mb,
                    **onnx_stats,
                }
            )

        # 3. ONNX INT8 Quantized
        onnx_int8 = onnx_p / f"{model_name}_fp32_int8.onnx"
        if onnx_int8.exists():
            int8_stats = benchmark_onnx(onnx_int8, warmup_iters=warmup, measure_iters=iterations)
            int8_size_mb = round(onnx_int8.stat().st_size / (1024 * 1024), 2)
            records.append(
                {
                    "model": model_name,
                    "engine": "ONNX Runtime INT8",
                    "size_mb": int8_size_mb,
                    **int8_stats,
                }
            )

    df = pd.DataFrame(records)

    # Save CSV
    out_dir_path = Path(output_dir)
    ensure_dir(out_dir_path)
    csv_file = out_dir_path / "benchmark_results.csv"
    df.to_csv(csv_file, index=False)
    logger.info("Exported benchmark metrics table to %s", csv_file)

    # Generate Figure
    fig_dir_path = Path(figures_dir)
    ensure_dir(fig_dir_path)
    fig_file = fig_dir_path / "latency_comparison.png"
    plot_latency_comparison(df, save_path=fig_file)
    plt.close("all")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeWeather: Inference Benchmark Profiler")
    parser.add_argument("--onnx_dir", type=Path, default=Path("outputs/onnx"))
    parser.add_argument("--checkpoints_dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--figures_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    run_benchmarks(
        onnx_dir=args.onnx_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.out_dir,
        figures_dir=args.figures_dir,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
