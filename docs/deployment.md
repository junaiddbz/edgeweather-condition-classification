# Production Deployment and Edge Serving Guide

## Overview

This guide details the deployment architecture for EdgeWeather, covering model export to Open Neural Network Exchange (ONNX), dynamic INT8 quantization, interactive web application serving via Streamlit, containerization using Docker, and automated continuous integration.

---

## 1. Model Export and Quantization Pipeline

To transition trained PyTorch checkpoints into high-throughput edge environments, EdgeWeather includes an automated ONNX export and quantization engine (`src/export_onnx.py`).

### 1.1 Exporting PyTorch Checkpoints to ONNX

The export script maps the PyTorch computational graph into an optimized static ONNX representation with explicit input and output tensors:

```bash
python -m src.export_onnx \
    --checkpoint outputs/checkpoints/mobilenet_v3_large_best.pt \
    --output_dir outputs/onnx \
    --quantize
```

Key technical parameters applied during export:
- **Opset Version**: 17 (standardized cross-platform compatibility).
- **Dynamic Axes**: Batch size is declared dynamic (`batch_size: 0`), permitting both single-image edge inference ($N=1$) and batch workloads.
- **Graph Optimization**: Constant folding and redundant operator elimination are executed prior to serialization.

### 1.2 Numerical Parity Verification

To ensure that graph export introduces no numerical drift, the export pipeline conducts automated parity testing:
$$\max_{i} |y_{\text{PyTorch}}^{(i)} - y_{\text{ONNX}}^{(i)}| < \epsilon \quad (\epsilon = 10^{-4})$$

Exports failing the parity check are rejected prior to deployment.

### 1.3 Dynamic INT8 Quantization

Dynamic quantization converts FP32 weights into signed 8-bit integers (`int8`) without requiring representative calibration data:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="outputs/onnx/mobilenet_v3_large_fp32.onnx",
    model_output="outputs/onnx/mobilenet_v3_large_fp32_int8.onnx",
    weight_type=QuantType.QInt8,
    per_channel=True,
)
```

Quantized models achieve:
- Up to **70.9% disk space reduction** on transformer weights (ViT-Tiny: 21.2 MB $\rightarrow$ 6.17 MB).
- Up to **4.33× throughput speedup** over native PyTorch on single-threaded CPUs.

---

## 2. Interactive Web Application Architecture

EdgeWeather includes an interactive web dashboard implemented in Streamlit (`app/streamlit_app.py`) designed for monitoring edge vision pipelines in real time.

### 2.1 Core Capabilities
- **Multi-Engine Switching**: Toggle between PyTorch FP32, ONNX Runtime FP32, and ONNX Runtime INT8 on the fly.
- **Dynamic Adaptive Palette**: The application theme re-tints dynamically based on the predicted weather condition (pewter for fog, pale blue for frost, ochre for sandstorm).
- **High-Precision Telemetry**: Measures live inference latency in milliseconds ($p50$), calculates frame rate (FPS), and reports model size on disk.
- **Explainable AI (XAI)**: Displays side-by-side comparisons of raw imagery, composite saliency heatmaps, and normalized attention masks using Grad-CAM, Grad-CAM++, and Attention Rollout.
- **Hazard Advisories**: Real-time operational risk cards notifying downstream vehicle systems of traction reduction or camera lens moisture.
- **Academic Benchmark Explorer**: Interactive Altair charts rendering the Domain Safety Confusion Cost Matrix, normalized confusion matrices, and 15-bin ECE reliability diagrams.

### 2.2 Local Execution

To start the dashboard locally:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Access the interface in any web browser at `http://localhost:8501`.

---

## 3. Containerization and Production Serving

For reproducible deployment in cloud clusters or edge orchestrators, EdgeWeather provides a multi-stage Docker configuration.

### 3.1 Dockerfile Structure (`docker/Dockerfile`)

```dockerfile
# Stage 1: Build & Dependency Resolution
FROM python:3.10-slim AS builder

WORKDIR /build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Lean Production Runtime
FROM python:3.10-slim AS runner

WORKDIR /app
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY . /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3.2 Deployment via Docker Compose

To build and run the service using Docker Compose:

```bash
docker-compose -f docker/docker-compose.yml up --build -d
```

To inspect runtime logs:

```bash
docker-compose -f docker/docker-compose.yml logs -f
```

To terminate the container:

```bash
docker-compose -f docker/docker-compose.yml down
```

---

## 4. Automated Testing and Continuous Integration

All components are verified under an automated Continuous Integration pipeline (`.github/workflows/ci.yml`) using GitHub Actions.

### 4.1 Test Scope (45 Tests, 100% Pass Rate)

| Test Module | Coverage Scope | Verified Properties |
| :--- | :--- | :--- |
| `tests/test_dataset.py` | Data loading and augmentation | Image tensor shapes, Albumentations pipelines, data loader collation. |
| `tests/test_models.py` | Model architectures | Forward pass output tensor shapes across all 3 backbones, layer freezing. |
| `tests/test_loss.py` | Loss functions and calibration | Domain safety cost matrix ordering, backward gradient pass, ECE calculation. |
| `tests/test_inference.py` | Export and benchmarking | ONNX numerical parity with PyTorch, INT8 quantization integrity, latency timers. |
| `tests/test_explainability.py` | Interpretability tools | Grad-CAM, Grad-CAM++, Attention Rollout forward/backward hooks, heatmap blend. |
| `tests/test_app.py` | Web dashboard utilities | Image preprocessing, cached model loading, advisory dictionary coverage. |
| `tests/test_train.py` | Training loop | Two-stage training step, mixed precision AMP execution, checkpoint persistence. |

### 4.2 Running the Test Suite Locally

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
