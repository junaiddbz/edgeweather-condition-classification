# syntax=docker/dockerfile:1
# Multi-stage build for a lean EdgeWeather production container

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch and all dependencies into /install
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        --prefix=/install \
        -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="EdgeWeather Project" \
      description="EdgeWeather: Real-time weather condition classification at the edge" \
      version="0.1.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY outputs/onnx/ outputs/onnx/
COPY .streamlit/ .streamlit/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash edgeweather
USER edgeweather

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
