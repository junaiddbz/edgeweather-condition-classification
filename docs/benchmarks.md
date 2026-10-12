# Empirical Hardware and Multi-Engine Benchmark Matrix

## Overview

In real-time automotive vision, automated UAV navigation, and edge IoT meteorological stations, classification algorithms must operate within strict computational budgets:
- **Latency Budget**: Sub-10 millisecond median inference latency ($p50 < 10.0\text{ ms}$) to enable 100+ Hz processing loops.
- **Resource Footprint**: Minimal storage requirements (<20 MB) to fit in embedded flash memory.
- **Throughput**: Maximum frames per second (FPS) on standard single-threaded CPU architectures without mandatory discrete GPU hardware.

This document details the multi-engine benchmarking methodology and empirical performance matrix across PyTorch CPU, PyTorch CUDA GPU, ONNX Runtime FP32, and ONNX Runtime INT8.

---

## 1. Benchmarking Environment and Test Methodology

### 1.1 Hardware Specifications
- **CPU Target**: Intel Core i7 / AMD Ryzen (single-threaded evaluation profile simulating modern edge x86/ARM compute boards).
- **GPU Target**: NVIDIA GeForce GTX 1050 Ti (4GB VRAM, Pascal Architecture, Compute Capability 6.1).
- **Host OS**: Windows 11 / Linux compatible.
- **Python Runtime**: Python 3.10, PyTorch 2.5.1 with CUDA 12.1 runtime.
- **Inference Engine**: ONNX Runtime v1.20 with `CPUExecutionProvider` (intra-op thread pool: 4 threads).

### 1.2 Measurement Protocol
- **Warm-Up Cycles**: 20 unmeasured inferences to ensure CPU cache residency and GPU kernel compilation.
- **Statistical Runs**: 100 consecutive timed inference runs on batch size $N = 1$ with $224 \times 224 \times 3$ input tensors.
- **Timing Primitive**: High-resolution monotonic clock (`time.perf_counter()`), complemented by `torch.cuda.Event` with CUDA stream synchronization when evaluating GPU runtimes.
- **Reported Percentiles**: Mean, standard deviation ($\sigma$), median ($p50$), 95th percentile ($p95$), and 99th percentile ($p99$).

---

## 2. Comprehensive Empirical Benchmark Matrix

The full experimental matrix evaluating all 3 architectures across 4 deployment configurations:

| Architecture | Inference Engine & Precision | Target Hardware | Model Size (MB) | Mean Latency (ms) | Std Dev ($\sigma$, ms) | Median Latency ($p50$, ms) | 95th Percentile ($p95$, ms) | 99th Percentile ($p99$, ms) | Throughput (FPS) | Speedup vs CPU Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | **ONNX Runtime INT8** | **Edge CPU** | **12.53** | **2.98** | **0.53** | **2.88** | **3.51** | **5.30** | **335.8** | **4.33×** |
| MobileNetV3-Large | ONNX Runtime FP32 | Edge CPU | 16.07 | 3.17 | 0.60 | 3.07 | 4.03 | 5.72 | 315.1 | 4.07× |
| MobileNetV3-Large | PyTorch CUDA FP32 | GTX 1050 Ti | 16.08 | 8.45 | 1.76 | 8.14 | 11.81 | 14.36 | 118.3 | 1.53× |
| MobileNetV3-Large | PyTorch FP32 | Edge CPU | 16.08 | 12.87 | 1.41 | 12.48 | 15.89 | 17.04 | 77.7 | 1.00× *(Baseline)* |
| **ResNet-50** | **PyTorch CUDA FP32** | **GTX 1050 Ti** | **89.76** | **11.51** | **0.71** | **11.17** | **13.00** | **14.28** | **86.9** | **3.77×** |
| ResNet-50 | ONNX Runtime INT8 | Edge CPU | 89.64 | 21.89 | 1.35 | 21.53 | 23.59 | 25.03 | 45.7 | 1.95× |
| ResNet-50 | ONNX Runtime FP32 | Edge CPU | 89.69 | 22.43 | 1.35 | 21.99 | 24.07 | 26.75 | 44.6 | 1.91× |
| ResNet-50 | PyTorch FP32 | Edge CPU | 89.76 | 42.07 | 1.65 | 42.08 | 44.39 | 46.26 | 23.8 | 1.00× *(Baseline)* |
| **ViT-Tiny** (`patch16_224`) | **PyTorch CUDA FP32** | **GTX 1050 Ti** | **21.08** | **5.46** | **0.39** | **5.31** | **6.34** | **6.60** | **183.0** | **2.52×** |
| ViT-Tiny (`patch16_224`) | **ONNX Runtime INT8** | **Edge CPU** | **6.17** | **7.24** | **1.16** | **7.27** | **9.32** | **11.69** | **138.0** | **1.84×** |
| ViT-Tiny (`patch16_224`) | ONNX Runtime FP32 | Edge CPU | 21.24 | 10.96 | 1.25 | 10.58 | 13.60 | 14.73 | 91.3 | 1.26× |
| ViT-Tiny (`patch16_224`) | PyTorch FP32 | Edge CPU | 21.08 | 13.41 | 1.05 | 13.37 | 14.79 | 17.50 | 74.6 | 1.00× *(Baseline)* |

---

## 3. Key Architectural and Hardware Insights

### 3.1 Edge CPU Champion: MobileNetV3-Large with INT8 Quantization
- Under ONNX Runtime dynamic INT8 quantization, MobileNetV3-Large achieves a **median latency of 2.88 ms** and a **throughput of 335.8 FPS** on standard CPU hardware.
- This represents a **4.33× speedup** over native PyTorch CPU execution and operates well within the 10 ms real-time operational budget.
- Tail latency remains tightly bounded, with 99% of requests completed in under 5.30 ms.

### 3.2 GPU Acceleration Dynamics on Discrete Hardware
- On deep convolutional backbones with large dense matrix operations (ResNet-50), hardware GPU acceleration provides substantial gains: latency drops from 42.08 ms on CPU to **11.17 ms on the GTX 1050 Ti** (a **3.77× speedup**, increasing throughput from 23.8 to 86.9 FPS).
- For ViT-Tiny, GPU execution achieves **5.31 ms median latency** and **183.0 FPS**, demonstrating the efficiency of parallel self-attention computation on tensor cores.
- For lightweight depthwise-separable CNNs (MobileNetV3), GPU overheads (host-to-device memory copies and kernel launch latency) limit the acceleration benefit (8.14 ms GPU vs. 2.88 ms CPU INT8). For edge deployments without discrete GPUs, CPU INT8 remains the preferred choice.

### 3.3 Dynamic INT8 Quantization Footprint Reduction
- Dynamic quantization converts FP32 linear and dense layer weights into signed 8-bit integers (`int8`), generating substantial storage and bandwidth reductions:
  - **ViT-Tiny**: Model size drops from **21.24 MB to 6.17 MB**, representing a **70.9% disk space reduction**, while latency drops from 13.37 ms to 7.27 ms.
  - **MobileNetV3-Large**: Disk footprint decreases from 16.07 MB to 12.53 MB.
  - **ResNet-50**: Standard dynamic quantization targets fully-connected layers, maintaining CNN convolutional layers in FP32 format; full int8 static quantization with calibration data can be applied for further reduction.

---

## 4. Latency Distribution Analysis

Analyzing the ratio between worst-case tail latency ($p99$) and median latency ($p50$) provides insight into execution determinism:

```
Latency Determinism Index (p99 / p50):
- MobileNetV3-Large (ONNX INT8): 5.30 ms / 2.88 ms = 1.84x  [Highly deterministic]
- ViT-Tiny (PyTorch CUDA GPU):    6.60 ms / 5.31 ms = 1.24x  [Extremely deterministic]
- ResNet-50 (PyTorch CPU):        46.26 ms / 42.08 ms = 1.10x [Consistent, high latency]
```

Low tail latency variance is vital in autonomous navigation systems where processing jitter can destabilize real-time control loops.

---

## 5. Deployment Recommendations by Operational Envelope

| Operational Scenario | Recommended Architecture | Recommended Engine | Target Hardware | Latency / FPS Profile | Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Battery-Powered Edge IoT / Low-Power Robotics** | **MobileNetV3-Large** | **ONNX Runtime INT8** | Low-power CPU (ARM / x86) | **2.88 ms / 335.8 FPS** | Minimal memory footprint (12.5 MB), sub-3ms latency, zero GPU power draw. |
| **Autonomous Vehicles with Embedded GPU** | **ResNet-50** | **PyTorch / TensorRT** | Embedded GPU (Jetson / dGPU) | **11.17 ms / 86.9 FPS** | Highest raw classification accuracy (87.72%), strong feature representation on deep residual layers. |
| **UAV Real-Time Environmental Vision** | **ViT-Tiny** | **PyTorch CUDA or ORT INT8** | Low-tier GPU or CPU | **5.31 ms / 183.0 FPS** | Global attention modeling, 6.17 MB quantized footprint, rapid self-attention throughput. |

---

## 6. Benchmarking Scope and Limitations

These measurements were taken on specific consumer hardware under controlled, single-image, single-threaded conditions. The numbers should be interpreted with the following caveats:

- **Not production-representative**: Real deployments involve concurrent processes, OS scheduling jitter, memory pressure from other applications, and camera capture overhead. Measured latency will be higher in practice.
- **Single-threaded CPU profiling**: ONNX Runtime was configured with an intra-op thread pool of 4 threads for the results shown. ARM-based edge boards (Raspberry Pi, Jetson Nano) will produce different throughput profiles.
- **No TensorRT baseline**: NVIDIA TensorRT optimization was not evaluated. TensorRT with FP16 or INT8 precision on a Jetson or dGPU would likely outperform all CPU results by a substantial margin for ResNet-50 and ViT-Tiny.
- **Dynamic INT8 quantization only**: Only dynamic quantization (weight-only, no activation quantization calibration) was applied. Static INT8 quantization with a representative calibration dataset would yield further latency reductions, particularly for ResNet-50 convolutional layers.
- **Batch size 1**: All benchmarks evaluate single-image inference. Batched inference would increase throughput substantially for GPU deployments.

