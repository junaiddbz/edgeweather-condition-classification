# Methodology and Theoretical Framework

## Overview

This document provides the mathematical foundations and design rationale for each component of the EdgeWeather pipeline. It covers dataset formulation, the deduplication algorithm, the custom loss function, the three model architectures and their fine-tuning strategy, calibration measurement, and explainability techniques.

The project is an academic portfolio exercise. The formulations here are implementations of established methods (perceptual hashing, Union-Find, Grad-CAM, ECE) applied to a specific weather classification problem, not novel theoretical contributions. The goal is to demonstrate principled design choices and rigorous evaluation, not to propose new algorithms.

---

## 1. Dataset Formulation and 11 Atmospheric Categories

The framework models meteorological phenomena across 11 distinct classes, representing adverse weather conditions and optical atmospheric events:

1. **Dew**: Condensation droplets on surfaces under calm, humid conditions.
2. **Fog**: Suspended microscopic water droplets reducing horizontal visibility below 1 kilometer.
3. **Frost**: Crystalline ice structures deposited on ground surfaces below freezing.
4. **Glaze**: Dense, transparent ice coating formed by freezing rain on freezing surfaces (black ice).
5. **Hail**: Solid precipitation pellets formed in strong convective updrafts.
6. **Lightning**: High-energy atmospheric electrical discharges producing intense illumination.
7. **Rain**: Liquid water droplets precipitating under convective or stratiform systems.
8. **Rainbow**: Optical refraction and reflection of sunlight through water droplets.
9. **Rime**: Rough, opaque white ice formed by rapid freezing of supercooled fog droplets.
10. **Sandstorm**: High-velocity particulate suspension drastically degrading visual optical range.
11. **Snow**: Atmospheric water vapor frozen into ice crystals precipitating as flakes.

The raw dataset originates from the [`dazzle-nu/autotrain-data-weather-classification`](https://huggingface.co/datasets/dazzle-nu/autotrain-data-weather-classification) repository on Hugging Face, comprising 6,862 real outdoor images across the 11 target classes. It can be automatically downloaded and extracted into `data/raw/` using `python -m src.extract_dataset`.

---

## 2. Dual Perceptual Hashing Deduplication Pipeline

### 2.1 The Problem of Cross-Split Data Leakage

Web-scraped and surveillance datasets contain pervasive duplicate images resulting from:
- High-frequency burst-mode captures (consecutive frames differing by minimal camera jitter).
- Minor compression and re-encoding artifacts.
- Mirror-flipped or slightly cropped visual scenes.

When random sampling is applied to create training, validation, and test splits, near-identical frames are distributed across partitions. This creates artificial test leakage, yielding artificially inflated evaluation metrics that do not generalize to genuine out-of-distribution deployments.

### 2.2 Dual Hashing Formulation

To eliminate data leakage, EdgeWeather implements a dual-hash perceptual distance metric combining discrete cosine transform (DCT) frequency hashing with horizontal gradient difference hashing:

1. **Perceptual Hash ($pHash$)**:
   The input image $I$ is converted to greyscale, downsampled to a $32 \times 32$ matrix, and transformed via a 2D Discrete Cosine Transform (DCT):
   $$D(u, v) = \frac{2}{N} C(u) C(v) \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} I(x, y) \cos \left[ \frac{(2x+1)u\pi}{2N} \right] \cos \left[ \frac{(2y+1)v\pi}{2N} \right]$$
   The top-left $8 \times 8$ low-frequency coefficients (excluding the DC component $D(0,0)$) are retained. The 64-bit hash is constructed by thresholding against the median coefficient value:
   $$h_{pHash}(u, v) = \begin{cases} 1 & \text{if } D(u, v) > \text{median}(D) \\ 0 & \text{otherwise} \end{cases}$$

2. **Difference Hash ($dHash$)**:
   The image is resized to an $8 \times 9$ grid. Gradients between adjacent horizontal pixels are computed:
   $$h_{dHash}(x, y) = \begin{cases} 1 & \text{if } P(x+1, y) > P(x, y) \\ 0 & \text{otherwise} \end{cases}$$

3. **Composite Distance Metric**:
   Two images $I_a$ and $I_b$ are classified as near-duplicates if both Hamming distances fall within threshold $\tau$:
   $$\mathcal{D}(I_a, I_b) = \min \Big( \text{Hamming}\big(h_{pHash}(I_a), h_{pHash}(I_b)\big), \; \text{Hamming}\big(h_{dHash}(I_a), h_{dHash}(I_b)\big) \Big) \le \tau$$
   A threshold of $\tau = 4$ bits out of 64 provides strict discrimination without false-positive grouping across visually distinct scenes of the same category.

### 2.3 Disjoint-Set Union-Find Clustering

Pairs satisfying $\mathcal{D}(I_a, I_b) \le \tau$ form edges in a duplicate graph $G = (V, E)$. Connected components are identified using a Disjoint-Set Union-Find data structure with path compression and union-by-rank, operating in near-linear time $O(N \cdot \alpha(N))$:

```
Algorithm: Near-Duplicate Graph Partitioning
1. Initialize parent[i] = i for all i in [0, N-1]
2. For each pair (i, j) with Hamming distance <= threshold:
     Union(i, j)
3. Form connected clusters C_k = {i | Find(i) == root_k}
4. Prune duplicates within each cluster, retaining the deterministic canonical representative
5. Assign clusters atomically to target partitions (Train: 70%, Val: 15%, Test: 15%)
```

**Results**: Out of 6,862 raw images, 400 duplicate clusters containing 459 duplicate frames were pruned. The clean dataset of 6,403 images was partitioned into:
- **Training**: 4,482 images (70%)
- **Validation**: 960 images (15%)
- **Test**: 961 images (15%)

Cross-split overlap verification confirms exactly 0 overlapping frames across all partitions.

---

## 3. Domain Safety Confusion-Aware Loss Engine

### 3.1 Limitation of Symmetric Cross-Entropy

Standard multi-class categorical cross-entropy treats all misclassifications as equidistant:
$$\mathcal{L}_{\text{CE}}(y, \hat{p}) = -\sum_{c=1}^{K} y_c \log \hat{p}_c$$

In safety-critical robotics and autonomous transport, the real-world operational cost of classification errors is highly asymmetric:
- Misclassifying `rime` as `frost` is a benign perceptual near-miss; vehicle traction controls and temperature alerts remain in winter operation modes.
- Misclassifying dense `fog`, black ice `glaze`, severe `sandstorm`, or torrential `rain` as `rainbow` or `dew` is catastrophic; vehicular control systems fail to adjust braking distance or engage visibility aids.

### 3.2 Formulation of Confusion-Aware Loss

We formulate the **Confusion-Aware Loss** ($\mathcal{L}_{\text{CAL}}$) as an additive risk-regularized objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LS-CE}}(y, \hat{p}) + \lambda \sum_{c=1}^{K} C_{y, c} \cdot \hat{p}_c$$

Where:
- $\mathcal{L}_{\text{LS-CE}}$ is cross-entropy with label smoothing parameter $\epsilon = 0.1$:
  $$y_k^{\text{smooth}} = (1 - \epsilon) y_k + \frac{\epsilon}{K}$$
- $\hat{p} = \text{Softmax}(z)$ is the predicted posterior probability distribution across $K = 11$ classes.
- $C \in \mathbb{R}^{K \times K}$ is the pre-computed asymmetric Domain Safety Cost Matrix.
- $\lambda$ is the risk penalty multiplier ($\lambda = 0.30$).

### 3.3 Domain Cost Tensor Definition

The cost matrix $C$ is structured as follows:

| Ground Truth Category ($y$) | Predicted Class ($c$) | Cost Weight $C_{y, c}$ | Rationale |
| :--- | :--- | :---: | :--- |
| Any category | Correct category ($c = y$) | 0.0 | Optimal classification |
| Perceptual adjacent pairs | `rime` $\leftrightarrow$ `frost`<br>`glaze` $\leftrightarrow$ `frost`<br>`rain` $\leftrightarrow$ `snow` | 0.5 | Shared physical response profile; minimal risk |
| Standard errors | Generic distinct pairs | 1.0 | Standard classification penalty |
| Severe hazards (`fog`, `glaze`, `hail`, `rain`, `sandstorm`, `snow`) | Reassuring / clear classes (`rainbow`, `dew`) | 3.0 | Critical hazard: false reassurance leading to failure to decelerate |

```
Asymmetric Cost Structure:
                  Predicted: Dew  Fog  Frost  Glaze  Hail  Lightn  Rain  Rainbow  Rime  Sand  Snow
Ground Truth: Dew            [0.0  1.0   1.0    1.0   1.0    1.0   1.0     1.0    1.0   1.0   1.0]
              Fog            [3.0  0.0   1.0    1.0   1.0    1.0   1.0     3.0    1.0   1.0   1.0]
              Frost          [1.0  1.0   0.0    0.5   1.0    1.0   1.0     1.0    0.5   1.0   1.0]
              Glaze          [3.0  1.0   0.5    0.0   1.0    1.0   1.0     3.0    1.0   1.0   1.0]
              Hail           [3.0  1.0   1.0    1.0   0.0    1.0   1.0     3.0    1.0   1.0   1.0]
              Lightning      [1.0  1.0   1.0    1.0   1.0    0.0   1.0     1.0    1.0   1.0   1.0]
              Rain           [3.0  1.0   1.0    1.0   1.0    1.0   0.0     3.0    1.0   1.0   0.5]
              Rainbow        [1.0  1.0   1.0    1.0   1.0    1.0   1.0     0.0    1.0   1.0   1.0]
              Rime           [1.0  1.0   0.5    1.0   1.0    1.0   1.0     1.0    0.0   1.0   1.0]
              Sandstorm      [3.0  1.0   1.0    1.0   1.0    1.0   1.0     3.0    1.0   0.0   1.0]
              Snow           [3.0  1.0   1.0    1.0   1.0    1.0   0.5     3.0    1.0   1.0   0.0]
```

---

## 4. Model Architectures and Transfer Learning Strategy

We analyze three distinct deep learning paradigms:

1. **MobileNetV3-Large (Edge Convolutional Network)**:
   - Architecture: Inverted residual bottleneck blocks with hard-swish non-linearities and Squeeze-and-Excitation (SE) channel attention modules.
   - Parameters: 4.22 Million.
   - Purpose: Low-latency, memory-efficient inference targeted for embedded processors.

2. **ResNet-50 (Deep Residual Network)**:
   - Architecture: 50-layer convolutional residual network featuring identity shortcut mappings and $3 \times 3$ bottleneck convolutions.
   - Parameters: 23.53 Million.
   - Purpose: High-capacity spatial feature baseline for benchmarking deep convolutional representations.

3. **ViT-Tiny (Vision Transformer)**:
   - Architecture: Vision Transformer (`vit_tiny_patch16_224`) dividing $224 \times 224$ images into 196 non-overlapping $16 \times 16$ patches, followed by linear projection, positional encoding, and 12 transformer encoder blocks with multi-head self-attention.
   - Parameters: 5.53 Million.
   - Purpose: Benchmark for non-convolutional, global receptive field self-attention on resource-constrained hardware.

### Two-Stage Fine-Tuning Pipeline

To prevent catastrophic forgetting of pre-trained feature extractors while adapting to weather imagery:

- **Stage 1 (Classifier Head Warmup)**: The pre-trained backbone parameters are frozen. Only the classification head (linear projection layer) is trained for 5 epochs using AdamW ($\eta = 10^{-3}$, weight decay $= 10^{-4}$).
- **Stage 2 (Full Network Differential Fine-Tuning)**: All layers are unfrozen. A differential learning rate is applied:
  - Backbone learning rate: $\eta_{\text{backbone}} = 10^{-5}$
  - Head learning rate: $\eta_{\text{head}} = 10^{-4}$
  - Cosine annealing learning rate schedule with minimum learning rate $\eta_{\text{min}} = 10^{-6}$.
  - Automatic Mixed Precision (AMP) using PyTorch `torch.amp.autocast`.

---

## 5. Model Calibration and Expected Calibration Error (ECE)

A classification system must not only predict the correct class label but also produce calibrated probabilities that represent true empirical likelihoods.

### 5.1 Formulation of ECE

Predictions over dataset $\mathcal{D}$ are partitioned into $M = 15$ equally-spaced confidence bins $B_m = \left( \frac{m-1}{M}, \frac{m}{M} \right]$ for $m \in \{1, \dots, M\}$.

The accuracy and average confidence within bin $B_m$ are computed as:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

The Expected Calibration Error is the weighted sum of bin discrepancies:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|$$

A perfectly calibrated classifier satisfies $\text{acc}(B_m) = \text{conf}(B_m)$ for all $m$, yielding $\text{ECE} = 0$.

---

## 6. Explainable AI (XAI) Formulations

To audit model decision boundaries and ensure the networks attend to valid meteorological visual cues rather than background artifacts, EdgeWeather integrates three interpretability techniques:

### 6.1 Grad-CAM (for MobileNetV3)

For class $c$ and feature map activations $A^k$ of the penultimate convolutional layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial z_c}{\partial A_{i, j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU} \left( \sum_{k} \alpha_k^c A^k \right)$$

### 6.2 Grad-CAM++ (for ResNet-50)

To handle complex scenes containing multiple meteorological phenomena, Grad-CAM++ computes second- and third-order gradient weightings:
$$\alpha_{i, j}^{k, c} = \frac{\frac{\partial^2 z_c}{(\partial A_{i, j}^k)^2}}{2 \frac{\partial^2 z_c}{(\partial A_{i, j}^k)^2} + \sum_{a, b} A_{a, b}^k \frac{\partial^3 z_c}{(\partial A_{a, b}^k)^3}}$$
$$w_k^c = \sum_{i} \sum_{j} \alpha_{i, j}^{k, c} \cdot \text{ReLU} \left( \frac{\partial z_c}{\partial A_{i, j}^k} \right)$$
$$L_{\text{Grad-CAM++}}^c = \text{ReLU} \left( \sum_{k} w_k^c A^k \right)$$

### 6.3 Attention Rollout (for Vision Transformers)

In ViT-Tiny, spatial tokens interact globally through self-attention matrices. Given raw attention weight matrix $A_l$ at transformer layer $l$, residual identity connections are incorporated:
$$\hat{A}_l = 0.5 A_l + 0.5 I$$
$$\hat{A}_l = \frac{\hat{A}_l}{\sum_j \hat{A}_{l, i, j}}$$

Attention Rollout recursively computes the cumulative token influence from input to output across all $L = 12$ layers:
$$\mathbf{R} = \prod_{l=1}^{L} \hat{A}_l = \hat{A}_L \cdot \hat{A}_{L-1} \dots \hat{A}_1$$

The attention map for the classification token (`[CLS]`, index 0) over all 196 spatial patch tokens is reshaped into a $14 \times 14$ grid and bilinearly upsampled to $224 \times 224$ pixels.
