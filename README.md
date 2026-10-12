# EdgeWeather: Image-Based Weather Condition Classification via Transfer Learning and Edge Optimization

[![CI](https://github.com/junaiddbz/edgeweather-condition-classification/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/junaiddbz/edgeweather-condition-classification/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/Code_Style-Ruff-8B5CF6)](https://docs.astral.sh/ruff/)

---

## About This Project

EdgeWeather is a portfolio and academic computer vision project exploring **weather condition classification from outdoor images** across 11 atmospheric categories. The project investigates how three different deep learning architectures — a lightweight edge CNN, a deep residual network, and a Vision Transformer — compare on a real-world, web-scraped dataset with all the complexity that entails: duplicate images, class imbalance, and out-of-distribution inputs.

Beyond raw accuracy, the project asks questions that motivate its design: *Does the model's confidence actually reflect its accuracy? Do its errors all matter equally, or are some misclassifications more dangerous than others? What does the model actually attend to when making a decision?*

The work is structured as a complete, reproducible ML pipeline — from dataset acquisition and deduplication, through transfer learning, ONNX export, INT8 quantization, and calibration analysis — and is demonstrated through an interactive Streamlit dashboard.

---

## Research Questions

Four concrete questions shaped every design decision in this project:

1. **Can a model trained on web-scraped data be trusted?** Web datasets contain burst-shot duplicates that contaminate train/test splits and inflate evaluation metrics. How can this be eliminated without manual annotation?

2. **Do all misclassifications have equal cost?** Mistaking `rime` for `frost` is a benign perceptual error. Mistaking dense `fog` or black ice `glaze` for clear weather is potentially hazardous. Can the loss function encode this asymmetry explicitly?

3. **Is a confident model a reliable model?** High accuracy does not guarantee calibrated probability estimates. What does the calibration profile of each architecture actually look like across 15 confidence bins?

4. **Where does each architecture look?** CNNs and Vision Transformers build representations differently. Do their attribution maps reflect genuine meteorological features, or spurious background correlations?

---

## Key Findings

All models were evaluated on a held-out test split of **961 images** after deduplication and leak-free partitioning.

### Accuracy and Calibration

| Architecture | Parameters | Test Top-1 Accuracy | Macro F1 | ECE (15 Bins) | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | **4.22 M** | 86.37% | 0.8661 | **0.0610** | **16.08 MB** |
| **ResNet-50** | 23.53 M | **87.72%** | **0.8847** | 0.0930 | 89.76 MB |
| **ViT-Tiny** (`patch16_224`) | 5.53 M | 83.25% | 0.8405 | 0.0636 | 21.08 MB |

Key observations:
- ResNet-50 achieves the highest raw accuracy but at 5.6× the parameter count of MobileNetV3. The 1.35% accuracy gain does not justify the footprint for any edge deployment scenario.
- ResNet-50 is also the most overconfident (ECE = 0.0930): it predicts >90% confidence on samples where empirical accuracy is closer to 81%.
- ViT-Tiny underperforms both CNNs, consistent with the known data-hunger of transformers when trained on small datasets without the spatial inductive biases that CNNs provide.

### Multi-Engine Inference Performance

| Architecture | Engine | Hardware | Median Latency ($p50$) | Tail Latency ($p99$) | Throughput (FPS) | Speedup |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | **ONNX Runtime INT8** | **Edge CPU** | **2.88 ms** | **5.30 ms** | **335.8 FPS** | **4.33×** |
| MobileNetV3-Large | ONNX Runtime FP32 | Edge CPU | 3.07 ms | 5.72 ms | 315.1 FPS | 4.07× |
| MobileNetV3-Large | PyTorch CUDA FP32 | GTX 1050 Ti | 8.14 ms | 14.36 ms | 118.3 FPS | 1.53× |
| MobileNetV3-Large | PyTorch FP32 | Edge CPU | 12.48 ms | 17.04 ms | 77.7 FPS | 1.00× |
| **ResNet-50** | **PyTorch CUDA FP32** | **GTX 1050 Ti** | **11.17 ms** | **14.28 ms** | **86.9 FPS** | **3.77×** |
| ResNet-50 | ONNX Runtime INT8 | Edge CPU | 21.53 ms | 25.03 ms | 45.7 FPS | 1.95× |
| ResNet-50 | PyTorch FP32 | Edge CPU | 42.08 ms | 46.26 ms | 23.8 FPS | 1.00× |
| **ViT-Tiny** | **PyTorch CUDA FP32** | **GTX 1050 Ti** | **5.31 ms** | **6.60 ms** | **183.0 FPS** | **2.52×** |
| ViT-Tiny | ONNX Runtime INT8 | Edge CPU | 7.27 ms | 11.69 ms | 138.0 FPS | 1.84× |
| ViT-Tiny | PyTorch FP32 | Edge CPU | 13.37 ms | 17.50 ms | 74.6 FPS | 1.00× |

A counter-intuitive result: for MobileNetV3, CPU INT8 outperforms GPU FP32 by nearly 3× (2.88 ms vs. 8.14 ms). The GPU overhead from host-device memory transfers dominates for a network this lightweight.

---

## Project Documentation

| Document | Contents |
| :--- | :--- |
| [Methodology](docs/methodology.md) | Mathematical formulations of dual perceptual hashing, Union-Find clustering, Confusion-Aware Loss, ECE, Grad-CAM, Grad-CAM++, and Attention Rollout |
| [Results and Analysis](docs/results_and_analysis.md) | Full 3-way comparative evaluation, per-class breakdown, confusion matrix analysis, calibration profiles, and OOD behavior |
| [Benchmarks](docs/benchmarks.md) | Multi-engine latency distributions (mean, std, p50, p95, p99), throughput, and quantization footprint analysis |
| [Deployment Guide](docs/deployment.md) | ONNX export, INT8 quantization, Docker containerization, and CI pipeline |

---

## Technical Approach

### Dual Perceptual Hashing Deduplication

The raw dataset contained 459 near-duplicate frames across 400 clusters — burst-mode shots that would have leaked across train/test splits under random partitioning. The deduplication pipeline pairs 64-bit DCT frequency hashing ($pHash$) with horizontal gradient difference hashing ($dHash$):

$$\mathcal{D}(I_a, I_b) = \min \Big( \text{Hamming}\big(h_{pHash}(I_a), h_{pHash}(I_b)\big),\ \text{Hamming}\big(h_{dHash}(I_a), h_{dHash}(I_b)\big) \Big) \le \tau$$

Duplicate clusters are then assigned atomically to a single partition via Disjoint-Set Union-Find, guaranteeing **0% cross-split leakage**. Using two hash types eliminates false positives that arise when two distinct fog or snow scenes happen to share similar DCT frequency profiles. See [Methodology](docs/methodology.md#2-dual-perceptual-hashing-deduplication-pipeline) for the full derivation.

### Confusion-Aware Loss

An asymmetric $11 \times 11$ cost matrix $C$ encodes the real-world hazard severity of each confusion pair. Misclassifying `fog` or black ice `glaze` as `rainbow` or `dew` carries a $3.0\times$ penalty; visually adjacent pairs like `rime` $\leftrightarrow$ `frost` carry only $0.5\times$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LS-CE}}(y, \hat{p}) + \lambda \sum_{c=1}^{K} C_{y,c} \cdot \hat{p}_c$$

In practice, this eliminated all catastrophic false-clear predictions on the test set. See [Methodology](docs/methodology.md#3-domain-safety-confusion-aware-loss-engine) for the cost matrix definition.

### Explainability Audit (XAI)

Three attribution techniques verify that the models attend to meteorological cues rather than spurious background correlations:

- **Grad-CAM** (MobileNetV3): localizes local surface textures — puddle reflections for `rain`, discharge channels for `lightning`.
- **Grad-CAM++** (ResNet-50): distributes activation across multi-region evidence — ice pellets for `hail`, horizon contrast gradient for `fog`.
- **Attention Rollout** (ViT-Tiny): captures global ambient illumination — entire sky chromaticity for `sandstorm`, rather than edge features.

See [Results and Analysis](docs/results_and_analysis.md#6-interpretability-and-feature-attribution-analysis) for comparative saliency discussion.

---

## Repository Structure

```text
Weather-Condition-Classification/
├── .github/
│   └── workflows/ci.yml              # GitHub Actions: Ruff lint + Pytest (45 tests)
├── app/
│   └── streamlit_app.py              # Interactive web dashboard
├── configs/
│   ├── base_config.yaml              # Dataset paths, augmentation, loss hyperparameters
│   ├── mobilenet_v3.yaml             # MobileNetV3-Large training config
│   ├── resnet50.yaml                 # ResNet-50 training config
│   └── vit_tiny.yaml                 # ViT-Tiny training config
├── data/
│   ├── raw/                          # Raw atmospheric images (11 classes; excluded from Git)
│   ├── processed/                    # Deduplicated split manifests: train.csv, val.csv, test.csv
│   └── synthetic_dryrun/             # Procedural images for offline CI validation
├── docker/
│   ├── Dockerfile                    # Multi-stage production container (builder + runtime)
│   └── docker-compose.yml            # Service definition with ONNX and CSV volume mounts
├── docs/
│   ├── methodology.md                # Mathematical framework and algorithm derivations
│   ├── results_and_analysis.md       # Empirical results, per-class metrics, and discussion
│   ├── benchmarks.md                 # Hardware benchmark matrix and latency analysis
│   └── deployment.md                 # Export, quantization, Docker, and CI guide
├── notebooks/
│   ├── 01_eda_and_deduplication.ipynb    # Class distributions and perceptual hash clustering
│   └── 02_xai_comparative_study.ipynb    # Grad-CAM vs. Attention Rollout comparison
├── outputs/
│   ├── checkpoints/                  # Trained PyTorch checkpoints (.pt; excluded from Git)
│   ├── figures/                      # Confusion matrices, ECE diagrams, latency plots
│   ├── onnx/                         # Exported ONNX models FP32 and INT8 (excluded from Git)
│   └── results/                      # Benchmark CSVs and evaluation JSON reports
├── src/
│   ├── __init__.py                   # Package entry point and version metadata
│   ├── benchmark.py                  # Latency profiler (p50/p95/p99, FPS)
│   ├── dataset.py                    # Albumentations augmentation pipeline and PyTorch Dataset
│   ├── deduplicate.py                # Dual pHash/dHash and Union-Find deduplication engine
│   ├── evaluate.py                   # Metrics, 15-bin ECE, and reliability diagram
│   ├── explainability.py             # Grad-CAM, Grad-CAM++, and ViT Attention Rollout
│   ├── export_onnx.py                # ONNX export, numerical parity check, INT8 quantization
│   ├── extract_dataset.py            # Automated Hugging Face dataset download and extraction
│   ├── loss.py                       # Domain Safety Cost Matrix and Confusion-Aware Loss
│   ├── models.py                     # MobileNetV3 / ResNet-50 / ViT-Tiny model factory
│   ├── train.py                      # Two-stage fine-tuning with AMP and Cosine Annealing
│   ├── train_comparative_study.py    # Automated sequential training of all three architectures
│   └── utils.py                      # Logging, reproducibility seeding, and config parsing
├── tests/                            # Pytest test suite — 45 tests, all passing
│   ├── test_app.py
│   ├── test_dataset.py
│   ├── test_evaluate.py
│   ├── test_explainability.py
│   ├── test_inference.py
│   ├── test_loss.py
│   ├── test_models.py
│   └── test_train.py
├── LICENSE                           # MIT License
├── pyproject.toml                    # Build system, Ruff, and Pytest configuration
└── requirements.txt                  # Pinned dependencies (Python 3.10)
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/junaiddbz/edgeweather-condition-classification.git
cd edgeweather-condition-classification

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# CPU-only install
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# GPU-enabled install (for training)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Dataset Acquisition

The project uses the [`dazzle-nu/autotrain-data-weather-classification`](https://huggingface.co/datasets/dazzle-nu/autotrain-data-weather-classification) dataset from Hugging Face — 6,862 outdoor images across 11 atmospheric classes.

```bash
python -m src.extract_dataset
# Places images into data/raw/<class_name>/
```

### 3. Deduplication and Split Generation

```bash
python -m src.deduplicate --raw_dir data/raw --out_dir data/processed --threshold 4
# Outputs data/processed/train.csv, val.csv, test.csv with 0% cross-split leakage
```

### 4. Training

```bash
# Train a single architecture
python -m src.train --base_config configs/base_config.yaml --model_config configs/mobilenet_v3.yaml

# Or train all three architectures sequentially
python -m src.train_comparative_study
```

### 5. Evaluation

```bash
python -m src.evaluate --checkpoint outputs/checkpoints/mobilenet_v3_large_best.pt --split test
```

### 6. ONNX Export and INT8 Quantization

```bash
python -m src.export_onnx --checkpoint outputs/checkpoints/mobilenet_v3_large_best.pt --quantize
```

### 7. Benchmarking

```bash
python -m src.benchmark \
    --checkpoint outputs/checkpoints/mobilenet_v3_large_best.pt \
    --onnx outputs/onnx/mobilenet_v3_large_fp32.onnx \
    --quantized outputs/onnx/mobilenet_v3_large_fp32_int8.onnx \
    --runs 100
```

### 8. Test Suite

```bash
pytest tests/ -v --cov=src
```

### 9. Dashboard

```bash
streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

## Design Decisions and Limitations

**Why dual hashing instead of a single hash?** A single hash type (e.g., only $pHash$) produces false positives on images that share globally similar DCT frequency content but depict distinct scenes — two different foggy landscapes can hash close under $pHash$ alone. Adding $dHash$ (which is sensitive to horizontal gradient direction rather than frequency magnitude) and taking their minimum Hamming distance increases precision substantially without manual threshold tuning.

**Why a custom cost matrix instead of class-weighted CE?** A standard class-weighted loss addresses frequency imbalance but encodes no information about *which* confusion pairs are dangerous. An explicit $K \times K$ cost matrix is a more expressive formulation: it can express that `fog` $\rightarrow$ `rainbow` is catastrophically wrong while `rime` $\rightarrow$ `frost` is a minor near-miss.

**Why two-stage fine-tuning?** Directly fine-tuning all layers from epoch 1 with a full learning rate causes catastrophic forgetting of the pre-trained feature extractor's low-level representations, especially on a small dataset of ~6,400 images. Warming up only the classification head for 5 epochs before unlocking the backbone is a standard and effective mitigation.

**Limitations:** The closed-world Softmax assumption means the system cannot abstain on out-of-distribution inputs (e.g., clear sunny skies, which have no corresponding class). Confidence thresholding at 60% is used as a proxy, but a proper open-set recognition layer — such as an energy-based OOD detector or a sigmoid multi-label output — would be the principled extension. The OOD behavior is analyzed in [Results and Analysis](docs/results_and_analysis.md#4-out-of-distribution-ood-behavior-and-closed-world-assumption).

---

## Automated Testing

The project includes **45 unit tests** across all major components, run automatically on every push via GitHub Actions:

- **Architecture integrity**: output tensor shapes, layer freezing, and two-stage parameter registration.
- **Numerical parity**: ONNX FP32 exports match PyTorch outputs within $\max|\Delta| < 10^{-4}$.
- **Loss ordering**: high-risk confusion pairs provably incur higher penalties than low-risk pairs.
- **Calibration correctness**: ECE = 0 for a perfectly calibrated synthetic model.
- **Gradient flow**: Grad-CAM and Attention Rollout hooks activate without degrading forward-pass performance.

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## References

```bibtex
@inproceedings{howard2019searching,
  title={Searching for MobileNetV3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={1314--1324},
  year={2019}
}

@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={770--778},
  year={2016}
}

@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{selvaraju2017grad,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  pages={618--626},
  year={2017}
}

@article{abnar2020quantifying,
  title={Quantifying Attention Flow in Transformers},
  author={Abnar, Samira and Zuidema, Willem},
  journal={arXiv preprint arXiv:2005.00928},
  year={2020}
}

@inproceedings{guo2017calibration,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2017}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
