"""
EdgeWeather — src package initialiser.

Public API surface for the data, modeling, and training pipeline.
"""

from importlib.metadata import PackageNotFoundError, version

from src.benchmark import benchmark_onnx, benchmark_pytorch, run_benchmarks
from src.dataset import WeatherDataset, create_dataloaders, get_eval_transforms, get_train_transforms
from src.deduplicate import compute_image_hashes, deduplicate_dataset
from src.evaluate import compute_ece, evaluate_model, plot_confusion_matrix, plot_reliability_diagram
from src.explainability import (
    AttentionRolloutViT,
    GradCAM,
    generate_comparative_xai_grid,
    overlay_heatmap_on_image,
)
from src.export_onnx import export_to_onnx, quantize_onnx_model, validate_onnx_parity
from src.loss import ConfusionAwareLoss, build_domain_confusion_cost_matrix, build_loss_fn
from src.models import WeatherClassifier, build_model
from src.train import Trainer, train_model
from src.utils import get_logger, load_config, load_full_config, set_global_seed

try:
    __version__ = version("edgeweather")
except PackageNotFoundError:  # running from source without installation
    __version__ = "0.1.0-dev"

__all__: list[str] = [
    "__version__",
    "WeatherDataset",
    "create_dataloaders",
    "get_train_transforms",
    "get_eval_transforms",
    "compute_image_hashes",
    "deduplicate_dataset",
    "WeatherClassifier",
    "build_model",
    "ConfusionAwareLoss",
    "build_domain_confusion_cost_matrix",
    "build_loss_fn",
    "Trainer",
    "train_model",
    "compute_ece",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_reliability_diagram",
    "GradCAM",
    "AttentionRolloutViT",
    "overlay_heatmap_on_image",
    "generate_comparative_xai_grid",
    "export_to_onnx",
    "validate_onnx_parity",
    "quantize_onnx_model",
    "benchmark_pytorch",
    "benchmark_onnx",
    "run_benchmarks",
    "set_global_seed",
    "load_config",
    "load_full_config",
    "get_logger",
]
