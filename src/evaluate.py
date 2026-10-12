"""
src/evaluate.py
===============
Comprehensive scientific evaluation suite for EdgeWeather.

Academic & Industry Features:
  - Multi-class Performance Metrics: Top-1 Accuracy, Macro/Weighted F1, Precision, Recall.
  - Expected Calibration Error (ECE): 15-bin formulation quantifying model overconfidence.
  - Reliability Diagrams: Visualizing calibration error gaps.
  - High-Resolution Confusion Matrices: Normalized heatmaps saved as publication-ready figures.
  - Error Analysis Suite: Extraction of high-confidence failures to reveal perceptual ambiguities.
  - Export: Complete structured results saved to JSON and CSV.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.dataset import DEFAULT_CLASSES, create_dataloaders
from src.models import build_model
from src.utils import ensure_dir, get_logger, load_full_config

logger = get_logger(__name__)


# ── Expected Calibration Error (ECE) ─────────────────────────────────────────

def compute_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, dict[str, np.ndarray]]:
    """Compute Expected Calibration Error (ECE) across M equal-width bins.

    Mathematical formulation:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Where B_m is the set of samples whose model confidence falls in bin m,
    acc(B_m) is the empirical accuracy of bin m, and conf(B_m) is the average
    predicted confidence of bin m.

    Args:
        probabilities: Softmax probability array of shape (N, K).
        targets: Ground-truth class indices of shape (N,).
        n_bins: Number of confidence bins (standard in literature is 15).

    Returns:
        Tuple of (ece_scalar, bin_statistics_dict).
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == targets).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    total_samples = len(targets)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > low) & (confidences <= high) if i > 0 else (confidences >= low) & (confidences <= high)
        count = np.sum(in_bin)
        bin_counts[i] = count

        if count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_accs[i] = bin_acc
            bin_confs[i] = bin_conf
            ece += (count / total_samples) * abs(bin_acc - bin_conf)

    stats = {
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges,
    }
    return float(ece), stats


# ── Plotting Utilities ───────────────────────────────────────────────────────

def plot_reliability_diagram(
    bin_stats: dict[str, np.ndarray],
    ece: float,
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate reliability diagram illustrating model calibration.

    Plots empirical accuracy vs. confidence bins, shading the calibration gap.

    Args:
        bin_stats: Dictionary containing bin statistics from :func:`compute_ece`.
        ece: Scalar ECE value.
        model_name: Model identifier string for plot title.
        save_path: Optional output path to persist PNG figure.

    Returns:
        Generated :class:`matplotlib.figure.Figure`.
    """
    bin_accs = bin_stats["bin_accuracies"]
    bin_edges = bin_stats["bin_edges"]
    bin_counts = bin_stats["bin_counts"]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 1.0 / len(bin_accs)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

    # Accuracy bars
    valid_mask = bin_counts > 0
    ax.bar(
        bin_centers[valid_mask],
        bin_accs[valid_mask],
        width=width * 0.9,
        alpha=0.7,
        color="#2b5c8f",
        edgecolor="black",
        label="Outputs",
    )

    # Gap shading
    for i in range(len(bin_accs)):
        if bin_counts[i] > 0:
            ax.bar(
                bin_centers[i],
                abs(bin_accs[i] - bin_centers[i]),
                bottom=min(bin_accs[i], bin_centers[i]),
                width=width * 0.9,
                color="#e74c3c",
                alpha=0.35,
                edgecolor="none",
                label="Calibration Gap" if i == 0 else "",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram: {model_name} (ECE = {ece:.4f})", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        logger.info("Saved reliability diagram to %s", save_path)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASSES,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot high-resolution normalized confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (K, K).
        class_names: List of class labels for axis ticks.
        normalize: Whether to normalize by row (true class proportions).
        title: Plot title string.
        save_path: Optional output path to save PNG figure.

    Returns:
        Generated :class:`matplotlib.figure.Figure`.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ground Truth Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        logger.info("Saved confusion matrix heatmap to %s", save_path)
    return fig


# ── Error Analysis ───────────────────────────────────────────────────────────

def extract_top_failure_cases(
    probabilities: np.ndarray,
    targets: np.ndarray,
    filepaths: Sequence[str],
    class_names: Sequence[str] = DEFAULT_CLASSES,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Identify top high-confidence misclassifications for qualitative error analysis.

    Highlights failure cases where the model was strongly confident yet incorrect.

    Args:
        probabilities: (N, K) softmax prediction probabilities.
        targets: (N,) ground-truth class indices.
        filepaths: List of image paths corresponding to each prediction.
        class_names: Ordered class names.
        top_k: Maximum number of worst-case failures to extract.

    Returns:
        List of dictionaries detailing the top misclassifications.
    """
    preds = np.argmax(probabilities, axis=1)
    confs = np.max(probabilities, axis=1)

    errors: list[dict[str, Any]] = []
    for idx, (p, t, c, path) in enumerate(zip(preds, targets, confs, filepaths, strict=False)):
        if p != t:
            errors.append({
                "index": idx,
                "filepath": str(path),
                "true_label": class_names[t],
                "pred_label": class_names[p],
                "confidence": float(c),
                "true_prob": float(probabilities[idx, t]),
            })

    # Sort descending by confidence (most overconfident mistakes first)
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    return errors[:top_k]


# ── Full Model Evaluator ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: Sequence[str] = DEFAULT_CLASSES,
    device: torch.device | None = None,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Execute complete scientific evaluation over a given DataLoader.

    Computes:
      - Top-1 Accuracy
      - Macro F1, Weighted F1
      - Macro Precision, Macro Recall
      - Expected Calibration Error (ECE)
      - Confusion Matrix
      - Per-class precision, recall, and F1 scores

    Args:
        model: Evaluated PyTorch model.
        dataloader: Test or validation DataLoader.
        class_names: List of class labels.
        device: PyTorch device (CPU or CUDA).
        n_bins: Confidence bin count for ECE.

    Returns:
        Dictionary containing all quantitative metrics and arrays.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()

    all_probs: list[np.ndarray] = []
    all_targets: list[int] = []
    all_paths: list[str] = []

    for batch in dataloader:
        images = batch["image"].to(dev, non_blocking=True)
        targets = batch["label"].to(dev, non_blocking=True)
        paths = batch.get("filepath", [""] * len(targets))

        logits = model(images)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        labels = targets.cpu().numpy()

        all_probs.append(probs)
        all_targets.extend(labels.tolist())
        all_paths.extend([str(p) for p in paths])

    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.array(all_targets)
    preds_arr = np.argmax(probs_arr, axis=1)

    # Core classification metrics
    acc = accuracy_score(targets_arr, preds_arr)
    macro_f1 = f1_score(targets_arr, preds_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets_arr, preds_arr, average="weighted", zero_division=0)
    macro_prec = precision_score(targets_arr, preds_arr, average="macro", zero_division=0)
    macro_rec = recall_score(targets_arr, preds_arr, average="macro", zero_division=0)

    # Calibration
    ece, bin_stats = compute_ece(probs_arr, targets_arr, n_bins=n_bins)

    # Confusion matrix
    cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(len(class_names))))

    # Detailed per-class report
    clf_report = classification_report(
        targets_arr,
        preds_arr,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    # Top failures
    failures = extract_top_failure_cases(
        probs_arr,
        targets_arr,
        all_paths,
        class_names=class_names,
        top_k=10,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "ece": float(ece),
        "confusion_matrix": cm,
        "bin_stats": bin_stats,
        "classification_report": clf_report,
        "top_failures": failures,
        "probabilities": probs_arr,
        "targets": targets_arr,
        "predictions": preds_arr,
        "filepaths": all_paths,
    }


# ── Pipeline Runner & CLI ────────────────────────────────────────────────────

def run_evaluation_pipeline(
    checkpoint_path: str | Path,
    config_path: str | Path = "configs/base_config.yaml",
    model_config_path: str | Path | None = None,
    split: str = "test",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute end-to-end evaluation pipeline and export figures and JSON report.

    Args:
        checkpoint_path: Path to model checkpoint file (.pt).
        config_path: Base config path.
        model_config_path: Model specific config path.
        split: Split to evaluate ('test', 'val').
        output_dir: Output directory for figures and results.

    Returns:
        Evaluation metrics summary dict.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model_name = ckpt_data.get("model_name", "mobilenet_v3_large")
    cfg = ckpt_data.get("config") or load_full_config(config_path, model_config_path)

    classes = cfg.get("dataset", {}).get("classes", DEFAULT_CLASSES)
    model = build_model(model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt_data["model_state_dict"])

    loaders = create_dataloaders(cfg)
    target_loader = loaders.get(split)
    if target_loader is None:
        raise ValueError(f"Dataloader for split '{split}' could not be constructed.")

    results = evaluate_model(model, target_loader, class_names=classes)

    # Figures and reports
    figures_dir = Path(output_dir or cfg.get("paths", {}).get("figures_dir", "outputs/figures"))
    results_dir = Path(output_dir or cfg.get("paths", {}).get("results_dir", "outputs/results"))
    ensure_dir(figures_dir)
    ensure_dir(results_dir)

    # 1. Confusion Matrix Plot
    cm_path = figures_dir / f"{model_name}_{split}_confusion_matrix.png"
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names=classes,
        normalize=True,
        title=f"Normalized Confusion Matrix: {model_name} ({split.title()})",
        save_path=cm_path,
    )
    plt.close("all")

    # 2. Reliability Diagram Plot
    rel_path = figures_dir / f"{model_name}_{split}_reliability_diagram.png"
    plot_reliability_diagram(
        results["bin_stats"],
        ece=results["ece"],
        model_name=model_name,
        save_path=rel_path,
    )
    plt.close("all")

    # 3. Export JSON report
    report_dict = {
        "model_name": model_name,
        "split": split,
        "checkpoint": str(ckpt_file),
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "macro_precision": results["macro_precision"],
        "macro_recall": results["macro_recall"],
        "ece": results["ece"],
        "classification_report": results["classification_report"],
        "top_failures": results["top_failures"],
    }
    json_path = results_dir / f"{model_name}_{split}_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2)

    logger.info(
        "[bold green]✓ Evaluation Complete[/bold green] | Acc: %.2f%% | Macro F1: %.4f | ECE: %.4f",
        results["accuracy"] * 100,
        results["macro_f1"],
        results["ece"],
    )
    return report_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeWeather: Scientific Evaluation Suite")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt model checkpoint.")
    parser.add_argument("--base_config", type=Path, default=Path("configs/base_config.yaml"))
    parser.add_argument("--model_config", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    run_evaluation_pipeline(
        checkpoint_path=args.checkpoint,
        config_path=args.base_config,
        model_config_path=args.model_config,
        split=args.split,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
