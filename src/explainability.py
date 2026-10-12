"""
src/explainability.py
=====================
Explainable AI (XAI) Suite for EdgeWeather.

Academic & Industry Features:
  - Grad-CAM & Grad-CAM++: Feature attribution for CNNs (MobileNetV3 and ResNet-50).
  - Attention Rollout: Layer-recursive self-attention attribution for Vision Transformers (ViT-Tiny).
  - Colormap Overlays: High-contrast thermal mapping (JET / TURBO) on RGB outdoor imagery.
  - 4-Panel Comparative Grid Generator:
      [Input Image] | [MobileNetV3 Grad-CAM] | [ResNet-50 Grad-CAM] | [ViT-Tiny Attention Rollout]
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.dataset import DEFAULT_CLASSES
from src.models import WeatherClassifier
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


# ── Colormap & Overlay Helpers ───────────────────────────────────────────────

def overlay_heatmap_on_image(
    image: np.ndarray | Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a 2D activation heatmap over an RGB image.

    Args:
        image: RGB input image as uint8 numpy array (H, W, 3) or PIL Image.
        heatmap: 2D float array in range [0, 1] of shape (H, W) or lower resolution.
        alpha: Blending weight for heatmap (1-alpha weight for original image).
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET, cv2.COLORMAP_TURBO).

    Returns:
        RGB blended image as uint8 numpy array of shape (H, W, 3).
    """
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    else:
        img_arr = image.copy()
        if img_arr.dtype in (np.float32, np.float64) and img_arr.max() <= 1.0:
            img_arr = (img_arr * 255.0).astype(np.uint8)

    h, w = img_arr.shape[:2]

    # Resize heatmap to match image dimensions
    if heatmap.shape[:2] != (h, w):
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap

    # Normalize to [0, 255] uint8
    norm_heat = np.clip(heatmap_resized, 0.0, 1.0)
    heat_uint8 = (norm_heat * 255.0).astype(np.uint8)

    # Apply colormap (OpenCV generates BGR)
    color_heat_bgr = cv2.applyColorMap(heat_uint8, colormap)
    color_heat_rgb = cv2.cvtColor(color_heat_bgr, cv2.COLOR_BGR2RGB)

    # Weighted blend
    blended = cv2.addWeighted(color_heat_rgb, alpha, img_arr, 1.0 - alpha, 0)
    return blended


# ── Grad-CAM & Grad-CAM++ for CNNs ───────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM & Grad-CAM++).

    Attaches hooks to the penultimate convolutional layer, computes gradients
    with respect to target class logits, and generates heatmaps of visual evidence.

    Attributes:
        model: Evaluated PyTorch model.
        target_layer: Convolutional module to intercept.
        variant: 'gradcam' (default) or 'gradcam++'.
    """

    def __init__(
        self,
        model: WeatherClassifier | nn.Module,
        target_layer: nn.Module | None = None,
        variant: str = "gradcam",
    ) -> None:
        """Initialize GradCAM.

        Args:
            model: PyTorch model or WeatherClassifier.
            target_layer: Target Conv module (auto-resolved if WeatherClassifier).
            variant: Either 'gradcam' or 'gradcam++'.
        """
        self.model = model
        self.variant = variant.lower()

        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_cam_target_layer"):
            self.target_layer = model.get_cam_target_layer()
        else:
            raise ValueError("Target layer must be explicitly provided for raw nn.Module.")

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target convolutional layer."""
        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            self.activations = out.detach()

        def backward_hook(module: nn.Module, grad_inp: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Clean up attached PyTorch hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Compute 2D Grad-CAM activation heatmap for input image.

        Args:
            input_tensor: 4D FloatTensor of shape (1, 3, H, W).
            target_class: Integer index of class to explain (None = top predicted class).

        Returns:
            Tuple of (heatmap_2d, predicted_class, prediction_probability).
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        device = next(self.model.parameters()).device
        x = input_tensor.to(device)
        x.requires_grad_(True)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)

        pred_class = int(logits.argmax(dim=-1).item())
        class_idx = target_class if target_class is not None else pred_class
        confidence = float(probs[0, class_idx].item())

        # Backward pass on target class score
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients. Check hook registration.")

        activations = self.activations[0]  # (C, H, W)
        gradients = self.gradients[0]      # (C, H, W)

        if self.variant == "gradcam++":
            # Grad-CAM++ formulation: higher-order derivative weighting
            g2 = gradients.pow(2)
            g3 = gradients.pow(3)
            sum_act = activations.sum(dim=(1, 2), keepdim=True)
            denom = 2.0 * g2 + sum_act * g3
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alpha_num = g2
            alphas = alpha_num / denom
            weights = (alphas * F.relu(gradients)).sum(dim=(1, 2))  # (C,)
        else:
            # Standard Grad-CAM: Global average pooling over spatial gradients
            weights = gradients.mean(dim=(1, 2))  # (C,)

        # Linear combination of activation channels
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)  # Discard negative contributions
        cam_np = cam.cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = cam_np.min(), cam_np.max()
        cam_np = (cam_np - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(cam_np)

        return cam_np, pred_class, confidence

    def __del__(self) -> None:
        self.remove_hooks()


# ── Attention Rollout for Vision Transformers (ViT-Tiny) ─────────────────────

class AttentionRolloutViT:
    """Attention Rollout algorithm for Vision Transformers (Abnar & Zuidema, 2020).

    Recursively multiplies attention matrices across all transformer blocks,
    incorporating residual identity connections, to compute the effective
    influence of each spatial image patch on the final classification [CLS] token.

    Attributes:
        model: WeatherClassifier wrapping ViT-Tiny.
    """

    def __init__(
        self,
        model: WeatherClassifier,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ) -> None:
        """Initialize AttentionRolloutViT.

        Args:
            model: WeatherClassifier instance with ViT-Tiny backbone.
            head_fusion: How to fuse multi-head attentions ('mean', 'min', 'max').
            discard_ratio: Fraction of lowest attention weights to discard (noise filtering).
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Compute 2D patch attention rollout heatmap (alias for generate_rollout).

        Args:
            input_tensor: FloatTensor of shape (1, 3, 224, 224).
            target_class: Optional target class (ViT CLS attention is class-agnostic).

        Returns:
            Tuple of (heatmap_14x14, predicted_class, prediction_probability).
        """
        return self.generate_rollout(input_tensor)

    def generate_rollout(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, int, float]:
        """Compute 2D patch attention rollout heatmap.

        Args:
            input_tensor: FloatTensor of shape (1, 3, 224, 224).

        Returns:
            Tuple of (heatmap_14x14, predicted_class, prediction_probability).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = input_tensor.to(device)

        # In PyTorch 2.x, timm attention blocks default to fused_attn = True (SDPA).
        # We temporarily disable fused_attn to expose explicit attention matrices.
        vit = self.model.backbone
        prev_fused = []
        for block in vit.blocks:
            prev_fused.append(getattr(block.attn, "fused_attn", False))
            block.attn.fused_attn = False

        captured_attentions: list[torch.Tensor] = []
        hooks = []

        def capture_hook(module: nn.Module, inp: Any, out: Any) -> None:
            # Input to attn_drop is the Softmax attention matrix: (B, num_heads, N, N)
            captured_attentions.append(inp[0].detach().cpu())

        for block in vit.blocks:
            hooks.append(block.attn.attn_drop.register_forward_hook(capture_hook))

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            pred_class = int(logits.argmax(dim=-1).item())
            confidence = float(probs[0, pred_class].item())

        # Clean up hooks and restore fused_attn flags
        for hook in hooks:
            hook.remove()
        for block, flag in zip(vit.blocks, prev_fused, strict=False):
            block.attn.fused_attn = flag

        if not captured_attentions:
            raise RuntimeError("Failed to capture attention weights from ViT transformer blocks.")

        # Execute recursive Attention Rollout
        # N = 197 tokens: 1 [CLS] token + 196 patch tokens (14x14 grid)
        num_tokens = captured_attentions[0].size(-1)
        rollout = torch.eye(num_tokens)

        for attn in captured_attentions:
            attn_batch = attn[0]  # (heads, N, N)

            if self.head_fusion == "mean":
                attn_fused = attn_batch.mean(dim=0)
            elif self.head_fusion == "max":
                attn_fused = attn_batch.max(dim=0)[0]
            elif self.head_fusion == "min":
                attn_fused = attn_batch.min(dim=0)[0]
            else:
                attn_fused = attn_batch.mean(dim=0)

            # Optional noise thresholding: discard lowest percentile weights
            if self.discard_ratio > 0.0:
                flat = attn_fused.flatten()
                k = int(len(flat) * self.discard_ratio)
                if k > 0:
                    thresh = torch.kthvalue(flat, k)[0]
                    attn_fused = torch.where(attn_fused < thresh, torch.zeros_like(attn_fused), attn_fused)

            # Account for residual connection: A_hat = 0.5 * A + 0.5 * I
            attn_fused = 0.5 * attn_fused + 0.5 * torch.eye(num_tokens)
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

            rollout = attn_fused @ rollout

        # Extract CLS token attention over spatial patches (row 0, columns 1 to end)
        mask = rollout[0, 1:].numpy()  # length 196
        grid_dim = int(np.sqrt(len(mask)))  # 14
        heatmap_2d = mask.reshape(grid_dim, grid_dim)

        # Normalize to [0, 1]
        h_min, h_max = heatmap_2d.min(), heatmap_2d.max()
        if h_max > h_min:
            heatmap_2d = (heatmap_2d - h_min) / (h_max - h_min)
        else:
            heatmap_2d = np.zeros_like(heatmap_2d)

        return heatmap_2d, pred_class, confidence


# ── 4-Panel Comparative Interpretability Grid ────────────────────────────────

def generate_comparative_xai_grid(
    image_input: str | Path | np.ndarray | Image.Image,
    mobilenet_model: WeatherClassifier,
    resnet_model: WeatherClassifier,
    vit_model: WeatherClassifier,
    class_names: Sequence[str] = DEFAULT_CLASSES,
    transform_fn: Any | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Produce a 4-panel side-by-side comparative explainability visualization.

    Panels:
      1. Original Input Image
      2. MobileNetV3-Large Grad-CAM (Edge CNN with Squeeze-and-Excitation attention)
      3. ResNet-50 Grad-CAM (Deep Residual CNN)
      4. ViT-Tiny Attention Rollout (Vision Transformer with multi-head self-attention)

    Args:
        image_input: Filepath, numpy array, or PIL Image.
        mobilenet_model: Loaded MobileNetV3 classifier.
        resnet_model: Loaded ResNet-50 classifier.
        vit_model: Loaded ViT-Tiny classifier.
        class_names: List of class names.
        transform_fn: Transform converting PIL/numpy into normalized tensor.
        save_path: Optional output path to save PNG figure.

    Returns:
        Tuple of (Figure, explanation_metadata_dict).
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input)
    else:
        pil_img = image_input.convert("RGB")

    raw_rgb = np.array(pil_img.resize((224, 224)))

    # Convert to normalized PyTorch tensor
    if transform_fn is not None:
        tensor = transform_fn(image=raw_rgb)["image"].unsqueeze(0)
    else:
        # Standard ImageNet normalization
        norm_arr = (raw_rgb / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).float().unsqueeze(0)

    # 1. MobileNet Grad-CAM
    cam_mobile = GradCAM(mobilenet_model)
    heat_mobile, pred_mobile, conf_mobile = cam_mobile.generate_heatmap(tensor)
    overlay_mobile = overlay_heatmap_on_image(raw_rgb, heat_mobile)
    cam_mobile.remove_hooks()

    # 2. ResNet Grad-CAM
    cam_res = GradCAM(resnet_model)
    heat_res, pred_res, conf_res = cam_res.generate_heatmap(tensor)
    overlay_res = overlay_heatmap_on_image(raw_rgb, heat_res)
    cam_res.remove_hooks()

    # 3. ViT Attention Rollout
    rollout_vit = AttentionRolloutViT(vit_model)
    heat_vit, pred_vit, conf_vit = rollout_vit.generate_rollout(tensor)
    overlay_vit = overlay_heatmap_on_image(raw_rgb, heat_vit)

    # Build 4-panel subplot figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(raw_rgb)
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay_mobile)
    axes[1].set_title(
        f"MobileNetV3 (Grad-CAM)\n{class_names[pred_mobile]} ({conf_mobile:.1%})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(overlay_res)
    axes[2].set_title(
        f"ResNet-50 (Grad-CAM)\n{class_names[pred_res]} ({conf_res:.1%})",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    axes[3].imshow(overlay_vit)
    axes[3].set_title(
        f"ViT-Tiny (Attention Rollout)\n{class_names[pred_vit]} ({conf_vit:.1%})",
        fontsize=12,
        fontweight="bold",
    )
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        ensure_dir(out_p.parent)
        fig.savefig(out_p, dpi=300)
        logger.info("Saved 4-panel comparative XAI figure to %s", out_p)

    metadata = {
        "mobilenet": {"pred": class_names[pred_mobile], "conf": conf_mobile},
        "resnet": {"pred": class_names[pred_res], "conf": conf_res},
        "vit": {"pred": class_names[pred_vit], "conf": conf_vit},
    }
    return fig, metadata
