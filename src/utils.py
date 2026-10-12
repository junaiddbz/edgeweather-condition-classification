"""
src/utils.py
============
Shared utility functions used across the EdgeWeather pipeline.

Covers:
  - Reproducibility: global seed setting with deterministic behaviour.
  - Configuration: loading and merging YAML config files.
  - Logging: a rich-formatted console logger factory.
  - Path helpers: ensuring output directories exist.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler

# ── Reproducibility ─────────────────────────────────────────────────────────


def set_global_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility.

    Sets seeds for Python's ``random``, NumPy, and PyTorch (CPU + CUDA).
    Also enables CuDNN deterministic mode and disables benchmark mode,
    which trades a small performance cost for bit-exact reproducibility.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Backward-compatible alias
set_seed = set_global_seed


# ── Configuration Loading ────────────────────────────────────────────────────


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict.

    Args:
        config_path: Absolute or relative path to the ``.yaml`` config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into ``base``, returning a new dictionary.

    Nested dicts are merged recursively; all other types are overwritten.

    Args:
        base: Base configuration dictionary (e.g., from ``base_config.yaml``).
        override: Model-specific config that overrides base values.

    Returns:
        Merged configuration dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_full_config(
    base_path: str | Path = "configs/base_config.yaml",
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load base config and optionally deep-merge a model-specific config.

    Args:
        base_path: Path to ``base_config.yaml``.
        model_config_path: Optional path to a model-specific YAML
            (e.g., ``configs/mobilenet_v3.yaml``).

    Returns:
        Fully merged configuration dictionary.
    """
    cfg = load_config(base_path)
    if model_config_path is not None:
        model_cfg = load_config(model_config_path)
        cfg = merge_configs(cfg, model_cfg)
    return cfg


# ── Logging ──────────────────────────────────────────────────────────────────

# Ensure UTF-8 streams on Windows to avoid charmap codec errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_console = Console(force_terminal=True, legacy_windows=False)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a Rich-formatted logger.

    Uses ``RichHandler`` for colourised, structured terminal output with
    automatic truncation of long messages and clickable tracebacks.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level. Defaults to ``logging.INFO``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_console, rich_tracebacks=True, markup=False)],
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# ── Path Helpers ─────────────────────────────────────────────────────────────


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and any missing parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        Resolved :class:`pathlib.Path` of the (now existing) directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device.

    Args:
        prefer_cuda: If ``True`` (default), use CUDA when available,
            otherwise fall back to CPU.

    Returns:
        :class:`torch.device` — either ``"cuda"`` or ``"cpu"``.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
