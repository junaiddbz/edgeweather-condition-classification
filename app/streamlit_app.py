"""
app/streamlit_app.py
====================
EdgeWeather: Image-Based Weather Condition Classification
Interactive Web Application & Academic Demo

Features:
  - 3-Way Comparative Architectures: MobileNetV3-Large, ResNet-50, ViT-Tiny.
  - Multi-Engine Inference: PyTorch FP32 vs. ONNX Runtime FP32 vs. ONNX Runtime INT8.
  - Live edge telemetry: Latency (ms), Throughput (FPS), Model Footprint (MB).
  - Explainable AI (XAI): Grad-CAM, Grad-CAM++, and ViT Attention Rollout.
  - Benchmark and calibration explorer.
  - Methodology documentation with interactive cost matrix heatmap.

Design notes
------------
The page takes on the atmosphere of whatever the model predicts: the sky wash,
the result panel and the accent colour all shift with the detected condition
(fog goes pewter, sandstorm goes ochre, frost goes pale blue, and so on).
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from src.dataset import DEFAULT_CLASSES, get_eval_transforms
from src.explainability import (
    AttentionRolloutViT,
    GradCAM,
    overlay_heatmap_on_image,
)
from src.loss import build_domain_confusion_cost_matrix
from src.models import WeatherClassifier, build_model

# ── Streamlit Page Configuration ─────────────────────────────────────────────

st.set_page_config(
    page_title="EdgeWeather: weather from a single photo",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Design Tokens ─────────────────────────────────────────────────────────────

INK = "#14232B"
MUTED = "#5D6E78"
LINE = "#D3DADF"


@dataclass(frozen=True)
class Sky:
    """Atmosphere palette that the whole page adopts for a predicted class."""
    accent: str   # gauge, bars, tab underline
    top: str      # sky colour at the top of the result panel / page wash
    bottom: str   # sky colour near the bottom of the result panel


DEFAULT_SKY = Sky("#3F6F8C", "#DCE6EC", "#F1F5F7")

SKIES: dict[str, Sky] = {
    "dew":       Sky("#2F8277", "#D8ECE6", "#F2F8F5"),
    "fog":       Sky("#5A6A78", "#D6DCE1", "#EEF0F2"),
    "frost":     Sky("#3A82B5", "#D3E7F4", "#F0F7FB"),
    "glaze":     Sky("#4A6BA3", "#D2DDEC", "#EDF1F7"),
    "hail":      Sky("#5B6A9C", "#D5D9E8", "#EEF0F6"),
    "lightning": Sky("#6B46C1", "#DCD3F0", "#F4F1FA"),
    "rain":      Sky("#2B6A96", "#CBDBE7", "#EAF0F5"),
    "rainbow":   Sky("#B8467F", "#F5E0D6", "#E4F0E8"),
    "rime":      Sky("#5F849C", "#D9E4EB", "#F1F5F7"),
    "sandstorm": Sky("#A8701A", "#EADBBF", "#F7F0E3"),
    "snow":      Sky("#52779A", "#E2EAF0", "#F8FAFB"),
}


# ── Global CSS ────────────────────────────────────────────────────────────────

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Instrument+Sans:wght@400;500;600;700&display=swap');

:root {
    --paper: #EDF0F2;
    --panel: #F8FAFB;
    --ink: #14232B;
    --muted: #5D6E78;
    --line: #D3DADF;
    --line-strong: #B9C4CB;
    --accent: #3F6F8C;
    --sky-top: #DCE6EC;
    --sky-bottom: #F1F5F7;
    --serif: 'Instrument Serif', 'Iowan Old Style', Georgia, serif;
    --sans: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    color-scheme: light;
}

/* ── Page surface: a wash of the current sky fading into paper */
.stApp {
    background-color: var(--paper);
    background-image: linear-gradient(180deg, var(--sky-top) 0px, var(--paper) 460px);
    background-repeat: no-repeat;
    color: var(--ink);
    font-family: var(--sans);
}
.block-container { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }

/* ── Typography */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: var(--serif);
    font-weight: 400;
    letter-spacing: -0.012em;
    color: var(--ink);
}
.stApp h3 { font-size: 2.05rem; line-height: 1.1; padding-top: 0.4rem; }
.stApp h4 { font-size: 1.55rem; line-height: 1.15; }
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp .stMarkdown { font-family: var(--sans); }
.stApp .stMarkdown p { line-height: 1.6; }
.stApp code {
    background: rgba(20, 35, 43, 0.07);
    color: var(--ink);
    border-radius: 5px;
    padding: 1px 6px;
    font-size: 0.85em;
}
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* ── Masthead */
.masthead { padding: 2px 0 26px; }
.wordmark {
    font-family: var(--serif);
    font-size: 3.7rem;
    line-height: 1;
    letter-spacing: -0.025em;
    color: var(--ink);
}
.mast-sub {
    max-width: 62ch;
    color: var(--muted);
    font-size: 1.06rem;
    line-height: 1.55;
    margin: 12px 0 0;
}

/* ── Sidebar */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.side-title { font-family: var(--serif); font-size: 1.6rem; line-height: 1.1; margin: 4px 0 2px; color: var(--ink); }
.side-note { font-size: 0.82rem; color: var(--muted); line-height: 1.55; }

/* ── Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 30px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 12px 0;
    background: transparent;
    font-weight: 500;
    color: var(--muted);
}
.stTabs [data-baseweb="tab"] p { font-size: 1rem; }
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p { color: var(--ink); font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Inputs */
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel);
    border: 1px dashed var(--line-strong);
    border-radius: 16px;
}
[data-baseweb="select"] > div { border-radius: 10px; border-color: var(--line-strong); background: #fff; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }
.stImage img { border-radius: 16px; }

/* ── Result panel: the memorable moment */
.sky-panel {
    background: linear-gradient(165deg, var(--sky-top) 0%, var(--sky-bottom) 72%, #FFFFFF 100%);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 30px 32px 26px;
    height: 100%;
}
.sky-label { color: var(--muted); font-size: 0.92rem; }
.sky-name {
    font-family: var(--serif);
    font-size: clamp(3.6rem, 7vw, 5.9rem);
    line-height: 0.95;
    letter-spacing: -0.03em;
    color: var(--ink);
    margin: 2px 0 18px;
}
.sky-meta { display: flex; align-items: flex-end; gap: 24px; flex-wrap: wrap; margin-bottom: 22px; }
.gauge { width: 190px; height: auto; flex: none; }
.gauge-track { fill: none; stroke: rgba(20, 35, 43, 0.12); stroke-width: 9; stroke-linecap: round; }
.gauge-fill {
    fill: none;
    stroke: var(--accent);
    stroke-width: 9;
    stroke-linecap: round;
    stroke-dasharray: 100 100;
    animation: sweep 1s cubic-bezier(0.2, 0.7, 0.2, 1);
}
.gauge-num { font-family: var(--serif); font-size: 38px; fill: var(--ink); }
.gauge-pct { font-size: 18px; fill: var(--muted); }
.gauge-cap { font-family: var(--sans); font-size: 10px; fill: var(--muted); }
@keyframes sweep { from { stroke-dashoffset: 100; } }
@media (prefers-reduced-motion: reduce) { .gauge-fill { animation: none; } }
.runner { font-size: 0.95rem; color: var(--muted); line-height: 1.5; padding-bottom: 8px; }
.runner strong { color: var(--ink); font-weight: 600; }

.ood {
    margin-top: 14px;
    font-size: 0.88rem;
    line-height: 1.55;
    color: #5C4412;
    background: rgba(245, 190, 80, 0.2);
    border: 1px solid rgba(180, 120, 20, 0.3);
    border-radius: 12px;
    padding: 12px 14px;
}
.ood strong { font-weight: 600; }

/* ── Telemetry readout */
.readout {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border-top: 1px solid var(--line-strong);
    border-bottom: 1px solid var(--line);
    margin: 30px 0 6px;
}
.ro-cell { padding: 18px 22px 16px; border-left: 1px solid var(--line); }
.ro-cell:first-child { border-left: none; padding-left: 2px; }
.ro-label { color: var(--muted); font-size: 0.9rem; }
.ro-value {
    font-family: var(--serif);
    font-size: 2.7rem;
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
}
.ro-value.text { font-size: 1.7rem; line-height: 1.45; letter-spacing: -0.01em; }
.ro-unit { font-family: var(--sans); font-size: 0.95rem; color: var(--muted); margin-left: 5px; letter-spacing: 0; }
.ro-note { color: var(--muted); font-size: 0.85rem; display: flex; align-items: center; gap: 7px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #C08A2E; display: inline-block; flex: none; }
.dot.ok { background: #3A8A5C; }
@media (max-width: 900px) {
    .readout { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .ro-cell:nth-child(3) { border-left: none; padding-left: 2px; }
}

/* ── Horizontal bars (class probabilities, latency comparison) */
.dist { display: flex; flex-direction: column; gap: 10px; max-width: 780px; margin-top: 8px; }
.dist-row { display: grid; grid-template-columns: 110px 1fr 64px; align-items: center; gap: 14px; font-size: 0.95rem; }
.dist.wide .dist-row { grid-template-columns: 200px 1fr 84px; }
.dist-name { color: var(--ink); }
.dist-track { display: block; height: 10px; background: rgba(20, 35, 43, 0.07); border-radius: 999px; overflow: hidden; }
.bar { display: block; height: 100%; border-radius: 999px; background: #A9B8C1; }
.dist-row.top .bar { background: var(--accent); }
.dist-val { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.dist-row.top .dist-name, .dist-row.top .dist-val { color: var(--ink); font-weight: 600; }

/* ── Spec table */
.spec { width: 100%; border-collapse: collapse; font-size: 0.95rem; margin-top: 6px; }
.spec th { text-align: left; font-weight: 600; padding: 12px 14px 12px 0; border-bottom: 1px solid var(--line-strong); color: var(--ink); }
.spec td { padding: 12px 14px 12px 0; border-bottom: 1px solid var(--line); vertical-align: top; color: var(--ink); }
.spec td:first-child { color: var(--muted); width: 22%; }
.spec td.best { font-weight: 700; box-shadow: inset 0 -2px 0 var(--accent); }

/* ── Empty state & section rule */
.empty {
    border: 1px dashed var(--line-strong);
    border-radius: 20px;
    padding: 56px 28px;
    text-align: center;
    color: var(--muted);
    background: var(--panel);
    margin-top: 18px;
}
.empty .empty-title { font-family: var(--serif); font-size: 1.9rem; color: var(--ink); margin-bottom: 6px; }
.rule { height: 1px; background: var(--line); margin: 34px 0 10px; }

/* ── Class chips & notes */
.class-bar {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 12px 18px;
    margin: 8px 0 20px;
}
.class-bar-head {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.class-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}
.class-chip {
    font-size: 0.84rem;
    font-weight: 500;
    color: var(--ink);
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 3px 11px;
    line-height: 1.4;
}
.class-chip.active {
    background: var(--accent);
    color: #FFFFFF;
    border-color: var(--accent);
    font-weight: 600;
}
.class-note {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 8px;
    line-height: 1.4;
}
</style>
"""


def html(markup: str) -> None:
    """Render raw HTML/CSS, flattened so Markdown never mistakes indentation for code."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def inject_theme(top_class: str | None) -> Sky:
    """Re-tint the page with the atmosphere of the predicted class."""
    sky = SKIES.get(top_class or "", DEFAULT_SKY)
    st.session_state["accent"] = sky.accent
    html(
        f"<style>:root{{--accent:{sky.accent};--sky-top:{sky.top};--sky-bottom:{sky.bottom};}}</style>"
    )
    return sky


html(BASE_CSS)


# ── Architecture Reference Data (single source for bars and tables) ──────────

ARCH_SPECS = [
    {
        "name": "MobileNetV3-Large", "key": "mobilenet_v3_large",
        "paradigm": "Hard-swish depthwise CNN",
        "params_m": 4.2, "fp32_mb": 16.06, "int8_mb": 12.53, "int8_label": "12.53 MB",
        "latency_ms": 2.87, "xai": "Grad-CAM / Grad-CAM++",
    },
    {
        "name": "ResNet-50", "key": "resnet50",
        "paradigm": "Residual bottleneck CNN",
        "params_m": 23.5, "fp32_mb": 89.72, "int8_mb": 23.0, "int8_label": "~23 MB",
        "latency_ms": 46.80, "xai": "Grad-CAM / Grad-CAM++",
    },
    {
        "name": "ViT-Tiny (timm)", "key": "vit_tiny_patch16_224",
        "paradigm": "Multi-head self-attention, 16×16 patches",
        "params_m": 5.5, "fp32_mb": 21.08, "int8_mb": 6.0, "int8_label": "~6 MB",
        "latency_ms": 14.13, "xai": "Attention rollout",
    },
]


def arch_table_html() -> str:
    """Spec table; the best value in each numeric row is marked in bold with an accent underline."""
    rows = [
        ("Paradigm", lambda s: s["paradigm"], None),
        ("Parameters", lambda s: f'{s["params_m"]} M', "params_m"),
        ("Disk size, FP32", lambda s: f'{s["fp32_mb"]:.2f} MB', "fp32_mb"),
        ("Disk size, INT8", lambda s: s["int8_label"], "int8_mb"),
        ("Edge CPU latency", lambda s: f'{s["latency_ms"]:.2f} ms', "latency_ms"),
        ("Explainability", lambda s: s["xai"], None),
    ]
    head = "".join(f"<th>{escape(s['name'])}</th>" for s in ARCH_SPECS)
    body = []
    for label, fmt, num_key in rows:
        best = min(s[num_key] for s in ARCH_SPECS) if num_key else None
        cells = "".join(
            f'<td class="{"best" if num_key and s[num_key] == best else ""}">{escape(fmt(s))}</td>'
            for s in ARCH_SPECS
        )
        body.append(f"<tr><td>{label}</td>{cells}</tr>")
    return f'<table class="spec"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


# ── Model & Engine Caching ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pytorch_model(
    model_name: str,
    checkpoint_path: str | None = None,
    num_classes: int = len(DEFAULT_CLASSES),
) -> WeatherClassifier:
    """Load or initialize PyTorch model with optional checkpoint weights."""
    model = build_model(config_or_name=model_name, num_classes=num_classes, pretrained=False)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_onnx_session(onnx_path: str) -> ort.InferenceSession | None:
    """Load ONNX Runtime InferenceSession with CPU Execution Provider."""
    if not HAS_ORT or not Path(onnx_path).exists():
        return None
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return session


def preprocess_image_for_model(
    img: Image.Image,
    target_size: int | tuple[int, int] = 224,
) -> tuple[np.ndarray, torch.Tensor]:
    """Convert PIL image to preprocessed RGB array and normalized PyTorch tensor."""
    rgb_img = np.array(img.convert("RGB"))
    size_int = target_size[0] if isinstance(target_size, (tuple, list)) else int(target_size)
    transform = get_eval_transforms(image_size=size_int)
    augmented = transform(image=rgb_img)
    tensor = augmented["image"].unsqueeze(0)  # Shape: (1, 3, H, W)
    return rgb_img, tensor


def run_onnx(path: Path, tensor: torch.Tensor) -> tuple[np.ndarray, float, float] | None:
    """Run one timed ONNX inference. Returns (probabilities, latency_ms, size_mb) or None."""
    session = load_onnx_session(str(path))
    if session is None:
        return None
    inp_name = session.get_inputs()[0].name
    inp_data = tensor.numpy().astype(np.float32)
    _ = session.run(None, {inp_name: inp_data})  # warm-up
    t0 = time.perf_counter()
    out = session.run(None, {inp_name: inp_data})[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    exp = np.exp(out[0] - np.max(out[0]))
    return exp / np.sum(exp), latency_ms, path.stat().st_size / (1024 * 1024)


def run_pytorch(arch: str, ckpt_file: Path, tensor: torch.Tensor) -> tuple[np.ndarray, float, float]:
    """Run one timed PyTorch inference. Returns (probabilities, latency_ms, size_mb)."""
    ckpt = str(ckpt_file) if ckpt_file.exists() else None
    model = load_pytorch_model(arch, checkpoint_path=ckpt)
    if ckpt:
        size_mb = ckpt_file.stat().st_size / (1024 * 1024)
    else:
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    with torch.no_grad():
        _ = model(tensor)  # warm-up
        t0 = time.perf_counter()
        logits = model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs = F.softmax(logits, dim=-1)[0].numpy()
    return probs, latency_ms, size_mb


# ── Preloaded Curated Samples ────────────────────────────────────────────────

def get_preloaded_sample_paths() -> dict[str, Path]:
    """Scan available real dataset or synthetic sample images."""
    samples = {}
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        for cls in DEFAULT_CLASSES:
            cls_folder = raw_dir / cls
            if cls_folder.exists():
                matches = sorted(cls_folder.glob("*.jpg"))
                if matches:
                    samples[cls] = matches[0]

    if not samples:
        sample_dir = Path("data/synthetic_dryrun/images")
        if sample_dir.exists():
            for cls in DEFAULT_CLASSES:
                matches = list(sample_dir.glob(f"{cls}_*.jpg"))
                if matches:
                    samples[cls] = matches[0]
    return samples


# ── Small HTML Component Builders ────────────────────────────────────────────

def gauge_svg(conf: float) -> str:
    """Barometer-style semicircle gauge for prediction confidence (0-100)."""
    c = max(0.0, min(100.0, conf))
    arc = "M 12 94 A 78 78 0 0 1 168 94"
    return (
        f'<svg class="gauge" viewBox="0 0 180 112" role="img" aria-label="Confidence {c:.0f} percent">'
        f'<path class="gauge-track" d="{arc}" pathLength="100"/>'
        f'<path class="gauge-fill" d="{arc}" pathLength="100" style="stroke-dashoffset:{100 - c:.2f}"/>'
        f'<text x="90" y="86" text-anchor="middle" class="gauge-num">{c:.0f}<tspan class="gauge-pct">%</tspan></text>'
        f'<text x="90" y="108" text-anchor="middle" class="gauge-cap">confidence</text>'
        f"</svg>"
    )


def readout_cell(label: str, value: str, unit: str = "", note: str = "", text: bool = False) -> str:
    unit_html = f'<span class="ro-unit">{escape(unit)}</span>' if unit else ""
    cls = "ro-value text" if text else "ro-value"
    return (
        f'<div class="ro-cell"><div class="ro-label">{escape(label)}</div>'
        f'<div class="{cls}">{escape(value)}{unit_html}</div>'
        f'<div class="ro-note">{note}</div></div>'
    )


def bars_html(
    names: list[str],
    values: list[float],
    fmt: str,
    highlight: int | None = 0,
    wide: bool = False,
    scale_max: float = 100.0,
) -> str:
    rows = []
    for i, (n, v) in enumerate(zip(names, values, strict=False)):
        width = max(v / scale_max * 100.0, 0.5)
        top = " top" if i == highlight else ""
        rows.append(
            f'<div class="dist-row{top}"><span class="dist-name">{escape(n)}</span>'
            f'<span class="dist-track"><span class="bar" style="width:{width:.2f}%"></span></span>'
            f'<span class="dist-val">{fmt.format(v)}</span></div>'
        )
    return f'<div class="dist{" wide" if wide else ""}">{"".join(rows)}</div>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

@dataclass
class Settings:
    arch: str
    arch_label: str
    engine: str
    xai_method: str
    alpha: float
    colormap: str
    onnx_int8: Path
    onnx_fp32: Path
    ckpt: Path


def render_sidebar() -> Settings:
    with st.sidebar:
        html('<div class="side-title">Model</div>')

        arch_options = {
            "MobileNetV3-Large (edge CNN)": "mobilenet_v3_large",
            "ResNet-50 (residual CNN)": "resnet50",
            "ViT-Tiny (vision transformer)": "vit_tiny_patch16_224",
        }
        arch_label = st.selectbox("Architecture", options=list(arch_options.keys()), index=0)
        arch = arch_options[arch_label]

        onnx_int8 = Path(f"outputs/onnx/{arch}_fp32_int8.onnx")
        onnx_fp32 = Path(f"outputs/onnx/{arch}_fp32.onnx")
        ckpt = Path(f"outputs/checkpoints/{arch}_best.pt")

        engine_options = [
            "ONNX Runtime INT8 (dynamic quantized)",
            "ONNX Runtime FP32",
            "PyTorch FP32",
        ]
        engine = st.selectbox(
            "Inference engine",
            options=engine_options,
            index=0 if onnx_int8.exists() else 2,
        )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        html('<div class="side-title">Explanation</div>')
        if "vit" in arch:
            xai_method = st.selectbox("Method", ["Attention Rollout (ViT)"])
        else:
            xai_method = st.selectbox("Method", ["Grad-CAM", "Grad-CAM++"])

        alpha = st.slider("Heatmap opacity", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        colormap = st.selectbox("Colormap", options=["JET", "VIRIDIS", "INFERNO", "TURBO"], index=0)

        st.markdown("&nbsp;", unsafe_allow_html=True)
        html(
            f"""
            <div class="side-note">
                EdgeWeather v0.1.0<br>
                Trained on {len(DEFAULT_CLASSES)} atmospheric classes with a confusion-aware loss
                and cosine annealing.
            </div>
            """
        )

    return Settings(arch, arch_label, engine, xai_method, alpha, colormap, onnx_int8, onnx_fp32, ckpt)


# ── Tab 1: Live Inference ────────────────────────────────────────────────────

def render_live_tab(cfg: Settings) -> None:
    st.markdown("### Choose a photo")

    in_left, in_right = st.columns([1.2, 1], gap="large")
    with in_left:
        uploaded_file = st.file_uploader(
            "Upload a sky or weather photo (JPG, PNG or WebP)",
            type=["jpg", "jpeg", "png", "webp"],
        )

    samples = get_preloaded_sample_paths()
    sample_key = None
    with in_right:
        if samples:
            sample_key = st.selectbox(
                "Or try a sample",
                options=list(samples.keys()),
                format_func=lambda k: k.title(),
                key="sample_choice",
            )
            st.caption("An uploaded photo takes priority over the sample.")
        else:
            st.caption("No sample images found in data/raw or data/synthetic_dryrun.")

    # Determine active image source
    active_pil_img: Image.Image | None = None
    source_desc = ""
    if uploaded_file is not None:
        active_pil_img = Image.open(uploaded_file).convert("RGB")
        source_desc = f"Uploaded: {uploaded_file.name}"
    elif sample_key is not None:
        sample_path = samples[sample_key]
        active_pil_img = Image.open(sample_path).convert("RGB")
        source_desc = f"Sample: {sample_path.name}"

    if active_pil_img is None:
        inject_theme(None)
        html(
            """
            <div class="empty">
                <div class="empty-title">Drop in a photo of the sky</div>
                Upload a JPG, PNG or WebP above and EdgeWeather will name the
                condition, time the prediction and show where the model looked.
            </div>
            """
        )
        return

    st.caption(f"{source_desc}, {active_pil_img.size[0]}×{active_pil_img.size[1]} px")

    # ── Preprocessing & inference
    rgb_raw, tensor_input = preprocess_image_for_model(active_pil_img)

    engine = cfg.engine
    result = None
    notice = ""
    if "INT8" in engine:
        if cfg.onnx_int8.exists():
            result = run_onnx(cfg.onnx_int8, tensor_input)
        if result is None:
            notice = f"The INT8 model is unavailable ({cfg.onnx_int8}), or onnxruntime is not installed. Using PyTorch FP32 instead."
            engine = "PyTorch FP32"
    elif "ONNX Runtime FP32" in engine:
        if cfg.onnx_fp32.exists():
            result = run_onnx(cfg.onnx_fp32, tensor_input)
        if result is None:
            notice = f"The ONNX FP32 model is unavailable ({cfg.onnx_fp32}), or onnxruntime is not installed. Using PyTorch FP32 instead."
            engine = "PyTorch FP32"
    if result is None:
        result = run_pytorch(cfg.arch, cfg.ckpt, tensor_input)

    probabilities, latency_ms, model_size_mb = result
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    top_idx = int(np.argmax(probabilities))
    top_class = DEFAULT_CLASSES[top_idx]
    top_conf = float(probabilities[top_idx]) * 100.0
    order = np.argsort(probabilities)[::-1]
    runner_idx = int(order[1]) if len(order) > 1 else top_idx
    runner_conf = float(probabilities[runner_idx]) * 100.0

    inject_theme(top_class)
    if notice:
        st.caption(notice)

    # ── Explainability (computed on the PyTorch model regardless of engine)
    py_model_for_xai = load_pytorch_model(
        cfg.arch,
        checkpoint_path=str(cfg.ckpt) if cfg.ckpt.exists() else None,
    )
    cv2_cmap = {
        "JET": cv2.COLORMAP_JET,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "TURBO": cv2.COLORMAP_TURBO,
    }[cfg.colormap]

    h, w = rgb_raw.shape[:2]
    raw_cam = np.zeros((h, w), dtype=np.float32)
    overlay_result = rgb_raw
    try:
        if "vit" in cfg.arch:
            rollout = AttentionRolloutViT(py_model_for_xai)
            raw_cam, _, _ = rollout.generate_rollout(tensor_input)
        elif cfg.xai_method == "Grad-CAM++":
            cam_tool = GradCAM(py_model_for_xai, variant="gradcam++")
            raw_cam, _, _ = cam_tool.generate_heatmap(tensor_input, target_class=top_idx)
        else:
            cam_tool = GradCAM(py_model_for_xai, variant="gradcam")
            raw_cam, _, _ = cam_tool.generate_heatmap(tensor_input, target_class=top_idx)

        overlay_result = overlay_heatmap_on_image(
            rgb_raw, raw_cam, alpha=cfg.alpha, colormap=cv2_cmap,
        )
    except Exception as xai_err:
        logging.warning("XAI extraction fallback: %s", xai_err)
        overlay_result = rgb_raw

    # ── Result panel + image
    st.markdown("&nbsp;", unsafe_allow_html=True)
    panel_col, image_col = st.columns([0.95, 1.05], gap="large")

    ood_html = ""
    if top_conf < 60.0:
        names = ", ".join(f"<code>{escape(c)}</code>" for c in DEFAULT_CLASSES)
        ood_html = (
            f'<div class="ood"><strong>Low confidence ({top_conf:.1f}%), possibly out of distribution.</strong> '
            f"The model is trained on {len(DEFAULT_CLASSES)} adverse atmospheric conditions: {names}. "
            f"Ordinary sunny or clear skies are not in the training set, so probability spreads across "
            f"the closest visual matches (e.g. direct golden sunlight may register as sandstorm).</div>"
        )

    with panel_col:
        html(
            f"""
            <section class="sky-panel">
                <div class="sky-label">Most likely condition</div>
                <div class="sky-name">{escape(top_class.title())}</div>
                <div class="sky-meta">
                    {gauge_svg(top_conf)}
                    <div class="runner">
                        Next most likely<br>
                        <strong>{escape(DEFAULT_CLASSES[runner_idx].title())}</strong>, {runner_conf:.1f}%
                    </div>
                </div>
                {ood_html}
            </section>
            """
        )

    with image_col:
        img_tabs = st.tabs(["Heatmap overlay", "Original", "Heatmap only"])
        with img_tabs[0]:
            st.image(
                overlay_result,
                caption=f"{cfg.xai_method} ({cfg.colormap}, opacity {cfg.alpha:.2f}) for class '{top_class}'",
                use_container_width=True,
            )
        with img_tabs[1]:
            st.image(rgb_raw, caption="Input photo", use_container_width=True)
        with img_tabs[2]:
            disp_cam = raw_cam if raw_cam.shape[:2] == (h, w) else cv2.resize(
                raw_cam, (w, h), interpolation=cv2.INTER_LINEAR
            )
            c_min, c_max = float(disp_cam.min()), float(disp_cam.max())
            if c_max > c_min:
                norm_cam = np.uint8(255.0 * (disp_cam - c_min) / (c_max - c_min + 1e-8))
            else:
                norm_cam = np.zeros((h, w), dtype=np.uint8)
            colored_cam_rgb = cv2.cvtColor(cv2.applyColorMap(norm_cam, cv2_cmap), cv2.COLOR_BGR2RGB)
            st.image(
                colored_cam_rgb,
                caption=f"Normalized saliency map ({cfg.xai_method})",
                use_container_width=True,
            )

    # ── Telemetry readout
    within = latency_ms < 10.0
    speed_note = (
        '<span class="dot ok"></span>Under the 10 ms target'
        if within
        else '<span class="dot"></span>Over the 10 ms target'
    )
    html(
        '<div class="readout">'
        + readout_cell("Latency", f"{latency_ms:.2f}", "ms", speed_note)
        + readout_cell("Throughput", f"{fps:.1f}", "FPS", "Batch size 1, CPU")
        + readout_cell("Model size", f"{model_size_mb:.2f}", "MB", "File on disk")
        + readout_cell("Engine", engine.split(" (")[0], "", escape(cfg.arch_label), text=True)
        + "</div>"
    )

    # ── Probability distribution
    st.markdown("### Confidence by class")
    sorted_names = [DEFAULT_CLASSES[i].title() for i in order]
    sorted_vals = [float(probabilities[i]) * 100.0 for i in order]
    html(bars_html(sorted_names, sorted_vals, "{:.1f}%", highlight=0))

    with st.expander("Show as table"):
        st.dataframe(
            pd.DataFrame({"Weather condition": sorted_names, "Probability (%)": [round(v, 2) for v in sorted_vals]}),
            use_container_width=True,
            hide_index=True,
        )


# ── Tab 2: Benchmark ─────────────────────────────────────────────────────────

def benchmark_chart(df: pd.DataFrame, accent: str) -> alt.Chart | None:
    label_cols = [c for c in ("model", "engine", "precision", "runtime", "backend") if c in df.columns]
    if "p50_ms" not in df.columns or not label_cols:
        return None
    d = df.copy()
    d["config"] = d[label_cols].astype(str).agg(" / ".join, axis=1)
    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=4, color=accent)
        .encode(
            y=alt.Y("config:N", sort="x", title=None, axis=alt.Axis(labelLimit=280)),
            x=alt.X("p50_ms:Q", title="Median latency (ms)"),
            tooltip=["config", "p50_ms"],
        )
        .properties(height=max(120, 34 * len(d)))
        .configure_axis(
            labelColor=INK, titleColor=MUTED, gridColor=LINE, domainColor=LINE,
            labelFont="Instrument Sans", titleFont="Instrument Sans",
        )
        .configure_view(strokeWidth=0)
    )


def render_benchmark_tab(cfg: Settings) -> None:
    accent = st.session_state.get("accent", DEFAULT_SKY.accent)

    st.markdown("### Edge inference benchmark")
    st.markdown(
        "Comparative latency, throughput, and memory footprint measured across three architectural paradigms "
        "(MobileNetV3-Large, ResNet-50, and ViT-Tiny) and three execution runtimes "
        "(PyTorch FP32, ONNX Runtime FP32, and INT8 Dynamic Quantization) running on CPU-constrained edge hardware. "
        "Each benchmark executes warm-up iterations followed by 100 timed trials; "
        "metrics report the mean, standard deviation, and p50 (median), p95, and p99 percentiles."
    )

    st.markdown("#### Latency at a glance")
    ordered = sorted(ARCH_SPECS, key=lambda s: s["latency_ms"])
    selected_pos = next((i for i, s in enumerate(ordered) if s["key"] == cfg.arch), None)
    html(
        bars_html(
            [s["name"] for s in ordered],
            [s["latency_ms"] for s in ordered],
            "{:.2f} ms",
            highlight=selected_pos,
            wide=True,
            scale_max=max(s["latency_ms"] for s in ordered),
        )
    )
    st.caption("Your selected architecture in the sidebar is highlighted above.")

    html('<div class="rule"></div>')
    st.markdown("#### Empirical multi-engine results")
    bench_csv = Path("outputs/results/benchmark_results.csv")
    if bench_csv.exists():
        df_bench = pd.read_csv(bench_csv)

        # Compute speedup relative to PyTorch CPU baseline for each model family
        if {"model", "engine", "p50_ms"} <= set(df_bench.columns):
            py_baselines = {}
            for m in df_bench["model"].unique():
                m_py = df_bench[(df_bench["model"] == m) & (df_bench["engine"].str.contains("PyTorch FP32", na=False))]
                if not m_py.empty:
                    py_baselines[m] = m_py["p50_ms"].values[0]

            if py_baselines:
                df_bench["speedup_vs_pytorch_cpu"] = df_bench.apply(
                    lambda r: f"{py_baselines[r['model']] / r['p50_ms']:.2f}×" if r["model"] in py_baselines else "-",
                    axis=1,
                )

        try:
            chart = benchmark_chart(df_bench, accent)
            if chart is not None:
                st.altair_chart(chart, use_container_width=True, theme=None)
        except Exception as chart_err:
            logging.warning("Benchmark chart skipped: %s", chart_err)

        st.dataframe(df_bench, use_container_width=True, hide_index=True)
    else:
        st.info("No benchmark results found. Run `python -m src.benchmark` to generate them.")

    chart_path = Path("outputs/figures/latency_comparison.png")
    if chart_path.exists():
        st.image(
            str(chart_path),
            caption="Comparative latency (p50 ms) and throughput (FPS) across architectures and engines",
            use_container_width=True,
        )

    st.markdown("#### Key benchmark findings")
    b1, b2, b3 = st.columns(3, gap="medium")
    with b1:
        st.markdown(
            """
            **MobileNetV3-Large**
            - **2.88 ms** CPU latency with INT8 quantization (335 FPS).
            - Well inside the strict **10 ms** real-time budget for robotics and dashcams.
            - 4.3× faster than PyTorch eager CPU execution.
            """
        )
    with b2:
        st.markdown(
            """
            **ViT-Tiny (timm)**
            - **3.4× model compression**: Footprint drops from 21.2 MB to **6.17 MB**.
            - Delivers **7.27 ms** CPU latency under INT8 quantization (138 FPS).
            - Demonstrates that lightweight vision transformers are practical on CPU edge runtimes.
            """
        )
    with b3:
        st.markdown(
            """
            **ResNet-50**
            - High parameter overhead (23.5M parameters, ~90 MB disk size).
            - Bottlenecked by memory bandwidth on CPU (~21.5 ms even with INT8).
            - Demonstrates the classic edge tradeoff between parameter capacity and latency.
            """
        )

    html('<div class="rule"></div>')
    st.markdown("### Calibration and error analysis")
    st.caption(f"Showing evaluation figures for {cfg.arch_label}.")
    cm_fig = Path(f"outputs/figures/{cfg.arch}_test_confusion_matrix.png")
    rel_fig = Path(f"outputs/figures/{cfg.arch}_test_reliability_diagram.png")

    cal_left, cal_right = st.columns(2, gap="large")
    with cal_left:
        st.markdown("#### Normalized confusion matrix")
        if cm_fig.exists():
            st.image(str(cm_fig), use_container_width=True)
        else:
            st.info("Confusion matrix not found. Run `python -m src.evaluate --checkpoint ...`")
    with cal_right:
        st.markdown("#### Reliability diagram (15-bin ECE)")
        if rel_fig.exists():
            st.image(str(rel_fig), use_container_width=True)
        else:
            st.info("Reliability diagram not found. Run `python -m src.evaluate --checkpoint ...`")


# ── Tab 3: Methodology ───────────────────────────────────────────────────────

def cost_heatmap(cost_df: pd.DataFrame, accent: str) -> alt.Chart:
    names = list(cost_df.index)
    long = (
        cost_df.rename_axis("True class")
        .reset_index()
        .melt(id_vars="True class", var_name="Predicted as", value_name="Cost")
    )
    threshold = float(long["Cost"].max()) * 0.55
    base = alt.Chart(long).encode(
        x=alt.X("Predicted as:N", sort=names, title=None, axis=alt.Axis(labelAngle=-40, orient="top")),
        y=alt.Y("True class:N", sort=names, title=None),
    )
    heat = base.mark_rect(stroke="#F8FAFB", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("Cost:Q", scale=alt.Scale(range=["#F1F5F7", accent]), legend=None),
        tooltip=["True class", "Predicted as", alt.Tooltip("Cost:Q", format=".2f")],
    )
    text = base.mark_text(fontSize=10, font="Instrument Sans").encode(
        text=alt.Text("Cost:Q", format=".1f"),
        color=alt.condition(alt.datum.Cost > threshold, alt.value("#FFFFFF"), alt.value(INK)),
    )
    return (
        (heat + text)
        .properties(height=400)
        .configure_axis(labelColor=INK, labelFont="Instrument Sans", domain=False, ticks=False)
        .configure_view(strokeWidth=0)
    )


def render_method_tab() -> None:
    accent = st.session_state.get("accent", DEFAULT_SKY.accent)

    st.markdown("### Research Methodology & Pipeline")

    # Deduplication
    prose, _ = st.columns([1.25, 1], gap="large")
    with prose:
        st.markdown("#### 1. Duplicate-aware dataset splitting")
        st.markdown(
            r"""
            Weather image datasets sourced from web crawls frequently contain near-identical frames:
            burst shots, dashcam footage, and re-encoded duplicates. A standard random split lets copies of the same
            physical scene land in both training and test sets, inflating test evaluation metrics.

            To eliminate data leakage, we compute two complementary perceptual hashes per image:
            $$\mathcal{D}(I_a, I_b) = \min \big( \text{Hamming}(pHash(I_a), pHash(I_b)), \text{Hamming}(dHash(I_a), dHash(I_b)) \big) \le \tau$$
            Pairs with distance $\le \tau$ (default $\tau = 6$) are grouped into connected components using a
            disjoint-set Union-Find algorithm with path compression. Each cluster is strictly allocated to a single split,
            preventing scene leakage.
            """
        )

    html('<div class="rule"></div>')

    # Loss
    loss_left, loss_right = st.columns([1, 1.05], gap="large")
    with loss_left:
        st.markdown("#### 2. Confusion-aware cost-sensitive loss")
        st.markdown(
            r"""
            Standard cross-entropy penalizes all classification errors equally. In automated vision systems,
            different error modes carry asymmetric real-world consequences: misidentifying dense fog as a rainbow
            is far more critical than confusing dew with frost.

            We incorporate an expected-cost regularizer into cross-entropy training:
            $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(y, \hat{p}) + \lambda \sum_{c=1}^{K} C_{y, c} \cdot \hat{p}_c$$
            where $C \in \mathbb{R}^{K \times K}$ is a domain safety cost matrix and $\lambda$ governs the penalty weight
            ($\lambda = 0.5$). The full matrix is visualized on the right (rows: ground truth class, columns: predicted class).
            """
        )
    with loss_right:
        cost_matrix_tensor = build_domain_confusion_cost_matrix(DEFAULT_CLASSES)
        cost_df = pd.DataFrame(
            cost_matrix_tensor.numpy(),
            index=[c.title() for c in DEFAULT_CLASSES],
            columns=[c.title() for c in DEFAULT_CLASSES],
        )
        try:
            st.altair_chart(cost_heatmap(cost_df, accent), use_container_width=True, theme=None)
        except Exception as chart_err:
            logging.warning("Cost heatmap fallback: %s", chart_err)
            st.dataframe(cost_df, use_container_width=True)

    html('<div class="rule"></div>')

    # Explainability (XAI)
    st.markdown("#### 3. Explainability: CNNs vs. Vision Transformers")
    xai_c1, xai_c2 = st.columns(2, gap="large")
    with xai_c1:
        st.markdown(
            r"""
            **Grad-CAM & Grad-CAM++ (CNNs)**

            For convolutional architectures (MobileNetV3 and ResNet-50), we compute class-discriminative saliency
            using the final convolutional stage (`features.16` and `layer4.2` respectively).

            Grad-CAM weights convolutional feature activation maps $A^k$ by the global-average-pooled gradients of the score:
            $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\Big(\sum_k \alpha_k^c A^k\Big)$$

            Grad-CAM++ incorporates higher-order partial derivatives to better localize multiple occurrences and fine dispersed particles (e.g. rain streaks and hail).
            """
        )
    with xai_c2:
        st.markdown(
            r"""
            **Attention Rollout (Vision Transformers)**

            Vision Transformers (ViT) divide input images into $16 \times 16$ non-overlapping patches and lack 2D convolutional feature maps.
            Standard Grad-CAM on transformer tokens produces noisy, disjointed activations because self-attention mixes token information across all layers.

            We implement **Attention Rollout** (Abnar & Zuidema, 2020), which recursively tracks attention flow across all 12 multi-head self-attention blocks:
            $$A^{(l)} = 0.5 A_{\text{raw}}^{(l)} + 0.5 I, \quad R^{(l)} = A^{(l)} R^{(l-1)}$$
            The resulting matrix tracks the attention of the classification token `[CLS]` across all spatial patch tokens from layer 1 through layer 12.
            """
        )

    html('<div class="rule"></div>')

    # ECE
    prose2, _ = st.columns([1.25, 1], gap="large")
    with prose2:
        st.markdown("#### 4. Expected calibration error & reliability")
        st.markdown(
            r"""
            A model deployed on physical edge systems must provide well-calibrated confidence estimates.
            Calibration is quantified by partitioning test predictions into $M=15$ equal-width confidence bins $B_m$
            and computing the weighted absolute deviation between empirical accuracy and mean confidence:
            $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
            We optimize confidence calibration via post-hoc temperature scaling on the validation set.
            """
        )

    html('<div class="rule"></div>')

    st.markdown("#### Architectures compared")
    html(arch_table_html())
    st.caption("Bold with an underline marks the best value in each row.")


# ── Tab 4: Dataset & About ───────────────────────────────────────────────────

def render_about_tab() -> None:
    st.markdown("### Project & Dataset Overview")

    st.markdown(
        """
        This project was developed as an academic and portfolio study in **edge-optimized computer vision**.
        It investigates how modern deep learning architectures handle real-world adverse atmospheric conditions under
        stringent compute, latency, and memory constraints.
        """
    )

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        st.markdown("#### The Multi-Class Weather Dataset")
        st.markdown(
            """
            The models are trained and evaluated on **11 outdoor meteorological classes**:
            - **dew**: Surface condensation micro-droplets on vegetation and glass.
            - **fog**: Dense atmospheric aerosol scattering causing optical attenuation.
            - **frost**: Sub-zero crystalline ice structures on terrestrial surfaces.
            - **glaze**: Transparent ice coatings formed by freezing rain (black ice risk).
            - **hail**: Solid frozen precipitation pellets and impact accumulation.
            - **lightning**: Convective electrical discharge and CMOS illumination flare.
            - **rain**: Active precipitation streaks, lens water droplets, and surface sheen.
            - **rainbow**: Post-precipitation optical refraction and atmospheric clearing.
            - **rime**: Supercooled water droplet accumulation forming opaque ice crusts.
            - **sandstorm**: Airborne mineral dust causing optical wavelength attenuation.
            - **snow**: Crystalline snow cover, blizzard conditions, and ground occlusion.
            """
        )

    with c2:
        st.markdown("#### Open-World & OOD Considerations")
        st.markdown(
            """
            **Why standard sunny skies are out-of-distribution:**
            The dataset intentionally isolates adverse and optical meteorological events. Standard clear sunny skies
            are absent from the class vocabulary.

            In closed-set softmax classification, feeding an ordinary sunny sky forces the network to allocate
            probability mass across its 11 classes, occasionally triggering confusion with sandstorm or glaze.
            The dashboard displays an out-of-distribution notice whenever maximum softmax confidence drops below **60%**.
            """
        )

    html('<div class="rule"></div>')

    st.markdown("#### End-to-end pipeline architecture")
    st.markdown(
        """
        1. **Data Audit & Deduplication**: Perceptual hashing (pHash + dHash) with Union-Find disjoint-set clustering eliminates duplicate image leakage across train/val/test splits.
        2. **Cost-Sensitive Training**: Cross-entropy regularized with an asymmetric domain confusion cost matrix penalizes dangerous error modes.
        3. **Post-Training Quantization**: Export to ONNX with INT8 dynamic quantization yields up to 4.3× speedup and 3.4× model size reduction.
        4. **Validation & Explainability**: 15-bin ECE calibration analysis alongside Grad-CAM / Attention Rollout heatmaps.
        5. **Edge Telemetry Dashboard**: Lightweight interactive runtime interface for real-time inference and explainability.
        """
    )

    html('<div class="rule"></div>')

    st.markdown("#### Replication & local execution")
    st.markdown(
        """
        All stages of the pipeline can be reproduced using standard CLI commands:

        ```bash
        # 1. Prepare data splits with perceptual deduplication
        python -m src.prepare_data --raw-dir data/raw --output-dir data/splits

        # 2. Train baseline architecture (MobileNetV3-Large)
        python -m src.train --config configs/mobilenet_v3.yaml

        # 3. Export model to ONNX & apply INT8 dynamic quantization
        python -m src.quantize --model mobilenet_v3_large --checkpoint outputs/checkpoints/mobilenet_v3_large_best.pt

        # 4. Run multi-engine CPU benchmark suite
        python -m src.benchmark --models mobilenet_v3_large resnet50 vit_tiny_patch16_224

        # 5. Launch interactive edge telemetry dashboard
        streamlit run app/streamlit_app.py
        ```
        """
    )


# ── Main Application ──────────────────────────────────────────────────────────

def main() -> None:
    class_chips_html = "".join(f'<span class="class-chip">{escape(c.title())}</span>' for c in DEFAULT_CLASSES)

    html(
        f"""
        <header class="masthead">
            <div class="wordmark">EdgeWeather</div>
            <p class="mast-sub">
                Real-time image-based atmospheric condition classification on edge hardware.
                Compare three architectures and three inference engines with dual-paradigm explainability.
            </p>
            <div class="class-bar">
                <div class="class-bar-head">Predictable Weather Classes (11)</div>
                <div class="class-tags">
                    {class_chips_html}
                </div>
                <div class="class-note">
                    <strong>Scope notice:</strong> Specialized on adverse and optical weather phenomena.
                    Ordinary clear/sunny skies are out-of-distribution and will exhibit elevated predictive uncertainty.
                </div>
            </div>
        </header>
        """
    )

    cfg = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Classify a photo",
        "Benchmarks",
        "Methodology",
        "Dataset & About",
    ])
    with tab1:
        render_live_tab(cfg)
    with tab2:
        render_benchmark_tab(cfg)
    with tab3:
        render_method_tab()
    with tab4:
        render_about_tab()


if __name__ == "__main__":
    main()
