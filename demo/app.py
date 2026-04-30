"""
Streamlit live demo for DiT-based anomaly detection.

Run from demo/:
    conda run -n base streamlit run app.py

Drag-and-drop a product image -> verdict + heatmap + reconstruction +
score-distribution context (where this image falls vs the full test set).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from demo_inference import (
    DEMO_DIR, DEVICE, available_categories, classify,
    _get_model, _get_diffusion, _get_feature_extractor,
)

st.set_page_config(
    page_title="DiT Anomaly Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .verdict-banner {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 700;
        letter-spacing: 0.04em;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        animation: slideIn 0.4s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .verdict-defective {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    .verdict-normal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .verdict-banner h1 { font-size: 2.2rem; margin: 0; color: white; }
    .verdict-banner p  { margin: 0.4rem 0 0 0; opacity: 0.95; font-size: 1.05rem; }
    .metric-card {
        background: rgba(255,255,255,0.04);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-card .label {
        font-size: 0.78rem;
        text-transform: uppercase;
        opacity: 0.7;
        letter-spacing: 0.05em;
    }
    .metric-card .value { font-size: 1.55rem; font-weight: 600; }
    .panel-title {
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
        opacity: 0.85;
        margin-bottom: 0.4rem;
    }
    .small-caption {
        font-size: 0.78rem;
        opacity: 0.65;
        text-align: center;
        margin-top: 0.2rem;
    }
    div[data-testid="stFileUploader"] > section {
        border: 2px dashed rgba(99, 102, 241, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: rgba(99, 102, 241, 0.85);
        background: rgba(99, 102, 241, 0.04);
    }
</style>
""",
    unsafe_allow_html=True,
)

THRESHOLDS_PATH = DEMO_DIR / "thresholds.json"
SAMPLES_DIR = DEMO_DIR / "sample_images"


@st.cache_data
def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text())
    return {}


@st.cache_data
def list_samples(category: str) -> dict:
    """Return {'normal': [paths], 'defective': [paths]} for a category."""
    out = {"normal": [], "defective": []}
    for kind in ("normal", "defective"):
        d = SAMPLES_DIR / category / kind
        if d.exists():
            out[kind] = sorted([str(p) for p in d.glob("*.png")])
    return out


@st.cache_resource
def warm_models(categories: tuple[str, ...]):
    """Load all models + feature extractor + diffusion ONCE on app start."""
    _get_diffusion()
    _get_feature_extractor()
    for c in categories:
        _get_model(c)
    return True


def render_verdict(label: str, score: float, threshold: float, confidence: float):
    margin = score - threshold
    if label == "DEFECTIVE":
        cls = "verdict-defective"
        icon = "⚠"
        sub = (f"Score {score:.4f} exceeds threshold {threshold:.4f} "
               f"by {margin:+.4f}")
    elif label == "NORMAL":
        cls = "verdict-normal"
        icon = "✓"
        sub = (f"Score {score:.4f} below threshold {threshold:.4f} "
               f"({margin:+.4f})")
    else:
        cls = ""
        icon = "?"
        sub = "No threshold available for this category"
    st.markdown(
        f"""
        <div class="verdict-banner {cls}">
          <h1>{icon} &nbsp; {label}</h1>
          <p>{sub} &nbsp;|&nbsp; confidence {confidence*100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_distribution_chart(category: str, query_score: float) -> go.Figure:
    """Plot the test-set score histogram with a marker for the query image."""
    th = load_thresholds().get(category, {})
    all_scores = th.get("all_scores", [])
    if not all_scores:
        return None

    norm_scores = [r["score"] for r in all_scores if r["label"] == 0]
    def_scores = [r["score"] for r in all_scores if r["label"] == 1]
    threshold = th.get("threshold", float("nan"))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=norm_scores, name="Normal (test set)",
        marker_color="rgba(16, 185, 129, 0.7)",
        nbinsx=30, opacity=0.85,
    ))
    fig.add_trace(go.Histogram(
        x=def_scores, name="Defective (test set)",
        marker_color="rgba(239, 68, 68, 0.7)",
        nbinsx=30, opacity=0.85,
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#a78bfa",
                  annotation_text=f"threshold {threshold:.3f}",
                  annotation_position="top right",
                  annotation_font_color="#c4b5fd")
    fig.add_vline(x=query_score, line_color="#fbbf24", line_width=3,
                  annotation_text=f"this image: {query_score:.3f}",
                  annotation_position="top left",
                  annotation_font_color="#fcd34d")

    fig.update_layout(
        barmode="overlay",
        height=320,
        margin=dict(l=10, r=10, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Image-level anomaly score",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    return fig


def confidence_gauge(score: float, threshold: float, label: str) -> go.Figure:
    margin = score - threshold
    val = float(1.0 / (1.0 + np.exp(-10 * margin)))
    if label == "DEFECTIVE":
        bar_color = "#ef4444"
    elif label == "NORMAL":
        val = 1.0 - val
        bar_color = "#10b981"
    else:
        bar_color = "#6b7280"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val * 100,
        number={"suffix": "%", "font": {"size": 32, "color": bar_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.7},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                {"range": [50, 100], "color": "rgba(255,255,255,0.10)"},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    return fig


st.markdown(
    "<h1 style='margin-bottom:0'>🔬 DiT Anomaly Detection</h1>"
    "<p style='margin-top:0; opacity:0.7'>"
    "Diffusion-reconstruction live demo &nbsp;·&nbsp; Group 6, DATA-MSML 612"
    "</p>",
    unsafe_allow_html=True,
)

all_ckpts = available_categories()
thresholds_dict = load_thresholds()
# Demo-ready categories need both a checkpoint and calibrated threshold.
ready = [c for c in all_ckpts if c in thresholds_dict and
         thresholds_dict[c].get("auroc") is not None]
cats = sorted(ready, key=lambda c: -thresholds_dict[c]["auroc"])
pending = [c for c in all_ckpts if c not in cats]

if not cats:
    st.error("No demo-ready categories found. Run `python extract_checkpoints.py` "
             "and `python compute_thresholds.py` first.")
    st.stop()

with st.sidebar:
    st.markdown("### Category")
    category = st.selectbox(
        "Product category",
        cats,
        help="The model is trained per-category. Choose the product type "
             "your image shows.",
    )

    th = thresholds_dict.get(category, {})
    if th:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="label">Test-set AUROC</div>
              <div class="value">{th.get('auroc', 0):.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:0.5rem">
              <div class="label">Threshold (Youden's J)</div>
              <div class="value">{th.get('threshold', 0):.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Try a sample")
    samples = list_samples(category)

    sample_state_key = "selected_sample"
    chosen = None
    cols = st.columns(2)
    with cols[0]:
        st.caption("Normal")
        for p in samples["normal"]:
            name = Path(p).name
            if st.button(name, key=f"good_{name}", use_container_width=True):
                chosen = p
    with cols[1]:
        st.caption("Defective")
        for p in samples["defective"]:
            name = Path(p).name
            if st.button(name, key=f"bad_{name}", use_container_width=True):
                chosen = p

    st.markdown("---")
    n_runs = st.slider("Inference runs (averaged)", 1, 10, 5,
                       help="More runs = lower variance, slower. Default 5.")

    if pending:
        st.markdown(
            f"<small style='opacity:0.6'>Pending threshold computation: "
            f"{', '.join(pending)}</small>",
            unsafe_allow_html=True,
        )
    st.markdown(f"<small>Device: <code>{DEVICE}</code></small>",
                unsafe_allow_html=True)

with st.spinner(f"Loading {len(cats)} model(s) into VRAM..."):
    warm_models(tuple(cats))

st.markdown("### Drop or pick an image")
upload = st.file_uploader(
    "PNG or JPG of the chosen product",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

image_path = None
image_pil = None
if chosen is not None:
    image_path = chosen
    image_pil = Image.open(chosen).convert("RGB")
    st.session_state.last_source = f"sample: {Path(chosen).name}"
elif upload is not None:
    image_pil = Image.open(upload).convert("RGB")
    st.session_state.last_source = f"upload: {upload.name}"
else:
    st.info("Drag-and-drop an image above, or click a sample in the sidebar.",
            icon="👆")
    st.stop()

with st.spinner(f"Running {n_runs}-pass DDIM reconstruction on {DEVICE}..."):
    t0 = time.time()
    result = classify(image_pil, category=category, n_runs=n_runs)
    elapsed = time.time() - t0

render_verdict(
    label=result["label"],
    score=result["score"],
    threshold=result["threshold"],
    confidence=result["confidence"],
)

m1, m2, m3 = st.columns(3)
m1.metric("Score", f"{result['score']:.3f}")
m2.metric("Threshold", f"{result['threshold']:.3f}")
m3.metric("Inference time", f"{elapsed:.2f}s")

st.markdown("---")

st.markdown("### Reconstruction & localization")
g1, g2, g3, g4 = st.columns(4)

display_size = (256, 256)
input_disp = result["input_resized"].resize(display_size, Image.NEAREST)
recon_disp = result["reconstruction"].resize(display_size, Image.NEAREST)
heat_disp = result["heatmap"].resize(display_size, Image.NEAREST)
over_disp = result["overlay"].resize(display_size, Image.NEAREST)

with g1:
    st.markdown("<div class='panel-title'>Input (128×128)</div>",
                unsafe_allow_html=True)
    st.image(input_disp)
    st.markdown("<div class='small-caption'>What the model sees</div>",
                unsafe_allow_html=True)

with g2:
    st.markdown("<div class='panel-title'>Reconstruction</div>",
                unsafe_allow_html=True)
    st.image(recon_disp)
    st.markdown("<div class='small-caption'>What 'normal' should look like</div>",
                unsafe_allow_html=True)

with g3:
    st.markdown("<div class='panel-title'>Anomaly heatmap</div>",
                unsafe_allow_html=True)
    st.image(heat_disp)
    st.markdown("<div class='small-caption'>Pixel + feature L2 (jet)</div>",
                unsafe_allow_html=True)

with g4:
    st.markdown("<div class='panel-title'>Overlay</div>",
                unsafe_allow_html=True)
    st.image(over_disp)
    st.markdown("<div class='small-caption'>Heatmap on input</div>",
                unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Score in context")
c1, c2 = st.columns([2, 1])

with c1:
    fig_hist = score_distribution_chart(category, result["score"])
    if fig_hist is not None:
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            f"Distribution of image-level scores across the full MVTec '{category}' "
            f"test set. Yellow line = this image. Dashed purple = decision threshold."
        )
    else:
        st.info("No reference distribution available for this category.")

with c2:
    fig_g = confidence_gauge(result["score"], result["threshold"], result["label"])
    st.plotly_chart(fig_g, use_container_width=True)
    st.caption(
        "Confidence in the verdict — sigmoid of the margin from the threshold."
    )

with st.expander("Run details"):
    st.json({
        "category": category,
        "device": str(DEVICE),
        "n_runs": n_runs,
        "elapsed_sec": round(elapsed, 3),
        "score": round(result["score"], 6),
        "threshold": round(result["threshold"], 6),
        "margin": round(result["margin"], 6),
        "label": result["label"],
        "confidence": round(result["confidence"], 4),
        "source": st.session_state.get("last_source", "?"),
        "image_size_internal": "128x128",
        "diffusion": "cosine schedule, T=1000",
        "ddim_steps": 50,
        "t_partial": 250,
        "alpha_pixel_vs_feature": 0.5,
    })
