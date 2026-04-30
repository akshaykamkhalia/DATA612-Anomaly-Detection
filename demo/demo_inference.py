"""
Demo inference: classify a single image as NORMAL or DEFECTIVE
using the trained DiT-Tiny diffusion-reconstruction model.

This module is the core backend for the live demo (Jupyter or Gradio).

Pipeline (per call):
  1. Load image -> resize 128x128 -> normalize to [-1, 1]
  2. Add t_partial=250 steps of noise (cosine schedule)
  3. DDIM denoise (50 steps) -> reconstruction
  4. Pixel anomaly map (L2) + ResNet-18 feature anomaly map (per-image normalized)
  5. Combined map = 0.5 * pixel + 0.5 * feature
  6. Image score = max of combined map
  7. Compare against per-category threshold -> label

Usage (programmatic):
    from demo_inference import classify
    out = classify("some/image.png", category="hazelnut")
    print(out["label"], out["score"], "vs threshold", out["threshold"])
    out["heatmap"].save("heatmap.png")          # PIL Image
    out["reconstruction"].save("recon.png")
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add the trained model package to the import path for the standalone demo.
DEMO_DIR = Path(__file__).resolve().parent
CODE_DIR = DEMO_DIR.parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.diffusion import GaussianDiffusion, cosine_beta_schedule  # noqa: E402
from src.dit import DiT_Tiny  # noqa: E402
from src.scoring import (  # noqa: E402
    FeatureExtractor,
    compute_pixel_anomaly_map_l2,
    compute_feature_anomaly_map,
    compute_combined_anomaly_map,
    compute_image_score,
)

# Inference settings match the committed evaluation configuration.
IMG_SIZE = 128
TIMESTEPS = 1000
T_PARTIAL = 250
NUM_DDIM_STEPS = 50
ALPHA = 0.5
SUPPORTED_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid",
                        "hazelnut", "leather", "metal_nut", "pill", "screw",
                        "tile", "toothbrush", "transistor", "wood", "zipper"]

CKPT_DIR = DEMO_DIR / "checkpoints"
THRESHOLDS_PATH = DEMO_DIR / "thresholds.json"

# Same normalization the training pipeline used: mean=0.5, std=0.5 -> [-1, 1]
_PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    return DEVICE


@lru_cache(maxsize=1)
def _get_diffusion() -> GaussianDiffusion:
    betas = cosine_beta_schedule(TIMESTEPS)
    return GaussianDiffusion(betas, device=DEVICE)


@lru_cache(maxsize=1)
def _get_feature_extractor() -> FeatureExtractor:
    fe = FeatureExtractor().to(DEVICE).eval()
    return fe


@lru_cache(maxsize=16)
def _get_model(category: str) -> torch.nn.Module:
    """Load and cache a per-category trained DiT-Tiny model."""
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. "
                         f"Supported: {SUPPORTED_CATEGORIES}")
    ckpt_path = CKPT_DIR / category / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint for '{category}' at {ckpt_path}. "
            f"Run extract_checkpoints.py --categories {category}"
        )
    model = DiT_Tiny(img_size=IMG_SIZE).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def available_categories() -> list[str]:
    """Categories that currently have a checkpoint extracted on disk."""
    if not CKPT_DIR.exists():
        return []
    return sorted([p.parent.name for p in CKPT_DIR.glob("*/best.pt")])


def _load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text())
    return {}


def _to_pil(img: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _heatmap_to_pil(amap: np.ndarray) -> Image.Image:
    """Normalize a (H, W) anomaly map to [0,1] and apply jet colormap -> PIL RGB."""
    import matplotlib.cm as cm
    a = amap.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        a = np.zeros_like(a)
    else:
        a = (a - lo) / (hi - lo)
    rgba = (cm.get_cmap("jet")(a) * 255).astype(np.uint8)
    return Image.fromarray(rgba[..., :3])


def _overlay(orig_pil: Image.Image, amap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend the original image with the heatmap to show *where* the defect is."""
    heat = _heatmap_to_pil(amap).resize(orig_pil.size, Image.BILINEAR)
    base = orig_pil.convert("RGB")
    return Image.blend(base, heat, alpha=alpha)


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """(C,H,W) in [-1,1] -> PIL RGB."""
    arr = ((x.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0).clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


@torch.no_grad()
def _score_once(x: torch.Tensor, category: str, seed: int | None) -> dict:
    """One stochastic forward pass. Internal."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = _get_model(category)
    diffusion = _get_diffusion()
    fe = _get_feature_extractor()

    x_hat = diffusion.reconstruct(model, x, t_partial=T_PARTIAL,
                                  num_ddim_steps=NUM_DDIM_STEPS)
    pixel_map = compute_pixel_anomaly_map_l2(x, x_hat)
    feat_map = compute_feature_anomaly_map(fe, x, x_hat, img_size=IMG_SIZE)
    combined = compute_combined_anomaly_map(pixel_map, feat_map, alpha=ALPHA)
    score = compute_image_score(combined).item()
    return {
        "score": float(score),
        "x_hat": x_hat,
        "anomaly_map": combined,
    }


@torch.no_grad()
def score_image(image: Union[str, Path, Image.Image, np.ndarray],
                category: str,
                n_runs: int = 5) -> dict:
    """
    Run the diffusion-reconstruction pipeline on one image and return the
    image-level anomaly score plus the last reconstruction and the AVERAGED
    anomaly map.

    n_runs averaging matches the protocol used to compute thresholds.json,
    so single-image scores at inference time are comparable to the threshold.
    Uses deterministic seeds 0..n_runs-1 for reproducibility.
    """
    pil = _to_pil(image)
    x = _PREPROCESS(pil).unsqueeze(0).to(DEVICE)  # (1, 3, H, W) in [-1, 1]

    scores = []
    amap_sum = None
    last_x_hat = None
    for k in range(n_runs):
        out = _score_once(x, category, seed=k)
        scores.append(out["score"])
        amap_sum = out["anomaly_map"] if amap_sum is None else amap_sum + out["anomaly_map"]
        last_x_hat = out["x_hat"]

    mean_score = float(np.mean(scores))
    amap_avg = (amap_sum / n_runs).squeeze().cpu().numpy()  # (H, W)

    return {
        "score": mean_score,
        "scores_per_run": scores,
        "input_tensor": x.squeeze(0).cpu(),
        "recon_tensor": last_x_hat.squeeze(0).cpu(),
        "anomaly_map": amap_avg,
        "pil_input": pil,
    }


@torch.no_grad()
def classify(image: Union[str, Path, Image.Image, np.ndarray],
             category: str,
             threshold: float | None = None,
             n_runs: int = 5) -> dict:
    """
    Top-level demo entrypoint.

    Args:
        image: file path, PIL Image, or HxWx3 ndarray
        category: MVTec category name (must have a checkpoint extracted)
        threshold: override threshold; if None, read from thresholds.json or
                   fall back to score itself (then the verdict is meaningless --
                   compute_thresholds.py must be run for real demos)

    Returns:
        dict with:
          label: "DEFECTIVE" or "NORMAL"
          score: image-level anomaly score (higher = more anomalous)
          threshold: decision boundary used
          margin: score - threshold (positive => DEFECTIVE)
          confidence: a soft 0..1 score (sigmoid of margin)
          heatmap: PIL Image (jet colormap of anomaly map)
          overlay: PIL Image (heatmap blended on input)
          reconstruction: PIL Image (model's "what it should look like")
          input_resized: PIL Image (what the model actually saw)
    """
    out = score_image(image, category, n_runs=n_runs)
    score = out["score"]

    if threshold is None:
        thresholds = _load_thresholds()
        if category in thresholds:
            threshold = float(thresholds[category]["threshold"])
        else:
            threshold = float("nan")  # unknown -> caller must handle
    threshold = float(threshold)

    margin = score - threshold if threshold == threshold else 0.0  # NaN check
    label = "DEFECTIVE" if (threshold == threshold and score > threshold) else \
            ("NORMAL" if threshold == threshold else "UNKNOWN")
    # Scale the threshold margin into a compact confidence value.
    confidence = float(1.0 / (1.0 + np.exp(-10 * margin))) if threshold == threshold else 0.5

    amap = out["anomaly_map"]
    pil_input = out["pil_input"]
    input_resized = _tensor_to_pil(out["input_tensor"])
    recon = _tensor_to_pil(out["recon_tensor"])
    heatmap = _heatmap_to_pil(amap)
    overlay = _overlay(input_resized, amap, alpha=0.5)

    return {
        "label": label,
        "score": score,
        "threshold": threshold,
        "margin": margin,
        "confidence": confidence,
        "heatmap": heatmap,
        "overlay": overlay,
        "reconstruction": recon,
        "input_resized": input_resized,
        "pil_input_original_size": pil_input,
        "category": category,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to input image")
    p.add_argument("category", help="MVTec category (e.g. hazelnut)")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--save-dir", default="output")
    args = p.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Available categories: {available_categories()}")

    res = classify(args.image, args.category, threshold=args.threshold)
    print(f"\n  label:      {res['label']}")
    print(f"  score:      {res['score']:.6f}")
    print(f"  threshold:  {res['threshold']}")
    print(f"  margin:     {res['margin']:+.6f}")
    print(f"  confidence: {res['confidence']:.3f}")

    save_dir = DEMO_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    res["heatmap"].save(save_dir / f"{stem}_{args.category}_heatmap.png")
    res["overlay"].save(save_dir / f"{stem}_{args.category}_overlay.png")
    res["reconstruction"].save(save_dir / f"{stem}_{args.category}_recon.png")
    res["input_resized"].save(save_dir / f"{stem}_{args.category}_input128.png")
    print(f"\nArtifacts saved to {save_dir}")
