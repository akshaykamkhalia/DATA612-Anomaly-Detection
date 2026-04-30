"""
Visualization: generate 6-panel comparison figures for anomaly detection.

Panel layout:
  [Input] [Reconstruction] [|Input - Recon|]
  [GT Mask] [Anomaly Map] [Overlay]
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [-1, 1] to (H, W, C) numpy in [0, 1]."""
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1).cpu().numpy()
    return np.transpose(img, (1, 2, 0))


def create_six_panel(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    gt_mask: torch.Tensor,
    anomaly_map: torch.Tensor,
    title: str = "",
    save_path: str = None,
) -> plt.Figure:
    """
    Create a 6-panel visualization figure.

    Args:
        original: (3, H, W) in [-1, 1]
        reconstruction: (3, H, W) in [-1, 1]
        gt_mask: (1, H, W) binary mask
        anomaly_map: (1, H, W) anomaly scores
        title: figure title
        save_path: if provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    orig_np = tensor_to_numpy(original)
    recon_np = tensor_to_numpy(reconstruction)
    diff_np = np.abs(orig_np - recon_np).mean(axis=-1)
    gt_np = gt_mask.squeeze().cpu().numpy()
    amap_np = anomaly_map.squeeze().cpu().numpy()

    # Normalize anomaly map for display
    amap_norm = (amap_np - amap_np.min()) / (amap_np.max() - amap_np.min() + 1e-8)

    # Overlay: anomaly map on original
    overlay = orig_np.copy()
    heatmap = plt.cm.jet(amap_norm)[:, :, :3]
    overlay = 0.5 * overlay + 0.5 * heatmap

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(orig_np)
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(recon_np)
    axes[0, 1].set_title("Reconstruction")
    axes[0, 2].imshow(diff_np, cmap="hot")
    axes[0, 2].set_title("|Input - Recon|")

    axes[1, 0].imshow(gt_np, cmap="gray")
    axes[1, 0].set_title("GT Mask")
    axes[1, 1].imshow(amap_norm, cmap="jet")
    axes[1, 1].set_title("Anomaly Map")
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Overlay")

    for ax in axes.flat:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def visualize_batch(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    gt_masks: torch.Tensor,
    anomaly_maps: torch.Tensor,
    labels: list,
    category: str,
    output_dir: str,
    max_samples: int = 8,
):
    """
    Generate 6-panel figures for a batch of test images.

    Args:
        originals: (B, 3, H, W) in [-1, 1]
        reconstructions: (B, 3, H, W) in [-1, 1]
        gt_masks: (B, 1, H, W) binary
        anomaly_maps: (B, 1, H, W)
        labels: list of int (0=normal, 1=anomalous)
        category: MVTec category name
        output_dir: save directory
        max_samples: max figures to generate
    """
    output_dir = Path(output_dir)
    n = min(len(originals), max_samples)

    for i in range(n):
        status = "anomalous" if labels[i] == 1 else "normal"
        title = f"{category} — {status} (sample {i})"
        save_path = output_dir / f"{category}_{status}_{i:03d}.png"

        create_six_panel(
            originals[i], reconstructions[i],
            gt_masks[i], anomaly_maps[i],
            title=title, save_path=str(save_path),
        )

    print(f"Saved {n} visualization panels to {output_dir}")


if __name__ == "__main__":
    # Smoke test with random data
    B, C, H, W = 2, 3, 128, 128
    orig = torch.randn(B, C, H, W)
    recon = orig + 0.1 * torch.randn_like(orig)
    mask = (torch.randn(B, 1, H, W) > 0.5).float()
    amap = torch.rand(B, 1, H, W)

    fig = create_six_panel(
        orig[0], recon[0], mask[0], amap[0],
        title="Smoke Test", save_path="output/figures/smoke_test.png"
    )
    print("Visualization smoke test passed.")
