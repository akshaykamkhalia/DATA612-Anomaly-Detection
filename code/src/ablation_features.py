

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.diffusion import GaussianDiffusion, cosine_beta_schedule
from src.dit import DiT_S, DiT_Tiny
from src.scoring import (
    FeatureExtractor,
    compute_pixel_anomaly_map,
    compute_feature_anomaly_map,
    compute_combined_anomaly_map,
    compute_image_score,
)


def reconstruct_and_compute_maps(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    test_loader,
    feature_extractor: FeatureExtractor,
    device: str,
    t_partial: int = 250,
    num_ddim_steps: int = 50,
    img_size: int = 128,
):
    """
    Reconstruct all test images and compute pixel + feature maps once.

    Returns:
        pixel_maps: list of (B, 1, H, W) tensors
        feat_maps: list of (B, 1, H, W) tensors
        masks_list: list of (B, 1, H, W) tensors
        labels_list: list of (B,) numpy arrays
    """
    model.eval()
    pixel_maps = []
    feat_maps = []
    masks_list = []
    labels_list = []

    for images, masks, labels in tqdm(test_loader, desc="Reconstructing + scoring"):
        images = images.to(device)
        masks = masks.to(device)

        x_0_hat = diffusion.reconstruct(
            model, images, t_partial=t_partial, num_ddim_steps=num_ddim_steps,
        )

        pixel_map = compute_pixel_anomaly_map(images, x_0_hat)
        feat_map = compute_feature_anomaly_map(
            feature_extractor, images, x_0_hat, img_size=img_size,
        )

        pixel_maps.append(pixel_map)
        feat_maps.append(feat_map)
        masks_list.append(masks)
        labels_list.append(labels.numpy())

    return pixel_maps, feat_maps, masks_list, labels_list


def evaluate_alpha(
    pixel_maps,
    feat_maps,
    masks_list,
    labels_list,
    alpha: float,
):
    """
    Compute AUROC metrics for a given alpha value.

    Args:
        alpha: weight for pixel map (1-alpha for feature map).
               1.0 = pixel-only, 0.0 = feature-only.

    Returns:
        dict with image_auroc and pixel_auroc
    """
    all_image_scores = []
    all_image_labels = []
    all_pixel_preds = []
    all_pixel_labels = []

    for pixel_map, feat_map, masks, labels in zip(
        pixel_maps, feat_maps, masks_list, labels_list,
    ):
        combined_map = compute_combined_anomaly_map(pixel_map, feat_map, alpha=alpha)
        img_scores = compute_image_score(combined_map)

        all_image_scores.append(img_scores.cpu().numpy())
        all_image_labels.append(labels)
        all_pixel_preds.append(combined_map.cpu().numpy().flatten())
        all_pixel_labels.append(masks.cpu().numpy().flatten())

    all_image_scores = np.concatenate(all_image_scores)
    all_image_labels = np.concatenate(all_image_labels)
    all_pixel_preds = np.concatenate(all_pixel_preds)
    all_pixel_labels = np.concatenate(all_pixel_labels)

    image_auroc = roc_auc_score(all_image_labels, all_image_scores)

    pixel_auroc = 0.0
    if len(np.unique(all_pixel_labels)) > 1:
        pixel_auroc = roc_auc_score(all_pixel_labels.astype(int), all_pixel_preds)

    return {
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation: feature-level scoring (pixel-only vs combined vs feature-only)",
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="small", choices=["small", "tiny"],
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--sweep", action="store_true", help="Sweep alpha 0.0-1.0")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--t_partial", type=int, default=250)
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.model == "small":
        model = DiT_S(img_size=args.img_size).to(device)
    else:
        model = DiT_Tiny(img_size=args.img_size).to(device)

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']})")

    betas = cosine_beta_schedule(args.timesteps)
    diffusion = GaussianDiffusion(betas, device=device)
    feat_extractor = FeatureExtractor().to(device)

    _, test_loader = get_dataloaders(
        args.data_root,
        args.category,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    print(f"\nCategory: {args.category}")
    pixel_maps, feat_maps, masks_list, labels_list = reconstruct_and_compute_maps(
        model, diffusion, test_loader, feat_extractor, device,
        t_partial=args.t_partial, num_ddim_steps=args.num_ddim_steps,
        img_size=args.img_size,
    )

    core_alphas = {
        "pixel_only": 1.0,
        "combined": 0.5,
        "feature_only": 0.0,
    }
    results = {}
    for label, alpha_val in core_alphas.items():
        res = evaluate_alpha(pixel_maps, feat_maps, masks_list, labels_list, alpha_val)
        results[label] = {"alpha": alpha_val, **res}

    print(f"\n{'='*55}")
    print(f"Feature-Level Ablation  --  Category: {args.category}")
    print(f"{'='*55}")
    print(f"{'Configuration':<16} {'Alpha':>6} {'Image AUROC':>12} {'Pixel AUROC':>12}")
    print(f"{'-'*46}")
    for label in core_alphas:
        r = results[label]
        print(
            f"{label:<16} {r['alpha']:>6.1f} "
            f"{r['image_auroc']:>12.4f} {r['pixel_auroc']:>12.4f}"
        )

    sweep_results = None
    if args.sweep:
        print(f"\nRunning alpha sweep (0.0 to 1.0, step 0.1) ...")
        alphas = [round(a, 1) for a in np.arange(0.0, 1.05, 0.1).tolist()]
        sweep_results = {}
        for alpha_val in alphas:
            res = evaluate_alpha(
                pixel_maps, feat_maps, masks_list, labels_list, alpha_val,
            )
            sweep_results[str(alpha_val)] = {"alpha": alpha_val, **res}
            print(
                f"  alpha={alpha_val:.1f}  "
                f"Image AUROC={res['image_auroc']:.4f}  "
                f"Pixel AUROC={res['pixel_auroc']:.4f}"
            )

        fig_dir = Path(args.output_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        alphas_plot = [sweep_results[str(a)]["alpha"] for a in alphas]
        img_aurocs = [sweep_results[str(a)]["image_auroc"] for a in alphas]
        pix_aurocs = [sweep_results[str(a)]["pixel_auroc"] for a in alphas]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas_plot, img_aurocs, "o-", label="Image AUROC")
        ax.plot(alphas_plot, pix_aurocs, "s--", label="Pixel AUROC")
        ax.set_xlabel("Alpha (1.0 = pixel-only, 0.0 = feature-only)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"Alpha Sweep -- {args.category}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.05)

        fig_path = fig_dir / "ablation_alpha_sweep.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Sweep plot saved to {fig_path}")

    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "ablation_features.json"

    payload = {
        "category": args.category,
        "t_partial": args.t_partial,
        "core_results": results,
    }
    if sweep_results is not None:
        payload["sweep_results"] = sweep_results

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
