"""
Evaluation: compute AUROC and PRO metrics for anomaly detection.

Usage:
    python -m src.evaluate --data_root data/mvtec --category hazelnut \
        --checkpoint output/checkpoints/hazelnut/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

from src.dataset import get_dataloaders, get_mvtec_categories
from src.diffusion import GaussianDiffusion, cosine_beta_schedule
from src.dit import DiT_S, DiT_Tiny
from src.scoring import (
    FeatureExtractor,
    compute_pixel_anomaly_map,
    compute_pixel_anomaly_map_l2,
    compute_feature_anomaly_map,
    compute_combined_anomaly_map,
    compute_image_score,
)


def evaluate_category(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    test_loader,
    feature_extractor: FeatureExtractor,
    device: str,
    t_partial: int = 250,
    num_ddim_steps: int = 50,
    alpha: float = 0.5,
    img_size: int = 128,
    scoring: str = "l2",
) -> dict:
    """
    Evaluate anomaly detection on one MVTec category.

    Args:
        scoring: pixel scoring method -- "l2" (MSE, recommended) or "ssim"

    Returns:
        dict with image_auroc, pixel_auroc, and per-image details
    """
    model.eval()
    all_image_scores = []
    all_image_labels = []
    all_pixel_preds = []
    all_pixel_labels = []

    for images, masks, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        # Reconstruct
        x_0_hat = diffusion.reconstruct(model, images, t_partial=t_partial, num_ddim_steps=num_ddim_steps)

        # Pixel anomaly map: L2 (default, faster + better image AUROC) or SSIM
        if scoring == "l2":
            pixel_map = compute_pixel_anomaly_map_l2(images, x_0_hat)
        else:
            pixel_map = compute_pixel_anomaly_map(images, x_0_hat)

        feat_map = compute_feature_anomaly_map(feature_extractor, images, x_0_hat, img_size=img_size)
        combined_map = compute_combined_anomaly_map(pixel_map, feat_map, alpha=alpha)

        # Image-level score: 95th percentile (robust, avoids outlier sensitivity of max)
        img_scores = compute_image_score(combined_map)

        all_image_scores.append(img_scores.cpu().numpy())
        all_image_labels.append(labels.numpy())

        # Pixel-level
        feat_flat = combined_map.cpu().numpy().flatten()
        mask_flat = masks.cpu().numpy().flatten()
        all_pixel_preds.append(feat_flat)
        all_pixel_labels.append(mask_flat)

    # Concatenate
    all_image_scores = np.concatenate(all_image_scores)
    all_image_labels = np.concatenate(all_image_labels)
    all_pixel_preds = np.concatenate(all_pixel_preds)
    all_pixel_labels = np.concatenate(all_pixel_labels)

    # Compute metrics
    image_auroc = roc_auc_score(all_image_labels, all_image_scores)

    # Pixel AUROC (only if there are both classes)
    pixel_auroc = 0.0
    if len(np.unique(all_pixel_labels)) > 1:
        pixel_auroc = roc_auc_score(all_pixel_labels.astype(int), all_pixel_preds)

    return {
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "num_test": len(all_image_labels),
        "num_anomalous": int(all_image_labels.sum()),
        "num_normal": int((all_image_labels == 0).sum()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DiT anomaly detection")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--category", type=str, default=None, help="Single category or 'all'")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output/results")
    parser.add_argument("--model", type=str, default="small", choices=["small", "tiny"])
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--t_partial", type=int, default=250)
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--scoring", type=str, default="l2", choices=["ssim", "l2"],
                        help="Pixel scoring method: l2 (recommended) or ssim")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if args.model == "small":
        model = DiT_S(img_size=args.img_size).to(device)
    else:
        model = DiT_Tiny(img_size=args.img_size).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']})")

    # Diffusion
    betas = cosine_beta_schedule(args.timesteps)
    diffusion = GaussianDiffusion(betas, device=device)

    # Feature extractor
    feat_extractor = FeatureExtractor().to(device)

    # Determine categories
    if args.category is None or args.category == "all":
        categories = get_mvtec_categories()
    else:
        categories = [args.category]

    # Evaluate
    results = {}
    for cat in categories:
        print(f"\n{'='*40}")
        print(f"Category: {cat}")

        _, test_loader = get_dataloaders(
            args.data_root, cat,
            img_size=args.img_size, batch_size=args.batch_size,
        )

        cat_results = evaluate_category(
            model, diffusion, test_loader, feat_extractor,
            device=device, t_partial=args.t_partial,
            num_ddim_steps=args.num_ddim_steps, alpha=args.alpha,
            img_size=args.img_size, scoring=args.scoring,
        )
        results[cat] = cat_results
        print(f"  Image AUROC: {cat_results['image_auroc']:.4f}")
        print(f"  Pixel AUROC: {cat_results['pixel_auroc']:.4f}")

    # Summary
    if len(categories) > 1:
        avg_image = np.mean([r["image_auroc"] for r in results.values()])
        avg_pixel = np.mean([r["pixel_auroc"] for r in results.values()])
        results["_average"] = {"image_auroc": float(avg_image), "pixel_auroc": float(avg_pixel)}
        print(f"\n{'='*40}")
        print(f"Average Image AUROC: {avg_image:.4f}")
        print(f"Average Pixel AUROC: {avg_pixel:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
