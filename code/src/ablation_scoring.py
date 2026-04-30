

import argparse
import json
from pathlib import Path

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
    compute_pixel_anomaly_map_l2,
    compute_pixel_anomaly_map_lpips,
    compute_feature_anomaly_map,
    compute_combined_anomaly_map,
    compute_image_score,
)


def reconstruct_all(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    test_loader,
    device: str,
    t_partial: int = 250,
    num_ddim_steps: int = 50,
):
    
    model.eval()
    originals = []
    reconstructions = []
    masks_list = []
    labels_list = []

    for images, masks, labels in tqdm(test_loader, desc="Reconstructing"):
        images = images.to(device)
        masks = masks.to(device)
        x_0_hat = diffusion.reconstruct(
            model, images, t_partial=t_partial, num_ddim_steps=num_ddim_steps,
        )
        originals.append(images)
        reconstructions.append(x_0_hat)
        masks_list.append(masks)
        labels_list.append(labels.numpy())

    return originals, reconstructions, masks_list, labels_list


def score_with_method(
    method_name: str,
    originals,
    reconstructions,
    masks_list,
    labels_list,
    feature_extractor: FeatureExtractor,
    alpha: float,
    img_size: int,
    device: str,
    lpips_model=None,
):
    
    all_image_scores = []
    all_image_labels = []
    all_pixel_preds = []
    all_pixel_labels = []

    for images, x_0_hat, masks, labels in zip(
        originals, reconstructions, masks_list, labels_list,
    ):
        if method_name == "ssim":
            pixel_map = compute_pixel_anomaly_map(images, x_0_hat)
        elif method_name == "l2":
            pixel_map = compute_pixel_anomaly_map_l2(images, x_0_hat)
        elif method_name == "lpips":
            pixel_map = compute_pixel_anomaly_map_lpips(
                images, x_0_hat, lpips_model=lpips_model,
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        feat_map = compute_feature_anomaly_map(
            feature_extractor, images, x_0_hat, img_size=img_size,
        )

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
        description="Ablation: scoring method comparison (SSIM vs L2 vs LPIPS)",
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="small", choices=["small", "tiny"],
    )
    parser.add_argument("--output_dir", type=str, default="output/results")
    parser.add_argument("--t_partial", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
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
    originals, reconstructions, masks_list, labels_list = reconstruct_all(
        model, diffusion, test_loader, device,
        t_partial=args.t_partial, num_ddim_steps=args.num_ddim_steps,
    )

    try:
        import lpips as lpips_lib
        lpips_model = lpips_lib.LPIPS(net="alex", spatial=True).to(device)
        methods = ["ssim", "l2", "lpips"]
    except ImportError:
        print("WARNING: lpips not installed, skipping LPIPS method.")
        lpips_model = None
        methods = ["ssim", "l2"]

    results = {}
    for method in methods:
        print(f"\nScoring with {method.upper()} ...")
        res = score_with_method(
            method_name=method,
            originals=originals,
            reconstructions=reconstructions,
            masks_list=masks_list,
            labels_list=labels_list,
            feature_extractor=feat_extractor,
            alpha=args.alpha,
            img_size=args.img_size,
            device=device,
            lpips_model=lpips_model,
        )
        results[method] = res
        print(f"  Image AUROC: {res['image_auroc']:.4f}")
        print(f"  Pixel AUROC: {res['pixel_auroc']:.4f}")

    print(f"\n{'='*50}")
    print(f"Scoring Method Ablation  --  Category: {args.category}")
    print(f"{'='*50}")
    print(f"{'Method':<10} {'Image AUROC':>12} {'Pixel AUROC':>12}")
    print(f"{'-'*34}")
    for method in methods:
        r = results[method]
        print(f"{method.upper():<10} {r['image_auroc']:>12.4f} {r['pixel_auroc']:>12.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ablation_scoring.json"

    payload = {
        "category": args.category,
        "alpha": args.alpha,
        "t_partial": args.t_partial,
        "methods": methods,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
