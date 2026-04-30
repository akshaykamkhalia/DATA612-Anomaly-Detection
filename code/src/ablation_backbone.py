"""
Backbone ablation: compare DiT vs UNet for anomaly detection.

Loads a DiT checkpoint and a UNet checkpoint, evaluates both on
the same MVTec category test set, and prints a comparison table.

Usage:
    python -m src.ablation_backbone \
        --data_root data/mvtec \
        --category hazelnut \
        --dit_checkpoint output/checkpoints/hazelnut/best.pt \
        --unet_checkpoint output/checkpoints_unet/hazelnut/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.diffusion import GaussianDiffusion, cosine_beta_schedule
from src.dit import DiT_S, DiT_Tiny
from src.unet import UNet
from src.evaluate import evaluate_category
from src.scoring import FeatureExtractor


def count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    sample_input: torch.Tensor,
    device: str,
    t_partial: int = 250,
    num_ddim_steps: int = 50,
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> float:
    """
    Measure average inference time (reconstruction) in seconds.

    Runs warmup iterations then averages over timed iterations.
    """
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            diffusion.reconstruct(
                model, sample_input, t_partial=t_partial, num_ddim_steps=num_ddim_steps,
            )

        # Timed
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(timed_runs):
            diffusion.reconstruct(
                model, sample_input, t_partial=t_partial, num_ddim_steps=num_ddim_steps,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

    return elapsed / timed_runs


def load_model(backbone: str, checkpoint_path: str, img_size: int, device: str) -> nn.Module:
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if backbone == "dit_small":
        model = DiT_S(img_size=img_size).to(device)
    elif backbone == "dit_tiny":
        model = DiT_Tiny(img_size=img_size).to(device)
    elif backbone == "unet":
        model = UNet(img_size=img_size).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Backbone ablation: DiT vs UNet")
    parser.add_argument("--data_root", type=str, required=True, help="Path to MVTec dataset")
    parser.add_argument("--dit_checkpoint", type=str, required=True, help="DiT checkpoint path")
    parser.add_argument("--unet_checkpoint", type=str, required=True, help="UNet checkpoint path")
    parser.add_argument("--category", type=str, required=True, help="MVTec category to evaluate")
    parser.add_argument("--output_dir", type=str, default="output/results",
                        help="Directory for saving results JSON")
    parser.add_argument("--dit_model", type=str, default="dit_small",
                        choices=["dit_small", "dit_tiny"],
                        help="DiT variant used in the checkpoint")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--t_partial", type=int, default=250)
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Diffusion
    betas = cosine_beta_schedule(args.timesteps)
    diffusion = GaussianDiffusion(betas, device=device)

    # Feature extractor (shared)
    feat_extractor = FeatureExtractor().to(device)

    # Data
    _, test_loader = get_dataloaders(
        args.data_root, args.category,
        img_size=args.img_size, batch_size=args.batch_size,
    )

    # Get a sample batch for inference timing
    sample_batch = None
    for images, _, _ in test_loader:
        sample_batch = images[:4].to(device)
        break

    # --- Evaluate DiT ---
    print(f"\n{'='*60}")
    print(f"Loading DiT ({args.dit_model}) from {args.dit_checkpoint}")
    dit_model = load_model(args.dit_model, args.dit_checkpoint, args.img_size, device)
    dit_params = count_params(dit_model)

    print(f"Evaluating DiT on {args.category}...")
    dit_results = evaluate_category(
        dit_model, diffusion, test_loader, feat_extractor,
        device=device, t_partial=args.t_partial,
        num_ddim_steps=args.num_ddim_steps, alpha=args.alpha,
        img_size=args.img_size,
    )

    dit_time = measure_inference_time(
        dit_model, diffusion, sample_batch, device,
        t_partial=args.t_partial, num_ddim_steps=args.num_ddim_steps,
    )
    del dit_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Evaluate UNet ---
    print(f"\n{'='*60}")
    print(f"Loading UNet from {args.unet_checkpoint}")
    unet_model = load_model("unet", args.unet_checkpoint, args.img_size, device)
    unet_params = count_params(unet_model)

    print(f"Evaluating UNet on {args.category}...")
    unet_results = evaluate_category(
        unet_model, diffusion, test_loader, feat_extractor,
        device=device, t_partial=args.t_partial,
        num_ddim_steps=args.num_ddim_steps, alpha=args.alpha,
        img_size=args.img_size,
    )

    unet_time = measure_inference_time(
        unet_model, diffusion, sample_batch, device,
        t_partial=args.t_partial, num_ddim_steps=args.num_ddim_steps,
    )
    del unet_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Comparison Table ---
    print(f"\n{'='*60}")
    print(f"Backbone Ablation Results -- Category: {args.category}")
    print(f"{'='*60}")
    header = f"{'Backbone':<15} | {'Image AUROC':>12} | {'Pixel AUROC':>12} | {'Params':>12} | {'Infer Time':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    dit_label = "DiT-S" if args.dit_model == "dit_small" else "DiT-Tiny"
    print(
        f"{dit_label:<15} | {dit_results['image_auroc']:>12.4f} | "
        f"{dit_results['pixel_auroc']:>12.4f} | {dit_params:>12,} | {dit_time:>10.3f}s"
    )
    print(
        f"{'UNet':<15} | {unet_results['image_auroc']:>12.4f} | "
        f"{unet_results['pixel_auroc']:>12.4f} | {unet_params:>12,} | {unet_time:>10.3f}s"
    )
    print(sep)

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ablation_backbone.json"

    results = {
        "category": args.category,
        "args": vars(args),
        "dit": {
            "backbone": args.dit_model,
            "image_auroc": dit_results["image_auroc"],
            "pixel_auroc": dit_results["pixel_auroc"],
            "params": dit_params,
            "inference_time_sec": round(dit_time, 4),
            "num_test": dit_results["num_test"],
            "num_anomalous": dit_results["num_anomalous"],
        },
        "unet": {
            "backbone": "unet",
            "image_auroc": unet_results["image_auroc"],
            "pixel_auroc": unet_results["pixel_auroc"],
            "params": unet_params,
            "inference_time_sec": round(unet_time, 4),
            "num_test": unet_results["num_test"],
            "num_anomalous": unet_results["num_anomalous"],
        },
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
