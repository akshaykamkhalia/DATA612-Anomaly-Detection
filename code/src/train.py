"""
Training loop for diffusion model (DiT or UNet backbone) with AMP support.

Usage:
    python -m src.train --category hazelnut --epochs 5
    python -m src.train --category hazelnut --epochs 100 --batch_size 16
    python -m src.train --category hazelnut --backbone unet --epochs 100
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.diffusion import GaussianDiffusion, cosine_beta_schedule
from src.dit import DiT_S, DiT_Tiny

# Backbone name mapping for backward compatibility
_MODEL_TO_BACKBONE = {
    "small": "dit_small",
    "tiny": "dit_tiny",
}


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    category: str,
    data_root: str = "data/mvtec",
    img_size: int = 128,
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    timesteps: int = 1000,
    checkpoint_dir: str = "output/checkpoints",
    device: str = "cuda",
    seed: int = 42,
    model_size: str = "small",
    num_workers: int = 4,
    warmup_epochs: int = 5,
    save_every: int = 10,
    gradient_clip: float = 1.0,
    backbone: str = None,
):
    """
    Train diffusion model on normal images from one MVTec category.

    Args:
        category: MVTec AD category name (e.g. "hazelnut")
        data_root: path to mvtec/ dataset folder
        img_size: image resize dimension
        batch_size: training batch size
        epochs: number of training epochs
        lr: learning rate for AdamW
        weight_decay: AdamW weight decay
        timesteps: total diffusion timesteps (T)
        checkpoint_dir: directory for saving checkpoints and logs
        device: "cuda" or "cpu"
        seed: random seed for reproducibility
        model_size: DEPRECATED -- use backbone instead. "small" or "tiny"
        num_workers: dataloader workers
        warmup_epochs: linear warmup epochs before cosine decay
        save_every: save checkpoint every N epochs
        gradient_clip: max gradient norm for clipping
        backbone: "dit_small", "dit_tiny", or "unet". If None, falls back to
                  model_size for backward compatibility.
    """
    set_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    use_amp = device == "cuda"

    print(f"Loading {category} from {data_root}...")
    train_loader, _ = get_dataloaders(
        data_root, category,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
    )
    print(f"  Train batches: {len(train_loader)} (batch_size={batch_size})")

    # Preserve older --model values while allowing explicit backbone selection.
    if backbone is None:
        backbone = _MODEL_TO_BACKBONE.get(model_size, "dit_small")

    if backbone == "dit_small":
        model = DiT_S(img_size=img_size).to(device)
        backbone_label = "DiT-S"
    elif backbone == "dit_tiny":
        model = DiT_Tiny(img_size=img_size).to(device)
        backbone_label = "DiT-Tiny"
    elif backbone == "unet":
        from src.unet import UNet
        model = UNet(img_size=img_size).to(device)
        backbone_label = "UNet"
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {backbone_label} ({params:,} parameters)")

    betas = cosine_beta_schedule(timesteps)
    diffusion = GaussianDiffusion(betas, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear warmup followed by cosine annealing.
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = Path(checkpoint_dir) / category
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loss_csv_path = ckpt_dir / "loss.csv"
    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "lr", "time_sec"])

    loss_fn = nn.MSELoss()
    print(f"\nTraining {category} for {epochs} epochs (AMP={'ON' if use_amp else 'OFF'})")
    print("-" * 50)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for x_0 in train_loader:
            x_0 = x_0.to(device)
            B = x_0.shape[0]

            t = torch.randint(0, timesteps, (B,), device=device, dtype=torch.long)

            noise = torch.randn_like(x_0)

            # Forward diffusion: x_t = q(x_0, t)
            x_t = diffusion.q_sample(x_0, t, noise)

            with torch.amp.autocast("cuda", enabled=use_amp):
                noise_pred = model(x_t, t)
                loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch:03d}/{epochs}] | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        with open(loss_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_loss:.6f}", f"{current_lr:.2e}", f"{epoch_time:.1f}"])

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "category": category,
                "backbone": backbone,
                "model_size": model_size,
                "img_size": img_size,
            }, ckpt_dir / "best.pt")

        if epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "category": category,
                "backbone": backbone,
                "model_size": model_size,
                "img_size": img_size,
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "category": category,
        "backbone": backbone,
        "model_size": model_size,
        "img_size": img_size,
    }, ckpt_dir / "final.pt")

    print("-" * 50)
    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Loss log saved to: {loss_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model for anomaly detection")
    parser.add_argument("--category", type=str, required=True,
                        help="MVTec AD category (e.g. hazelnut)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str, default="data/mvtec")
    parser.add_argument("--checkpoint_dir", type=str, default="output/checkpoints")
    parser.add_argument("--model", type=str, default="small", choices=["small", "tiny"],
                        help="(Deprecated) Use --backbone instead. Maps small->dit_small, tiny->dit_tiny")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["dit_small", "dit_tiny", "unet"],
                        help="Backbone architecture (default: dit_small). Overrides --model if set.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Explicit --backbone takes precedence; --model remains for older commands.
    resolved_backbone = args.backbone
    if resolved_backbone is None:
        resolved_backbone = _MODEL_TO_BACKBONE.get(args.model, "dit_small")

    train(
        category=args.category,
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
        model_size=args.model,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        save_every=args.save_every,
        backbone=resolved_backbone,
    )
