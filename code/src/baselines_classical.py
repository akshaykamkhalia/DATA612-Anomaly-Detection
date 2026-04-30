"""
Classical baselines for MVTec AD anomaly detection: PCA and Convolutional Autoencoder.

Provides two reconstruction-based baselines for comparison against the DiT model:
  1. PCA -- flatten images, fit PCA on normals, reconstruct, MSE anomaly map
  2. ConvAutoencoder -- symmetric Conv/ConvTranspose encoder-decoder, MSE anomaly map

Usage:
    python -m src.baselines_classical --data_root data/mvtec --category hazelnut
    python -m src.baselines_classical --data_root data/mvtec --category all
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.dataset import get_dataloaders, get_mvtec_categories


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

IMG_SIZE = 128
IMG_CHANNELS = 3



class PCABaseline:
    """
    PCA reconstruction baseline for anomaly detection.

    Flattens training images to vectors, fits PCA, then scores test images
    by the per-pixel MSE between original and PCA-reconstructed image.
    """

    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    @staticmethod
    def _loader_to_flat_numpy(loader, is_train: bool = True) -> np.ndarray:
        """Collect all images from a DataLoader into a flat numpy array.

        Args:
            loader: DataLoader yielding images (train) or (images, masks, labels) (test).
            is_train: if True, loader returns only images.

        Returns:
            images as (N, C*H*W) float32 numpy array.
        """
        all_imgs = []
        for batch in loader:
            imgs = batch if is_train else batch[0]
            all_imgs.append(imgs.numpy())
        all_imgs = np.concatenate(all_imgs, axis=0)  # (N, C, H, W)
        N = all_imgs.shape[0]
        return all_imgs.reshape(N, -1).astype(np.float32)

    @staticmethod
    def _collect_test_labels(loader):
        """Collect masks and labels from test loader."""
        all_masks, all_labels = [], []
        for _, masks, labels in loader:
            all_masks.append(masks.numpy())
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
        return np.concatenate(all_masks, axis=0), np.concatenate(all_labels, axis=0)

    def fit(self, train_loader):
        """Fit PCA on normal training images."""
        X_train = self._loader_to_flat_numpy(train_loader, is_train=True)
        self.pca.fit(X_train)
        return self

    def predict(self, test_loader):
        """Compute anomaly maps and collect ground truth for evaluation.

        Returns:
            anomaly_maps: (N, 1, H, W) float32 numpy array
            masks: (N, 1, H, W) binary numpy array
            labels: (N,) int numpy array  (0=normal, 1=anomalous)
        """
        X_test = self._loader_to_flat_numpy(test_loader, is_train=False)
        masks, labels = self._collect_test_labels(test_loader)

        Z = self.pca.transform(X_test)
        X_recon = self.pca.inverse_transform(Z)

        # Per-pixel MSE anomaly map  (N, C, H, W) -> mean over C -> (N, 1, H, W)
        N = X_test.shape[0]
        diff = (X_test - X_recon).reshape(N, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        anomaly_maps = (diff ** 2).mean(axis=1, keepdims=True)  # (N, 1, H, W)

        return anomaly_maps, masks, labels



class ConvAutoencoder(nn.Module):
    """
    Symmetric convolutional autoencoder for reconstruction-based anomaly detection.

    Architecture:
        Encoder: Conv2d(3->32) -> Conv2d(32->64) -> Conv2d(64->128) -> Conv2d(128->256)
        Decoder: ConvTranspose2d(256->128) -> ConvTranspose2d(128->64) ->
                 ConvTranspose2d(64->32) -> ConvTranspose2d(32->3, Tanh)
        Input/output: (B, 3, 128, 128), bottleneck: (B, 256, 8, 8)
    """

    def __init__(self):
        super().__init__()
        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Decoder: 8 -> 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(
    model: ConvAutoencoder,
    train_loader,
    device: str = "cpu",
    epochs: int = 50,
    lr: float = 1e-3,
) -> ConvAutoencoder:
    """Train the ConvAutoencoder on normal images with MSE loss."""
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for images in train_loader:
            images = images.to(device)
            recon = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}")

    return model


@torch.no_grad()
def ae_predict(
    model: ConvAutoencoder,
    test_loader,
    device: str = "cpu",
):
    """Compute anomaly maps from autoencoder reconstruction error.

    Returns:
        anomaly_maps: (N, 1, H, W) float32 numpy array
        masks: (N, 1, H, W) binary numpy array
        labels: (N,) int numpy array
    """
    model.eval()
    all_maps, all_masks, all_labels = [], [], []

    for images, masks, labels in test_loader:
        images = images.to(device)
        recon = model(images)

        # |input - reconstruction| mean over channels -> (B, 1, H, W)
        diff = (images - recon).abs().mean(dim=1, keepdim=True)
        all_maps.append(diff.cpu().numpy())
        all_masks.append(masks.numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    anomaly_maps = np.concatenate(all_maps, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return anomaly_maps, masks, labels



def compute_aurocs(anomaly_maps: np.ndarray, masks: np.ndarray, labels: np.ndarray):
    """Compute image-level and pixel-level AUROC.

    Args:
        anomaly_maps: (N, 1, H, W) predicted anomaly scores
        masks: (N, 1, H, W) ground truth binary masks
        labels: (N,) image-level labels (0=normal, 1=anomalous)

    Returns:
        dict with image_auroc, pixel_auroc
    """
    # Image AUROC: score = max of anomaly map per image
    image_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)

    image_auroc = 0.0
    if len(np.unique(labels)) > 1:
        image_auroc = roc_auc_score(labels.astype(int), image_scores)

    # Pixel AUROC
    pixel_preds = anomaly_maps.flatten()
    pixel_labels = masks.flatten().astype(int)

    pixel_auroc = 0.0
    if len(np.unique(pixel_labels)) > 1:
        pixel_auroc = roc_auc_score(pixel_labels, pixel_preds)

    return {
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "num_test": int(len(labels)),
        "num_anomalous": int(labels.sum()),
        "num_normal": int((labels == 0).sum()),
    }



def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical baselines (PCA + ConvAutoencoder) for MVTec AD"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to mvtec/ dataset folder")
    parser.add_argument("--category", type=str, default=None,
                        help="Single MVTec category or 'all' (default: all)")
    parser.add_argument("--output_dir", type=str, default="output/results",
                        help="Directory to save results JSON")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', or 'cuda:0' (auto-detect if omitted)")
    parser.add_argument("--ae_epochs", type=int, default=50,
                        help="Training epochs for ConvAutoencoder (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for data loaders (default: 16)")
    parser.add_argument("--pca_components", type=int, default=100,
                        help="Number of PCA components (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for ConvAutoencoder (default: 1e-3)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    return parser.parse_args()


def format_table(results: dict) -> str:
    """Format results as a simple ASCII table (no tabulate dependency required).

    Falls back to tabulate if available, otherwise uses manual formatting.
    """
    try:
        from tabulate import tabulate
        headers = ["Category", "PCA Img AUROC", "PCA Pix AUROC",
                    "AE Img AUROC", "AE Pix AUROC"]
        rows = []
        for cat in sorted(results.keys()):
            if cat.startswith("_"):
                continue
            pca = results[cat].get("pca", {})
            ae = results[cat].get("autoencoder", {})
            rows.append([
                cat,
                f"{pca.get('image_auroc', 0.0):.4f}",
                f"{pca.get('pixel_auroc', 0.0):.4f}",
                f"{ae.get('image_auroc', 0.0):.4f}",
                f"{ae.get('pixel_auroc', 0.0):.4f}",
            ])
        if "_average" in results:
            avg = results["_average"]
            rows.append([
                "AVERAGE",
                f"{avg['pca_image_auroc']:.4f}",
                f"{avg['pca_pixel_auroc']:.4f}",
                f"{avg['ae_image_auroc']:.4f}",
                f"{avg['ae_pixel_auroc']:.4f}",
            ])
        return tabulate(rows, headers=headers, tablefmt="grid")
    except ImportError:
        # Manual fallback
        line_fmt = "{:<14s} {:>14s} {:>14s} {:>14s} {:>14s}"
        sep = "-" * 76
        lines = [
            sep,
            line_fmt.format("Category", "PCA Img AUROC", "PCA Pix AUROC",
                            "AE Img AUROC", "AE Pix AUROC"),
            sep,
        ]
        for cat in sorted(results.keys()):
            if cat.startswith("_"):
                continue
            pca = results[cat].get("pca", {})
            ae = results[cat].get("autoencoder", {})
            lines.append(line_fmt.format(
                cat,
                f"{pca.get('image_auroc', 0.0):.4f}",
                f"{pca.get('pixel_auroc', 0.0):.4f}",
                f"{ae.get('image_auroc', 0.0):.4f}",
                f"{ae.get('pixel_auroc', 0.0):.4f}",
            ))
        if "_average" in results:
            avg = results["_average"]
            lines.append(sep)
            lines.append(line_fmt.format(
                "AVERAGE",
                f"{avg['pca_image_auroc']:.4f}",
                f"{avg['pca_pixel_auroc']:.4f}",
                f"{avg['ae_image_auroc']:.4f}",
                f"{avg['ae_pixel_auroc']:.4f}",
            ))
        lines.append(sep)
        return "\n".join(lines)


def run_category(
    category: str,
    data_root: str,
    device: str,
    ae_epochs: int,
    lr: float,
    pca_components: int,
    batch_size: int,
    num_workers: int,
) -> dict:
    """Run both PCA and AE baselines on a single category.

    Returns:
        dict with 'pca' and 'autoencoder' sub-dicts containing AUROC metrics.
    """
    print(f"\n{'='*60}")
    print(f"Category: {category}")
    print(f"{'='*60}")

    train_loader, test_loader = get_dataloaders(
        data_root, category,
        img_size=IMG_SIZE, batch_size=batch_size, num_workers=num_workers,
    )

    cat_results = {}

    print(f"\n[PCA] Fitting with {pca_components} components ...")
    pca_baseline = PCABaseline(n_components=pca_components)
    pca_baseline.fit(train_loader)
    pca_maps, pca_masks, pca_labels = pca_baseline.predict(test_loader)
    pca_metrics = compute_aurocs(pca_maps, pca_masks, pca_labels)
    cat_results["pca"] = pca_metrics
    print(f"[PCA]  Image AUROC: {pca_metrics['image_auroc']:.4f}")
    print(f"[PCA]  Pixel AUROC: {pca_metrics['pixel_auroc']:.4f}")

    print(f"\n[AE] Training ConvAutoencoder for {ae_epochs} epochs ...")
    ae_model = ConvAutoencoder()
    ae_model = train_autoencoder(ae_model, train_loader, device=device,
                                 epochs=ae_epochs, lr=lr)
    ae_maps, ae_masks, ae_labels = ae_predict(ae_model, test_loader, device=device)
    ae_metrics = compute_aurocs(ae_maps, ae_masks, ae_labels)
    cat_results["autoencoder"] = ae_metrics
    print(f"[AE]  Image AUROC: {ae_metrics['image_auroc']:.4f}")
    print(f"[AE]  Pixel AUROC: {ae_metrics['pixel_auroc']:.4f}")

    return cat_results


def main():
    args = parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.category is None or args.category.lower() == "all":
        categories = get_mvtec_categories()
    else:
        categories = [args.category]

    results = {}
    for cat in categories:
        cat_results = run_category(
            category=cat,
            data_root=args.data_root,
            device=device,
            ae_epochs=args.ae_epochs,
            lr=args.lr,
            pca_components=args.pca_components,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        results[cat] = cat_results

    if len(categories) > 1:
        pca_img = np.mean([r["pca"]["image_auroc"] for r in results.values()])
        pca_pix = np.mean([r["pca"]["pixel_auroc"] for r in results.values()])
        ae_img = np.mean([r["autoencoder"]["image_auroc"] for r in results.values()])
        ae_pix = np.mean([r["autoencoder"]["pixel_auroc"] for r in results.values()])
        results["_average"] = {
            "pca_image_auroc": float(pca_img),
            "pca_pixel_auroc": float(pca_pix),
            "ae_image_auroc": float(ae_img),
            "ae_pixel_auroc": float(ae_pix),
        }

    print(f"\n\n{'='*60}")
    print("CLASSICAL BASELINE RESULTS")
    print(f"{'='*60}")
    print(format_table(results))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "baselines_classical.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
