"""
Export anomaly maps as .tiff files and run official MVTec PRO evaluation.

Exports full-resolution anomaly maps matching MVTec directory structure,
then optionally runs the official MVTec AD evaluation script to compute
AU-PRO and AU-ROC metrics.

Usage:
    python -m src.export_tiff --checkpoint output/checkpoints/hazelnut/best.pt \
        --category hazelnut --run_eval

    python -m src.export_tiff --checkpoint output/checkpoints/hazelnut/best.pt \
        --category hazelnut --data_root data/mvtec --output_dir output/predictions
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MVTecDataset, MVTEC_CATEGORIES
from src.diffusion import GaussianDiffusion, cosine_beta_schedule
from src.dit import DiT_S, DiT_Tiny
from src.scoring import (
    FeatureExtractor,
    compute_pixel_anomaly_map,
    compute_feature_anomaly_map,
    compute_combined_anomaly_map,
)


class MVTecDatasetWithPaths(MVTecDataset):
    """
    Extends MVTecDataset to also return the original image file path.

    For the test split, __getitem__ returns:
        (image_tensor, mask, label, image_path_str)
    """

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        if self.split == "train":
            return image

        # Test split: return (image, mask, label, path)
        label = self.labels[idx]
        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)

        return image, mask, label, str(self.image_paths[idx])


def collate_with_paths(batch):
    """Custom collate function that handles the path string in the 4th position."""
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    paths = [item[3] for item in batch]
    return images, masks, labels, paths


def get_original_resolution(data_root: str, category: str) -> tuple:
    """
    Determine the original image resolution for a given MVTec category
    by reading one test image from disk with PIL.

    Args:
        data_root: path to mvtec/ folder
        category: MVTec category name

    Returns:
        (height, width) of the original image
    """
    test_dir = Path(data_root) / category / "test"
    # Find the first .png in any subdirectory
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for img_file in sorted(subdir.glob("*.png")):
            with Image.open(img_file) as img:
                width, height = img.size
            return (height, width)

    raise FileNotFoundError(
        f"No test images found for category '{category}' in {test_dir}"
    )


def derive_tiff_path(
    original_path: str,
    data_root: str,
    output_dir: str,
) -> Path:
    """
    Convert an original MVTec image path to its corresponding .tiff output path.

    Maps: data/mvtec/hazelnut/test/crack/000.png
       -> output/predictions/hazelnut/test/crack/000.tiff

    Args:
        original_path: absolute path to the original test image
        data_root: path to mvtec/ folder (e.g., "data/mvtec")
        output_dir: base output directory for predictions

    Returns:
        Path object for the output .tiff file
    """
    original = Path(original_path)
    root = Path(data_root)

    # Get relative path from the data root: e.g., hazelnut/test/crack/000.png
    try:
        rel_path = original.relative_to(root.resolve())
    except ValueError:
        # Fallback: try without resolve
        rel_path = original.relative_to(root)

    # Change extension to .tiff
    tiff_path = Path(output_dir) / rel_path.with_suffix(".tiff")
    return tiff_path


def export_category(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    feature_extractor: FeatureExtractor,
    data_root: str,
    category: str,
    output_dir: str,
    device: str,
    img_size: int = 128,
    t_partial: int = 250,
    num_ddim_steps: int = 50,
    alpha: float = 0.5,
    batch_size: int = 8,
) -> int:
    """
    Export anomaly maps for one MVTec category as .tiff files.

    For each test image:
      1. Reconstruct via DDIM
      2. Compute combined anomaly map at model resolution (img_size x img_size)
      3. Upsample to original MVTec resolution
      4. Save as float32 .tiff

    Args:
        model: trained DiT model
        diffusion: GaussianDiffusion instance
        feature_extractor: pretrained ResNet-18 feature extractor
        data_root: path to mvtec/ folder
        category: MVTec category name
        output_dir: base output directory for .tiff files
        device: torch device
        img_size: model input resolution
        t_partial: partial diffusion timestep
        num_ddim_steps: number of DDIM sampling steps
        alpha: weight for pixel map in combined scoring
        batch_size: batch size for processing

    Returns:
        Number of .tiff files exported
    """
    model.eval()

    # Determine original resolution for this category
    orig_h, orig_w = get_original_resolution(data_root, category)
    print(f"  Original resolution: {orig_h}x{orig_w}")

    # Create dataset with paths
    test_ds = MVTecDatasetWithPaths(
        data_root, category, split="test", img_size=img_size, augment=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues with path strings
        pin_memory=True,
        collate_fn=collate_with_paths,
    )

    num_exported = 0

    for images, masks, labels, paths in tqdm(test_loader, desc=f"  Exporting {category}"):
        images = images.to(device)

        # Reconstruct via DDIM
        with torch.no_grad():
            x_0_hat = diffusion.reconstruct(
                model, images, t_partial=t_partial, num_ddim_steps=num_ddim_steps
            )

        # Compute anomaly maps at model resolution
        pixel_map = compute_pixel_anomaly_map(images, x_0_hat)
        feat_map = compute_feature_anomaly_map(
            feature_extractor, images, x_0_hat, img_size=img_size
        )
        combined_map = compute_combined_anomaly_map(pixel_map, feat_map, alpha=alpha)
        # combined_map shape: (B, 1, img_size, img_size)

        # Upsample to original MVTec resolution
        combined_map_upsampled = F.interpolate(
            combined_map,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        # Shape: (B, 1, orig_h, orig_w)

        # Save each image's anomaly map as .tiff
        for i, path_str in enumerate(paths):
            anomaly_np = combined_map_upsampled[i, 0].cpu().numpy().astype(np.float32)

            tiff_path = derive_tiff_path(path_str, data_root, output_dir)
            tiff_path.parent.mkdir(parents=True, exist_ok=True)

            tifffile.imwrite(str(tiff_path), anomaly_np)
            num_exported += 1

    return num_exported


def run_mvtec_evaluation(
    data_root: str,
    output_dir: str,
    results_dir: str,
    category: str = None,
    eval_script_dir: str = None,
) -> dict:
    """
    Run the official MVTec AD evaluation script via subprocess.

    Args:
        data_root: path to mvtec/ folder (dataset_base_dir)
        output_dir: path to predictions directory (anomaly_maps_dir)
        results_dir: path to store evaluation results
        category: single category or None for all
        eval_script_dir: directory containing evaluate_experiment.py.
            Defaults to data/mvtec_ad_evaluation relative to code/ working dir.

    Returns:
        dict with parsed evaluation results, or empty dict on failure
    """
    if eval_script_dir is None:
        eval_script_dir = str(
            Path(data_root).parent / "mvtec_ad_evaluation"
        )

    eval_script = Path(eval_script_dir) / "evaluate_experiment.py"
    if not eval_script.exists():
        print(f"ERROR: MVTec evaluation script not found at {eval_script}")
        return {}

    cmd = [
        sys.executable,
        str(eval_script),
        "--dataset_base_dir", str(data_root),
        "--anomaly_maps_dir", str(output_dir),
        "--output_dir", str(results_dir),
    ]

    if category is not None:
        cmd.extend(["--evaluated_objects", category])

    print(f"\nRunning MVTec evaluation:")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working dir: {eval_script_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=eval_script_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )

        print(result.stdout)
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")

        if result.returncode != 0:
            print(f"ERROR: Evaluation script exited with code {result.returncode}")
            return {}

    except subprocess.TimeoutExpired:
        print("ERROR: Evaluation script timed out (600s)")
        return {}
    except Exception as e:
        print(f"ERROR: Failed to run evaluation script: {e}")
        return {}

    # Parse output JSON
    metrics_path = Path(results_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print(f"\nEvaluation results loaded from {metrics_path}")

        # Print summary
        if "mean_au_pro" in metrics:
            print(f"  Mean AU-PRO: {metrics['mean_au_pro']:.4f}")
        if "mean_classification_au_roc" in metrics:
            print(f"  Mean AU-ROC: {metrics['mean_classification_au_roc']:.4f}")

        for obj_name in MVTEC_CATEGORIES:
            if obj_name in metrics:
                obj_data = metrics[obj_name]
                pro = obj_data.get("au_pro", "N/A")
                roc = obj_data.get("classification_au_roc", "N/A")
                if isinstance(pro, float):
                    print(f"  {obj_name}: AU-PRO={pro:.4f}, AU-ROC={roc:.4f}")

        return metrics
    else:
        print(f"WARNING: Expected metrics file not found at {metrics_path}")
        return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export anomaly maps as .tiff and run MVTec PRO evaluation"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/mvtec",
        help="Path to MVTec AD dataset root (default: data/mvtec)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained DiT model checkpoint",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Single MVTec category to export, or None for all categories",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/predictions",
        help="Base directory for exported .tiff anomaly maps (default: output/predictions)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="output/results",
        help="Directory for MVTec evaluation results (default: output/results)",
    )
    parser.add_argument(
        "--model", type=str, default="small", choices=["small", "tiny"],
        help="DiT model variant (default: small)",
    )
    parser.add_argument(
        "--img_size", type=int, default=128,
        help="Model input resolution (default: 128)",
    )
    parser.add_argument(
        "--t_partial", type=int, default=250,
        help="Partial diffusion timestep for reconstruction (default: 250)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight for pixel anomaly map in combined scoring (default: 0.5)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for processing test images (default: 8)",
    )
    parser.add_argument(
        "--num_ddim_steps", type=int, default=50,
        help="Number of DDIM sampling steps (default: 50)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000,
        help="Total diffusion timesteps for cosine schedule (default: 1000)",
    )
    parser.add_argument(
        "--run_eval", action="store_true",
        help="Also run official MVTec AD evaluation after exporting .tiff files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    if args.model == "small":
        model = DiT_S(img_size=args.img_size).to(device)
    else:
        model = DiT_Tiny(img_size=args.img_size).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Diffusion
    betas = cosine_beta_schedule(args.timesteps)
    diffusion = GaussianDiffusion(betas, device=device)

    # Feature extractor
    feat_extractor = FeatureExtractor().to(device)

    # Determine categories
    if args.category is not None:
        categories = [args.category]
    else:
        categories = list(MVTEC_CATEGORIES)

    # Export .tiff files for each category
    total_exported = 0
    for cat in categories:
        print(f"\n{'='*50}")
        print(f"Category: {cat}")

        num_exported = export_category(
            model=model,
            diffusion=diffusion,
            feature_extractor=feat_extractor,
            data_root=args.data_root,
            category=cat,
            output_dir=args.output_dir,
            device=device,
            img_size=args.img_size,
            t_partial=args.t_partial,
            num_ddim_steps=args.num_ddim_steps,
            alpha=args.alpha,
            batch_size=args.batch_size,
        )
        total_exported += num_exported
        print(f"  Exported {num_exported} .tiff files")

    print(f"\n{'='*50}")
    print(f"Total exported: {total_exported} .tiff files")
    print(f"Output directory: {args.output_dir}")

    # Run official MVTec evaluation if requested
    if args.run_eval:
        print(f"\n{'='*50}")
        print("Running official MVTec AD evaluation...")
        metrics = run_mvtec_evaluation(
            data_root=args.data_root,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            category=args.category,
        )
        if not metrics:
            print("Evaluation did not produce results.")
    else:
        print("\nSkipping evaluation (use --run_eval to run MVTec evaluation)")


if __name__ == "__main__":
    main()
