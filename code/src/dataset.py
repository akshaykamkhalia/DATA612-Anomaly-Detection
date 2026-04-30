"""
MVTec AD Dataset loader for PyTorch.

Loads per-category train/test splits from the MVTec Anomaly Detection dataset.
Training uses only normal ("good") images. Test returns images, masks, and labels.

Reference: Bergmann et al., "MVTec AD -- A Comprehensive Real-World Dataset
for Unsupervised Anomaly Detection", CVPR 2019.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

MVTEC_CATEGORIES = sorted([
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
])


def get_mvtec_categories() -> list:
    """Return sorted list of all 15 MVTec AD category names."""
    return list(MVTEC_CATEGORIES)


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec Anomaly Detection.

    Args:
        root: path to mvtec/ folder containing category subdirectories
        category: one of the 15 MVTec AD categories
        split: "train" (normal only) or "test" (normal + anomalous)
        img_size: resize images to (img_size, img_size)
        augment: apply training augmentations (only when split="train")
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        img_size: int = 128,
        augment: bool = True,
    ):
        assert category in MVTEC_CATEGORIES, f"Unknown category: {category}"
        assert split in ("train", "test"), f"split must be 'train' or 'test'"

        self.root = Path(root) / category
        self.split = split
        self.img_size = img_size

        # Build file lists
        self.image_paths = []
        self.mask_paths = []
        self.labels = []  # 0 = normal, 1 = anomalous

        if split == "train":
            good_dir = self.root / "train" / "good"
            for img_file in sorted(good_dir.glob("*.png")):
                self.image_paths.append(img_file)
                self.mask_paths.append(None)
                self.labels.append(0)
        else:
            test_dir = self.root / "test"
            gt_dir = self.root / "ground_truth"
            for defect_type in sorted(test_dir.iterdir()):
                if not defect_type.is_dir():
                    continue
                is_good = defect_type.name == "good"
                for img_file in sorted(defect_type.glob("*.png")):
                    self.image_paths.append(img_file)
                    if is_good:
                        self.mask_paths.append(None)
                        self.labels.append(0)
                    else:
                        # Ground truth mask: ground_truth/{defect}/{stem}_mask.png
                        mask_file = gt_dir / defect_type.name / f"{img_file.stem}_mask.png"
                        self.mask_paths.append(mask_file if mask_file.exists() else None)
                        self.labels.append(1)

        # Transforms
        if split == "train" and augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        if self.split == "train":
            return image

        # Test split: return (image, mask, label)
        label = self.labels[idx]
        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)

        return image, mask, label


def get_dataloaders(
    root: str,
    category: str,
    img_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for a single MVTec AD category.

    Args:
        root: path to mvtec/ folder
        category: one of the 15 category names
        img_size: resize dimension
        batch_size: batch size for both loaders
        num_workers: dataloader workers

    Returns:
        (train_loader, test_loader)
    """
    train_ds = MVTecDataset(root, category, split="train", img_size=img_size, augment=True)
    test_ds = MVTecDataset(root, category, split="test", img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick smoke test
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/mvtec"
    category = sys.argv[2] if len(sys.argv) > 2 else "hazelnut"

    print(f"MVTec AD categories: {get_mvtec_categories()}")
    print(f"\nLoading category: {category}")

    train_ds = MVTecDataset(root, category, split="train", img_size=128)
    test_ds = MVTecDataset(root, category, split="test", img_size=128)

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"  Test anomalous: {sum(test_ds.labels)}")
    print(f"  Test normal:    {len(test_ds) - sum(test_ds.labels)}")

    # Check shapes
    img = train_ds[0]
    print(f"\n  Train image shape: {img.shape}")
    print(f"  Train image range: [{img.min():.2f}, {img.max():.2f}]")

    img, mask, label = test_ds[0]
    print(f"  Test image shape:  {img.shape}")
    print(f"  Test mask shape:   {mask.shape}")
    print(f"  Test label:        {label}")
