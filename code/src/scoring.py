

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from skimage.metrics import structural_similarity as skimage_ssim


class FeatureExtractor(nn.Module):


    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list:

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        x = (x - mean) / std

        h = self.layer0(x)
        f1 = self.layer1(h)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


def compute_pixel_anomaly_map(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    win_size: int = 11,
) -> torch.Tensor:

    # Convert to CPU numpy in [0, 1] for skimage SSIM full-map computation.
    device = original.device
    original_np = ((original.detach().cpu() + 1.0) / 2.0).clamp(0, 1).numpy()
    reconstruction_np = ((reconstruction.detach().cpu() + 1.0) / 2.0).clamp(0, 1).numpy()

    batch_maps = []
    for i in range(original_np.shape[0]):
        orig_img = np.transpose(original_np[i], (1, 2, 0))   # (H, W, C)
        recon_img = np.transpose(reconstruction_np[i], (1, 2, 0))

        _, ssim_map = skimage_ssim(
            orig_img,
            recon_img,
            data_range=1.0,
            win_size=win_size,
            channel_axis=2,
            full=True,
            gaussian_weights=True,
            sigma=1.5,
        )

        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)
        batch_maps.append((1.0 - ssim_map).astype(np.float32))

    anomaly_map = torch.from_numpy(np.stack(batch_maps, axis=0)).unsqueeze(1).to(device)
    return anomaly_map


def compute_pixel_anomaly_map_l2(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    diff = (original - reconstruction) ** 2
    return diff.mean(dim=1, keepdim=True)


def compute_pixel_anomaly_map_lpips(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    lpips_model=None,
) -> torch.Tensor:
    import lpips as lpips_lib
    if lpips_model is None:
        lpips_model = lpips_lib.LPIPS(net='alex', spatial=True).to(original.device)
    with torch.no_grad():
        dist = lpips_model(original, reconstruction)
    if dist.shape[-2:] != original.shape[-2:]:
        dist = F.interpolate(dist, size=original.shape[-2:], mode='bilinear', align_corners=False)
    return dist


def compute_feature_anomaly_map(
    feature_extractor: FeatureExtractor,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    img_size: int = 128,
) -> torch.Tensor:

    feats_orig = feature_extractor(original)
    feats_recon = feature_extractor(reconstruction)

    anomaly_maps = []
    for f_orig, f_recon in zip(feats_orig, feats_recon):
        # L2 distance per spatial location
        diff = (f_orig - f_recon) ** 2
        diff = diff.mean(dim=1, keepdim=True)  # average over channels
        diff = F.interpolate(diff, size=(img_size, img_size), mode="bilinear", align_corners=False)
        anomaly_maps.append(diff)

    combined = torch.stack(anomaly_maps, dim=0).mean(dim=0)
    return combined


def compute_combined_anomaly_map(
    pixel_map: torch.Tensor,
    feature_map: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:

    B = pixel_map.shape[0]
    pixel_norm = pixel_map.clone()
    feature_norm = feature_map.clone()

    for i in range(B):
        pmin, pmax = pixel_norm[i].min(), pixel_norm[i].max()
        if pmax > pmin:
            pixel_norm[i] = (pixel_norm[i] - pmin) / (pmax - pmin)

        fmin, fmax = feature_norm[i].min(), feature_norm[i].max()
        if fmax > fmin:
            feature_norm[i] = (feature_norm[i] - fmin) / (fmax - fmin)

    return alpha * pixel_norm + (1 - alpha) * feature_norm


def compute_image_score(anomaly_map: torch.Tensor) -> torch.Tensor:

    return anomaly_map.flatten(1).max(dim=1).values


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, C, H, W = 2, 3, 128, 128
    original = torch.randn(B, C, H, W, device=device)
    recon = original + 0.1 * torch.randn_like(original)  # slight perturbation

    pixel_map = compute_pixel_anomaly_map(original, recon)
    print(f"Pixel anomaly map shape: {pixel_map.shape}")
    print(f"Pixel anomaly map range: [{pixel_map.min():.4f}, {pixel_map.max():.4f}]")

    feat_extractor = FeatureExtractor().to(device)
    feat_map = compute_feature_anomaly_map(feat_extractor, original, recon, img_size=H)
    print(f"Feature anomaly map shape: {feat_map.shape}")
    print(f"Feature anomaly map range: [{feat_map.min():.4f}, {feat_map.max():.4f}]")

    combined_map = compute_combined_anomaly_map(pixel_map, feat_map, alpha=0.5)
    print(f"Combined anomaly map shape: {combined_map.shape}")

    scores = compute_image_score(combined_map)
    print(f"Image scores: {scores}")

    l2_map = compute_pixel_anomaly_map_l2(original, recon)
    print(f"\nL2 anomaly map shape: {l2_map.shape}")
    print(f"L2 anomaly map range: [{l2_map.min():.4f}, {l2_map.max():.4f}]")

    try:
        lpips_map = compute_pixel_anomaly_map_lpips(original, recon)
        print(f"LPIPS anomaly map shape: {lpips_map.shape}")
        print(f"LPIPS anomaly map range: [{lpips_map.min():.4f}, {lpips_map.max():.4f}]")
    except ImportError:
        print("LPIPS not installed, skipping LPIPS smoke test.")

    print("\nScoring smoke test passed.")
