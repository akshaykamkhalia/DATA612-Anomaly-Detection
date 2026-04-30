"""
Gaussian Diffusion with cosine noise schedule and DDIM sampling.

Implements:
  - Cosine beta schedule (Nichol & Dhariwal, 2021)
  - Forward process q(x_t | x_0)
  - DDIM reverse sampling (Song et al., 2021)
  - Reconstruction pipeline for anomaly detection
"""

import math
import torch
import torch.nn as nn
import numpy as np


def cosine_beta_schedule(timesteps: int = 1000, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from "Improved Denoising Diffusion Probabilistic
    Models" (Nichol & Dhariwal, 2021).

    Returns:
        betas: (timesteps,) tensor, values in (0, 0.999)
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    f_t = torch.cos(((t / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


class GaussianDiffusion:
    """
    Manages diffusion schedule, forward noising, and DDIM reverse sampling.

    Args:
        betas: noise schedule tensor of shape (T,)
        device: torch device
    """

    def __init__(self, betas: torch.Tensor, device: str = "cpu"):
        self.timesteps = len(betas)
        self.device = device

        # Precompute schedule quantities
        betas = betas.float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Index tensor at timestep t, reshape for broadcasting with x."""
        batch_size = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward process: compute x_t from x_0.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0: clean images (B, C, H, W)
            t: timesteps (B,) in [0, T-1]
            noise: optional pre-sampled Gaussian noise (B, C, H, W)

        Returns:
            x_t: noised images (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_start: int,
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM reverse sampling from x_{t_start} to x_0.

        Args:
            model: noise prediction network (DiT or UNet)
            x_t: noised image at timestep t_start (B, C, H, W)
            t_start: starting timestep (e.g., 250)
            num_steps: number of DDIM steps (default 50)
            eta: stochasticity (0 = deterministic DDIM, 1 = DDPM)

        Returns:
            x_0_hat: reconstructed image (B, C, H, W)
        """
        # Build subsequence of timesteps from t_start down to 0
        step_size = max(t_start // num_steps, 1)
        timestep_seq = list(range(t_start, 0, -step_size))
        if timestep_seq[-1] != 0:
            timestep_seq.append(0)

        x = x_t
        for i in range(len(timestep_seq) - 1):
            t_cur = timestep_seq[i]
            t_prev = timestep_seq[i + 1]

            t_batch = torch.full((x.shape[0],), t_cur, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = model(x, t_batch)

            # Get alpha values
            alpha_bar_t = self.alphas_cumprod[t_cur]
            alpha_bar_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)

            # Predict x_0
            predicted_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

            # Compute sigma for stochasticity
            sigma = eta * torch.sqrt(
                (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_t_prev)
            )

            # Direction pointing to x_t
            direction = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_t_prev - sigma ** 2, min=0.0)
            ) * predicted_noise

            # Random noise (only if not the last step and eta > 0)
            if t_prev > 0 and eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # DDIM update
            x = torch.sqrt(alpha_bar_t_prev) * predicted_x0 + direction + sigma * noise

        return x

    @torch.no_grad()
    def reconstruct(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t_partial: int = 250,
        num_ddim_steps: int = 50,
    ) -> torch.Tensor:
        """
        Anomaly detection reconstruction: noise then denoise.

        1. Add t_partial steps of noise to x_0
        2. DDIM reverse to get x_0_hat

        Args:
            model: trained noise prediction network
            x_0: clean test images (B, C, H, W) in [-1, 1]
            t_partial: noise steps to add (tune via ablation, default 250)
            num_ddim_steps: DDIM sampling steps (default 50)

        Returns:
            x_0_hat: reconstruction (B, C, H, W)
        """
        batch_size = x_0.shape[0]
        t = torch.full((batch_size,), t_partial - 1, device=self.device, dtype=torch.long)

        # Forward: add noise
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Reverse: DDIM denoise
        x_0_hat = self.ddim_sample(model, x_t, t_start=t_partial - 1, num_steps=num_ddim_steps)

        return x_0_hat


def get_diffusion(timesteps: int = 1000, device: str = "cpu") -> GaussianDiffusion:
    """Create a GaussianDiffusion instance with cosine schedule."""
    betas = cosine_beta_schedule(timesteps)
    return GaussianDiffusion(betas, device=device)


if __name__ == "__main__":
    # Smoke test
    print("Testing cosine beta schedule...")
    betas = cosine_beta_schedule(1000)
    print(f"  betas shape: {betas.shape}")
    print(f"  betas range: [{betas.min():.6f}, {betas.max():.6f}]")

    diff = GaussianDiffusion(betas)
    print(f"  alphas_cumprod[0]:   {diff.alphas_cumprod[0]:.6f} (should be ~1.0)")
    print(f"  alphas_cumprod[999]: {diff.alphas_cumprod[999]:.6f} (should be ~0.0)")

    # Test forward process
    x_0 = torch.randn(2, 3, 128, 128)
    t = torch.tensor([0, 999])
    x_t = diff.q_sample(x_0, t)
    print(f"\n  q_sample shapes: x_0={x_0.shape}, t={t.shape}, x_t={x_t.shape}")

    diff_t0 = (x_t[0] - x_0[0]).abs().mean().item()
    diff_t999 = (x_t[1] - x_0[1]).abs().mean().item()
    print(f"  |x_t - x_0| at t=0:   {diff_t0:.4f} (should be ~0)")
    print(f"  |x_t - x_0| at t=999: {diff_t999:.4f} (should be large)")
    print("\nDiffusion smoke test passed.")
