"""
Diffusion Transformer (DiT) backbone for noise prediction.

Adapted from Peebles & Xie, "Scalable Diffusion Models with Transformers"
(ICCV 2023). Reference: https://github.com/facebookresearch/DiT

Architecture:
  Patchify -> Positional Embedding -> N x DiTBlock(AdaLN-Zero) -> Unpatchify
  Input: noised image (B, 3, H, W) + timestep (B,)
  Output: predicted noise (B, 3, H, W)
"""

import math
import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Adaptive Layer Norm modulation: scale * x + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Image to patch embedding via Conv2d."""

    def __init__(self, img_size: int = 128, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP -> hidden_size vector."""

    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_dim = frequency_dim

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal positional embeddings for timesteps."""
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal_embedding(t, self.frequency_dim)
        return self.mlp(t_emb)


class DiTBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning.

    Produces 6 modulation parameters (shift, scale, gate for both
    attention and MLP branches) from the timestep embedding.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                          batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # AdaLN-Zero: 6 modulation params (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

        # Zero gates make each block start as an identity-style residual path.
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token sequence (B, N, D)
            c: conditioning vector (B, D) from timestep embedder
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with AdaLN
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * h

        # MLP with AdaLN
        h = modulate(self.norm2(x), shift2, scale2)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x


class FinalLayer(nn.Module):
    """Final AdaLN + linear projection to patch pixels."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for noise prediction.

    Args:
        img_size: input image resolution (default 128)
        patch_size: patch size for patchification (default 4)
        in_channels: input channels (default 3 for RGB)
        hidden_size: transformer hidden dimension
        depth: number of transformer blocks
        num_heads: number of attention heads
        mlp_ratio: MLP hidden dim multiplier
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        self.time_embed = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following DiT paper conventions."""
        # Positional embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

        # Timestep MLP
        for layer in self.time_embed.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

        for block in self.blocks:
            # Attention output projection
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            # MLP
            for layer in block.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (B, num_patches, patch_size^2 * C) -> (B, C, H, W).
        """
        p = self.patch_size
        c = self.in_channels
        h = w = self.img_size // p

        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict noise from noised image + timestep.

        Args:
            x: noised images (B, 3, img_size, img_size)
            t: timesteps (B,) integers in [0, T-1]

        Returns:
            predicted noise (B, 3, img_size, img_size)
        """
        x = self.patch_embed(x) + self.pos_embed

        c = self.time_embed(t)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x


def DiT_S(img_size: int = 128) -> DiT:
    """DiT-Small: 384 hidden, 12 blocks, 6 heads (~33M params)."""
    return DiT(
        img_size=img_size,
        hidden_size=384,
        depth=12,
        num_heads=6,
    )


def DiT_Tiny(img_size: int = 128) -> DiT:
    """DiT-Tiny: 192 hidden, 6 blocks, 3 heads (~4.4M params)."""
    return DiT(
        img_size=img_size,
        hidden_size=192,
        depth=6,
        num_heads=3,
    )


if __name__ == "__main__":
    import time

    print("DiT Backbone Smoke Test")
    print("=" * 40)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_s = DiT_S(img_size=128).to(device)
    params_s = sum(p.numel() for p in model_s.parameters())
    print(f"\nDiT-S parameters: {params_s:,} (~33M expected)")

    x = torch.randn(2, 3, 128, 128, device=device)
    t = torch.tensor([0, 500], device=device)

    start = time.time()
    with torch.no_grad():
        out = model_s(x, t)
    elapsed = (time.time() - start) * 1000
    print(f"DiT-S output shape: {out.shape} (expected [2, 3, 128, 128])")
    print(f"DiT-S forward time: {elapsed:.1f}ms")
    print(f"DiT-S output has NaN: {torch.isnan(out).any().item()}")

    model_t = DiT_Tiny(img_size=128).to(device)
    params_t = sum(p.numel() for p in model_t.parameters())
    print(f"\nDiT-Tiny parameters: {params_t:,} (~4.4M expected)")

    with torch.no_grad():
        out_t = model_t(x, t)
    print(f"DiT-Tiny output shape: {out_t.shape}")
    print(f"DiT-Tiny output has NaN: {torch.isnan(out_t).any().item()}")

    print("\nDiT smoke test passed.")
