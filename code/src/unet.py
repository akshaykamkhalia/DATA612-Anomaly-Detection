"""
UNet backbone for diffusion noise prediction.

Drop-in replacement for DiT with the SAME forward interface:
    forward(x: (B, 3, 128, 128), t: (B,)) -> (B, 3, 128, 128)

Architecture (128x128 input):
  Encoder: 4 levels, channel_mults (1,2,4,8), base_channels=64
    128 -> 64 -> 32 -> 16 -> 8 (bottleneck)
  Bottleneck: ResBlock -> SelfAttention -> ResBlock at 8x8
  Decoder: 4 levels symmetric with skip connections (concat)
  Time conditioning: sinusoidal -> MLP -> added in each ResBlock
  GroupNorm(32) throughout, SiLU activations
  Target: 25-40M parameters for fair comparison with DiT-S (33M)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding -> MLP projection."""

    def __init__(self, time_dim: int = 256, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal positional embeddings for timesteps."""
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal_embedding(t, self.frequency_dim)
        return self.mlp(t_emb)


class ResBlock(nn.Module):
    """
    Residual block with time-embedding injection.

    GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + skip + time_emb
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        # Time embedding projection -> added after first conv
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        # Skip connection (1x1 conv if channel mismatch)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            t_emb: (B, time_dim)
        Returns:
            (B, out_channels, H, W)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Add time embedding (broadcast over spatial dims)
        t = self.time_proj(t_emb)[:, :, None, None]
        h = h + t

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample(nn.Module):
    """Spatial downsample via strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsample via nearest interpolation + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SelfAttention(nn.Module):
    """
    Self-attention for spatial feature maps.
    Reshape (B, C, H, W) -> (B, H*W, C), apply MHA, reshape back.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        # Reshape to sequence
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        return x + h


class UNet(nn.Module):
    """
    UNet backbone for diffusion noise prediction.

    Drop-in replacement for DiT with the same forward signature:
        forward(x: (B, 3, 128, 128), t: (B,)) -> (B, 3, 128, 128)

    Args:
        img_size: input image resolution (default 128)
        in_channels: input channels (default 3 for RGB)
        base_channels: base channel count (default 64)
        channel_mults: channel multipliers per encoder level
        time_dim: timestep embedding dimension
    """

    def __init__(
        self,
        img_size: int = 128,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        time_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_dim=time_dim)

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_in = base_channels
        encoder_channels = [ch_in]  # Track channels for skip connections

        for i, mult in enumerate(channel_mults):
            ch_out = base_channels * mult
            self.encoder_blocks.append(ResBlock(ch_in, ch_out, time_dim))
            encoder_channels.append(ch_out)
            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch_out))
            else:
                # No downsample at last encoder level (we downsample into bottleneck)
                self.downsamples.append(Downsample(ch_out))
            ch_in = ch_out

        # --- Bottleneck (at 8x8 for 128->64->32->16->8) ---
        bottleneck_ch = base_channels * channel_mults[-1]
        self.bottleneck_res1 = ResBlock(bottleneck_ch, bottleneck_ch, time_dim)
        self.bottleneck_attn = SelfAttention(bottleneck_ch, num_heads=8)
        self.bottleneck_res2 = ResBlock(bottleneck_ch, bottleneck_ch, time_dim)

        # --- Decoder ---
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        reversed_mults = list(reversed(channel_mults))
        for i in range(len(channel_mults)):
            ch_out = base_channels * reversed_mults[i]
            # Skip connection doubles input channels
            if i == 0:
                # First decoder level: upsample from bottleneck, skip from last encoder
                skip_ch = encoder_channels[-(i + 1)]
            else:
                skip_ch = encoder_channels[-(i + 1)]

            self.upsamples.append(Upsample(ch_in))
            self.decoder_blocks.append(ResBlock(ch_in + skip_ch, ch_out, time_dim))
            ch_in = ch_out

        # --- Output ---
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=ch_in)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(ch_in, in_channels, kernel_size=3, padding=1)

        # Zero-init output conv for stable training start
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict noise from noised image + timestep.

        Args:
            x: noised images (B, 3, 128, 128)
            t: timesteps (B,) integers in [0, T-1]

        Returns:
            predicted noise (B, 3, 128, 128)
        """
        # Timestep embedding
        t_emb = self.time_embed(t)

        # Initial conv
        h = self.conv_in(x)

        # Encoder (save skip connections)
        skips = [h]
        for i, (res_block, down) in enumerate(zip(self.encoder_blocks, self.downsamples)):
            h = res_block(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, t_emb)

        # Decoder (consume skip connections in reverse)
        for i, (up, res_block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            h = up(h)
            skip = skips[-(i + 1)]
            # Handle spatial size mismatch (edge case from stride arithmetic)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = res_block(h, t_emb)

        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        return h


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 3, 128, 128)
    t = torch.tensor([0, 500])
    out = model(x, t)
    print(f"UNet params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shape: {out.shape}")
    print(f"NaN: {torch.isnan(out).any().item()}")
