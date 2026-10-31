"""
backend/engines/moon_image/image_autoencoder.py
Original Spatial Variational Autoencoder (VAE) for Moon Image.
Compresses RGB images into a compact 4-channel latent space and reconstructs them.
Trained from scratch from random initialization.
"""
import torch
import torch.nn as nn
from typing import Tuple

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class MoonImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_dim: int = 64):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # 3 downsampling stages (total downscale = 2^3 = 8)
        self.down1 = nn.Sequential(
            ResidualConvBlock(base_dim, base_dim),
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1)
        )
        self.down2 = nn.Sequential(
            ResidualConvBlock(base_dim * 2, base_dim * 2),
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1)
        )
        self.down3 = nn.Sequential(
            ResidualConvBlock(base_dim * 4, base_dim * 4),
            nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=3, stride=2, padding=1)
        )
        self.mid = ResidualConvBlock(base_dim * 4, base_dim * 4)
        # Predict mean and log-variance for VAE latent
        self.to_moments = nn.Conv2d(base_dim * 4, 2 * latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.init_conv(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        moments = self.to_moments(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar


class MoonImageDecoder(nn.Module):
    def __init__(self, latent_channels: int = 4, out_channels: int = 3, base_dim: int = 64):
        super().__init__()
        self.from_latent = nn.Conv2d(latent_channels, base_dim * 4, kernel_size=1)
        self.mid = ResidualConvBlock(base_dim * 4, base_dim * 4)

        # 3 upsampling stages (total upscale = 2^3 = 8)
        self.up1 = nn.Sequential(
            ResidualConvBlock(base_dim * 4, base_dim * 4),
            nn.ConvTranspose2d(base_dim * 4, base_dim * 4, kernel_size=4, stride=2, padding=1)
        )
        self.up2 = nn.Sequential(
            ResidualConvBlock(base_dim * 4, base_dim * 2),
            nn.ConvTranspose2d(base_dim * 2, base_dim * 2, kernel_size=4, stride=2, padding=1)
        )
        self.up3 = nn.Sequential(
            ResidualConvBlock(base_dim * 2, base_dim),
            nn.ConvTranspose2d(base_dim, base_dim, kernel_size=4, stride=2, padding=1)
        )
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Outputs bounded to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.mid(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return self.final_conv(h)


class MoonImageAutoencoder(nn.Module):
    """
    Complete learned Spatial Autoencoder for Moon Image.
    Encodes images into latents and decodes latents back to image space.
    """
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_dim: int = 64):
        super().__init__()
        self.encoder = MoonImageEncoder(in_channels, latent_channels, base_dim)
        self.decoder = MoonImageDecoder(latent_channels, in_channels, base_dim)
        self.latent_channels = latent_channels
        self.scale_factor = 0.18215  # Matches standard latent normalization variance

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes an RGB image tensor in [-1, 1] to scaled latent representation."""
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return z * self.scale_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent representation back to [-1, 1] RGB image tensor."""
        z = z / self.scale_factor
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)
        return recon, mean, logvar
