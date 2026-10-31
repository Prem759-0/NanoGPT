"""
backend/engines/moon_video/video_autoencoder.py
Original Spatiotemporal Video Autoencoder for Moon Video.
Encodes video clips [B, C, F, H, W] to compact latent space [B, 4, F, H/8, W/8] and reconstructs them.
Trained from scratch from random initialization.
"""
import torch
import torch.nn as nn
from typing import Tuple

from ..moon_image.image_autoencoder import MoonImageEncoder, MoonImageDecoder

class MoonVideoAutoencoder(nn.Module):
    """
    Spatiotemporal Video Autoencoder for Moon Video.
    Applies spatial autoencoding across all frames with temporal smoothing.
    """
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_dim: int = 48):
        super().__init__()
        self.encoder = MoonImageEncoder(in_channels=in_channels, latent_channels=latent_channels, base_dim=base_dim)
        self.decoder = MoonImageDecoder(latent_channels=latent_channels, out_channels=in_channels, base_dim=base_dim)
        self.temporal_smooth = nn.Conv3d(latent_channels, latent_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.scale_factor = 0.18215

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B, C, F, H, W] in [-1, 1]
        Returns latents: [B, 4, F, H/8, W/8]
        """
        b, c, f, h, w = video.shape
        # Flatten frames into batch dimension
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        mean, logvar = self.encoder(frames)
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        z = mean + torch.randn_like(std) * std

        # Reshape to [B, latent_channels, F, H/8, W/8]
        _, lc, lh, lw = z.shape
        z_video = z.view(b, f, lc, lh, lw).permute(0, 2, 1, 3, 4)
        z_smooth = self.temporal_smooth(z_video)
        return z_smooth * self.scale_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, 4, F, H/8, W/8]
        Returns video: [B, C, F, H, W] in [-1, 1]
        """
        z = z / self.scale_factor
        b, lc, f, lh, lw = z.shape
        frames_latent = z.permute(0, 2, 1, 3, 4).reshape(b * f, lc, lh, lw)
        decoded_frames = self.decoder(frames_latent)
        _, c, h, w = decoded_frames.shape
        return decoded_frames.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
