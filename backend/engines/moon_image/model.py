"""
backend/engines/moon_image/model.py
MoonImageModel: Unified Model Container for Moon Image Engine.
Strict Ownership: 100% Random Initialization, Zero External Weights.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .config import MoonImageConfig
from .text_encoder import MoonTextEncoder
from .image_autoencoder import MoonImageAutoencoder
from .diffusion import MoonUNet2D, MoonDiffusionScheduler

class MoonImageModel(nn.Module):
    """
    Complete Sovereign Image Generation Model.
    Contains:
    1. Text Conditioning Transformer (MoonTextEncoder)
    2. Spatial Autoencoder (MoonImageAutoencoder)
    3. Latent UNet with Cross-Attention (MoonUNet2D)
    4. Diffusion Noise Scheduler (MoonDiffusionScheduler)
    """
    def __init__(self, config: MoonImageConfig = None):
        super().__init__()
        self.config = config or MoonImageConfig()

        self.text_encoder = MoonTextEncoder(
            vocab_size=self.config.text_vocab_size,
            embed_dim=self.config.text_embed_dim,
            num_heads=self.config.text_num_heads,
            num_layers=self.config.text_num_layers,
            max_length=self.config.text_max_length
        )

        self.autoencoder = MoonImageAutoencoder(
            in_channels=self.config.image_channels,
            latent_channels=self.config.latent_channels,
            base_dim=self.config.unet_base_channels
        )

        self.unet = MoonUNet2D(
            in_channels=self.config.unet_in_channels,
            out_channels=self.config.unet_out_channels,
            base_channels=self.config.unet_base_channels,
            channel_mult=self.config.unet_channel_mult,
            context_dim=self.config.cross_attention_dim,
            time_embed_dim=self.config.text_embed_dim
        )

        self.scheduler = MoonDiffusionScheduler(
            timesteps=self.config.timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end
        )

        # Initialize all model weights from scratch with truncated Gaussian
        self.apply(self._init_weights)
        self.provenance = "moon_ai_random_init"

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter counts of all sub-networks."""
        text_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        ae_params = sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
        unet_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total = text_params + ae_params + unet_params
        return {
            "text_encoder_params": text_params,
            "autoencoder_params": ae_params,
            "unet_params": unet_params,
            "total_trainable_params": total
        }

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Computes the diffusion training loss:
        1. Encodes RGB image into latent space.
        2. Adds noise at timestep t.
        3. Encodes prompt with text_encoder.
        4. UNet predicts the added noise.
        5. Returns predicted noise and target noise for MSE loss.
        """
        # Encode to latent
        latents = self.autoencoder.encode(pixel_values)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Text conditioning context
        context, _ = self.text_encoder(input_ids)

        # Predict noise
        pred_noise = self.unet(noisy_latents, timesteps, context)
        return pred_noise, noise
