"""
backend/engines/moon_video/model.py
MoonVideoModel: Unified Spatiotemporal Model Container for Moon Video Engine.
Strict Ownership: 100% Random Initialization, Zero External Weights.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .config import MoonVideoConfig
from .text_encoder import MoonVideoTextEncoder
from .video_autoencoder import MoonVideoAutoencoder
from .diffusion import MoonUNet3D
from ..moon_image.diffusion import MoonDiffusionScheduler

class MoonVideoModel(nn.Module):
    """
    Complete Sovereign Video Generation Model.
    Contains:
    1. Text & Camera Motion Conditioning Encoder (MoonVideoTextEncoder)
    2. Spatiotemporal Video Autoencoder (MoonVideoAutoencoder)
    3. Spatiotemporal Diffusion UNet (MoonUNet3D)
    4. Diffusion Noise Scheduler (MoonDiffusionScheduler)
    """
    def __init__(self, config: MoonVideoConfig = None):
        super().__init__()
        self.config = config or MoonVideoConfig()

        self.text_encoder = MoonVideoTextEncoder(
            vocab_size=self.config.text_vocab_size,
            embed_dim=self.config.text_embed_dim,
            num_heads=self.config.text_num_heads,
            num_layers=self.config.text_num_layers,
            max_length=self.config.text_max_length,
            num_camera_motions=len(self.config.camera_motions) + 1
        )

        self.autoencoder = MoonVideoAutoencoder(
            in_channels=self.config.channels,
            latent_channels=self.config.latent_channels,
            base_dim=self.config.unet_base_channels
        )

        self.unet = MoonUNet3D(
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

        self.apply(self._init_weights)
        self.provenance = "moon_ai_random_init"

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> Dict[str, int]:
        text_p = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        ae_p = sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
        unet_p = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total = text_p + ae_p + unet_p
        return {
            "text_encoder_params": text_p,
            "autoencoder_params": ae_p,
            "spatiotemporal_unet_params": unet_p,
            "total_trainable_params": total
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        camera_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pixel_values: [B, C, F, H, W]
        input_ids: [B, seq_len]
        timesteps: [B]
        """
        latents = self.autoencoder.encode(pixel_values)
        noise = torch.randn_like(latents)

        # Broadcast noise addition across frames
        b, c, f, h, w = latents.shape
        latents_flat = latents.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        noise_flat = noise.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        t_flat = timesteps.repeat_interleave(f)

        noisy_flat = self.scheduler.add_noise(latents_flat, noise_flat, t_flat)
        noisy_latents = noisy_flat.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)

        context, _ = self.text_encoder(input_ids, camera_motion_id=camera_id)
        pred_noise = self.unet(noisy_latents, timesteps, context)
        return pred_noise, noise
