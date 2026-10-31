"""
backend/engines/moon_image/config.py
Configuration and Hyperparameters for Moon Image Engine.
Strict Model Ownership: 100% Random Initialization, Zero External Weights.
"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class MoonImageConfig:
    # Model Identity & Provenance
    model_name: str = "moon-image-v0"
    version: str = "0.1.0"
    provenance: str = "moon_ai_random_init"
    strict_ownership: bool = True

    # Spatial Resolutions
    image_height: int = 256
    image_width: int = 256
    image_channels: int = 3
    latent_channels: int = 4
    downscale_factor: int = 8  # 256 -> 32x32 latent space

    # Text Conditioning Architecture
    text_vocab_size: int = 8192
    text_embed_dim: int = 256
    text_num_heads: int = 4
    text_num_layers: int = 4
    text_max_length: int = 77

    # Latent Diffusion UNet Architecture
    unet_in_channels: int = 4
    unet_out_channels: int = 4
    unet_base_channels: int = 64
    unet_channel_mult: Tuple[int, ...] = (1, 2, 4)
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: Tuple[int, ...] = (2, 4)  # At 16x16 and 8x8 in latent space
    unet_num_heads: int = 4
    cross_attention_dim: int = 256  # Matches text_embed_dim

    # Diffusion Schedule
    timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.0120
    beta_schedule: str = "scaled_linear"
    default_inference_steps: int = 25
    default_guidance_scale: float = 7.5

    @property
    def latent_height(self) -> int:
        return self.image_height // self.downscale_factor

    @property
    def latent_width(self) -> int:
        return self.image_width // self.downscale_factor
