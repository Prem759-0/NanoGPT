"""
backend/engines/moon_video/config.py
Configuration and Hyperparameters for Moon Video Engine.
Strict Model Ownership: 100% Random Initialization, Zero External Weights.
"""
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class MoonVideoConfig:
    # Model Identity & Provenance
    model_name: str = "moon-video-v0"
    version: str = "0.1.0"
    provenance: str = "moon_ai_random_init"
    strict_ownership: bool = True

    # Video Temporal & Spatial Dimensions
    num_frames: int = 16
    fps: int = 8
    frame_height: int = 128
    frame_width: int = 128
    channels: int = 3
    latent_channels: int = 4
    downscale_factor: int = 8  # 128 -> 16x16 spatial latent

    # Camera Motion Types supported by conditioning
    camera_motions: Tuple[str, ...] = ("static", "orbit_left", "orbit_right", "dolly_in", "dolly_out", "pan_left", "pan_right", "tilt_up", "tilt_down")

    # Text Conditioning Architecture
    text_vocab_size: int = 8192
    text_embed_dim: int = 256
    text_num_heads: int = 4
    text_num_layers: int = 4
    text_max_length: int = 77

    # Spatiotemporal UNet Architecture
    unet_in_channels: int = 4
    unet_out_channels: int = 4
    unet_base_channels: int = 48
    unet_channel_mult: Tuple[int, ...] = (1, 2, 4)
    temporal_attention_heads: int = 4
    cross_attention_dim: int = 256

    # Diffusion Schedule
    timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.0120
    default_inference_steps: int = 25
    default_guidance_scale: float = 7.5

    @property
    def latent_height(self) -> int:
        return self.frame_height // self.downscale_factor

    @property
    def latent_width(self) -> int:
        return self.frame_width // self.downscale_factor
