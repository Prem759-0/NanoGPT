"""
backend/engines/moon_video/diffusion.py
Spatiotemporal Latent Diffusion UNet (MoonUNet3D) and Diffusion Noise Scheduler.
Combines spatial cross-attention with learned temporal attention for motion synthesis.
Trained from scratch from random initialization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..moon_image.diffusion import SinusoidalTimestepEmbedding, SpatialCrossAttention, ResBlock2D, MoonDiffusionScheduler
from .temporal_model import TemporalTransformerBlock

class MoonUNet3D(nn.Module):
    """
    Spatiotemporal UNet for Video Diffusion.
    Integrates spatial 2D convolutions + text cross-attention with 1D temporal attention blocks.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 48,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        context_dim: int = 256,
        time_embed_dim: int = 256
    ):
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbedding(time_embed_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling stages
        self.down_blocks = nn.ModuleList()
        curr_channels = base_channels
        self.skip_channels = [base_channels]

        for mult in channel_mult:
            out_ch = base_channels * mult
            res = ResBlock2D(curr_channels, out_ch, time_embed_dim)
            attn = SpatialCrossAttention(out_ch, context_dim)
            temp = TemporalTransformerBlock(out_ch)
            down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            self.down_blocks.append(nn.ModuleList([res, attn, temp, down]))
            curr_channels = out_ch
            self.skip_channels.append(out_ch)

        # Mid block
        self.mid_res1 = ResBlock2D(curr_channels, curr_channels, time_embed_dim)
        self.mid_attn = SpatialCrossAttention(curr_channels, context_dim)
        self.mid_temp = TemporalTransformerBlock(curr_channels)
        self.mid_res2 = ResBlock2D(curr_channels, curr_channels, time_embed_dim)

        # Upsampling stages
        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            skip_ch = self.skip_channels.pop()
            res = ResBlock2D(curr_channels + skip_ch, out_ch, time_embed_dim)
            attn = SpatialCrossAttention(out_ch, context_dim)
            temp = TemporalTransformerBlock(out_ch)
            up = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
            self.up_blocks.append(nn.ModuleList([res, attn, temp, up]))
            curr_channels = out_ch

        self.final_norm = nn.GroupNorm(8, curr_channels)
        self.final_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, F, H, W]
        timesteps: [B]
        context: [B, seq_len, context_dim]
        Returns: [B, C, F, H, W] predicted noise
        """
        b, c, f, h, w = x.shape
        # Flatten frames into batch dimension for spatial operations: [B * F, C, H, W]
        spatial_x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)

        # Replicate timesteps and context across all frames
        t_rep = timesteps.repeat_interleave(f)
        ctx_rep = context.repeat_interleave(f, dim=0)

        t_emb = self.time_embed(t_rep)
        feat = self.init_conv(spatial_x)

        skips = [feat]
        for res, attn, temp, down in self.down_blocks:
            feat = res(feat, t_emb)
            feat = attn(feat, ctx_rep)
            feat = temp(feat, num_frames=f)
            skips.append(feat)
            feat = down(feat)

        feat = self.mid_res1(feat, t_emb)
        feat = self.mid_attn(feat, ctx_rep)
        feat = self.mid_temp(feat, num_frames=f)
        feat = self.mid_res2(feat, t_emb)

        for res, attn, temp, up in self.up_blocks:
            skip = skips.pop()
            if feat.shape[2:] != skip.shape[2:]:
                feat = F.interpolate(feat, size=skip.shape[2:], mode="nearest")
            feat = torch.cat([feat, skip], dim=1)
            feat = res(feat, t_emb)
            feat = attn(feat, ctx_rep)
            feat = temp(feat, num_frames=f)
            feat = up(feat)

        if feat.shape[2:] != spatial_x.shape[2:]:
            feat = F.interpolate(feat, size=spatial_x.shape[2:], mode="bilinear", align_corners=False)

        pred_noise = self.final_conv(F.silu(self.final_norm(feat)))
        # Reshape back to [B, C, F, H, W]
        return pred_noise.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
