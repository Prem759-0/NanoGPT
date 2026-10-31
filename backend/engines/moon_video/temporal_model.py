"""
backend/engines/moon_video/temporal_model.py
Temporal Modeling Layers for Moon Video Engine.
Learns frame-to-frame temporal relationships, smooth camera motion, and object consistency.
Trained from scratch from random initialization.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSelfAttention(nn.Module):
    """
    1D Self-Attention across the temporal (frame) axis.
    Input shape: [batch_size * num_frames, channels, height, width] or [batch_size, channels, num_frames, height, width]
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        x: [B * F, C, H, W]
        """
        bf, c, h, w = x.shape
        b = bf // num_frames
        residual = x

        # [B * F, C, H, W] -> [B, F, C, H, W] -> [B, H, W, F, C]
        norm_x = self.norm(x)
        norm_x = norm_x.view(b, num_frames, c, h, w).permute(0, 3, 4, 1, 2)  # [B, H, W, F, C]
        norm_x = norm_x.reshape(b * h * w, num_frames, c)                     # [B*HW, F, C]

        q = self.q_proj(norm_x).view(b * h * w, num_frames, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(norm_x).view(b * h * w, num_frames, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(norm_x).view(b * h * w, num_frames, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        probs = F.softmax(scores, dim=-1)

        out = torch.matmul(probs, v)  # [B*HW, heads, F, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(b * h * w, num_frames, c)
        out = self.out_proj(out)      # [B*HW, F, C]

        # Reshape back to [B * F, C, H, W]
        out = out.view(b, h, w, num_frames, c).permute(0, 3, 4, 1, 2)  # [B, F, C, H, W]
        out = out.reshape(bf, c, h, w)
        return residual + out


class TemporalConvBlock(nn.Module):
    """
    1D Temporal Convolution layer across the frame dimension.
    Learns continuous optical flow dynamics and momentum.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1d = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        bf, c, h, w = x.shape
        b = bf // num_frames
        residual = x

        # [B * F, C, H, W] -> [B, C, F, H*W] -> [B * HW, C, F]
        h_feat = x.view(b, num_frames, c, h * w).permute(0, 3, 2, 1)  # [B, HW, C, F]
        h_feat = h_feat.reshape(b * h * w, c, num_frames)

        h_feat = self.conv1d(h_feat)
        h_feat = self.norm(h_feat)
        h_feat = self.act(h_feat)

        # Reshape back
        out = h_feat.view(b, h * w, c, num_frames).permute(0, 3, 2, 1) # [B, F, C, HW]
        out = out.permute(0, 1, 2, 3).view(bf, c, h, w)
        return residual + out


class TemporalTransformerBlock(nn.Module):
    """
    Composite block pairing temporal convolution with temporal self-attention.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.conv = TemporalConvBlock(channels)
        self.attn = TemporalSelfAttention(channels, num_heads=num_heads)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        x = self.conv(x, num_frames)
        x = self.attn(x, num_frames)
        return x
