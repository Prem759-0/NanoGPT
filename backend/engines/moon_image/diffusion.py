"""
backend/engines/moon_image/diffusion.py
Original Latent Diffusion Network (MoonUNet2D) and Diffusion Noise Scheduler.
Features learned cross-attention conditioning from MoonTextEncoder.
Trained from scratch from random initialization.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class SpatialCrossAttention(nn.Module):
    """Cross-attention mechanism between spatial latent features and text tokens."""
    def __init__(self, in_channels: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.norm = nn.GroupNorm(8, in_channels)
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Linear(context_dim, in_channels)
        self.v_proj = nn.Linear(context_dim, in_channels)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        norm_x = self.norm(x)

        q = self.q_proj(norm_x).reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # [b, heads, hw, dim]
        k = self.k_proj(context).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)     # [b, heads, seq, dim]
        v = self.v_proj(context).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)     # [b, heads, seq, dim]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_probs, v)  # [b, heads, hw, dim]
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return residual + self.out_proj(out)


class ResBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        t = self.time_proj(F.silu(t_emb))
        h = h + t.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class MoonUNet2D(nn.Module):
    """
    Original Latent UNet with spatial cross-attention conditioning.
    Predicts the noise component added to latent representations.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        context_dim: int = 256,
        time_embed_dim: int = 256
    ):
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbedding(time_embed_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        curr_channels = base_channels
        self.skip_channels = [base_channels]

        for mult in channel_mult:
            out_ch = base_channels * mult
            res = ResBlock2D(curr_channels, out_ch, time_embed_dim)
            attn = SpatialCrossAttention(out_ch, context_dim)
            down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            self.down_blocks.append(nn.ModuleList([res, attn, down]))
            curr_channels = out_ch
            self.skip_channels.append(out_ch)

        # Mid block
        self.mid_res1 = ResBlock2D(curr_channels, curr_channels, time_embed_dim)
        self.mid_attn = SpatialCrossAttention(curr_channels, context_dim)
        self.mid_res2 = ResBlock2D(curr_channels, curr_channels, time_embed_dim)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            skip_ch = self.skip_channels.pop()
            res = ResBlock2D(curr_channels + skip_ch, out_ch, time_embed_dim)
            attn = SpatialCrossAttention(out_ch, context_dim)
            up = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
            self.up_blocks.append(nn.ModuleList([res, attn, up]))
            curr_channels = out_ch

        self.final_norm = nn.GroupNorm(8, curr_channels)
        self.final_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        h = self.init_conv(x)

        skips = [h]
        for res, attn, down in self.down_blocks:
            h = res(h, t_emb)
            h = attn(h, context)
            skips.append(h)
            h = down(h)

        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, t_emb)

        for res, attn, up in self.up_blocks:
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = res(h, t_emb)
            h = attn(h, context)
            h = up(h)

        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)

        return self.final_conv(F.silu(self.final_norm(h)))


class MoonDiffusionScheduler:
    """
    DDPM and DDIM Gaussian Noise Scheduler.
    Controls forward diffusion noise addition and reverse denoising steps.
    """
    def __init__(self, timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Scaled linear schedule
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def add_noise(self, original_latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward diffusion process q(x_t | x_0)."""
        device = original_latents.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]).sqrt()

        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)

        return sqrt_alpha_prod * original_latents + sqrt_one_minus_alpha_prod * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: Optional[int] = None
    ) -> torch.Tensor:
        """Performs one step of reverse DDIM denoising."""
        device = sample.device
        alphas_cumprod = self.alphas_cumprod.to(device)

        prev_t = timestep - 1 if prev_timestep is None else prev_timestep
        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)

        # Predict original x_0
        pred_original_sample = (sample - (1.0 - alpha_prod_t).sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_sample_direction = (1.0 - alpha_prod_t_prev).sqrt() * model_output

        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        return prev_sample
