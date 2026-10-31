"""
backend/engines/moon_image/pipeline.py
Inference Pipeline for Moon Image Engine.
Supports Text-to-Image, Image-to-Image, and Inpainting.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from PIL import Image
import numpy as np

from .model import MoonImageModel
from .config import MoonImageConfig

class MoonImagePipeline:
    def __init__(self, model: MoonImageModel):
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "MoonImagePipeline":
        """Loads a Moon Image checkpoint verified for sovereign random initialization."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        provenance = ckpt.get("provenance")
        if provenance != "moon_ai_random_init":
            raise ValueError(f"Strict Ownership Violation: Checkpoint provenance '{provenance}' is invalid. Must be 'moon_ai_random_init'.")
        config_data = ckpt.get("config", {})
        config = MoonImageConfig(**config_data) if isinstance(config_data, dict) else MoonImageConfig()

        model = MoonImageModel(config=config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(model)

    @torch.no_grad()
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Synthesizes a visual image from text prompt via learned latent diffusion.
        """
        device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Encode conditional prompt
        pos_tokens = self.model.text_encoder.tokenizer.encode(prompt).unsqueeze(0).to(device)
        pos_context, _ = self.model.text_encoder(pos_tokens)

        # 2. Encode unconditional prompt (for classifier-free guidance)
        neg_tokens = self.model.text_encoder.tokenizer.encode(negative_prompt).unsqueeze(0).to(device)
        neg_context, _ = self.model.text_encoder(neg_tokens)

        # 3. Sample initial random latent
        lat_h = self.config.latent_height
        lat_w = self.config.latent_width
        latents = torch.randn(1, self.config.latent_channels, lat_h, lat_w, device=device)

        # 4. Schedule timesteps
        step_ratio = self.config.timesteps // num_inference_steps
        timesteps = list(range(0, self.config.timesteps, step_ratio))[::-1]

        # 5. Reverse Denoising Loop
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)

            if guidance_scale > 1.0:
                # Combined batch for classifier-free guidance
                lat_input = torch.cat([latents, latents], dim=0)
                t_input = torch.cat([t_tensor, t_tensor], dim=0)
                ctx_input = torch.cat([neg_context, pos_context], dim=0)

                noise_pred = self.model.unet(lat_input, t_input, ctx_input)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.model.unet(latents, t_tensor, pos_context)

            # DDIM reverse step
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            latents = self.model.scheduler.step(noise_pred, t, latents, prev_timestep=prev_t)

        # 6. Decode latent into RGB pixel image
        decoded = self.model.autoencoder.decode(latents)
        image_np = (decoded.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        image_uint8 = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_uint8)

    @torch.no_grad()
    def image_to_image(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.75,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Transforms a reference image conditioned on a new text prompt."""
        device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        # Preprocess input image to [-1, 1] tensor
        img_resized = image.resize((self.config.image_width, self.config.image_height))
        img_arr = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

        # Encode reference image to latent space
        init_latents = self.model.autoencoder.encode(img_tensor)

        # Determine start timestep based on strength
        start_step = int(num_inference_steps * (1.0 - strength))
        step_ratio = self.config.timesteps // num_inference_steps
        timesteps = list(range(0, self.config.timesteps, step_ratio))[::-1]
        active_timesteps = timesteps[start_step:]

        # Add noise up to the starting timestep
        start_t = active_timesteps[0] if active_timesteps else 0
        noise = torch.randn_like(init_latents)
        latents = self.model.scheduler.add_noise(init_latents, noise, torch.tensor([start_t], device=device))

        # Encode text prompts
        pos_tokens = self.model.text_encoder.tokenizer.encode(prompt).unsqueeze(0).to(device)
        pos_context, _ = self.model.text_encoder(pos_tokens)
        neg_tokens = self.model.text_encoder.tokenizer.encode("").unsqueeze(0).to(device)
        neg_context, _ = self.model.text_encoder(neg_tokens)

        # Denoise remaining steps
        for i, t in enumerate(active_timesteps):
            t_tensor = torch.tensor([t], device=device)
            lat_input = torch.cat([latents, latents], dim=0)
            t_input = torch.cat([t_tensor, t_tensor], dim=0)
            ctx_input = torch.cat([neg_context, pos_context], dim=0)

            noise_pred = self.model.unet(lat_input, t_input, ctx_input)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_t = active_timesteps[i + 1] if i + 1 < len(active_timesteps) else -1
            latents = self.model.scheduler.step(noise_pred, t, latents, prev_timestep=prev_t)

        decoded = self.model.autoencoder.decode(latents)
        image_np = (decoded.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        return Image.fromarray((image_np * 255).astype(np.uint8))
