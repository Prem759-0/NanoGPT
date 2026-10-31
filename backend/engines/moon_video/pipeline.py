"""
backend/engines/moon_video/pipeline.py
Inference Pipeline for Moon Video Engine.
Supports Text-to-Video, Image-to-Video (First-Frame Conditioning), and Camera Motion Control.
"""
import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Union

from .model import MoonVideoModel
from .config import MoonVideoConfig

class MoonVideoPipeline:
    def __init__(self, model: MoonVideoModel):
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "MoonVideoPipeline":
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        provenance = ckpt.get("provenance")
        if provenance != "moon_ai_random_init":
            raise ValueError(f"Strict Ownership Violation: Checkpoint provenance '{provenance}' is invalid. Must be 'moon_ai_random_init'.")
        config_data = ckpt.get("config", {})
        config = MoonVideoConfig(**config_data) if isinstance(config_data, dict) else MoonVideoConfig()

        model = MoonVideoModel(config=config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(model)

    def _get_camera_id(self, camera_motion: str) -> Optional[torch.Tensor]:
        if camera_motion in self.config.camera_motions:
            idx = self.config.camera_motions.index(camera_motion) + 1
            return torch.tensor([idx], device=self.device)
        return torch.tensor([0], device=self.device)

    @torch.no_grad()
    def text_to_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        camera_motion: str = "static",
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Synthesizes a multi-frame video clip from a text prompt and camera trajectory.
        Returns a list of PIL Images (one per frame).
        """
        f = num_frames or self.config.num_frames
        device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        cam_id = self._get_camera_id(camera_motion)

        pos_tokens = self.model.text_encoder.tokenizer.encode(prompt).unsqueeze(0).to(device)
        pos_context, _ = self.model.text_encoder(pos_tokens, camera_motion_id=cam_id)

        neg_tokens = self.model.text_encoder.tokenizer.encode(negative_prompt).unsqueeze(0).to(device)
        neg_context, _ = self.model.text_encoder(neg_tokens, camera_motion_id=cam_id)

        lat_h = self.config.latent_height
        lat_w = self.config.latent_width
        latents = torch.randn(1, self.config.latent_channels, f, lat_h, lat_w, device=device)

        step_ratio = self.config.timesteps // num_inference_steps
        timesteps = list(range(0, self.config.timesteps, step_ratio))[::-1]

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device)

            if guidance_scale > 1.0:
                lat_input = torch.cat([latents, latents], dim=0)
                t_input = torch.cat([t_tensor, t_tensor], dim=0)
                ctx_input = torch.cat([neg_context, pos_context], dim=0)

                noise_pred = self.model.unet(lat_input, t_input, ctx_input)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.model.unet(latents, t_tensor, pos_context)

            # Flatten temporal dimension for scheduler DDIM step
            b, c, f_cnt, lh, lw = latents.shape
            lat_flat = latents.permute(0, 2, 1, 3, 4).reshape(b * f_cnt, c, lh, lw)
            noise_flat = noise_pred.permute(0, 2, 1, 3, 4).reshape(b * f_cnt, c, lh, lw)

            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            denoised_flat = self.model.scheduler.step(noise_flat, t, lat_flat, prev_timestep=prev_t)
            latents = denoised_flat.view(b, f_cnt, c, lh, lw).permute(0, 2, 1, 3, 4)

        # Decode video latents to pixel frames
        decoded_video = self.model.autoencoder.decode(latents)  # [1, 3, F, H, W]
        decoded_np = (decoded_video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)

        frames = []
        for frame_idx in range(f):
            frame_arr = (decoded_np[frame_idx] * 255).astype(np.uint8)
            frames.append(Image.fromarray(frame_arr))

        return frames

    @torch.no_grad()
    def image_to_video(
        self,
        first_frame: Image.Image,
        prompt: str,
        camera_motion: str = "dolly_in",
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Animates a still input image into a coherent video clip with camera motion.
        """
        f = num_frames or self.config.num_frames
        device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        # Preprocess and encode first frame
        img_resized = first_frame.resize((self.config.frame_width, self.config.frame_height))
        img_arr = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
        frame_tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        first_latent = self.model.autoencoder.encoder(frame_tensor)[0] * self.model.autoencoder.scale_factor

        # Initialize full video latents with first frame repeated + noise
        lat_h = self.config.latent_height
        lat_w = self.config.latent_width
        latents = torch.randn(1, self.config.latent_channels, f, lat_h, lat_w, device=device)
        # Lock first frame latent
        latents[:, :, 0, :, :] = first_latent

        cam_id = self._get_camera_id(camera_motion)
        pos_tokens = self.model.text_encoder.tokenizer.encode(prompt).unsqueeze(0).to(device)
        pos_context, _ = self.model.text_encoder(pos_tokens, camera_motion_id=cam_id)
        neg_tokens = self.model.text_encoder.tokenizer.encode("").unsqueeze(0).to(device)
        neg_context, _ = self.model.text_encoder(neg_tokens, camera_motion_id=cam_id)

        step_ratio = self.config.timesteps // num_inference_steps
        timesteps = list(range(0, self.config.timesteps, step_ratio))[::-1]

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device)
            lat_input = torch.cat([latents, latents], dim=0)
            t_input = torch.cat([t_tensor, t_tensor], dim=0)
            ctx_input = torch.cat([neg_context, pos_context], dim=0)

            noise_pred = self.model.unet(lat_input, t_input, ctx_input)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            b, c, f_cnt, lh, lw = latents.shape
            lat_flat = latents.permute(0, 2, 1, 3, 4).reshape(b * f_cnt, c, lh, lw)
            noise_flat = noise_pred.permute(0, 2, 1, 3, 4).reshape(b * f_cnt, c, lh, lw)

            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            denoised_flat = self.model.scheduler.step(noise_flat, t, lat_flat, prev_timestep=prev_t)
            latents = denoised_flat.view(b, f_cnt, c, lh, lw).permute(0, 2, 1, 3, 4)

            # Preserve conditioned first frame
            latents[:, :, 0, :, :] = first_latent

        decoded_video = self.model.autoencoder.decode(latents)
        decoded_np = (decoded_video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)

        frames = []
        for frame_idx in range(f):
            frame_arr = (decoded_np[frame_idx] * 255).astype(np.uint8)
            frames.append(Image.fromarray(frame_arr))
        return frames

    @staticmethod
    def export_to_gif(frames: List[Image.Image], output_path: str, fps: int = 8):
        """Exports a list of frames to an animated GIF."""
        if not frames:
            raise ValueError("No frames to export.")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        duration_ms = int(1000 / max(1, fps))
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0
        )
