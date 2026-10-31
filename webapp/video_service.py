import os
import math
import time
import hashlib
import random
from typing import Dict, Any, List, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    from backend.engines.moon_video import MoonVideoInferenceService
    MOON_VIDEO_ENGINE_AVAILABLE = True
except ImportError:
    MOON_VIDEO_ENGINE_AVAILABLE = False


class VideoService:
    """
    Moon Video Sovereign Spatio-Temporal Synthesis Engine (moon-video-v0).
    Features:
    - Sovereign Spatiotemporal UNet3D Architecture (Spatial cross-attention + 1D Temporal Self-Attention)
    - 8-DoF Camera Motion Conditioning (Pan, Tilt, Zoom, Dolly, Orbit, Crane)
    - Strict Model Ownership: Zero external video APIs (no Kling, Runway, Luma, ModelScope)
    - Weight Provenance: Strictly initialized from random weights ('moon_ai_random_init')
    - Truthful MODEL_NOT_TRAINED diagnostics when checkpoint is not yet trained
    - Architect: Prem Gaikwad (India 🇮🇳)
    """

    def __init__(self, output_dir: str, checkpoint_path: Optional[str] = None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = checkpoint_path or os.environ.get(
            "MOON_VIDEO_CHECKPOINT_PATH", "models/checkpoints/moon_video_v0.pt"
        )
        self.inference_service: Optional[MoonVideoInferenceService] = None
        if MOON_VIDEO_ENGINE_AVAILABLE:
            try:
                self.inference_service = MoonVideoInferenceService(checkpoint_path=self.checkpoint_path)
            except Exception:
                self.inference_service = None

    def generate_video(
        self,
        prompt: str,
        duration_sec: float = 3.0,
        fps: int = 15,
        width: int = 480,
        height: int = 320
    ) -> Dict[str, Any]:
        """
        Synthesizes a multi-frame spatio-temporal video sequence.
        If a trained checkpoint exists, runs MoonVideoPipeline.
        If untrained, returns a truthful diagnostic response with a local diagnostic artifact.
        """
        t0 = time.time()
        num_frames = int(duration_sec * fps)
        prompt_hash = hashlib.md5(f"{prompt}_{time.time()}".encode()).hexdigest()[:10]
        
        # Determine camera motion from prompt keywords
        p_low = prompt.lower()
        if "zoom" in p_low:
            camera_motion = "zoom_in"
        elif "pan" in p_low or "left" in p_low:
            camera_motion = "pan_left"
        elif "orbit" in p_low or "rotate" in p_low:
            camera_motion = "orbit"
        else:
            camera_motion = "static"

        # 1. Attempt trained neural checkpoint
        if self.inference_service and self.inference_service.is_loaded:
            gif_filename = f"video_{prompt_hash}.gif"
            gif_filepath = os.path.join(self.output_dir, gif_filename)
            res = self.inference_service.generate_video(
                prompt=prompt,
                camera_motion=camera_motion,
                num_frames=num_frames,
                export_gif_path=gif_filepath
            )
            if res.get("success"):
                return {
                    "video_url": f"/static/generated/{gif_filename}",
                    "filename": gif_filename,
                    "prompt": prompt,
                    "duration": duration_sec,
                    "fps": fps,
                    "mode": f"neural_spatiotemporal ({camera_motion})",
                    "status": "READY",
                    "model_name": "moon-video-v0",
                    "provenance": "moon_ai_random_init",
                    "message": "Synthesized with trained sovereign spatiotemporal UNet3D weights."
                }

        # 2. Sovereign Untrained Diagnostic Mode
        # Render clean diagnostic frames watermarked with sovereign status
        seed = sum(ord(c) for c in prompt)
        rng = random.Random(seed)

        frames = self._render_sovereign_diagnostic_frames(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            camera_motion=camera_motion,
            rng=rng
        )

        filename = f"video_{prompt_hash}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        if IMAGEIO_AVAILABLE:
            try:
                writer = imageio.get_writer(
                    filepath,
                    fps=fps,
                    codec="libx264",
                    format="FFMPEG",
                    ffmpeg_params=["-pix_fmt", "yuv420p"]
                )
                for f in frames:
                    writer.append_data(f)
                writer.close()
            except Exception:
                filename = f"video_{prompt_hash}.gif"
                filepath = os.path.join(self.output_dir, filename)
                pil_frames = [Image.fromarray(f) for f in frames]
                pil_frames[0].save(
                    filepath,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=int(1000 / fps),
                    loop=0
                )
        else:
            filename = f"video_{prompt_hash}.gif"
            filepath = os.path.join(self.output_dir, filename)
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                filepath,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps),
                loop=0
            )

        return {
            "video_url": f"/static/generated/{filename}",
            "filename": filename,
            "prompt": prompt,
            "duration": duration_sec,
            "fps": fps,
            "mode": f"sovereign_spatiotemporal_diagnostic ({camera_motion})",
            "status": "MODEL_NOT_TRAINED",
            "model_name": "moon-video-v0",
            "message": "Moon Video model initialized (Random Init). Requires training checkpoint to synthesize learned neural features.",
            "checkpoint_required": self.checkpoint_path,
            "training_command": "python training/video/train.py --manifest datasets/video/manifest.jsonl --output_dir checkpoints/moon_video",
            "provenance": "moon_ai_random_init"
        }

    def _render_sovereign_diagnostic_frames(
        self,
        prompt: str,
        num_frames: int,
        width: int,
        height: int,
        camera_motion: str,
        rng: random.Random
    ) -> List[np.ndarray]:
        frames = []
        p_disp = prompt[:45] + ("..." if len(prompt) > 45 else "")

        for frame_idx in range(num_frames):
            t = frame_idx / max(1, num_frames - 1)
            img = Image.new("RGB", (width, height), color=(8, 12, 22))
            draw = ImageDraw.Draw(img)

            # Animated grid to illustrate camera motion
            offset = int(t * 40)
            for x in range(0, width, 40):
                x_pos = (x + offset) % width
                draw.line([(x_pos, 0), (x_pos, height - 55)], fill=(15, 25, 45), width=1)
            for y in range(0, height - 55, 30):
                draw.line([(0, y), (width, y)], fill=(15, 25, 45), width=1)

            # Central focal target
            cx = int(width / 2 + math.sin(t * math.pi * 2) * 20)
            cy = int((height - 55) / 2)
            draw.ellipse([(cx - 25, cy - 25), (cx + 25, cy + 25)], outline=(56, 189, 248), width=2)
            draw.line([(cx - 35, cy), (cx + 35, cy)], fill=(245, 158, 11), width=1)
            draw.line([(cx, cy - 35), (cx, cy + 35)], fill=(245, 158, 11), width=1)

            # Diagnostic labels
            draw.text((15, 15), "MOON VIDEO v0 — SOVEREIGN SPATIO-TEMPORAL UNet3D", fill=(240, 240, 245))
            draw.text((15, 35), "Status: MODEL_NOT_TRAINED (Random Initialization Mode)", fill=(245, 158, 11))
            draw.text((15, 55), f"Motion Conditioning: {camera_motion.upper()} | Frame {frame_idx + 1}/{num_frames}", fill=(148, 163, 184))
            draw.text((15, 75), f"Prompt: \"{p_disp}\"", fill=(226, 232, 240))

            # Studio footer
            draw.rectangle([(0, height - 50), (width, height)], fill=(5, 8, 15))
            draw.line([(0, height - 50), (width, height - 50)], fill=(56, 189, 248), width=2)
            draw.text((12, height - 42), "🌙 Moon AI Sovereign Video Studio • Chief Architect: Prem Gaikwad (India 🇮🇳)", fill=(241, 245, 249))
            draw.text((12, height - 24), "Train weights: python training/video/train.py --manifest datasets/video/manifest.jsonl", fill=(148, 163, 184))

            frames.append(np.array(img))

        return frames
