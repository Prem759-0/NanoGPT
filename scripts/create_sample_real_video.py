"""
scripts/create_sample_real_video.py
Generates a real benchmark video file (.mp4) with true temporal motion
for verifying Moon Video real-video decoding and training pipelines.
"""
import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw
import imageio

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_benchmark_mp4(output_path: str, num_frames: int = 24, fps: int = 8, width: int = 128, height: int = 128):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frames = []

    for f_idx in range(num_frames):
        t = f_idx / max(1, num_frames - 1)
        im = Image.new("RGB", (width, height), color=(10, 15, 25))
        draw = ImageDraw.Draw(im)

        # Dynamic moving geometry
        cx = int(width / 2 + math.sin(t * math.pi * 2) * (width * 0.25))
        cy = int(height / 2 + math.cos(t * math.pi * 2) * (height * 0.2))
        r = int(width * 0.18)

        # Draw glowing orbital circle
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(245, 158, 11), outline=(56, 189, 248), width=2)

        # Secondary counter-rotating element
        cx2 = int(width / 2 - math.sin(t * math.pi * 2) * (width * 0.2))
        cy2 = int(height / 2 - math.cos(t * math.pi * 2) * (height * 0.15))
        r2 = int(width * 0.1)
        draw.ellipse([(cx2 - r2, cy2 - r2), (cx2 + r2, cy2 + r2)], fill=(168, 85, 247), outline=(236, 72, 153), width=1)

        # Moving horizon line
        h_y = int(height * 0.75 + math.sin(t * math.pi * 4) * 4)
        draw.line([(0, h_y), (width, h_y)], fill=(30, 45, 70), width=1)

        frames.append(np.array(im))

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"✅ Generated real benchmark video: {output_path} ({num_frames} frames, {fps} fps, {width}x{height})")
    return output_path

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "videos")
    out_file = os.path.join(out_dir, "sample_real_video.mp4")
    generate_benchmark_mp4(out_file)
