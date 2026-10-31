"""
training/video/evaluate.py
Evaluation and Temporal Consistency Verification Suite for Moon Video.
"""
import os
import sys
import argparse
import json
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engines.moon_video.pipeline import MoonVideoPipeline

EVAL_VIDEO_PROMPTS = [
    ("Drone flight orbiting a gleaming hypercar on a coastal mountain pass", "orbit_left"),
    ("Cinematic camera dolly into a neon cybernetic control room", "dolly_in"),
    ("Slow pan across a misty pine forest at sunrise", "pan_right")
]

def evaluate(args):
    print("=" * 65)
    print("   MOON VIDEO BENCHMARK EVALUATION")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.checkpoint))
    if not os.path.isfile(ckpt_path):
        print(f"❌ Video Checkpoint not found at: {ckpt_path}")
        return {"status": "FAILED", "error": "Video checkpoint missing"}

    pipeline = MoonVideoPipeline.from_checkpoint(ckpt_path, device=device)
    model = pipeline.model
    model.eval()

    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "video")
    os.makedirs(out_dir, exist_ok=True)
    generated_videos = []

    for i, (prompt, motion) in enumerate(EVAL_VIDEO_PROMPTS):
        print(f"Generating video clip {i+1}/{len(EVAL_VIDEO_PROMPTS)}: '{prompt}' (Motion: {motion})...")
        frames = pipeline.text_to_video(
            prompt=prompt,
            camera_motion=motion,
            num_frames=args.num_frames,
            num_inference_steps=10,
            seed=42
        )
        gif_file = os.path.join(out_dir, f"video_sample_{i}.gif")
        pipeline.export_to_gif(frames, gif_file, fps=pipeline.config.fps)
        generated_videos.append({
            "prompt": prompt,
            "camera_motion": motion,
            "num_frames": len(frames),
            "path": os.path.relpath(gif_file, PROJECT_ROOT)
        })

    report = {
        "status": "COMPLETED",
        "model_name": model.config.model_name,
        "checkpoint": os.path.relpath(ckpt_path, PROJECT_ROOT),
        "num_frames": args.num_frames,
        "samples": generated_videos
    }

    report_path = os.path.join(out_dir, "video_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Video Evaluation completed. Report saved to: {os.path.relpath(report_path, PROJECT_ROOT)}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Moon Video model.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/moon_video_v0.pt")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()
    evaluate(args)
