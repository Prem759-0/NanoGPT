"""
training/image/evaluate.py
Evaluation and Quality Verification Suite for Moon Image Engine.
Computes validation loss and synthesizes visual evaluation samples.
"""
import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engines.moon_image.pipeline import MoonImagePipeline
from backend.engines.moon_image.model import MoonImageModel
from training.image.dataset import MoonImageDataset
from training.common.checkpoint import load_checkpoint

EVAL_PROMPTS = [
    "A futuristic hypercar driving along a cybernetic highway at sunset",
    "Close up portrait of an astronaut on Mars looking at Earth in the sky",
    "A serene zen garden with glowing cherry blossoms in the rain"
]

def evaluate(args):
    print("=" * 65)
    print("   MOON IMAGE BENCHMARK EVALUATION")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.checkpoint))
    if not os.path.isfile(ckpt_path):
        print(f"❌ Checkpoint not found at: {ckpt_path}")
        return {"status": "FAILED", "error": "Checkpoint missing"}

    pipeline = MoonImagePipeline.from_checkpoint(ckpt_path, device=device)
    model = pipeline.model
    model.eval()

    # 1. Validation Loss
    manifest_full = os.path.normpath(os.path.join(PROJECT_ROOT, args.manifest))
    val_dataset = MoonImageDataset(manifest_path=manifest_full, base_dir=PROJECT_ROOT, split="val")
    val_loss = 0.0

    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=1)
        with torch.no_grad():
            for batch in val_loader:
                pv = batch["pixel_values"].to(device)
                ids = batch["input_ids"].to(device)
                t = torch.tensor([500], device=device).long()
                pred, target = model(pv, ids, t)
                val_loss += F.mse_loss(pred, target).item()
        val_loss /= len(val_dataset)

    print(f"Validation Loss: {val_loss:.4f}")

    # 2. Sample Generation
    out_dir = os.path.join(PROJECT_ROOT, "evaluation", "image")
    os.makedirs(out_dir, exist_ok=True)
    generated_samples = []

    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"Generating sample {i+1}/{len(EVAL_PROMPTS)}: '{prompt}'...")
        img = pipeline.text_to_image(prompt=prompt, num_inference_steps=10, seed=42)
        sample_file = os.path.join(out_dir, f"sample_{i}.png")
        img.save(sample_file)
        generated_samples.append({
            "prompt": prompt,
            "path": os.path.relpath(sample_file, PROJECT_ROOT)
        })

    report = {
        "status": "COMPLETED",
        "model_name": model.config.model_name,
        "checkpoint": os.path.relpath(ckpt_path, PROJECT_ROOT),
        "validation_loss": round(val_loss, 4),
        "samples": generated_samples
    }

    report_path = os.path.join(out_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Evaluation completed. Report saved to: {os.path.relpath(report_path, PROJECT_ROOT)}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Moon Image model.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/moon_image_v0.pt")
    parser.add_argument("--manifest", type=str, default="datasets/image/manifest.jsonl")
    args = parser.parse_args()
    evaluate(args)
