"""
training/image/train.py
Runnable Sovereign Training Script for Moon Image Engine.
Trains latent diffusion network from scratch with random initialization.
Zero pretrained weights imported.
"""
import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engines.moon_image.config import MoonImageConfig
from backend.engines.moon_image.model import MoonImageModel
from training.common.seed import set_seed
from training.common.distributed import setup_device, get_vram_usage_mb
from training.common.metrics import MetricTracker
from training.common.checkpoint import save_checkpoint
from training.image.dataset import MoonImageDataset

def train(args):
    print("=" * 65)
    print("   MOON IMAGE SOVEREIGN TRAINING (Random Initialization)")
    print("=" * 65)

    set_seed(args.seed)
    device, dtype = setup_device(args.device)
    print(f"Device:            {device}")
    print(f"Compute Precision: {dtype}")
    print(f"Manifest:          {args.manifest}")
    print(f"Output Directory:  {args.output_dir}\n")

    # 1. Initialize Model from Scratch
    config = MoonImageConfig(
        image_height=args.image_size,
        image_width=args.image_size
    )
    model = MoonImageModel(config=config).to(device)
    param_counts = model.count_parameters()
    print("Model Architecture Initialized (Random Gaussian Initialization):")
    for k, v in param_counts.items():
        print(f"  • {k}: {v:,} parameters")
    print(f"  • Provenance Tag: {model.provenance}\n")

    # 2. Setup Datasets
    manifest_full = os.path.normpath(os.path.join(PROJECT_ROOT, args.manifest))
    train_dataset = MoonImageDataset(
        manifest_path=manifest_full,
        base_dir=PROJECT_ROOT,
        split="train",
        image_size=args.image_size
    )

    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in manifest '{args.manifest}' with split='train'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=False
    )

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    tracker = MetricTracker()
    total_steps = args.max_steps if not args.dry_run else 2
    step = 0
    t0 = time.time()

    model.train()
    print(f"Starting training run: max_steps={total_steps}, batch_size={args.batch_size}...")

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            step += 1

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            b = pixel_values.shape[0]

            # Sample random timesteps t in [0, timesteps-1]
            timesteps = torch.randint(0, config.timesteps, (b,), device=device).long()

            # Forward pass: predict noise added to image latents
            pred_noise, target_noise = model(pixel_values, input_ids, timesteps)
            loss = F.mse_loss(pred_noise, target_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            tracker.update("loss", loss.item())

            if step % args.log_every == 0 or step == total_steps:
                avg_loss = tracker.get_recent_avg("loss")
                vram_mb = get_vram_usage_mb()
                print(f"[Step {step:4d}/{total_steps:4d}] Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) | VRAM: {vram_mb:.1f} MB")

    elapsed = time.time() - t0
    final_loss = tracker.get_recent_avg("loss")
    print(f"\n✅ Training completed in {elapsed:.2f}s | Final Loss: {final_loss:.4f}")

    # 4. Save Checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, args.output_dir, "moon_image_v0.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        epoch=1,
        loss=final_loss,
        config_dict=config.__dict__,
        output_path=ckpt_path
    )
    print(f"✅ Checkpoint saved to: {os.path.relpath(ckpt_path, PROJECT_ROOT)}")
    return ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Moon Image Latent Diffusion Model.")
    parser.add_argument("--manifest", type=str, default="datasets/image/manifest.jsonl")
    parser.add_argument("--image_size", type=int, default=128)  # Default 128 for lightweight/fast convergence
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Run 2 quick steps to verify pipeline")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    args = parser.parse_args()

    train(args)
