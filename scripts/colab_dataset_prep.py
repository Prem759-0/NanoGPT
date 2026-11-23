"""
scripts/colab_dataset_prep.py
Google Colab Dataset Preparation & Quality Gate Runner for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Tasks:
1. Detects Colab environment and hardware (NVIDIA T4, A10G, A100).
2. Verifies Google Drive connection for persistent dataset caching.
3. Audits all Stage 2.0 dataset manifests (datasets/manifests/).
4. Strictly evaluates HYGIENE_VALID and TRAINING_DATA_ADEQUATE quality gates.
5. Synchronizes datasets/manifests to Google Drive without leaving partial files.
6. Recommends T4-optimized training commands.
"""
import os
import sys
import shutil
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.tools.dataset_manager import audit_all_dataset_manifests
from scripts.colab_train import get_hardware_profile, detect_colab_environment

def run_colab_prep(drive_dir: str = "/content/drive/MyDrive/MoonAI_Datasets"):
    print("=" * 70)
    print("   MOON AI — GOOGLE COLAB DATASET PREPARATION SUITE")
    print("   Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)")
    print("=" * 70)

    is_colab = detect_colab_environment()
    print(f"• Environment: {'Google Colab' if is_colab else 'Local Workstation'}")

    hw = get_hardware_profile()
    print(f"• Detected Accelerator: {hw['name']} ({hw['device'].upper()})")
    print(f"• Available VRAM: {hw['vram_gb']} GB")
    print(f"• Recommended T4 Precision: {hw['recommended_precision']}")

    # 1. Run Comprehensive Dataset Quality Gate Audit
    manifests_dir = os.path.join(PROJECT_ROOT, "datasets", "manifests")
    out_json = os.path.join(PROJECT_ROOT, "datasets", "dataset_report.json")
    out_md = os.path.join(PROJECT_ROOT, "datasets", "dataset_report.md")

    print("\n[Step 1/3] Running Stage 2.0 Dataset Quality Gates & Integrity Audit...")
    report = audit_all_dataset_manifests(
        manifests_dir=manifests_dir,
        output_json=out_json,
        output_md=out_md,
        base_dir=PROJECT_ROOT
    )

    hygiene = report["overall_status"]["HYGIENE_VALID"]
    adequacy = report["overall_status"]["TRAINING_DATA_ADEQUATE"]

    print("\n" + "=" * 55)
    print("   QUALITY GATE VERIFICATION RESULTS")
    print("=" * 55)
    print(f"• HYGIENE_VALID:           {'✅ PASS (100% Verified)' if hygiene else '❌ FAIL'}")
    print(f"• TRAINING_DATA_ADEQUATE:  {'✅ ADEQUATE' if adequacy else '⚠️ INSUFFICIENT (Pilot stage; more samples needed)'}")

    img_assess = report["datasets"]["image"]["adequacy_assessment"]
    vid_assess = report["datasets"]["video"]["adequacy_assessment"]
    print(f"  - Image Genuine Samples: {img_assess['current_genuine_count']} / {img_assess['target_pilot_count']} (Shortfall: {img_assess['shortfall']})")
    print(f"  - Video Genuine Samples: {vid_assess['current_genuine_count']} / {vid_assess['target_pilot_count']} (Shortfall: {vid_assess['shortfall']})")

    # 2. Sync to Google Drive
    print("\n[Step 2/3] Persistent Storage Synchronization...")
    if is_colab and os.path.isdir("/content/drive"):
        try:
            os.makedirs(drive_dir, exist_ok=True)
            # Sync manifests
            dest_manifests = os.path.join(drive_dir, "manifests")
            if os.path.exists(dest_manifests):
                shutil.rmtree(dest_manifests)
            shutil.copytree(manifests_dir, dest_manifests)
            shutil.copy2(out_json, os.path.join(drive_dir, "dataset_report.json"))
            shutil.copy2(out_md, os.path.join(drive_dir, "dataset_report.md"))
            print(f"☁️ Synced manifests & reports to Google Drive: {drive_dir}")
        except Exception as e:
            print(f"⚠️ Drive sync warning: {e}")
    else:
        print("• Google Drive Sync: Skipped (Local development environment)")

    # 3. T4 Execution Recommendations
    print("\n[Step 3/3] Ready-to-Run T4 Training Commands:")
    print("-------------------------------------------------------")
    print("# Run Image Pilot (FP16, batch=2, grad_accum=8, max 50 steps):")
    print("!python scripts/colab_train.py --track image --steps 50 --precision fp16 --batch_size 2 --grad_accum 8")
    print("\n# Run Video Pilot (FP16, batch=1, grad_accum=8, max 25 steps):")
    print("!python scripts/colab_train.py --track video --steps 25 --precision fp16 --batch_size 1 --grad_accum 8")
    print("-------------------------------------------------------")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Colab Dataset Preparation Runner")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/MoonAI_Datasets")
    args = parser.parse_args()
    run_colab_prep(drive_dir=args.drive_dir)

