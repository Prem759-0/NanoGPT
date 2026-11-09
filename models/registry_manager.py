"""
models/registry_manager.py
Dynamic Model Registry and Provenance Verification Manager for Moon AI.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Derives exact parameter counts dynamically from instantiated models,
validates sovereign provenance ('moon_ai_random_init'), and maintains models/registry.json.
"""
import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "models", "registry.json")

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Returns exact count of parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dynamic_model_stats() -> Dict[str, Dict[str, Any]]:
    """Instantiates each Moon AI sovereign model and returns exact dynamically derived stats."""
    stats = {}

    # 1. Moon Image Model
    try:
        from backend.engines.moon_image.model import MoonImageModel
        from backend.engines.moon_image.config import MoonImageConfig
        img_cfg = MoonImageConfig(image_height=128, image_width=128)
        img_model = MoonImageModel(config=img_cfg)
        param_breakdown = img_model.count_parameters()
        stats["moon-image-v0"] = {
            "trainable_parameters": param_breakdown["total_trainable_params"],
            "breakdown": param_breakdown,
            "architecture": "MoonUNet2D + MoonTextEncoder + MoonImageAutoencoder",
            "provenance": "moon_ai_random_init"
        }
    except Exception as e:
        stats["moon-image-v0"] = {"error": str(e)}

    # 2. Moon Video Model
    try:
        from backend.engines.moon_video.model import MoonVideoModel
        from backend.engines.moon_video.config import MoonVideoConfig
        vid_cfg = MoonVideoConfig(num_frames=8, frame_height=128, frame_width=128)
        vid_model = MoonVideoModel(config=vid_cfg)
        param_breakdown = vid_model.count_parameters()
        stats["moon-video-v0"] = {
            "trainable_parameters": param_breakdown["total_trainable_params"],
            "breakdown": param_breakdown,
            "architecture": "MoonUNet3D + MoonVideoTextEncoder + MoonVideoAutoencoder",
            "provenance": "moon_ai_random_init"
        }
    except Exception as e:
        stats["moon-video-v0"] = {"error": str(e)}

    # 3. Moon LLM Scratch
    try:
        ckpt_path = os.path.join(PROJECT_ROOT, "out-assistant", "ckpt.pt")
        if os.path.isfile(ckpt_path):
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = raw.get("model", raw.get("model_state_dict", {}))
            llm_params = sum(p.numel() for p in state_dict.values())
        else:
            from model import GPT, GPTConfig
            cfg = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=128, bias=False, vocab_size=97, dropout=0.0)
            llm_model = GPT(cfg)
            llm_params = count_trainable_parameters(llm_model)

        stats["moon-llm-scratch"] = {
            "trainable_parameters": llm_params,
            "architecture": "NanoGPT (4 layers, 4 heads, 128 embed dim, 128 block size)",
            "provenance": "moon_ai_random_init"
        }
    except Exception as e:
        stats["moon-llm-scratch"] = {"error": str(e)}

    return stats

def verify_checkpoint_provenance(checkpoint_path: str) -> bool:
    """Verifies that a checkpoint has sovereign 'moon_ai_random_init' provenance."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    provenance = data.get("provenance")
    if provenance != "moon_ai_random_init":
        raise ValueError(
            f"Provenance validation failed! Expected 'moon_ai_random_init', found '{provenance}'. "
            "Third-party pretrained checkpoints are strictly prohibited under sovereign ownership policy."
        )
    return True

def update_registry(save: bool = True) -> Dict[str, Any]:
    """Dynamically updates models/registry.json with precise parameter counts and metadata."""
    if os.path.isfile(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {
            "registry_version": "2.0.0",
            "ownership_policy": "STRICT_SOVEREIGN_RANDOM_INIT_ONLY",
            "rules": {
                "prohibit_third_party_weights": True,
                "prohibit_external_apis": True,
                "require_random_initialization": True,
                "require_manifest_hash_provenance": True
            },
            "models": {}
        }

    stats = get_dynamic_model_stats()

    for model_key, model_stat in stats.items():
        if "error" in model_stat:
            continue
        if model_key in registry.get("models", {}):
            entry = registry["models"][model_key]
            entry["trainable_parameters"] = model_stat["trainable_parameters"]
            entry["architecture"] = model_stat["architecture"]
            entry["provenance"] = model_stat["provenance"]
            if "breakdown" in model_stat:
                entry["parameter_breakdown"] = model_stat["breakdown"]

    if save:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        print(f"✅ Registry updated with dynamic parameter counts: {REGISTRY_PATH}")

    return registry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moon AI Dynamic Model Registry Manager")
    parser.add_argument("--update", action="store_true", help="Update models/registry.json with dynamic parameter counts")
    parser.add_argument("--verify", type=str, help="Verify provenance of a given checkpoint file")
    args = parser.parse_args()

    if args.verify:
        try:
            verify_checkpoint_provenance(args.verify)
            print(f"✅ Checkpoint verified: '{args.verify}' has valid sovereign provenance ('moon_ai_random_init').")
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            sys.exit(1)
    else:
        update_registry(save=True)

