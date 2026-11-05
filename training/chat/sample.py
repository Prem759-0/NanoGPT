"""
Moon AI — Sovereign Chat Generation and Sampling Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Loads a trained Moon Chat checkpoint, initializes the model architecture,
and runs text generation with configurable temperature, top-k filtering, and diagnostics.
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPTConfig, GPT
from datasets.chat.tokenizer import SovereignTokenizer


def generate_chat_response(
    model: GPT,
    tokenizer: SovereignTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cpu"
) -> str:
    """Generates completion for a given user prompt using the trained Moon model."""
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.encode_chat(messages)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(1e-5, temperature)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)

    generated_text = tokenizer.decode(x[0].tolist())
    return generated_text


def load_chat_model(checkpoint_path: str, device: str = "cpu") -> tuple[GPT, Dict[str, Any]]:
    """Loads model from checkpoint and returns (model, checkpoint_payload)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["model_args"]
    gpt_cfg = GPTConfig(
        block_size=args["block_size"],
        vocab_size=args["vocab_size"],
        n_layer=args["n_layer"],
        n_head=args["n_head"],
        n_embd=args["n_embd"],
        dropout=args.get("dropout", 0.0),
        bias=args.get("bias", False)
    )
    model = GPT(gpt_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="Moon AI Chat Sampling Utility")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/chat/moon_chat_pilot_v0.pt")
    parser.add_argument("--tokenizer", type=str, default="models/chat/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="What is Python?")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    tok_path = os.path.abspath(args.tokenizer)

    tokenizer = SovereignTokenizer.load(tok_path)
    model, ckpt = load_chat_model(ckpt_path, device=args.device)

    print("=" * 60)
    print("   MOON AI -- CHAT MODEL INFERENCE")
    print("=" * 60)
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Status:     {ckpt.get('status')}")
    print(f"Iterations: {ckpt.get('iter_num')}")
    print(f"Val Loss:   {ckpt.get('best_val_loss', 'N/A')}")
    print("-" * 60)
    print(f"Prompt:     {args.prompt}")
    output = generate_chat_response(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
    print(f"Output:\n{output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

