"""
Moon AI — Chat Dataset Loader for Sovereign Autoregressive Training
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Implements Assistant-Only Loss Masking:
- System and user prompt tokens are masked out with target = -1 (ignore_index).
- Assistant response tokens are active targets.
- Padding tokens are masked out with target = -1.
"""

import json
import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple

from datasets.chat.tokenizer import SovereignTokenizer

class SovereignChatDataset(Dataset):
    """
    Loads multi-turn ChatML conversations from manifest, encodes with
    SovereignTokenizer, and yields autoregressive (x, y) tensor batches
    with strict assistant-only response loss masking.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer: SovereignTokenizer,
        block_size: int = 512,
        pad_token_id: int = 1
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.samples: List[Tuple[List[int], List[bool]]] = []

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        raw_records: List[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        # Encode conversations with assistant-only loss masks
        for rec in raw_records:
            messages = rec.get("messages", [])
            tokens, loss_mask = self.tokenizer.encode_chat_with_loss_mask(messages)
            if len(tokens) < 2:
                continue
            self.samples.append((tokens, loss_mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, loss_mask = self.samples[idx]

        target_len = self.block_size + 1
        if len(tokens) >= target_len:
            tokens_chunk = tokens[:target_len]
            mask_chunk = loss_mask[:target_len]
        else:
            pad_len = target_len - len(tokens)
            tokens_chunk = tokens + [self.pad_token_id] * pad_len
            mask_chunk = loss_mask + [False] * pad_len

        x = torch.tensor(tokens_chunk[:-1], dtype=torch.long)

        # Assistant-only targets:
        # At position t, x[t] predicts tokens_chunk[t + 1].
        # Target y[t] is tokens_chunk[t + 1] ONLY if mask_chunk[t + 1] is True.
        # User prompt tokens, system prompts, turn headers, and padding receive target = -1.
        y_list = []
        for t in range(self.block_size):
            next_token = tokens_chunk[t + 1]
            is_target = mask_chunk[t + 1]
            if is_target and next_token != self.pad_token_id:
                y_list.append(next_token)
            else:
                y_list.append(-1)

        y = torch.tensor(y_list, dtype=torch.long)
        return x, y
