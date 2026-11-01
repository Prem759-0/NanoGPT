"""
Moon AI — Chat Dataset Loader for Sovereign Autoregressive Training
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY
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
    SovereignTokenizer, and yields autoregressive (x, y) tensor batches.
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
        self.samples: List[torch.Tensor] = []

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        raw_records: List[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        # Encode conversations
        for rec in raw_records:
            messages = rec.get("messages", [])
            tokens = self.tokenizer.encode_chat(messages)
            if len(tokens) < 2:
                continue
            self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.samples[idx]

        # Truncate or pad to block_size + 1 (for causal next-token prediction)
        target_len = self.block_size + 1
        if len(tokens) >= target_len:
            chunk = tokens[:target_len]
        else:
            pad_len = target_len - len(tokens)
            chunk = torch.cat([tokens, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])

        x = chunk[:-1]
        y = chunk[1:]

        # Mask padding in loss computation with -1
        y = torch.where(y == self.pad_token_id, torch.tensor(-1, dtype=torch.long), y)

        return x, y
