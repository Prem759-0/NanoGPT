"""
Moon AI — Chat Dataset Loader with Multi-Window Chunking & Loss Masking
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Implements:
1. Multi-window sequential chunking for long conversations (zero silent truncation).
2. Strict assistant-only response loss masking across every window.
3. Proper padding and prompt token masking with target = -1 (ignore_index).
4. Full metadata and record tracking (record_id, window_idx, supervised_tokens).
5. Accurate accounting of supervised training tokens.
"""

import json
import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Set

from datasets.chat.tokenizer import SovereignTokenizer

class SovereignChatDataset(Dataset):
    """
    Loads multi-turn ChatML conversations from manifest, encodes with
    SovereignTokenizer, and yields autoregressive (x, y) tensor batches
    with strict assistant-only response loss masking and multi-window chunking.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer: SovereignTokenizer,
        block_size: int = 512,
        pad_token_id: int = 1,
        stride: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.stride = stride or block_size  # Contiguous autoregressive sliding stride

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.sample_metadata: List[Dict[str, Any]] = []
        self.unique_record_ids: Set[str] = set()
        self.total_supervised_tokens: int = 0
        self.raw_record_count: int = 0

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        raw_records: List[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        self.raw_record_count = len(raw_records)
        target_len = self.block_size + 1

        for rec in raw_records:
            rec_id = rec.get("id", f"record_{len(self.unique_record_ids)}")
            messages = rec.get("messages", [])
            tokens, loss_mask = self.tokenizer.encode_chat_with_loss_mask(messages)
            if len(tokens) < 2:
                continue

            self.unique_record_ids.add(rec_id)
            L = len(tokens)

            # Window extraction
            if L <= target_len:
                # Single window with end padding
                pad_len = target_len - L
                tokens_chunk = tokens + [self.pad_token_id] * pad_len
                mask_chunk = loss_mask + [False] * pad_len

                x = torch.tensor(tokens_chunk[:-1], dtype=torch.long)
                y_list = []
                supervised_in_window = 0
                for t in range(self.block_size):
                    nxt = tokens_chunk[t + 1]
                    is_active = mask_chunk[t + 1]
                    if is_active and nxt != self.pad_token_id:
                        y_list.append(nxt)
                        supervised_in_window += 1
                    else:
                        y_list.append(-1)

                y = torch.tensor(y_list, dtype=torch.long)
                self.samples.append((x, y))
                self.sample_metadata.append({
                    "record_id": rec_id,
                    "window_idx": 0,
                    "total_windows": 1,
                    "token_length": L,
                    "supervised_tokens": supervised_in_window
                })
                self.total_supervised_tokens += supervised_in_window
            else:
                # Multi-window chunking: slice conversation into contiguous windows
                windows_data = []
                start = 0
                while start < L - 1:
                    end = min(start + target_len, L)
                    sub_tokens = tokens[start:end]
                    sub_mask = loss_mask[start:end]

                    if len(sub_tokens) < target_len:
                        pad_len = target_len - len(sub_tokens)
                        sub_tokens = sub_tokens + [self.pad_token_id] * pad_len
                        sub_mask = sub_mask + [False] * pad_len

                    x = torch.tensor(sub_tokens[:-1], dtype=torch.long)
                    y_list = []
                    supervised_in_window = 0
                    for t in range(self.block_size):
                        nxt = sub_tokens[t + 1]
                        is_active = sub_mask[t + 1]
                        if is_active and nxt != self.pad_token_id:
                            y_list.append(nxt)
                            supervised_in_window += 1
                        else:
                            y_list.append(-1)

                    y = torch.tensor(y_list, dtype=torch.long)
                    windows_data.append((x, y, supervised_in_window))

                    start += self.stride
                    if start + 1 >= L:
                        break

                num_windows = len(windows_data)
                for w_idx, (x, y, sup_tokens) in enumerate(windows_data):
                    self.samples.append((x, y))
                    self.sample_metadata.append({
                        "record_id": rec_id,
                        "window_idx": w_idx,
                        "total_windows": num_windows,
                        "token_length": L,
                        "supervised_tokens": sup_tokens
                    })
                    self.total_supervised_tokens += sup_tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Returns tracking metadata for a given sample window."""
        return self.sample_metadata[idx]

    def get_total_supervised_tokens(self) -> int:
        """Returns the total number of supervised target tokens (y != -1) across all sample windows."""
        return self.total_supervised_tokens

    def get_unique_record_count(self) -> int:
        """Returns the number of unique conversation records represented in this dataset."""
        return len(self.unique_record_ids)
