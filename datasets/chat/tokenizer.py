"""
Moon AI — Pure-Sovereign Byte-Level BPE Tokenizer
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Zero pretrained Hugging Face tokenizers.
Zero external third-party vocabulary files.
Trained from scratch from raw byte statistics and deterministic pair merges.
Byte-level fallback ensures 100% vocabulary coverage with zero <unk> tokens.
"""

import json
import os
from typing import List, Dict, Tuple, Optional, Any

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

class SovereignTokenizer:
    """
    Byte-Level Pair Encoding (BPE) tokenizer trained purely from scratch.
    Guarantees sovereign independence:
    - Base alphabet: 256 byte values (0-255).
    - Special tokens: chat delimiters (<|im_start|>, <|im_end|>, etc.).
    - Merges: trained iteratively from corpus frequency statistics.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None, merges: Optional[List[Tuple[int, int]]] = None):
        self.special_tokens = list(SPECIAL_TOKENS)
        self.special_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_special = {idx: tok for idx, tok in enumerate(self.special_tokens)}

        self.byte_offset = len(self.special_tokens)
        # Token IDs:
        # 0 .. len(SPECIAL_TOKENS)-1: special tokens
        # len(SPECIAL_TOKENS) .. len(SPECIAL_TOKENS)+255: raw bytes 0..255
        # len(SPECIAL_TOKENS)+256 .. : merged tokens

        if vocab is not None and merges is not None:
            self.vocab = dict(vocab)
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.merges = [tuple(m) for m in merges]
            self.bpe_ranks = {tuple(m): idx for idx, m in enumerate(self.merges)}
        else:
            self.vocab: Dict[str, int] = {}
            self.id_to_token: Dict[int, str] = {}
            self.merges: List[Tuple[int, int]] = []
            self.bpe_ranks: Dict[Tuple[int, int], int] = {}
            self._init_base_vocab()

    def _init_base_vocab(self):
        self.vocab.clear()
        self.id_to_token.clear()
        # Add special tokens
        for idx, tok in enumerate(self.special_tokens):
            self.vocab[tok] = idx
            self.id_to_token[idx] = tok

        # Add 256 bytes
        for b in range(256):
            token_id = self.byte_offset + b
            token_repr = f"<byte_{b:02x}>"
            self.vocab[token_repr] = token_id
            self.id_to_token[token_id] = token_repr

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, texts: List[str], target_vocab_size: int = 512, min_frequency: int = 2):
        """
        Trains byte-pair encoding merges on the provided corpus from scratch.
        """
        self._init_base_vocab()
        self.merges.clear()
        self.bpe_ranks.clear()

        # Convert each text to a list of byte token IDs
        sequences: List[List[int]] = []
        for text in texts:
            if not text:
                continue
            raw_bytes = text.encode("utf-8")
            sequences.append([self.byte_offset + b for b in raw_bytes])

        current_vocab_size = len(self.vocab)

        while current_vocab_size < target_vocab_size:
            # Count adjacent pairs across all sequences
            pair_counts: Dict[Tuple[int, int], int] = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])
            if best_count < min_frequency:
                break

            new_token_id = current_vocab_size
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = len(self.merges) - 1

            # Construct token representation
            tok_repr = f"({self.id_to_token[best_pair[0]]}+{self.id_to_token[best_pair[1]]})"
            self.vocab[tok_repr] = new_token_id
            self.id_to_token[new_token_id] = tok_repr
            current_vocab_size += 1

            # Apply merge to all sequences
            new_sequences: List[List[int]] = []
            for seq in sequences:
                i = 0
                new_seq: List[int] = []
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                        new_seq.append(new_token_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

    def encode_text(self, text: str) -> List[int]:
        """Encodes raw text into token IDs using learned byte-level BPE merges."""
        if not text:
            return []
        raw_bytes = text.encode("utf-8")
        tokens = [self.byte_offset + b for b in raw_bytes]

        if not self.merges or len(tokens) < 2:
            return tokens

        # Iteratively apply merges in order of priority (bpe_ranks)
        while len(tokens) >= 2:
            min_rank = None
            min_pair = None
            min_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None:
                    if min_rank is None or rank < min_rank:
                        min_rank = rank
                        min_pair = pair
                        min_idx = i

            if min_pair is None:
                break

            # Merge all occurrences of min_pair
            new_tokens = []
            i = 0
            merged_id = self.byte_offset + 256 + min_rank
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == min_pair:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode_tokens(self, tokens: List[int]) -> str:
        """Decodes token IDs back into UTF-8 text."""
        # Unpack merged tokens recursively into raw byte values
        byte_list: List[int] = []
        for tok in tokens:
            if tok in self.id_to_special:
                # Append special token string representation as UTF-8
                spec_str = self.id_to_special[tok]
                byte_list.extend(spec_str.encode("utf-8"))
            elif self.byte_offset <= tok < self.byte_offset + 256:
                byte_list.append(tok - self.byte_offset)
            else:
                # Merged token
                sub_bytes = self._unpack_token_to_bytes(tok)
                byte_list.extend(sub_bytes)

        return bytes(byte_list).decode("utf-8", errors="replace")

    def _unpack_token_to_bytes(self, token_id: int) -> List[int]:
        """Recursively unpacks a merged token ID into its constituent raw bytes."""
        if self.byte_offset <= token_id < self.byte_offset + 256:
            return [token_id - self.byte_offset]
        if token_id in self.id_to_special:
            return list(self.id_to_special[token_id].encode("utf-8"))

        merge_idx = token_id - (self.byte_offset + 256)
        if 0 <= merge_idx < len(self.merges):
            p0, p1 = self.merges[merge_idx]
            return self._unpack_token_to_bytes(p0) + self._unpack_token_to_bytes(p1)
        return []

    def format_chat_prompt(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        """Formats multi-turn messages into standard ChatML string."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"
        return formatted

    def encode_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> List[int]:
        """Encodes multi-turn conversation into token IDs with special token boundaries."""
        tokens: List[int] = []
        im_start = self.special_to_id["<|im_start|>"]
        im_end = self.special_to_id["<|im_end|>"]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            tokens.append(im_start)
            tokens.extend(self.encode_text(f"{role}\n"))
            tokens.extend(self.encode_text(content))
            tokens.append(im_end)
            tokens.extend(self.encode_text("\n"))

        if add_generation_prompt:
            tokens.append(im_start)
            tokens.extend(self.encode_text("assistant\n"))

        return tokens

    def save(self, filepath: str):
        """Saves tokenizer state to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "version": "1.0.0",
            "type": "byte_level_bpe",
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": self.merges
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SovereignTokenizer":
        """Loads tokenizer state from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab=data.get("vocab"), merges=data.get("merges"))
        return tok
